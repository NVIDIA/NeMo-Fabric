# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared raw-evidence reduction for Python collectors.

Normalization, aggregate/model reconciliation and pricing belong to core.
Collectors supply settled snapshots and explicit coverage gaps, never inferred
zero usage from HTTP status. One record represents one deduplicated source unit.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
import math

from nemo_fabric_adapter_contract.models import AccountedModel
from nemo_fabric_adapter_contract.models import AgentUsage
from nemo_fabric_adapter_contract.models import GapReason
from nemo_fabric_adapter_contract.models import ReportedUsage
from nemo_fabric_adapter_contract.models import UsageAccounting
from nemo_fabric_adapter_contract.models import UsageDimension
from nemo_fabric_adapter_contract.models import UsageGap
from nemo_fabric_adapter_contract.models import UsageRequest

ACCOUNTING_KEY = "nemo_fabric.usage_accounting.v1"
DETAIL_KEY = "nemo_fabric.detailed_token_usage.v1"


@dataclass(frozen=True, kw_only=True)
class UsageObservation:
    """One source unit's deduplicated raw snapshot and completeness boundary."""

    reported: ReportedUsage
    model: str | None = None
    provider: str | None = None
    gap: GapReason | None = None
    tokens_aligned: bool = False
    is_model_call: bool = False
    gaps: tuple[UsageGap, ...] = ()


def _reduce(
    records: list[UsageObservation],
) -> tuple[ReportedUsage, ReportedUsage, list[UsageGap]]:
    complete: dict[str, object] = {}
    observed: dict[str, object] = {}
    gaps: dict[GapReason, list[UsageDimension]] = defaultdict(list)
    for dimension in UsageDimension:
        key = dimension.value
        values = [getattr(record.reported, key) for record in records]
        known = [value for value in values if value is not None]
        total = None
        if known:
            if dimension is UsageDimension.COST_USD:
                amount = math.fsum(known)
                if math.isfinite(amount) and 0 <= amount * 1e9 < 1 << 64:
                    total = amount
            elif dimension is UsageDimension.PEAK_REQUEST_INPUT_TOKENS:
                total = max(known)
            else:
                amount = sum(known)
                if amount <= (1 << 64) - 1:
                    total = amount
        if total is not None:
            observed[key] = total
        causes = {record.gap for record in records if record.gap is not None}
        causes.update(
            gap.reason
            for record in records
            for gap in record.gaps
            if dimension in gap.dimensions
        )
        if len(known) != len(records):
            causes.add(GapReason.MISSING_USAGE)
        if known and total is None:
            causes.add(GapReason.SCOPE_UNVERIFIED)
        if total is not None and not causes:
            complete[key] = total
        elif not causes:
            causes.add(GapReason.SCOPE_UNVERIFIED if not records else GapReason.MISSING_USAGE)
        for cause in sorted(causes):
            gaps[cause].append(dimension)
    for flag, counter in (
        ("input_tokens_include_cache", "input_tokens"),
        ("output_tokens_include_reasoning", "output_tokens"),
    ):
        flags = {
            getattr(record.reported, flag)
            for record in records
            if getattr(record.reported, counter) is not None
        }
        if len(flags) == 1 and None not in flags:
            observed[flag] = next(iter(flags))
            if counter in complete:
                complete[flag] = observed[flag]
        elif len(
            [
                record
                for record in records
                if getattr(record.reported, counter) is not None
            ]
        ) == 1 and flags == {None}:
            # One raw source remains useful even when its convention is unknown.
            # Core withholds normalized values; do not discard literal evidence.
            pass
        elif counter in observed:
            # Mixed conventions cannot be combined as one raw input/output count.
            complete.pop(counter, None)
            observed.pop(counter, None)
            dimension = UsageDimension(counter)
            if not all(
                record.is_model_call and record.model is not None for record in records
            ):
                if dimension not in gaps[GapReason.SCOPE_UNVERIFIED]:
                    gaps[GapReason.SCOPE_UNVERIFIED].append(dimension)
    return (
        ReportedUsage.from_mapping(complete),
        ReportedUsage.from_mapping(observed),
        [
            UsageGap(dimensions=dimensions, reason=reason)
            for reason, dimensions in sorted(gaps.items())
        ],
    )


def _observed_tokens_aligned(records: list[UsageObservation]) -> bool:
    # Shape alone never proves scope. The collector must first identify a
    # coherent source snapshot; matching presence then keeps reductions aligned.
    fields = [
        "input_tokens",
        "cached_input_tokens",
        "cache_write_input_tokens",
        "output_tokens",
        "peak_request_input_tokens",
    ]
    if any(r.reported.output_tokens_include_reasoning is False for r in records):
        fields.append("reasoning_tokens")
    relevant = [
        r
        for r in records
        if any(getattr(r.reported, field) is not None for field in fields)
    ]
    return (
        bool(relevant)
        and all(r.tokens_aligned and not r.gaps for r in relevant)
        and all(
            len({getattr(r.reported, field) is not None for r in relevant}) == 1
            for field in fields
        )
    )


def _request_usage(record: UsageObservation) -> ReportedUsage:
    # A scoped observation is not a complete per-call count for that dimension.
    # Keep the literal value in model observations without allowing requests to
    # re-certify it or use it as a complete pricing component.
    fields = record.reported.to_mapping()
    for gap in record.gaps:
        for dimension in gap.dimensions:
            fields.pop(dimension.value, None)
    return ReportedUsage.from_mapping(fields)


def usage_from_observations(records: Iterable[UsageObservation]) -> AgentUsage:
    """Reduce snapshots once, preserving field presence and independently known values.

    Empty source coverage is unverified rather than assumed to mean zero calls.
    A collector that proves no model work occurred can report an explicit zero
    snapshot. This helper does not change scheduling or wait for late work.
    """

    records = list(records)
    reported, observed, gaps = _reduce(records)
    groups: dict[tuple[str | None, str], list[UsageObservation]] = defaultdict(list)
    for record in records:
        if record.model is not None:
            groups[(record.provider, record.model)].append(record)
    models = []
    for (provider, model), group in sorted(
        groups.items(), key=lambda item: (item[0][0] or "", item[0][1])
    ):
        complete_model, observed_model, model_gaps = _reduce(group)
        models.append(
            AccountedModel(
                model=model,
                provider=provider,
                reported=complete_model,
                observed=observed_model,
                observed_tokens_aligned=_observed_tokens_aligned(group),
                gaps=model_gaps,
                requests=[
                    UsageRequest(
                        usage=_request_usage(record),
                        gap=record.gap,
                        tokens_aligned=record.tokens_aligned,
                    )
                    for record in group
                ]
                if all(record.is_model_call for record in group)
                else [],
                requests_exhaustive=all(record.is_model_call for record in group),
            )
        )
    accounting = UsageAccounting(
        models_exhaustive=bool(records)
        and all(record.model is not None and record.gap is None for record in records),
        gaps=gaps,
        observed=observed,
        models=models,
    )
    base_keys = {
        "input_tokens",
        "cached_input_tokens",
        "input_tokens_include_cache",
        "output_tokens",
        "total_tokens",
        "cost_usd",
    }
    complete_mapping = reported.to_mapping()
    detail_keys = {
        "input_tokens",
        "cached_input_tokens",
        "cache_write_input_tokens",
        "input_tokens_include_cache",
        "output_tokens",
        "reasoning_tokens",
        "output_tokens_include_reasoning",
        "peak_request_input_tokens",
    }
    detail = {
        key: value for key, value in complete_mapping.items() if key in detail_keys
    }
    return AgentUsage(
        **{key: value for key, value in complete_mapping.items() if key in base_keys},
        extensions={ACCOUNTING_KEY: accounting.to_mapping(), DETAIL_KEY: detail},
    )
