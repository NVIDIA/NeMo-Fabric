# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed model accounting and invocation summaries with explicit coverage.

This module never backfills from ATIF or interprets absent values as zero.
Observed values are deduplicated evidence, not guaranteed mathematical bounds.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any, TypedDict
from typing import NotRequired

from nemo_fabric.types import FabricMapping
from nemo_fabric.types import ModelUsage
from nemo_fabric.types import (
    _MODEL_USAGE_DIMENSIONS,
    _usage_value,
)


class DimensionSummary(TypedDict):
    """An aggregate together with its explicit invocation coverage."""

    expected_count: int
    complete_count: int
    incomplete_count: int
    missing_result_count: int
    complete_total: int | None
    observed_sum: int | None
    reasons: dict[str, int]
    aggregation_error: NotRequired[str]


class UsageSummary(TypedDict):
    """Coverage of an explicit invocation manifest, including missing results."""

    unit: str
    dimensions: dict[str, DimensionSummary]
    pricing: NotRequired[dict[str, Any]]
    pricing_groups: NotRequired[list[dict[str, Any]]]


def _supported_catalog_basis(basis: Mapping[str, Any] | None) -> bool:
    return basis is not None and all(
        basis.get(key) == value
        for key, value in {
            "basis": "catalog",
            "currency": "USD",
            "normalization_version": "cache_reasoning_inclusive_v1",
            "assumption_policy": "applicable_tariff_no_missing_counts_v1",
        }.items()
    )


def summarize_usage(
    results: Iterable[Mapping[str, Any]],
    *,
    expected: Iterable[tuple[str, str]],
) -> UsageSummary:
    """Summarize invocations against an explicit expected identity manifest.

    Include failed and cancelled results. Identical duplicates count once;
    conflicts and unexpected results are errors. A complete total exists only
    when every expected invocation has a complete value. Observations remain
    visible alongside coverage. No observations is null, never zero.
    """

    identities = list(expected)
    if any(
        not isinstance(key, tuple)
        or len(key) != 2
        or any(not isinstance(v, str) or not v for v in key)
        for key in identities
    ):
        raise ValueError(
            "expected identities must be (runtime_id, invocation_id) string tuples"
        )
    expected_set = set(identities)
    if len(expected_set) != len(identities):
        raise ValueError("duplicate expected identity")
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for result in results:
        record = (
            result.to_mapping() if isinstance(result, FabricMapping) else dict(result)
        )
        key = (record.get("runtime_id"), record.get("invocation_id"))
        if key not in expected_set:
            raise ValueError(f"unexpected invocation identity: {key!r}")
        if key in unique and unique[key] != record:
            raise ValueError(f"conflicting duplicate invocation: {key!r}")
        unique[key] = record
    parsed = {
        key: ModelUsage.from_mapping(record["model_usage"])
        if record.get("model_usage") is not None
        else None
        for key, record in unique.items()
    }
    price_groups: dict[tuple[Any, ...], list[tuple[str, str]]] = {}
    contributing_bases: set[tuple[Any, ...]] = set()
    for identity, usage in parsed.items():
        if usage is not None:
            basis = usage.cost.pricing
            if basis is not None and _supported_catalog_basis(basis):
                key = tuple(basis.get(field) for field in sorted(basis._fields))
                price_groups.setdefault(key, []).append(identity)
                estimate = usage.cost.estimated
                if (
                    estimate.amount_nano_usd is not None
                    or estimate.observed_nano_usd is not None
                ):
                    contributing_bases.add(key)
    # An absent estimate adds no incompatible amount. Retain unpriced members
    # of contributing groups so their incomplete coverage remains visible.
    price_groups = {
        key: group for key, group in price_groups.items() if key in contributing_bases
    }
    dimensions: dict[str, DimensionSummary] = {}
    for dimension in _MODEL_USAGE_DIMENSIONS:
        complete: list[int] = []
        observed: list[int] = []
        reasons: Counter[str] = Counter()
        for model_usage in parsed.values():
            if model_usage is None:
                reasons["legacy_unverified"] += 1
                continue
            if dimension == "estimated_cost":
                if not _supported_catalog_basis(model_usage.cost.pricing):
                    reasons["pricing_unverified"] += 1
                    continue
            value = _usage_value(model_usage, dimension)
            if value is not None:
                complete.append(value)
                observed.append(value)
            else:
                reasons.update(
                    {
                        gap.reason
                        for gap in model_usage.gaps
                        if dimension in gap.dimensions
                    }
                    or {"missing_usage"}
                )
                observation = _usage_value(model_usage, dimension, observed=True)
                if observation is not None:
                    observed.append(observation)
        complete_total = (
            sum(complete) if identities and len(complete) == len(identities) else None
        )
        observed_sum = sum(observed) if observed else None
        overflow = any(
            value is not None and value > (1 << 64) - 1
            for value in (complete_total, observed_sum)
        )
        if complete_total is not None and complete_total > (1 << 64) - 1:
            complete_total = None
        if observed_sum is not None and observed_sum > (1 << 64) - 1:
            observed_sum = None
        missing = len(identities) - len(unique)
        if missing:
            reasons["missing_result"] = missing
        dimensions[dimension] = {
            "expected_count": len(identities),
            "complete_count": len(complete),
            "incomplete_count": len(unique) - len(complete),
            "missing_result_count": missing,
            "complete_total": complete_total,
            "observed_sum": observed_sum,
            "reasons": dict(sorted(reasons.items())),
        }
        if overflow:
            dimensions[dimension]["aggregation_error"] = "uint64_overflow"
    result: UsageSummary = {"unit": "invocation", "dimensions": dimensions}
    if len(price_groups) == 1:
        identity = next(iter(price_groups.values()))[0]
        result["pricing"] = parsed[identity].cost.pricing.to_mapping()
    elif len(price_groups) > 1:
        # Each value can be complete within its own basis, while their sum is
        # not a comparable catalog amount. Keep coverage and expose groups.
        dimension = dimensions["estimated_cost"]
        dimension["complete_total"] = dimension["observed_sum"] = None
        dimension["aggregation_error"] = "incompatible_pricing"
        result["pricing_groups"] = [
            summarize_usage([unique[key] for key in group], expected=group)
            for group in price_groups.values()
        ]
    return result
