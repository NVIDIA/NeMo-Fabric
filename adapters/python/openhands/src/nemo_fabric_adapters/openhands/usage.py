# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenHands completion-log evidence; shared code owns reduction and pricing."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import json
from typing import Any, Iterator

from nemo_fabric_adapter_contract.models import (
    AgentUsage,
    GapReason,
    ReportedUsage,
    UsageGap,
    UsageDimension,
)
from nemo_fabric_adapter_contract.usage import UsageObservation, usage_from_observations

_OWNER: ContextVar[InvocationReports | None] = ContextVar(
    "fabric_openhands_usage_owner", default=None
)


def _counter(value: Any, *, positive: bool = False) -> int | None:
    if (
        type(value) is int
        and (0 < value if positive else 0 <= value)
        and value < 1 << 64
    ):
        return value
    return None


def _detail_counter(
    usage: dict[str, Any], groups: tuple[str, ...], key: str
) -> int | None:
    counts = {_counter((usage.get(group) or {}).get(key)) for group in groups}
    counts.discard(None)
    return counts.pop() if len(counts) == 1 else None


@dataclass
class InvocationReports:
    """One invocation's terminal model observations, frozen before return."""

    source: CompletionReports
    responses: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    gaps: list[GapReason] = field(default_factory=list)
    closed: bool = False

    def usage(
        self, metrics: Any, *, events: list[Any] | tuple[Any, ...] = ()
    ) -> AgentUsage:
        self.closed = True
        records = list(metrics.token_usages)
        latencies = list(metrics.response_latencies)
        response_ids = [response_id for response_id, _ in self.responses]
        if response_ids != [latency.response_id for latency in latencies]:
            self.gaps.append(GapReason.SCOPE_UNVERIFIED)
        # A latency is recorded even for a missing or explicit 0/0 report, while
        # token_usages omits those responses. Match individual responses by ID.
        by_id: dict[str, list[Any]] = {}
        for record in records:
            by_id.setdefault(record.response_id, []).append(record)
        if any(record.response_id not in response_ids for record in records):
            self.gaps.append(GapReason.SCOPE_UNVERIFIED)
        observations = []
        for response_id, usage in self.responses:
            matches = by_id.get(response_id, [])
            record = matches.pop(0) if matches else None
            prompt = _counter(
                usage.get("prompt_tokens", usage.get("input_tokens")), positive=True
            )
            output = _counter(
                usage.get("completion_tokens", usage.get("output_tokens")),
                positive=True,
            )
            if record is not None:
                prompt = (
                    prompt
                    if prompt is not None
                    else _counter(record.prompt_tokens, positive=True)
                )
                output = (
                    output
                    if output is not None
                    else _counter(record.completion_tokens, positive=True)
                )
            cache = _detail_counter(
                usage,
                ("prompt_tokens_details", "input_tokens_details"),
                "cached_tokens",
            )
            if cache is None and self.source.omitted_cache_is_zero:
                cache = 0
            writes = _detail_counter(
                usage,
                ("prompt_tokens_details", "input_tokens_details"),
                "cache_creation_tokens",
            )
            reasoning = _detail_counter(
                usage,
                ("completion_tokens_details", "output_tokens_details"),
                "reasoning_tokens",
            )
            if prompt is not None and cache is not None and cache > prompt:
                cache = None
            if prompt is not None and writes is not None and writes > prompt:
                writes = None
            presence_gaps = []
            if self.source.anthropic_usage:
                # LiteLLM fills absent Anthropic cache buckets with zero and
                # estimates/clamps reasoning. The callback cannot distinguish
                # those values from literal provider reports.
                cache = cache if cache is not None and cache > 0 else None
                writes = writes if writes is not None and writes > 0 else None
                reasoning = None
                uncached = _detail_counter(
                    usage, ("prompt_tokens_details",), "text_tokens"
                )
                input_known = (
                    cache is not None
                    and writes is not None
                    and uncached is not None
                    and uncached > 0
                )
                dimensions = []
                if not input_known:
                    dimensions.extend(
                        [
                            UsageDimension.INPUT_TOKENS,
                            UsageDimension.PEAK_REQUEST_INPUT_TOKENS,
                            UsageDimension.TOTAL_TOKENS,
                        ]
                    )
                if usage.get("_anthropic_iterations"):
                    # The serialized list preserves source entries, but this
                    # callback has no qualification of server-iteration coverage.
                    dimensions.extend(
                        [
                            UsageDimension.INPUT_TOKENS,
                            UsageDimension.CACHED_INPUT_TOKENS,
                            UsageDimension.CACHE_WRITE_INPUT_TOKENS,
                            UsageDimension.OUTPUT_TOKENS,
                            UsageDimension.TOTAL_TOKENS,
                            UsageDimension.PEAK_REQUEST_INPUT_TOKENS,
                        ]
                    )
                if dimensions:
                    presence_gaps.append(
                        UsageGap(
                            dimensions=list(dict.fromkeys(dimensions)),
                            reason=GapReason.SCOPE_UNVERIFIED,
                        )
                    )
            observations.append(
                UsageObservation(
                    gaps=tuple(presence_gaps),
                    model=self.source.model,
                    provider=self.source.provider,
                    is_model_call=True,
                    tokens_aligned=True,
                    reported=ReportedUsage(
                        input_tokens=prompt,
                        cached_input_tokens=cache,
                        cache_write_input_tokens=writes,
                        input_tokens_include_cache=True if prompt is not None else None,
                        output_tokens=output,
                        reasoning_tokens=reasoning,
                        output_tokens_include_reasoning=True
                        if output is not None
                        else None,
                        peak_request_input_tokens=prompt,
                    ),
                )
            )
        if any(type(event).__name__ == "InterruptEvent" for event in events):
            # The event does not identify whether model I/O or a tool was
            # interrupted, so it establishes a coverage limit, not a model start.
            self.gaps.append(GapReason.SCOPE_UNVERIFIED)
        if any(
            getattr(event, "tool_name", None) == "inspect_image_with_vision"
            for event in events
        ):
            # SDK1.50 can auto-attach this profile-backed tool. Its separate LLM
            # has no callback on the constructed primary/condenser client.
            self.gaps.append(GapReason.SCOPE_UNVERIFIED)
        observations.extend(
            UsageObservation(
                model=self.source.model,
                provider=self.source.provider,
                reported=ReportedUsage(),
                gap=gap,
            )
            for gap in set(self.gaps)
        )
        return usage_from_observations(observations)


class CompletionReports:
    """Use the supported callback without retaining completion-log payloads."""

    def __init__(
        self,
        model: str,
        provider: str | None,
        *,
        omitted_cache_is_zero: bool = False,
        anthropic_usage: bool = False,
    ) -> None:
        self.model = model
        self.provider = provider
        self.omitted_cache_is_zero = omitted_cache_is_zero
        self.anthropic_usage = anthropic_usage or provider == "anthropic"

    @contextmanager
    def invocation(self) -> Iterator[InvocationReports]:
        owner = InvocationReports(self)
        token = _OWNER.set(owner)
        try:
            yield owner
        finally:
            owner.closed = True
            _OWNER.reset(token)

    def retry(self, attempt: int, maximum: int, error: BaseException | None) -> None:
        """Record intermediate failures consumed by the SDK retry decorator."""
        owner = _OWNER.get()
        if owner is not None and owner.source is self and not owner.closed:
            if not _counted_empty_reply(owner, type(error).__name__, str(error)):
                owner.gaps.append(GapReason.MISSING_USAGE)

    def record(self, filename: str, log_data: str) -> None:
        owner = _OWNER.get()
        if owner is None or owner.source is not self or owner.closed:
            return
        try:
            document = json.loads(log_data)
            if filename.endswith("-error.json"):
                error = document.get("error") or {}
                # This exact SDK validation branch logs usage before raising.
                # Other no-response errors occur before telemetry and are gaps.
                counted_empty_reply = _counted_empty_reply(
                    owner, error.get("type"), str(error.get("message", ""))
                )
                if not counted_empty_reply:
                    owner.gaps.append(GapReason.MISSING_USAGE)
                return
            response = document.get("response") or {}
            usage = response.get("usage") or {}
            if not isinstance(usage, dict):
                raise ValueError("invalid completion usage")
            # Retain counts only. Never keep request text, replies, errors or cost
            # values whose provider versus price-table origin is unspecified.
            fields = {
                key: usage[key]
                for key in (
                    "prompt_tokens",
                    "input_tokens",
                    "completion_tokens",
                    "output_tokens",
                )
                if key in usage
            }
            for key in (
                "prompt_tokens_details",
                "input_tokens_details",
                "completion_tokens_details",
                "output_tokens_details",
            ):
                details = usage.get(key)
                if details is not None and not isinstance(details, dict):
                    raise ValueError("invalid completion token details")
                if details is not None:
                    fields[key] = {
                        name: details[name]
                        for name in (
                            "cached_tokens",
                            "cache_creation_tokens",
                            "reasoning_tokens",
                            "text_tokens",
                        )
                        if name in details
                    }
            if self.anthropic_usage:
                fields["_anthropic_iterations"] = bool(usage.get("iterations"))
            owner.responses.append((str(response.get("id", "")), fields))
        except (ValueError, TypeError, AttributeError):
            owner.gaps.append(GapReason.SCOPE_UNVERIFIED)


def _counted_empty_reply(
    owner: InvocationReports, error_type: str | None, message: str
) -> bool:
    return (
        error_type == "LLMNoResponseError"
        and message.startswith("Response choices is less than 1. Response:")
        and bool(owner.responses)
    )
