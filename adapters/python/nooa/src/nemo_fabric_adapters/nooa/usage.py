# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Invocation-owned evidence from NOOA's constructed model clients.

The pinned NOOA transport accessor is isolated in ``install_model_observer``.
JSON response hooks preserve provider field presence before LiteLLM zero-fills
usage. They do not replace clients, callbacks, retries or background scheduling.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import json
from threading import RLock
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import httpx
from nemo_fabric_adapter_contract.models import AgentUsage
from nemo_fabric_adapter_contract.models import GapReason
from nemo_fabric_adapter_contract.models import ReportedUsage
from nemo_fabric_adapter_contract.usage import UsageObservation
from nemo_fabric_adapter_contract.usage import usage_from_observations

_OWNER: ContextVar[InvocationUsage | None] = ContextVar(
    "fabric_nooa_usage", default=None
)
_CALL: ContextVar[_Call | None] = ContextVar("fabric_nooa_call", default=None)
_ATTEMPT = "nemo_fabric.nooa_usage_attempt"


def _count(value: Any) -> int | None:
    return value if type(value) is int and 0 <= value <= (1 << 64) - 1 else None


def _report(
    usage: dict[str, Any], *, messages_api: bool = False, responses_api: bool = False
) -> ReportedUsage:
    """Map literal API fields only; normalization and arithmetic belong to core."""
    if messages_api or responses_api:
        # Responses and Messages use the same primary names but different cache
        # conventions. A Messages response carries native cache counters.
        anthropic = messages_api
        input_name, output_name = "input_tokens", "output_tokens"
        input_details, output_details = "input_tokens_details", "output_tokens_details"
    else:
        anthropic = False
        input_name, output_name = "prompt_tokens", "completion_tokens"
        input_details, output_details = (
            "prompt_tokens_details",
            "completion_tokens_details",
        )
    details = usage.get(input_details)
    details = details if isinstance(details, dict) else {}
    output = usage.get(output_details)
    output = output if isinstance(output, dict) else {}
    input_count, output_count = (
        _count(usage.get(input_name)),
        _count(usage.get(output_name)),
    )
    return ReportedUsage(
        input_tokens=input_count,
        cached_input_tokens=_count(
            usage.get("cache_read_input_tokens")
            if anthropic
            else details.get("cached_tokens")
        ),
        cache_write_input_tokens=_count(usage.get("cache_creation_input_tokens")),
        input_tokens_include_cache=not anthropic if input_count is not None else None,
        output_tokens=output_count,
        output_tokens_include_reasoning=True if output_count is not None else None,
        reasoning_tokens=_count(output.get("reasoning_tokens")),
        total_tokens=_count(usage.get("total_tokens")),
        peak_request_input_tokens=input_count if not anthropic else None,
    )


@dataclass
class _Attempt:
    model: str | None
    provider: str | None
    messages_api: bool = False
    responses_api: bool = False
    usage: ReportedUsage | None = None
    response_id: str | None = None
    source_unverified: bool = False


def _source_units(attempts: list[_Attempt]) -> list[tuple[_Attempt, bool]]:
    """Deduplicate identical reports; conflicting identities establish no total."""
    units: list[tuple[_Attempt, bool]] = []
    identities: dict[str, int] = {}
    for attempt in attempts:
        if attempt.response_id is None or attempt.response_id not in identities:
            if attempt.response_id is not None:
                identities[attempt.response_id] = len(units)
            units.append((attempt, False))
            continue
        index = identities[attempt.response_id]
        previous, _ = units[index]
        if previous != attempt:
            # Keep one genuine snapshot, without summing ambiguous repeated IDs
            # or certifying either snapshot as this call's complete usage.
            units[index] = (previous if previous.usage is not None else attempt, True)
    return units


@dataclass
class _Call:
    model: str | None
    attempts: list[_Attempt] = field(default_factory=list)
    finished: bool = False
    lock: RLock = field(default_factory=RLock)


class InvocationUsage:
    """Freeze evidence at return; late calls retain their initiating owner."""

    def __init__(self) -> None:
        self._calls: list[_Call] = []
        self._snapshot: AgentUsage | None = None
        self._lock = RLock()

    @contextmanager
    def scope(self) -> Iterator[None]:
        token = _OWNER.set(self)
        try:
            yield
        finally:
            _OWNER.reset(token)

    def start(self, model: str | None) -> _Call:
        call = _Call(model)
        with self._lock:
            if self._snapshot is None:
                self._calls.append(call)
        return call

    def finish(
        self,
        *,
        unfinished: bool = False,
        scope_unverified: bool = False,
        no_calls_verified: bool = False,
    ) -> AgentUsage:
        with self._lock:
            if self._snapshot is not None:
                return self._snapshot
            records = []
            for call in self._calls:
                with call.lock:
                    gap = GapReason.UNFINISHED_WORK if not call.finished else None
                    for attempt, conflicting in _source_units(call.attempts):
                        records.append(
                            UsageObservation(
                                reported=attempt.usage or ReportedUsage(),
                                model=attempt.model,
                                provider=attempt.provider,
                                gap=gap
                                or (
                                    GapReason.SCOPE_UNVERIFIED
                                    if attempt.source_unverified or conflicting
                                    else GapReason.MISSING_USAGE
                                    if attempt.usage is None
                                    else None
                                ),
                                tokens_aligned=True,
                                is_model_call=True,
                            )
                        )
                    if not call.attempts:
                        records.append(
                            UsageObservation(
                                reported=ReportedUsage(),
                                model=call.model,
                                gap=gap or GapReason.SCOPE_UNVERIFIED,
                                tokens_aligned=True,
                                is_model_call=True,
                            )
                        )
            if (
                not records
                and no_calls_verified
                and not unfinished
                and not scope_unverified
            ):
                records.append(
                    UsageObservation(
                        reported=ReportedUsage(
                            input_tokens=0,
                            cached_input_tokens=0,
                            cache_write_input_tokens=0,
                            input_tokens_include_cache=True,
                            output_tokens=0,
                            output_tokens_include_reasoning=True,
                            reasoning_tokens=0,
                            total_tokens=0,
                        )
                    )
                )
            if unfinished or scope_unverified:
                records.append(
                    UsageObservation(
                        reported=ReportedUsage(),
                        gap=GapReason.UNFINISHED_WORK
                        if unfinished
                        else GapReason.SCOPE_UNVERIFIED,
                    )
                )
            self._snapshot = usage_from_observations(records)
            return self._snapshot


def _request(request: httpx.Request) -> None:
    import httpx

    call = _CALL.get()
    if call is None:
        return
    model = call.model
    try:
        body = json.loads(request.content)
    except (ValueError, UnicodeDecodeError, httpx.RequestNotRead):
        body = None
    if (
        isinstance(body, dict)
        and isinstance(body.get("model"), str)
        and body["model"].strip()
    ):
        model = body["model"]
    provider = (
        {"api.openai.com": "openai", "api.anthropic.com": "anthropic"}.get(
            request.url.host
        )
        if request.url.scheme == "https" and request.url.port in (None, 443)
        else None
    )
    attempt = _Attempt(
        model,
        provider,
        messages_api=request.url.path.rstrip("/").endswith("/messages"),
        responses_api=request.url.path.rstrip("/").endswith("/responses"),
    )
    with call.lock:
        call.attempts.append(attempt)
    request.extensions[_ATTEMPT] = (call, attempt)


def _record_response(response: httpx.Response) -> None:
    item = response.request.extensions.get(_ATTEMPT)
    if item is None:
        return
    call, attempt = item
    try:
        body = response.json()
    except (ValueError, UnicodeDecodeError):
        return
    if not isinstance(body, dict):
        return
    with call.lock:
        usage = body.get("usage")
        if isinstance(usage, dict):
            attempt.usage = _report(
                usage,
                messages_api=attempt.messages_api,
                responses_api=attempt.responses_api,
            )
        identity = body.get("id")
        attempt.response_id = (
            identity if isinstance(identity, str) and identity else None
        )


def _unverified_response(response: httpx.Response) -> None:
    item = response.request.extensions.get(_ATTEMPT)
    if item is not None:
        call, attempt = item
        with call.lock:
            attempt.source_unverified = True


def _response(response: httpx.Response) -> None:
    if "application/json" in response.headers.get("content-type", ""):
        response.read()
        _record_response(response)
    else:
        _unverified_response(response)


async def _async_request(request: httpx.Request) -> None:
    _request(request)


async def _async_response(response: httpx.Response) -> None:
    if "application/json" in response.headers.get("content-type", ""):
        await response.aread()
        _record_response(response)
    else:
        _unverified_response(response)


def install_model_observer(model: Any) -> None:
    """Observe this NOOA 0.0.10 client without replacing its HTTP transport.

    Changed/custom clients still execute with a scope gap. Their normalized
    counters are not promoted: LiteLLM can estimate missing stream usage.
    """
    import httpx

    transport = getattr(model, "_http", None)
    sync = getattr(transport, "httpx_sync", None)
    async_client = getattr(transport, "httpx_async", None)
    observed = isinstance(sync, httpx.Client) and isinstance(
        async_client, httpx.AsyncClient
    )
    if observed:
        sync.event_hooks.setdefault("request", []).append(_request)
        sync.event_hooks.setdefault("response", []).append(_response)
        async_client.event_hooks.setdefault("request", []).append(_async_request)
        async_client.event_hooks.setdefault("response", []).append(_async_response)
    configured = getattr(model, "model", None)
    configured = (
        configured if isinstance(configured, str) and configured.strip() else None
    )
    original_call = getattr(model, "call", None)
    original_acall = getattr(model, "acall", None)
    if not callable(original_call) or not callable(original_acall):
        return

    def invoke(*args: Any, **kwargs: Any) -> Any:
        owner = _OWNER.get()
        if owner is None:
            return original_call(*args, **kwargs)
        call = owner.start(configured)
        token = _CALL.set(call)
        try:
            return original_call(*args, **kwargs)
        finally:
            with call.lock:
                call.finished = True
            _CALL.reset(token)

    async def ainvoke(*args: Any, **kwargs: Any) -> Any:
        owner = _OWNER.get()
        if owner is None:
            return await original_acall(*args, **kwargs)
        call = owner.start(configured)
        token = _CALL.set(call)
        try:
            return await original_acall(*args, **kwargs)
        finally:
            with call.lock:
                call.finished = True
            _CALL.reset(token)

    model.call, model.acall = invoke, ainvoke


def running_jobs(agent: Any) -> list[Any]:
    """Read NOOA's public queue handles without waiting or cancelling."""
    read = getattr(getattr(agent, "queue_manager", None), "running_handles", None)
    return list(read()) if callable(read) else []


def owned_work_pending(
    agent: Any, owner: InvocationUsage, jobs_before: list[Any]
) -> bool:
    """Include scheduled summaries/jobs that can dispatch their first call later."""
    for summarizer in getattr(agent, "_summarizers", None) or []:
        task = getattr(summarizer, "_pending_task", None)
        if (
            isinstance(task, asyncio.Task)
            and not task.done()
            and task.get_context().get(_OWNER) is owner
        ):
            return True
    for job in running_jobs(agent):
        task = getattr(job, "_task", None)
        if isinstance(task, asyncio.Task):
            if not task.done() and task.get_context().get(_OWNER) is owner:
                return True
        elif all(job is not previous for previous in jobs_before):
            # A changed handle hides ownership; retain the conservative boundary.
            return True
    return False
