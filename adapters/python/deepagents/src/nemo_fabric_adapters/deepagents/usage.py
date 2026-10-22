# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Invocation-owned LangChain observations and passive HTTP attempt evidence."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass, field
from threading import RLock
from typing import Any
from uuid import UUID

from langchain_core.caches import BaseCache
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.globals import get_llm_cache
from openai import LengthFinishReasonError
from nemo_fabric_adapter_contract.models import AgentUsage, GapReason, ReportedUsage
from nemo_fabric_adapter_contract.usage import UsageObservation, usage_from_observations

_OWNER: ContextVar[ModelCallUsage | None] = ContextVar(
    "fabric_deepagents_usage_owner", default=None
)
_ACTIVE_CALL: ContextVar[tuple[ModelCallUsage, UUID] | None] = ContextVar(
    "fabric_deepagents_active_call", default=None
)
_ATTEMPT = "nemo_fabric.usage_attempt"
_CACHE_HIT = "nemo_fabric.local_cache_hit"
_OBSERVER = "nemo_fabric.usage_observer"


@dataclass
class _Call:
    attempts: list[int] = field(default_factory=list)
    tracked: bool = False
    openai_format: bool = False
    finished: bool = False
    failed: bool = False
    cached: bool = False
    reported: ReportedUsage = field(default_factory=ReportedUsage)
    model: str | None = None
    provider: str | None = None
    cache_scope_unverified: bool = False
    stream_updates: int = 0
    stream_observed: ReportedUsage = field(default_factory=ReportedUsage)


class ModelCallUsage(BaseCallbackHandler):
    """One handler per invocation; returned snapshots never change afterwards.

    HTTP starts/statuses expose SDK-internal retry gaps without counting tokens
    again. No HTTP status proves zero processed tokens. Callbacks on unsupported
    clients remain observations with unverified scope. Local cache replay is
    identified by the supported cache wrapper, not by absence of HTTP alone.
    """

    run_inline = True

    def __init__(
        self,
        observer: str | None,
        remote_names: frozenset[str] = frozenset(),
    ) -> None:
        super().__init__()
        self._observer = observer
        self._remote_names = remote_names
        self._calls: dict[UUID, _Call] = {}
        self._statuses: list[int | None] = []
        self._unassigned_request = False
        self._remote_used = False
        self._closed = False
        self._snapshot: AgentUsage | None = None
        self._lock = RLock()

    @contextmanager
    def scope(self) -> Iterator[None]:
        """Propagate this owner into graph descendants and worker threads."""
        token = _OWNER.set(self)
        call_token = _ACTIVE_CALL.set(None)
        try:
            yield
        finally:
            _ACTIVE_CALL.reset(call_token)
            _OWNER.reset(token)

    def on_chat_model_start(
        self, serialized: Any, messages: Any, *, run_id: UUID, **kwargs: Any
    ) -> None:
        metadata = kwargs.get("metadata") or {}
        with self._lock:
            if not self._closed:
                identity = (
                    serialized.get("id", []) if isinstance(serialized, dict) else []
                )
                tracked = (
                    _OWNER.get() is self
                    and self._observer is not None
                    and metadata.get(_OBSERVER) == self._observer
                )
                self._calls.setdefault(
                    run_id,
                    _Call(
                        tracked=tracked,
                        cache_scope_unverified=not tracked
                        and get_llm_cache() is not None,
                        openai_format=bool(identity and identity[-1] == "ChatOpenAI"),
                    ),
                )
                if tracked:
                    _ACTIVE_CALL.set((self, run_id))

    def on_llm_start(
        self, serialized: Any, prompts: Any, *, run_id: UUID, **kwargs: Any
    ) -> None:
        self.on_chat_model_start(serialized, prompts, run_id=run_id, **kwargs)

    def on_llm_new_token(
        self, token: str, *, run_id: UUID, chunk: Any = None, **kwargs: Any
    ) -> None:
        message = getattr(chunk, "message", None)
        if not isinstance(getattr(message, "usage_metadata", None), dict):
            return
        with self._lock:
            call = self._calls.get(run_id)
            if self._closed or call is None:
                return
            call.stream_updates += 1
            counts = _raw_openai_usage(message, known_convention=call.openai_format)
            call.stream_observed = ReportedUsage.from_mapping(
                {**call.stream_observed.to_mapping(), **counts.to_mapping()}
            )

    def on_llm_end(self, response: Any, *, run_id: UUID, **kwargs: Any) -> None:
        with self._lock:
            if self._closed:
                return
            call = self._calls.setdefault(run_id, _Call())
            generations = response.generations[0] if response.generations else []
            message = getattr(generations[0], "message", None) if generations else None
            metadata = getattr(message, "response_metadata", {}) or {}
            call.cached = bool(metadata.get(_CACHE_HIT))
            model = metadata.get("model_name")
            call.model = model if isinstance(model, str) and model.strip() else None
            # Retain counters and identity only, not prompts or response bodies.
            call.reported = _raw_openai_usage(
                message, known_convention=call.openai_format
            )
            if call.stream_updates > 1:
                # The callback does not establish whether repeated usage frames
                # are deltas or cumulative snapshots. Never charge their sum.
                call.reported = call.stream_observed
            if call.cache_scope_unverified and not call.cached:
                call.reported = ReportedUsage()
            call.finished = True
            if _ACTIVE_CALL.get() == (self, run_id):
                _ACTIVE_CALL.set(None)

    def on_llm_error(
        self, error: BaseException, *, run_id: UUID, **kwargs: Any
    ) -> None:
        with self._lock:
            if self._closed:
                return
            call = self._calls.setdefault(run_id, _Call())
            if call.openai_format and isinstance(error, LengthFinishReasonError):
                # SDK schema parsing failed after a terminal provider response.
                # Its public completion retains literal usage, including omissions.
                completion = error.completion
                raw = (
                    completion.usage.model_dump(exclude_unset=True)
                    if completion.usage is not None
                    else {}
                )
                call.reported = _openai_counts(raw, known_convention=True)
                model = completion.model
                call.model = model if isinstance(model, str) and model.strip() else None
                call.finished = True
            else:
                partial = kwargs.get("response")
                if partial is not None:
                    self.on_llm_end(partial, run_id=run_id)
                call.finished = call.failed = True
            if _ACTIVE_CALL.get() == (self, run_id):
                _ACTIVE_CALL.set(None)

    def on_tool_start(self, serialized: Any, input_str: str, **kwargs: Any) -> None:
        inputs = kwargs.get("inputs")
        name = serialized.get("name") if isinstance(serialized, dict) else None
        if (
            isinstance(inputs, dict)
            and self._remote_names
            and (
                name in {"task", "start_async_task"}
                and inputs.get("subagent_type") in self._remote_names
                # Updating a remote task creates another run, without a
                # subagent_type argument. Checking its status does not.
                or name == "update_async_task"
            )
        ):
            with self._lock:
                if not self._closed:
                    self._remote_used = True

    def request_started(self, request: Any) -> None:
        """Keep only an owner/index association, never request payloads or credentials."""
        with self._lock:
            if not self._closed:
                index = len(self._statuses)
                self._statuses.append(None)
                active = _ACTIVE_CALL.get()
                call = (
                    self._calls.get(active[1])
                    if active is not None and active[0] is self
                    else None
                )
                if call is None:
                    self._unassigned_request = True
                else:
                    call.attempts.append(index)
                    url = getattr(request, "url", None)
                    provider = (
                        "openai"
                        if getattr(url, "scheme", None) == "https"
                        and getattr(url, "host", None) == "api.openai.com"
                        and getattr(url, "port", None) in (None, 443)
                        else None
                    )
                    # Configured labels establish the protocol, not billing.
                    # Every physical attempt must use the native endpoint.
                    if len(call.attempts) == 1:
                        call.provider = provider
                    elif call.provider != provider:
                        call.provider = None
                request.extensions[_ATTEMPT] = (self, index)

    def response_received(self, index: int, status: int) -> None:
        with self._lock:
            if not self._closed:
                self._statuses[index] = status

    def finish(self) -> AgentUsage:
        """Freeze a complete/observed projection at the harness's ordinary return boundary."""
        with self._lock:
            if self._snapshot is not None:
                return self._snapshot
            self._closed = True
            root_gap = self._remote_used or self._unassigned_request
            observations = []
            for call in self._calls.values():
                counts = call.reported
                if call.cached:
                    counts = ReportedUsage(
                        input_tokens=0,
                        cached_input_tokens=0,
                        cache_write_input_tokens=0,
                        input_tokens_include_cache=True,
                        output_tokens=0,
                        reasoning_tokens=0,
                        output_tokens_include_reasoning=True,
                        total_tokens=0,
                        peak_request_input_tokens=0,
                    )
                statuses = [self._statuses[index] for index in call.attempts]
                gap = (
                    GapReason.UNFINISHED_WORK
                    if not call.finished
                    else GapReason.MISSING_USAGE
                    if call.failed
                    or any(
                        status is None or not 200 <= status < 300 for status in statuses
                    )
                    else GapReason.SCOPE_UNVERIFIED
                    if not call.cached
                    and (
                        not call.tracked
                        or call.stream_updates > 1
                        or len(statuses) != 1
                    )
                    else None
                )
                observations.append(
                    UsageObservation(
                        is_model_call=True,
                        reported=counts,
                        model=call.model,
                        provider=call.provider if call.tracked else None,
                        gap=gap,
                        tokens_aligned=call.cached
                        or (
                            call.tracked and call.finished and call.stream_updates <= 1
                        ),
                    )
                )
            if root_gap:
                observations.append(
                    UsageObservation(
                        reported=ReportedUsage(), gap=GapReason.SCOPE_UNVERIFIED
                    )
                )
            self._snapshot = usage_from_observations(observations)
            return self._snapshot


def _count(value: Any) -> int | None:
    return value if type(value) is int and 0 <= value <= (1 << 64) - 1 else None


def _positive_count(value: Any) -> int | None:
    return value if _count(value) is not None and value > 0 else None


def _raw_openai_usage(message: Any, *, known_convention: bool) -> ReportedUsage:
    metadata = getattr(message, "response_metadata", {}) or {}
    raw = metadata.get("token_usage")
    if not isinstance(raw, dict):
        # ChatOpenAI streaming drops raw field presence. Positive input/output
        # counters still come directly from provider usage; zero can be a filled
        # default. Keep independently valid positives and never claim its derived
        # total is a provider-reported total.
        usage = getattr(message, "usage_metadata", None)
        if not known_convention or not isinstance(usage, dict):
            return ReportedUsage()
        input_tokens = _positive_count(usage.get("input_tokens"))
        output_tokens = _positive_count(usage.get("output_tokens"))
        input_details = usage.get("input_token_details") or {}
        output_details = usage.get("output_token_details") or {}
        return ReportedUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_tokens_include_cache=True if input_tokens is not None else None,
            output_tokens_include_reasoning=True if output_tokens is not None else None,
            cached_input_tokens=_positive_count(input_details.get("cache_read")),
            reasoning_tokens=_positive_count(output_details.get("reasoning")),
            peak_request_input_tokens=input_tokens,
        )
    return _openai_counts(raw, known_convention=known_convention)


def _openai_counts(raw: dict[str, Any], *, known_convention: bool) -> ReportedUsage:
    prompt = _count(raw.get("prompt_tokens"))
    completion = _count(raw.get("completion_tokens"))
    prompt_details = raw.get("prompt_tokens_details")
    completion_details = raw.get("completion_tokens_details")
    prompt_details = prompt_details if isinstance(prompt_details, dict) else {}
    completion_details = (
        completion_details if isinstance(completion_details, dict) else {}
    )
    return ReportedUsage(
        input_tokens=prompt,
        cached_input_tokens=_count(prompt_details.get("cached_tokens")),
        input_tokens_include_cache=True
        if prompt is not None and known_convention
        else None,
        output_tokens=completion,
        output_tokens_include_reasoning=True
        if completion is not None and known_convention
        else None,
        reasoning_tokens=_count(completion_details.get("reasoning_tokens")),
        total_tokens=_count(raw.get("total_tokens")),
        peak_request_input_tokens=prompt if known_convention else None,
    )


def request_hook(request: Any) -> None:
    owner = _OWNER.get()
    if owner is not None:
        owner.request_started(request)


def response_hook(response: Any) -> None:
    attempt = response.request.extensions.get(_ATTEMPT)
    if attempt is not None:
        owner, index = attempt
        owner.response_received(index, response.status_code)


async def async_request_hook(request: Any) -> None:
    request_hook(request)


async def async_response_hook(response: Any) -> None:
    response_hook(response)


class AccountingCache(BaseCache):
    """Delegate to the ambient cache without recharging cached historical usage.

    Returned generations are copied before marking a hit. Stored entries and
    the global cache are not changed; a disabled cache remains a miss.
    """

    @staticmethod
    def _mark(generations: Any) -> Any:
        if not isinstance(generations, list):
            return generations
        result = deepcopy(generations)
        for generation in result:
            message = getattr(generation, "message", None)
            if message is not None:
                message.response_metadata[_CACHE_HIT] = True
        return result

    def lookup(self, prompt: str, llm_string: str) -> Any:
        cache = get_llm_cache()
        return None if cache is None else self._mark(cache.lookup(prompt, llm_string))

    async def alookup(self, prompt: str, llm_string: str) -> Any:
        cache = get_llm_cache()
        return (
            None
            if cache is None
            else self._mark(await cache.alookup(prompt, llm_string))
        )

    def update(self, prompt: str, llm_string: str, return_val: Any) -> None:
        cache = get_llm_cache()
        if cache is not None:
            cache.update(prompt, llm_string, return_val)

    async def aupdate(self, prompt: str, llm_string: str, return_val: Any) -> None:
        cache = get_llm_cache()
        if cache is not None:
            await cache.aupdate(prompt, llm_string, return_val)

    def clear(self, **kwargs: Any) -> None:
        cache = get_llm_cache()
        if cache is not None:
            cache.clear(**kwargs)


def model_metadata(observer: str | None) -> dict[str, str]:
    """Model-local metadata is not inherited by separately constructed child models."""
    return {} if observer is None else {_OBSERVER: observer}


def observed_http_clients() -> tuple[Any, Any] | None:
    """Use the supported OpenAI client injection with LangChain's default timeout.

    LangChain 1.1.10 supplies timeout=None to these same OpenAI default client
    classes. Hooks observe dispatch only; they do not replace transports,
    disable SDK retries, or consume response streams.
    """
    try:
        from openai import DefaultAsyncHttpxClient, DefaultHttpxClient
    except ImportError:
        # An older optional observer surface limits accounting scope rather than
        # preventing an otherwise supported harness from running.
        return None

    return (
        DefaultHttpxClient(
            timeout=None,
            event_hooks={"request": [request_hook], "response": [response_hook]},
        ),
        DefaultAsyncHttpxClient(
            timeout=None,
            event_hooks={
                "request": [async_request_hook],
                "response": [async_response_hook],
            },
        ),
    )
