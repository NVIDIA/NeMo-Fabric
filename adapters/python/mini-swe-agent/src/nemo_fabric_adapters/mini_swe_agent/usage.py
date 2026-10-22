# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Invocation-owned raw LiteLLM observations; shared core normalizes and prices."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import importlib
import importlib.metadata
import inspect
import json
import os
from threading import RLock
from typing import Any
from urllib.parse import urlsplit

import httpx
import litellm
from minisweagent.models.litellm_model import LitellmModel
from nemo_fabric_adapter_contract.models import AgentUsage, GapReason, ReportedUsage
from nemo_fabric_adapter_contract.usage import UsageObservation, usage_from_observations

_OWNER: ContextVar[MiniInvocationUsage | None] = ContextVar(
    "fabric_mini_usage", default=None
)
_ACTIVE: ContextVar[_Call | None] = ContextVar("fabric_mini_attempt", default=None)


def _count(value: Any) -> int | None:
    return value if type(value) is int and 0 <= value < 1 << 64 else None


def _reported(raw: dict[str, Any], *, responses: bool = False) -> ReportedUsage:
    if responses:
        cache = raw.get("input_tokens_details")
        cache = cache if isinstance(cache, dict) else {}
        reasoning = raw.get("output_tokens_details")
        reasoning = reasoning if isinstance(reasoning, dict) else {}
        return ReportedUsage(
            input_tokens=_count(raw.get("input_tokens")),
            cached_input_tokens=_count(cache.get("cached_tokens")),
            input_tokens_include_cache=True
            if _count(raw.get("input_tokens")) is not None
            else None,
            output_tokens=_count(raw.get("output_tokens")),
            output_tokens_include_reasoning=True
            if _count(raw.get("output_tokens")) is not None
            else None,
            reasoning_tokens=_count(reasoning.get("reasoning_tokens")),
            total_tokens=_count(raw.get("total_tokens")),
        )
    if raw.keys() & {
        "prompt_tokens",
        "completion_tokens",
        "prompt_tokens_details",
        "completion_tokens_details",
        "total_tokens",
    }:
        cache = raw.get("prompt_tokens_details")
        cache = cache if isinstance(cache, dict) else {}
        reasoning = raw.get("completion_tokens_details")
        reasoning = reasoning if isinstance(reasoning, dict) else {}
        return ReportedUsage(
            input_tokens=_count(raw.get("prompt_tokens")),
            cached_input_tokens=_count(cache.get("cached_tokens")),
            input_tokens_include_cache=True
            if _count(raw.get("prompt_tokens")) is not None
            else None,
            output_tokens=_count(raw.get("completion_tokens")),
            output_tokens_include_reasoning=True
            if _count(raw.get("completion_tokens")) is not None
            else None,
            reasoning_tokens=_count(reasoning.get("reasoning_tokens")),
            total_tokens=_count(raw.get("total_tokens")),
        )
    if raw.keys() & {
        "input_tokens",
        "output_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
    }:
        return ReportedUsage(
            input_tokens=_count(raw.get("input_tokens")),
            cached_input_tokens=_count(raw.get("cache_read_input_tokens")),
            cache_write_input_tokens=_count(raw.get("cache_creation_input_tokens")),
            input_tokens_include_cache=False
            if _count(raw.get("input_tokens")) is not None
            else None,
            output_tokens=_count(raw.get("output_tokens")),
            output_tokens_include_reasoning=True
            if _count(raw.get("output_tokens")) is not None
            else None,
        )
    # Bedrock Converse's input excludes the independently optional cache counts.
    if raw.keys() & {
        "inputTokens",
        "outputTokens",
        "totalTokens",
        "cacheReadInputTokens",
        "cacheWriteInputTokens",
    }:
        return ReportedUsage(
            input_tokens=_count(raw.get("inputTokens")),
            cached_input_tokens=_count(raw.get("cacheReadInputTokens")),
            cache_write_input_tokens=_count(raw.get("cacheWriteInputTokens")),
            input_tokens_include_cache=False
            if _count(raw.get("inputTokens")) is not None
            else None,
            output_tokens=_count(raw.get("outputTokens")),
            output_tokens_include_reasoning=True
            if _count(raw.get("outputTokens")) is not None
            else None,
            total_tokens=_count(raw.get("totalTokens")),
        )
    return ReportedUsage()


@dataclass
class _Call:
    model: str | None
    provider: str | None
    responses: list[list[ReportedUsage]] = field(default_factory=list)
    identities: dict[str, list[ReportedUsage]] = field(default_factory=dict)
    attempts: list[int | None] = field(default_factory=list)
    pre_calls: int = 0
    finished: bool = False
    conflict: bool = False
    scope_verified: bool = True
    http_expected: bool = False
    responses_api: bool = False
    lock: RLock = field(default_factory=RLock)

    def log(self, details: dict[str, Any]) -> None:
        event = details.get("log_event_type")
        if event == "pre_api_call":
            with self.lock:
                self.pre_calls += 1
            return
        if event != "post_api_call":
            return
        body = details.get("original_response")
        if isinstance(body, str):
            try:
                body = json.loads(body)
            except ValueError:
                return
        if not isinstance(body, dict) or not isinstance(body.get("usage"), dict):
            return
        raw = body["usage"]
        iterations = raw.get("iterations")
        if iterations:
            if not isinstance(iterations, list) or not all(
                isinstance(item, dict) for item in iterations
            ):
                snapshots = [ReportedUsage()]
            else:
                snapshots = [_reported(item) for item in iterations]
        else:
            snapshots = [
                _reported(
                    raw,
                    responses=self.responses_api or body.get("object") == "response",
                )
            ]
        identity = body.get("id")
        # Bedrock has no response id. Repeated identical post-call callbacks
        # within one _query are the same response, not separate model work.
        identity = identity if isinstance(identity, str) else "no-id"
        with self.lock:
            previous = self.identities.get(identity)
            if previous is not None:
                self.conflict |= previous != snapshots
                return
            self.identities[identity] = snapshots
            self.responses.append(snapshots)


class MiniInvocationUsage:
    """Freeze each invocation's source records before its result can escape."""

    def __init__(self) -> None:
        self.calls: list[_Call] = []
        self._snapshot: AgentUsage | None = None
        self._lock = RLock()

    @contextmanager
    def scope(self) -> Iterator[None]:
        token = _OWNER.set(self)
        try:
            yield
        finally:
            _OWNER.reset(token)

    def start(self, model: str | None, provider: str | None) -> _Call:
        call = _Call(model=model, provider=provider)
        with self._lock:
            if self._snapshot is None:
                self.calls.append(call)
        return call

    def finish(self, *, no_calls: bool = False) -> AgentUsage:
        with self._lock:
            if self._snapshot is not None:
                return self._snapshot
            records = []
            for call in self.calls:
                with call.lock:
                    gap = None
                    if not call.finished:
                        gap = GapReason.UNFINISHED_WORK
                    elif not call.scope_verified or (
                        call.http_expected and not call.attempts
                    ):
                        gap = GapReason.SCOPE_UNVERIFIED
                    elif call.conflict or any(
                        status is None or status >= 300 for status in call.attempts
                    ):
                        gap = GapReason.MISSING_USAGE
                    elif not call.http_expected and call.pre_calls > len(
                        call.responses
                    ):
                        gap = GapReason.MISSING_USAGE
                    elif not call.responses:
                        gap = GapReason.MISSING_USAGE
                    for snapshots in call.responses or [[ReportedUsage()]]:
                        records.extend(
                            UsageObservation(
                                reported=raw,
                                model=call.model,
                                provider=call.provider,
                                gap=gap,
                                tokens_aligned=True,
                                is_model_call=len(snapshots) == 1,
                            )
                            for raw in snapshots
                        )
            if not self.calls and no_calls:
                # The fixed agent/model pair was observed for its entire run.
                # No _query entry proves no model request was attempted.
                records.append(
                    UsageObservation(
                        reported=ReportedUsage(
                            input_tokens=0,
                            cached_input_tokens=0,
                            cache_write_input_tokens=0,
                            input_tokens_include_cache=True,
                            output_tokens=0,
                            output_tokens_include_reasoning=True,
                            total_tokens=0,
                        )
                    )
                )
            self._snapshot = usage_from_observations(records)
            return self._snapshot


def _native_endpoint(url: Any, host: str) -> bool:
    """Qualify a native HTTPS endpoint without inferring billing from its protocol."""
    try:
        parts = urlsplit(str(url))
        return (
            parts.scheme == "https"
            and parts.hostname == host
            and parts.port in (None, 443)
        )
    except ValueError:
        return False


def _http_start(request: Any) -> None:
    if (call := _ACTIVE.get()) is not None:
        with call.lock:
            index = len(call.attempts)
            call.attempts.append(None)
            if not _native_endpoint(request.url, "api.openai.com"):
                call.provider = None
        request.extensions["fabric_mini_usage_call"] = (call, index)


def _http_end(response: Any) -> None:
    tracked = response.request.extensions.get("fabric_mini_usage_call")
    if tracked is not None:
        call, index = tracked
        with call.lock:
            call.attempts[index] = response.status_code
        # These adapter calls are non-streaming. Capture the literal terminal
        # body before SDK/Responses conversion can lose field presence; hooks
        # on status alone cannot recover a report the bridge never logs.
        response.read()
        call.log(
            {"log_event_type": "post_api_call", "original_response": response.text}
        )


def _responses_mode(model: str, provider: str, params: dict[str, Any]) -> bool | None:
    checker = getattr(
        importlib.import_module("litellm.main"), "responses_api_bridge_check", None
    )
    if not callable(checker):
        return None
    tool = getattr(
        importlib.import_module("minisweagent.models.litellm_model"), "BASH_TOOL", None
    )
    if tool is None:
        return None
    arguments = {
        "model": model,
        "custom_llm_provider": provider,
        "tools": params.get("tools", [tool]),
        "reasoning_effort": params.get("reasoning_effort"),
        "reasoning_summary": params.get("reasoning_summary"),
        "web_search_options": params.get("web_search_options"),
        "api_base": params.get("api_base"),
    }
    try:
        signature = inspect.signature(checker)
        # Qualified 1.89.3 predates endpoint-aware bridge routing. Delegate to
        # its actual rule without passing the newer api_base argument.
        if "api_base" not in signature.parameters:
            arguments.pop("api_base")
        signature.bind(**arguments)
    except (TypeError, ValueError):
        return None
    bridge, _ = checker(**arguments)
    return bridge.get("mode") == "responses"


def _http_client() -> httpx.Client:
    from litellm.llms.custom_httpx.http_handler import get_ssl_configuration

    # Match LiteLLM's default OpenAI client settings, including custom CA roots.
    return httpx.Client(
        verify=get_ssl_configuration(),
        follow_redirects=True,
        event_hooks={"request": [_http_start], "response": [_http_end]},
    )


class UsageRecordingLitellmModel(LitellmModel):
    """Observe raw bodies before zero filling and retain mini's retry policy.

    Native OpenAI Chat and Responses use adapter-owned clients with passive
    HTTP hooks, so a retry cannot hide failed attempts behind a final response.
    Anthropic's default Python HTTP path issues one request per _query. Other
    provider transports retain their raw reports without a scope certification.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        try:
            self._source_qualified = importlib.metadata.version(
                "mini-swe-agent"
            ) == "2.4.6" and importlib.metadata.version("litellm") in {
                "1.89.3",
                "1.96.2",
                "1.100.1",
            }
        except importlib.metadata.PackageNotFoundError:
            self._source_qualified = False
        self._usage_client: Any = None
        self._usage_client_key: tuple[str | None, str, bool] | None = None

    def close_usage_client(self) -> None:
        if self._usage_client is not None:
            self._usage_client.close()
            self._usage_client = None
            self._usage_client_key = None

    def _openai_client(self, params: dict[str, Any], *, responses: bool) -> Any:
        from openai import OpenAI
        from litellm.utils import get_api_key
        from litellm.secret_managers.main import get_secret

        base = (
            params.get("api_base")
            or litellm.api_base
            or get_secret("OPENAI_BASE_URL")
            or get_secret("OPENAI_API_BASE")
            or "https://api.openai.com/v1"
        )
        key = get_api_key("openai", params.get("api_key"))
        if self._usage_client_key != (key, base, responses):
            self.close_usage_client()
            if responses:
                from litellm.llms.custom_httpx.http_handler import HTTPHandler

                self._usage_client = HTTPHandler()
                self._usage_client.client.event_hooks["request"].append(_http_start)
                self._usage_client.client.event_hooks["response"].append(_http_end)
            else:
                client = _http_client()
                try:
                    self._usage_client = OpenAI(
                        api_key=key, base_url=base, http_client=client
                    )
                except BaseException:
                    client.close()
                    raise
            self._usage_client_key = (key, base, responses)
        return self._usage_client

    def _query(self, messages: list[dict[str, str]], **kwargs: Any) -> Any:
        owner = _OWNER.get()
        if owner is None:
            return super()._query(messages, **kwargs)
        params = self.config.model_kwargs | kwargs
        provider, _, model = self.config.model_name.partition("/")
        identity = model or self.config.model_name
        # Ambient aliases/fallbacks can route an otherwise fixed configured
        # model elsewhere. Keep counts, but do not invent one-model allocation.
        routed = bool(
            self.config.model_name in (getattr(litellm, "model_alias_map", None) or {})
            or getattr(litellm, "model_fallbacks", None)
            or params.get("fallbacks")
        )
        native = (
            self._source_qualified
            and provider in {"openai", "anthropic"}
            and not routed
        )
        call = owner.start(
            identity if not routed and self._source_qualified else None,
            provider if native else None,
        )
        # Default Anthropic Python HTTP transport performs one send. The
        # opt-in Rust bridge and arbitrary provider transports are separate
        # qualification boundaries, not grounds to erase raw observations.
        rust = params.get("rust") is True or os.environ.get(
            "LITELLM_RUST", ""
        ).lower() in {"1", "true", "yes", "on"}
        call.scope_verified = native and not (provider == "anthropic" and rust)
        opaque_client = (
            params.get("client") is not None
            or getattr(litellm, "client_session", None) is not None
            or getattr(litellm, "network_mock", False)
        )
        if provider == "openai" and opaque_client:
            call.scope_verified = (
                native
                and params.get("client") is self._usage_client
                and self._usage_client is not None
            )
            call.http_expected = call.scope_verified
            if not call.scope_verified:
                call.provider = None
        if provider == "anthropic" and params.get("client") is not None:
            # An external handler may retry or route requests inside post().
            # Its terminal logger cannot certify the default single-send path.
            call.scope_verified = False
            call.provider = None
        if provider == "anthropic" and native:
            from litellm.secret_managers.main import get_secret

            base = (
                params.get("api_base")
                or litellm.api_base
                or get_secret("ANTHROPIC_API_BASE")
                or get_secret("ANTHROPIC_BASE_URL")
                or "https://api.anthropic.com/v1/messages"
            )
            if not _native_endpoint(base, "api.anthropic.com"):
                call.provider = None
        previous = params.get("logger_fn")

        def log(details: dict[str, Any]) -> None:
            call.log(details)
            if callable(previous):
                previous(details)

        token = _ACTIVE.set(call)
        try:
            if provider == "openai" and native and not opaque_client:
                responses = _responses_mode(identity, provider, params)
                if responses is None:
                    # Unsupported observer seam must not change execution or
                    # claim this source version has certified call coverage.
                    call.scope_verified = False
                    call.model = call.provider = None
                else:
                    call.responses_api = responses
                    params["client"] = self._openai_client(params, responses=responses)
                    call.http_expected = True
                    base = self._usage_client_key[1]
                    # OpenAI-compatible endpoints do not establish native billing.
                    if not _native_endpoint(base, "api.openai.com"):
                        call.provider = None
            response = super()._query(messages, **{**params, "logger_fn": log})
            return response
        finally:
            with call.lock:
                call.finished = True
            _ACTIVE.reset(token)
