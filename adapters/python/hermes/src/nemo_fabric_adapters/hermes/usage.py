# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Invocation-owned Hermes evidence from leased hooks and owned HTTP clients.

The HTTP seam is qualified against Hermes a62979dc. It preserves the upstream
client, pool, proxy/TLS policy and lazy body consumption. Public middleware is a
fallback observation source, never added to an already observed wire response.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from hashlib import sha256
import inspect
import json
from pathlib import Path
from threading import RLock
from typing import Any
from urllib.parse import urlsplit
from weakref import ReferenceType, ref

import httpx
from nemo_fabric_adapter_contract.models import AgentUsage, GapReason, ReportedUsage
from nemo_fabric_adapter_contract.usage import UsageObservation, usage_from_observations

_OWNER: ContextVar[InvocationUsage | None] = ContextVar(
    "fabric_hermes_owner", default=None
)
_OPERATION: ContextVar[Operation | None] = ContextVar(
    "fabric_hermes_operation", default=None
)
_REQUEST_KEY = "nemo_fabric.hermes_usage"
_MAX_EVENT_BYTES = 1 << 20
_SOURCE_HASHES = {
    "client_lifecycle.py": "4e8a80556184c415256934c50e6191b840910867fe68d36efc263440ea7b7dd2",
    "agent_runtime_helpers.py": "140e1c8d520f207bc66258407199d8c0b1f85f9d77f1cc6a7185936b459eb912",
}


def qualified_provider(url: Any) -> str | None:
    """Only an observed native HTTPS endpoint authorizes bundled provider prices."""
    if not isinstance(url, str):
        return None
    try:
        parts = urlsplit(url)
        if parts.scheme != "https" or parts.port not in (None, 443):
            return None
        return {"api.openai.com": "openai", "api.anthropic.com": "anthropic"}.get(
            parts.hostname
        )
    except ValueError:
        return None


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dump(exclude_unset=True)
    # Hermes' default streaming response is a SimpleNamespace, not an SDK model.
    return vars(value) if hasattr(value, "__dict__") else {}


def _count(value: Any, *, presence_lost: bool = False) -> int | None:
    return (
        value
        if type(value) is int
        and (value > 0 if presence_lost else value >= 0)
        and value < 1 << 64
        else None
    )


def raw_usage(response: Any, api_mode: str) -> tuple[ReportedUsage, str | None]:
    body = _mapping(response)
    raw = _mapping(body.get("usage"))
    model = body.get("model")
    model = model if isinstance(model, str) and model else None
    messages = api_mode == "anthropic_messages"
    details = _mapping(
        raw.get("input_tokens_details", raw.get("prompt_tokens_details"))
    )
    output_details = _mapping(
        raw.get("output_tokens_details", raw.get("completion_tokens_details"))
    )
    return ReportedUsage(
        input_tokens=_count(raw.get("input_tokens", raw.get("prompt_tokens"))),
        cached_input_tokens=_count(
            raw.get("cache_read_input_tokens", details.get("cached_tokens"))
        ),
        cache_write_input_tokens=_count(raw.get("cache_creation_input_tokens")),
        input_tokens_include_cache=not messages,
        output_tokens=_count(raw.get("output_tokens", raw.get("completion_tokens"))),
        output_tokens_include_reasoning=True,
        reasoning_tokens=_count(output_details.get("reasoning_tokens")),
        total_tokens=_count(raw.get("total_tokens")),
    ), model


@dataclass
class Call:
    model: str | None = None
    provider: str | None = None
    reported: ReportedUsage = field(default_factory=ReportedUsage)
    settled: bool = False
    gap: GapReason | None = None


@dataclass
class Operation:
    owner: InvocationUsage
    model: str | None
    provider: str | None
    calls: list[int] = field(default_factory=list)


@dataclass
class InvocationUsage:
    registration_tag: object
    calls: list[Call] = field(default_factory=list)
    children: set[str] = field(default_factory=set)
    lock: RLock = field(default_factory=RLock)
    snapshot: AgentUsage | None = None

    @contextmanager
    def activate(self) -> Iterator[None]:
        token = _OWNER.set(self)
        try:
            yield
        finally:
            _OWNER.reset(token)

    def start(self, model: str | None = None, provider: str | None = None) -> int:
        with self.lock:
            index = len(self.calls)
            self.calls.append(Call(model=model, provider=provider))
            return index

    def settle(
        self,
        index: int,
        reported: ReportedUsage,
        *,
        model: str | None = None,
        gap: GapReason | None = None,
    ) -> None:
        with self.lock:
            call = self.calls[index]
            call.reported = reported
            call.model = model or call.model
            call.settled = True
            call.gap = gap

    def finish(self) -> AgentUsage:
        with self.lock:
            if self.snapshot is None:
                records = [
                    UsageObservation(
                        reported=call.reported,
                        model=call.model,
                        provider=call.provider,
                        gap=call.gap if call.settled else GapReason.UNFINISHED_WORK,
                        tokens_aligned=True,
                        is_model_call=True,
                    )
                    for call in self.calls
                ]
                if self.children:
                    records.append(
                        UsageObservation(
                            reported=ReportedUsage(), gap=GapReason.UNFINISHED_WORK
                        )
                    )
                self.snapshot = usage_from_observations(records)
            return self.snapshot


class ObservedBody(httpx.SyncByteStream):
    """Read exactly the bytes the SDK asks for; retain only bounded usage events."""

    def __init__(
        self,
        stream: httpx.SyncByteStream,
        owner: InvocationUsage,
        index: int,
        *,
        streaming: bool,
        api_mode: str,
    ) -> None:
        self.stream, self.owner, self.index = stream, owner, index
        self.streaming, self.api_mode = streaming, api_mode
        self.buffer = b""
        self.event: list[bytes] = []
        self.event_bytes = 0
        self.lost = False
        self.parse_loss = False
        self.seen_usage = False
        self.finished = False
        self.terminal = False
        self.reported = ReportedUsage()
        self.model: str | None = None

    def _response(self, value: Any) -> None:
        if not isinstance(value, dict):
            self.parse_loss = True
            return
        terminal_response = value.get("type") in {
            "response.completed",
            "response.incomplete",
            "response.failed",
        }
        if terminal_response:
            self.terminal = True
        # Responses SSE carries the terminal response inside the event envelope.
        body = value.get("response", value)
        if isinstance(body, dict) and isinstance(body.get("usage"), dict):
            self.reported, self.model = raw_usage(body, self.api_mode)
            self.seen_usage = True
        elif terminal_response or (
            isinstance(body, dict) and "usage" in body and self.seen_usage
        ):
            self.parse_loss = True

    def _json(self, payload: bytes) -> None:
        try:
            self._response(json.loads(payload))
        except (ValueError, UnicodeError):
            self.parse_loss = True

    def _event(self) -> None:
        if self.event and not self.lost:
            payload = b"\n".join(self.event)
            if payload.strip() == b"[DONE]":
                self.terminal = True
            else:
                self._json(payload)
        self.parse_loss |= self.lost
        self.event, self.event_bytes = [], 0
        self.lost = False

    def _feed(self, chunk: bytes) -> None:
        if not self.streaming:
            if len(self.buffer) + len(chunk) <= _MAX_EVENT_BYTES:
                self.buffer += chunk
            else:
                self.lost = True
                self.parse_loss = True
            return
        self.buffer += chunk
        while b"\n" in self.buffer:
            line, self.buffer = self.buffer.split(b"\n", 1)
            line = line.rstrip(b"\r")
            if not line:
                self._event()
            elif line.startswith(b"data:"):
                data = line[5:].lstrip(b" ")
                self.event_bytes += len(data)
                if self.event_bytes <= _MAX_EVENT_BYTES:
                    self.event.append(data)
                else:
                    self.lost = True
                    self.parse_loss = True
        if len(self.buffer) > _MAX_EVENT_BYTES:
            self.buffer = b""
            self.lost = True
            self.parse_loss = True

    def __iter__(self) -> Iterator[bytes]:
        completed = False
        try:
            for chunk in self.stream:
                self._feed(chunk)
                yield chunk
            if self.streaming:
                self._feed(b"\n\n")
            elif not self.lost:
                self._json(self.buffer)
            completed = True
        finally:
            self.finished = True
            self.owner.settle(
                self.index,
                self.reported,
                model=self.model,
                gap=self._gap(completed=completed),
            )

    def close(self) -> None:
        try:
            self.stream.close()
        finally:
            if not self.finished:
                self.finished = True
                self.owner.settle(
                    self.index,
                    self.reported,
                    model=self.model,
                    gap=self._gap(completed=False),
                )

    def _gap(self, *, completed: bool) -> GapReason | None:
        if self.parse_loss:
            return GapReason.SCOPE_UNVERIFIED
        if self.streaming:
            return None if self.terminal else GapReason.UNFINISHED_WORK
        return None if completed else GapReason.MISSING_USAGE


def observe_request(request: httpx.Request) -> None:
    owner = _OWNER.get()
    if owner is None:
        return
    try:
        body = json.loads(request.content)
    except (ValueError, UnicodeError, httpx.RequestNotRead):
        return
    if not isinstance(body, dict) or not isinstance(body.get("model"), str):
        return
    operation = _OPERATION.get()
    index = owner.start(body["model"], qualified_provider(str(request.url)))
    if operation is not None:
        operation.calls.append(index)
    request.extensions[_REQUEST_KEY] = (
        owner,
        index,
        bool(body.get("stream")),
        "anthropic_messages"
        if request.url.path.endswith("/messages")
        else "chat_completions",
    )


def observe_response(response: httpx.Response) -> None:
    context = response.request.extensions.get(_REQUEST_KEY)
    if context is None:
        return
    owner, index, streaming, api_mode = context
    if not response.is_success:
        owner.settle(index, ReportedUsage(), gap=GapReason.MISSING_USAGE)
        return
    observed = ObservedBody(
        response.stream, owner, index, streaming=streaming, api_mode=api_mode
    )
    if response.is_stream_consumed:
        # Middleware/test transports may have consumed the body before this hook.
        # Read the cached bytes only; this never causes an early network read.
        observed._feed(response.content)
        if streaming:
            observed._feed(b"\n\n")
        elif not observed.lost:
            observed._json(observed.buffer)
        owner.settle(
            index,
            observed.reported,
            model=observed.model,
            gap=observed._gap(completed=True),
        )
    else:
        response.stream = observed


def transport_qualified(agent_type: type) -> bool:
    """Qualify only the two files owning this private per-client construction seam."""
    try:
        method = agent_type._build_keepalive_http_client
        parameters = inspect.signature(method).parameters
        if (
            list(parameters) != ["base_url", "verify"]
            or parameters["verify"].kind is not inspect.Parameter.KEYWORD_ONLY
        ):
            return False
        source = Path(inspect.getfile(method)).parent
        return all(
            sha256((source / name).read_bytes()).hexdigest() == digest
            for name, digest in _SOURCE_HASHES.items()
        )
    except (AttributeError, TypeError, OSError):
        return False


def observed_agent_type(agent_type: type) -> type:
    """Keep execution on an incompatible Hermes; its middleware remains observed."""
    if not transport_qualified(agent_type):
        return agent_type

    class ObservedAgent(agent_type):
        @staticmethod
        def _build_keepalive_http_client(
            base_url: str = "", *, verify: Any = True
        ) -> Any:
            client = agent_type._build_keepalive_http_client(base_url, verify=verify)
            if isinstance(client, httpx.Client):
                client.event_hooks["request"].append(observe_request)
                client.event_hooks["response"].append(observe_response)
            return client

    return ObservedAgent


def install_observer(registration_tag: object) -> list[Any]:
    try:
        from hermes_cli.plugins import PluginContext, PluginManifest, get_plugin_manager
    except ImportError:
        return []

    context = PluginContext(
        PluginManifest(name="nemo-fabric-usage"), get_plugin_manager()
    )
    auxiliary: dict[tuple[str, int], tuple[InvocationUsage, int]] = {}
    child_owners: dict[str, ReferenceType[InvocationUsage]] = {}
    lock = RLock()

    def owner_for(payload: dict[str, Any]) -> InvocationUsage | None:
        session_id = payload.get("session_id")
        with lock:
            child_owner = (
                child_owners.get(session_id) if isinstance(session_id, str) else None
            )
        # A collected child's route must not fall through to the current turn.
        owner = child_owner() if child_owner is not None else _OWNER.get()
        return (
            owner
            if owner is not None and owner.registration_tag is registration_tag
            else None
        )

    def execution(
        *, request: dict[str, Any], next_call: Callable[..., Any], **payload: Any
    ) -> Any:
        owner = owner_for(payload)
        if owner is None:
            return next_call(request)
        operation = Operation(
            owner, payload.get("model"), qualified_provider(payload.get("base_url"))
        )
        token = _OPERATION.set(operation)
        owner_token = _OWNER.set(owner)
        try:
            response = next_call(request)
        except BaseException:
            if not operation.calls:
                owner.settle(
                    owner.start(operation.model, operation.provider),
                    ReportedUsage(),
                    gap=GapReason.MISSING_USAGE,
                )
            raise
        else:
            if not operation.calls:
                reported, model = raw_usage(
                    response, str(payload.get("api_mode") or "")
                )
                owner.settle(
                    owner.start(operation.model, operation.provider),
                    reported,
                    model=model,
                    gap=GapReason.SCOPE_UNVERIFIED,
                )
            return response
        finally:
            _OWNER.reset(owner_token)
            _OPERATION.reset(token)

    def auxiliary_start(**payload: Any) -> None:
        owner = owner_for(payload)
        key = (payload.get("api_request_id"), payload.get("api_call_count"))
        if owner is not None and isinstance(key[0], str) and type(key[1]) is int:
            with lock:
                if key not in auxiliary:
                    auxiliary[key] = (
                        owner,
                        owner.start(
                            payload.get("model"),
                            qualified_provider(payload.get("base_url")),
                        ),
                    )

    def auxiliary_end(**payload: Any) -> None:
        key = (payload.get("api_request_id"), payload.get("api_call_count"))
        with lock:
            context = auxiliary.pop(key, None)
        if context is None:
            return
        owner, index = context
        raw = _mapping(payload.get("usage"))
        # CanonicalUsage has already zero-filled omission. Positive values survive;
        # zero presence cannot be recovered. A positive uncached bucket proves
        # the OpenAI normalizer did not clamp prompt-minus-cache at zero; in that
        # case its public prompt_tokens preserves the original inclusive prompt.
        uncached = _count(raw.get("input_tokens"), presence_lost=True)
        inclusive = (
            _count(raw.get("prompt_tokens"), presence_lost=True)
            if uncached is not None
            and payload.get("api_mode") in {"chat_completions", "codex_responses"}
            else None
        )
        reported = ReportedUsage(
            input_tokens=inclusive if inclusive is not None else uncached,
            input_tokens_include_cache=inclusive is not None,
            cached_input_tokens=_count(
                raw.get("cache_read_tokens"), presence_lost=True
            ),
            cache_write_input_tokens=_count(
                raw.get("cache_write_tokens"), presence_lost=True
            ),
            output_tokens=_count(raw.get("output_tokens"), presence_lost=True),
            output_tokens_include_reasoning=True,
        )
        owner.settle(
            index,
            reported,
            model=payload.get("response_model"),
            gap=GapReason.MISSING_USAGE if payload.get("error") or not raw else None,
        )

    def child_start(**payload: Any) -> None:
        owner = owner_for(payload)
        identity = payload.get("child_session_id")
        if owner is not None and isinstance(identity, str):
            with lock, owner.lock:
                child_owners[identity] = ref(owner)
                owner.children.add(identity)

    def child_stop(**payload: Any) -> None:
        identity = payload.get("child_session_id")
        with lock:
            child_owner = child_owners.get(identity)
        owner = child_owner() if child_owner is not None else None
        if owner is not None:
            with owner.lock:
                owner.children.discard(identity)

    return [
        context.register_middleware("llm_execution", execution),
        context.register_hook("pre_auxiliary_call", auxiliary_start),
        context.register_hook("post_auxiliary_call", auxiliary_end),
        context.register_hook("subagent_start", child_start),
        context.register_hook("subagent_stop", child_stop),
    ]


@dataclass
class MethodObservation:
    target: Any
    original: Callable[..., Any]
    installed: Callable[..., Any]
    scope_unverified: bool = False

    def dispose(self) -> None:
        if (
            self.target is not None
            and getattr(self.target, "compress", None) is self.installed
        ):
            self.target.compress = self.original


def observe_context_engine(
    agent: Any, registration_tag: object
) -> MethodObservation | None:
    """Observe actual calls to an owned plugin engine's public compression method."""
    from functools import wraps
    from agent.context_compressor import ContextCompressor

    engine = getattr(agent, "context_compressor", None)
    if engine is None or type(engine) is ContextCompressor:
        return None
    from hermes_cli.plugins import get_plugin_context_engine

    original = getattr(engine, "compress", None)
    if not callable(original):
        return None
    if engine is get_plugin_context_engine():
        # An engine that returned its registry singleton from clone_for_agent
        # violates the per-agent ownership premise. Preserve execution untouched.
        return MethodObservation(None, original, original, scope_unverified=True)

    @wraps(original)
    def compress(*args: Any, **kwargs: Any) -> Any:
        owner = _OWNER.get()
        if owner is None or owner.registration_tag is not registration_tag:
            return original(*args, **kwargs)
        index = owner.start()
        try:
            return original(*args, **kwargs)
        finally:
            # A plugin may call outside Hermes' auxiliary-client contract.
            owner.settle(index, ReportedUsage(), gap=GapReason.SCOPE_UNVERIFIED)

    if inspect.iscoroutinefunction(original):

        @wraps(original)
        async def async_compress(*args: Any, **kwargs: Any) -> Any:
            owner = _OWNER.get()
            if owner is None or owner.registration_tag is not registration_tag:
                return await original(*args, **kwargs)
            index = owner.start()
            try:
                return await original(*args, **kwargs)
            finally:
                owner.settle(index, ReportedUsage(), gap=GapReason.SCOPE_UNVERIFIED)

        observed = async_compress
    else:
        observed = compress
    try:
        engine.compress = observed
    except (AttributeError, TypeError):
        return MethodObservation(None, original, original, scope_unverified=True)
    return MethodObservation(engine, original, observed)
