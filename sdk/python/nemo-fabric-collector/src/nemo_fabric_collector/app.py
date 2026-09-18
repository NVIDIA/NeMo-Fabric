# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Starlette application for collecting and routing ATOF records."""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, NewType

from starlette.applications import Starlette
from starlette.requests import ClientDisconnect, Request
from starlette.responses import (
    JSONResponse,
    PlainTextResponse,
    Response,
    StreamingResponse,
)
from starlette.routing import Route

RequestId = NewType("RequestId", str)
ScopeUuid = NewType("ScopeUuid", str)

_MAX_RECORD_BYTES = 1024 * 1024
_QUEUE_MAX_BYTES = 16 * 1024 * 1024
_QUEUE_MAXSIZE = 1024
_QUEUE_PUT_TIMEOUT_SECONDS = 30.0
_COMPLETION_WAIT_TIMEOUT_SECONDS = 1.0
_PI_TURN_WINDOW = "pi_turn_window"
_PI_BOUNDARY_ACTIONS = frozenset({"preserve", "release", "wait"})
_MAX_CANCELLED_REGISTRATION_TOKENS = 1024

logger = logging.getLogger(__name__)


class _RecordTooLarge(ValueError):
    pass


class _AtofQueueFull(Exception):
    pass


class _SubscriptionPhase(Enum):
    REGISTERED = auto()
    STREAMING = auto()
    DISCONNECTED = auto()
    DRAINING = auto()
    CLOSED = auto()


class _TerminationReason(Enum):
    COMPLETED = auto()
    DEREGISTERED = auto()
    CLIENT_CANCELLED = auto()
    LEASE_EXPIRED = auto()
    PUBLISHER_FAILED = auto()
    COLLECTOR_SHUTDOWN = auto()
    COLLECTOR_ERROR = auto()


class _AtofQueueClosed(Exception):
    def __init__(
        self,
        reason: _TerminationReason,
        error: BaseException | None = None,
    ) -> None:
        super().__init__(f"ATOF record queue closed: {reason.name.lower()}")
        self.reason = reason
        self.error = error


class _AtofRecordQueue:
    def __init__(
        self,
        *,
        maxsize: int,
        max_bytes: int,
        put_timeout: float = _QUEUE_PUT_TIMEOUT_SECONDS,
    ) -> None:
        self._records: deque[tuple[dict[str, Any], int]] = deque()
        self._maxsize = maxsize
        self._max_bytes = max_bytes
        self._put_timeout = put_timeout
        self._queued_bytes = 0
        self._closed = False
        self._drain_on_close = False
        self._termination_reason: _TerminationReason | None = None
        self._termination_error: BaseException | None = None
        self._changed = asyncio.Event()

    @property
    def closed(self) -> bool:
        return self._closed

    def empty(self) -> bool:
        return not self._records

    async def put(
        self,
        record: dict[str, Any],
        *,
        byte_size: int | None = None,
    ) -> None:
        size = byte_size if byte_size is not None else _record_size(record)
        if size > self._max_bytes:
            raise _RecordTooLarge

        while True:
            if self._closed:
                raise self._closed_error()
            if (
                len(self._records) < self._maxsize
                and self._queued_bytes + size <= self._max_bytes
            ):
                self._records.append((record, size))
                self._queued_bytes += size
                self._changed.set()
                return
            self._changed.clear()
            try:
                await asyncio.wait_for(self._changed.wait(), self._put_timeout)
            except TimeoutError:
                raise _AtofQueueFull from None

    async def get(self) -> dict[str, Any]:
        while True:
            if self._records and (not self._closed or self._drain_on_close):
                record, size = self._records.popleft()
                self._queued_bytes -= size
                self._changed.set()
                return record
            if self._closed:
                raise self._closed_error()
            self._changed.clear()
            await self._changed.wait()

    def close(
        self,
        *,
        drain: bool,
        reason: _TerminationReason,
        error: BaseException | None = None,
    ) -> None:
        if self._closed:
            return
        self._closed = True
        self._drain_on_close = drain
        self._termination_reason = reason
        self._termination_error = error
        if not drain:
            self._records.clear()
            self._queued_bytes = 0
        self._changed.set()

    def _closed_error(self) -> _AtofQueueClosed:
        reason = self._termination_reason or _TerminationReason.COLLECTOR_ERROR
        return _AtofQueueClosed(reason, self._termination_error)


@dataclass
class _RequestState:
    phase: _SubscriptionPhase = _SubscriptionPhase.REGISTERED
    stream_token: object | None = None
    correlation_mode: str | None = None
    routing_ready: bool = True
    capture_records: bool = True
    boundary_preserved: bool = False
    boundary_timed_out: bool = False
    boundary_generation: int | None = None
    registration_token: str | None = None
    completion_seen: asyncio.Event = field(default_factory=asyncio.Event)


class _StreamAlreadyAttached(Exception):
    pass


class AtofCollector:
    """Maintain in-memory request registrations and route ATOF records."""

    def __init__(
        self,
        *,
        queue_maxsize: int = _QUEUE_MAXSIZE,
        queue_max_bytes: int = _QUEUE_MAX_BYTES,
        standalone: bool = False,
        completion_wait_timeout: float = _COMPLETION_WAIT_TIMEOUT_SECONDS,
    ):
        # Consider moving these to a database allowing for multiple workers
        self.request_uuids: dict[RequestId, set[ScopeUuid]] = {}
        self.uuid_to_request: dict[ScopeUuid, RequestId] = {}
        self.request_messages: dict[RequestId, _AtofRecordQueue] = {}
        self.request_states: dict[RequestId, _RequestState] = {}
        self.state_lock = asyncio.Lock()
        self._queue_maxsize = queue_maxsize
        self._queue_max_bytes = queue_max_bytes
        self._standalone = standalone
        self._completion_wait_timeout = completion_wait_timeout
        self._pi_boundary_ready = asyncio.Event()
        self._pi_boundary_ready.set()
        self._pi_boundary_generation = 0
        self._pi_boundary_owner: int | None = None
        self._cancelled_registration_tokens: dict[tuple[RequestId, str], None] = {}

    async def register(
        self,
        request_id: RequestId,
        *,
        correlation_mode: str | None = None,
        capture_records: bool = True,
        registration_token: str | None = None,
    ) -> None:
        if correlation_mode is not None and correlation_mode != _PI_TURN_WINDOW:
            raise RuntimeError(f"unsupported correlation mode {correlation_mode!r}")
        if not capture_records and correlation_mode != _PI_TURN_WINDOW:
            raise RuntimeError("discarding records requires the Pi correlation mode")
        if registration_token is not None and correlation_mode != _PI_TURN_WINDOW:
            raise RuntimeError("registration tokens require the Pi correlation mode")
        if correlation_mode == _PI_TURN_WINDOW:
            try:
                await asyncio.wait_for(
                    self._pi_boundary_ready.wait(),
                    timeout=self._completion_wait_timeout,
                )
            except TimeoutError:
                raise RuntimeError(
                    "previous Pi invocation boundary is unresolved"
                ) from None

        async with self.state_lock:
            if (
                registration_token is not None
                and (request_id, registration_token)
                in self._cancelled_registration_tokens
            ):
                raise RuntimeError("registration attempt was cancelled")
            if request_id in self.request_states:
                raise RuntimeError(f"request_id {request_id!r} is already registered")

            if self._standalone and self.request_uuids:
                raise RuntimeError(
                    "standalone collector already has a registered request"
                )
            if correlation_mode is not None and not self._standalone:
                raise RuntimeError("correlation modes require a standalone collector")
            if correlation_mode == _PI_TURN_WINDOW and (
                not self._pi_boundary_ready.is_set()
                or self._pi_boundary_owner is not None
            ):
                raise RuntimeError("previous Pi invocation boundary is unresolved")

            boundary_generation = None
            if correlation_mode == _PI_TURN_WINDOW:
                self._pi_boundary_generation += 1
                boundary_generation = self._pi_boundary_generation
            self.request_uuids[request_id] = set()
            self.request_messages[request_id] = _AtofRecordQueue(
                maxsize=self._queue_maxsize,
                max_bytes=self._queue_max_bytes,
            )
            self.request_states[request_id] = _RequestState(
                correlation_mode=correlation_mode,
                routing_ready=correlation_mode != _PI_TURN_WINDOW,
                capture_records=capture_records,
                boundary_generation=boundary_generation,
                registration_token=registration_token,
            )
            if correlation_mode == _PI_TURN_WINDOW:
                self._pi_boundary_owner = boundary_generation
                self._pi_boundary_ready.clear()

    async def attach_stream(
        self,
        request_id: RequestId,
        *,
        registration_token: str | None = None,
    ) -> tuple[_AtofRecordQueue, object] | None:
        async with self.state_lock:
            state = self.request_states.get(request_id)
            queue = self.request_messages.get(request_id)
            if state is None or queue is None:
                return None
            if state.registration_token != registration_token:
                return None
            if state.stream_token is not None:
                raise _StreamAlreadyAttached

            token = object()
            state.stream_token = token
            if state.phase in {
                _SubscriptionPhase.REGISTERED,
                _SubscriptionPhase.DISCONNECTED,
            }:
                state.phase = _SubscriptionPhase.STREAMING
            return queue, token

    async def detach_stream(
        self,
        request_id: RequestId,
        queue: _AtofRecordQueue,
        token: object,
    ) -> None:
        async with self.state_lock:
            state = self.request_states.get(request_id)
            if state is None or state.stream_token is not token:
                return
            state.stream_token = None
            if queue.closed and queue.empty():
                if state.boundary_preserved:
                    state.phase = _SubscriptionPhase.DISCONNECTED
                else:
                    state.phase = _SubscriptionPhase.CLOSED
                    self.request_messages.pop(request_id, None)
                    self.request_states.pop(request_id, None)
            elif state.phase is _SubscriptionPhase.STREAMING:
                state.phase = _SubscriptionPhase.DISCONNECTED

    async def deregister(
        self,
        request_id: RequestId,
        *,
        remove_queue: bool,
        pi_boundary: str | None = None,
        registration_token: str | None = None,
    ) -> None:
        if pi_boundary is not None and pi_boundary not in _PI_BOUNDARY_ACTIONS:
            raise RuntimeError(f"unsupported Pi boundary action {pi_boundary!r}")
        if pi_boundary == "wait":
            await self._wait_for_completion(request_id, registration_token)
        async with self.state_lock:
            queue = self.request_messages.get(request_id)
            state = self.request_states.get(request_id)
            if registration_token is not None and (
                state is None or state.registration_token != registration_token
            ):
                if pi_boundary == "release":
                    self._remember_cancelled_registration(
                        request_id,
                        registration_token,
                    )
                return
            if queue is None or state is None:
                return
            if pi_boundary is not None and state.correlation_mode != _PI_TURN_WINDOW:
                raise RuntimeError(
                    "Pi boundary actions require the Pi correlation mode"
                )

            if pi_boundary == "preserve":
                # A consumer can disconnect while native execution is still in
                # flight. Stop capture but retain the Pi lease and its routing so
                # only the invocation outcome may release the boundary.
                if request_id not in self.request_uuids:
                    # The outcome won the race and already released this lease.
                    # A late stream finalizer may clean its queue, but must not
                    # re-arm the preserved state.
                    if remove_queue:
                        state.phase = _SubscriptionPhase.CLOSED
                        self.request_messages.pop(request_id, None)
                        self.request_states.pop(request_id, None)
                        queue.close(
                            drain=False,
                            reason=_TerminationReason.DEREGISTERED,
                        )
                    return
                state.capture_records = False
                state.boundary_preserved = True
                state.phase = _SubscriptionPhase.DISCONNECTED
                if remove_queue:
                    queue.close(
                        drain=False,
                        reason=_TerminationReason.DEREGISTERED,
                    )
                return

            state.boundary_preserved = False
            self._remove_routes(request_id)
            if pi_boundary == "release":
                # Native invocation failures and Pi results that never started
                # an agent run have no terminal hook to await.
                self._resolve_pi_boundary(state)
            elif pi_boundary == "wait" and (
                not state.boundary_timed_out or state.completion_seen.is_set()
            ):
                # Publish lease availability only after its routes are removed,
                # so a waiting registration cannot race the completed owner.
                self._resolve_pi_boundary(state)

            if remove_queue:
                state.phase = _SubscriptionPhase.CLOSED
                self.request_messages.pop(request_id, None)
                self.request_states.pop(request_id, None)
                queue.close(
                    drain=False,
                    reason=_TerminationReason.DEREGISTERED,
                )
                return

            state.phase = _SubscriptionPhase.DRAINING
            queue.close(
                drain=True,
                reason=_TerminationReason.DEREGISTERED,
            )
            if queue.empty() and state.stream_token is None:
                self.request_messages.pop(request_id, None)
                self.request_states.pop(request_id, None)

    async def route(self, record: dict[str, Any], *, byte_size: int) -> None:
        async with self.state_lock:
            request_id = self._route_request(record)
            if request_id is None:
                return
            queue = self.request_messages.get(request_id)
            state = self.request_states.get(request_id)

        if queue is None or state is None:
            return
        pi_completion = state.correlation_mode == _PI_TURN_WINDOW and _is_pi_completion(
            record
        )
        if state.capture_records:
            try:
                await queue.put(record, byte_size=byte_size)
            except _AtofQueueClosed:
                # Preserve the successful publisher response for a partially
                # processed NDJSON payload rather than causing a retry that could
                # duplicate records already enqueued from that payload.
                pass
            except _AtofQueueFull:
                logger.warning(
                    "Dropping ATOF record after queue backpressure timeout",
                    extra={
                        "request_id": request_id,
                        "byte_size": byte_size,
                        "timeout_seconds": _QUEUE_PUT_TIMEOUT_SECONDS,
                    },
                )
                if not pi_completion:
                    return

        if pi_completion:
            async with self.state_lock:
                current = (
                    self.request_states.get(request_id) is state
                    and self.request_messages.get(request_id) is queue
                )
                if current:
                    state.completion_seen.set()
                    if state.boundary_timed_out and not self.request_uuids:
                        # The terminal marker was selected before the timeout
                        # but could not enter a backpressured queue until its
                        # routes were removed. Do not reopen a newer Pi lease.
                        self._resolve_pi_boundary(state)
                elif not self.request_uuids:
                    # A record selected for a timed-out request can finish
                    # queueing after that request is removed. It still closes
                    # the quarantine unless another Pi lease already owns the
                    # collector.
                    self._resolve_pi_boundary(state)

    async def close(self) -> None:
        async with self.state_lock:
            queues = tuple(self.request_messages.values())
            self.request_uuids.clear()
            self.uuid_to_request.clear()
            self.request_messages.clear()
            self.request_states.clear()
            self._cancelled_registration_tokens.clear()
            self._pi_boundary_owner = None
            self._pi_boundary_ready.set()
            for queue in queues:
                queue.close(
                    drain=False,
                    reason=_TerminationReason.COLLECTOR_SHUTDOWN,
                )

    def _route_request(self, record: dict[str, Any]) -> RequestId | None:
        if self._standalone:
            if len(self.request_uuids) == 0:
                if not self._pi_boundary_ready.is_set() and _is_pi_completion(record):
                    # A previous Pi batch arrived after its bounded completion
                    # wait. Drop the entire batch and reopen registration only
                    # at its ordered terminal marker.
                    self._pi_boundary_owner = None
                    self._pi_boundary_ready.set()
                return None

            request_id = next(iter(self.request_uuids))
            state = self.request_states.get(request_id)
            if state is None:
                return None
            if state.correlation_mode == _PI_TURN_WINDOW and not state.routing_ready:
                # Pi has no Fabric request ID. Serialized leases guarantee that
                # the first turn start, or a zero-turn terminal marker, belongs
                # to this invocation rather than to a preceding agent run.
                if not (_is_pi_turn_start(record) or _is_pi_completion(record)):
                    return None
                state.routing_ready = True
            return request_id

        uuid = _record_uuid(record)
        if uuid is None:
            return None

        # In the current streaming.py implementation, there was specific handling
        # for Hermes turns. Ask Yuchen why this was needed.
        root_request_id = _root_request_id(record)
        if root_request_id is not None and self._accepts_records(root_request_id):
            if not self._associate_scope(uuid, root_request_id):
                return None
            return root_request_id

        request_id = self.uuid_to_request.get(uuid)
        if request_id is None:
            parent_uuid = _parent_uuid(record)
            if parent_uuid is not None:
                request_id = self.uuid_to_request.get(parent_uuid)
        if request_id is None or not self._accepts_records(request_id):
            return None

        if _is_scope_start(record):
            if not self._associate_scope(uuid, request_id):
                return None
        return request_id

    def _associate_scope(self, uuid: ScopeUuid, request_id: RequestId) -> bool:
        existing_request = self.uuid_to_request.get(uuid)
        if existing_request is not None and existing_request != request_id:
            return False
        request_uuids = self.request_uuids.get(request_id)
        if request_uuids is None:
            return False
        request_uuids.add(uuid)
        self.uuid_to_request[uuid] = request_id
        return True

    def _resolve_pi_boundary(self, state: _RequestState) -> None:
        if (
            state.boundary_generation is not None
            and self._pi_boundary_owner == state.boundary_generation
        ):
            self._pi_boundary_owner = None
            self._pi_boundary_ready.set()

    def _remember_cancelled_registration(
        self,
        request_id: RequestId,
        registration_token: str,
    ) -> None:
        key = (request_id, registration_token)
        self._cancelled_registration_tokens[key] = None
        while (
            len(self._cancelled_registration_tokens)
            > _MAX_CANCELLED_REGISTRATION_TOKENS
        ):
            oldest = next(iter(self._cancelled_registration_tokens))
            self._cancelled_registration_tokens.pop(oldest)

    def _accepts_records(self, request_id: RequestId) -> bool:
        state = self.request_states.get(request_id)
        return state is not None and state.phase in {
            _SubscriptionPhase.REGISTERED,
            _SubscriptionPhase.STREAMING,
            _SubscriptionPhase.DISCONNECTED,
        }

    def _remove_routes(self, request_id: RequestId) -> None:
        for uuid in self.request_uuids.pop(request_id, set()):
            if self.uuid_to_request.get(uuid) == request_id:
                self.uuid_to_request.pop(uuid, None)

    async def _wait_for_completion(
        self,
        request_id: RequestId,
        registration_token: str | None,
    ) -> None:
        async with self.state_lock:
            state = self.request_states.get(request_id)
            if (
                state is None
                or state.correlation_mode != _PI_TURN_WINDOW
                or (
                    registration_token is not None
                    and state.registration_token != registration_token
                )
            ):
                return
            completion_seen = state.completion_seen
        if completion_seen.is_set():
            return
        try:
            await asyncio.wait_for(
                completion_seen.wait(),
                timeout=self._completion_wait_timeout,
            )
        except TimeoutError:
            async with self.state_lock:
                state = self.request_states.get(request_id)
                if (
                    state is None
                    or state.completion_seen.is_set()
                    or (
                        registration_token is not None
                        and state.registration_token != registration_token
                    )
                ):
                    return
                state.boundary_timed_out = True
                self._pi_boundary_ready.clear()
            # Do not hold the completed invocation open indefinitely. A later
            # registration fails closed until this invocation's ordered terminal
            # marker arrives; all records in that delayed batch are discarded.
            logger.warning(
                "Timed out waiting for the Pi ATOF invocation boundary",
                extra={
                    "request_id": request_id,
                    "timeout_seconds": self._completion_wait_timeout,
                },
            )


def _record_size(record: dict[str, Any]) -> int:
    return len(json.dumps(record, separators=(",", ":"), ensure_ascii=False).encode())


def _is_scope_start(record: dict[str, Any]) -> bool:
    return record.get("kind") == "scope" and record.get("scope_category") == "start"


def _record_uuid(record: dict[str, Any]) -> ScopeUuid | None:
    value = record.get("uuid")
    return ScopeUuid(value) if isinstance(value, str) else None


def _parent_uuid(record: dict[str, Any]) -> ScopeUuid | None:
    value = record.get("parent_uuid")
    return ScopeUuid(value) if isinstance(value, str) else None


def _root_request_id(record: dict[str, Any]) -> RequestId | None:
    if not _is_scope_start(record):
        return None
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        return None
    value = metadata.get("nemo_fabric_request_id")
    return RequestId(value) if isinstance(value, str) and value else None


def _record_metadata(record: dict[str, Any]) -> dict[str, Any]:
    metadata = record.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _is_pi_turn_start(record: dict[str, Any]) -> bool:
    metadata = _record_metadata(record)
    return (
        _is_scope_start(record)
        and metadata.get("agent_kind") == "pi"
        and metadata.get("nemo_relay_scope_role") == "turn"
        and metadata.get("hook_event_name") == "turn_start"
        and metadata.get("turn_source") == "turn_start"
    )


def _is_pi_completion(record: dict[str, Any]) -> bool:
    metadata = _record_metadata(record)
    return (
        record.get("kind") == "mark"
        and metadata.get("agent_kind") == "pi"
        and metadata.get("hook_event_name") == "agent_settled"
    )


def _collector(request: Request) -> AtofCollector:
    return request.app.state.collector


def _authorize(request: Request, token: str | None) -> Response | None:
    if token is None:
        return None
    authorization = request.headers.get("authorization", "")
    parts = authorization.split(None, 1)
    if (
        len(parts) == 2
        and parts[0].lower() == "bearer"
        and parts[1].isascii()
        # Use constant-time comparison to prevent timing attacks
        and secrets.compare_digest(parts[1], token)
    ):
        return None
    return JSONResponse(
        {"detail": "Unauthorized"},
        status_code=401,
        headers={"WWW-Authenticate": "Bearer"},
    )


async def healthz(_: Request) -> Response:
    return PlainTextResponse("ok")


async def register(request: Request) -> Response:
    unauthorized = _authorize(request, request.app.state.control_token)
    if unauthorized is not None:
        return unauthorized
    payload = await _request_json(request)
    if payload is None:
        return _error_response(400, "Request body must be a JSON object")
    request_id = _request_id(payload)
    if request_id is None:
        return _error_response(400, "request_id must be a non-empty string")
    correlation_mode = payload.get("correlation_mode")
    if correlation_mode is not None and correlation_mode != _PI_TURN_WINDOW:
        return _error_response(400, "correlation_mode is not supported")
    capture_records = payload.get("capture_records", True)
    if not isinstance(capture_records, bool):
        return _error_response(400, "capture_records must be a boolean")
    if not capture_records and correlation_mode != _PI_TURN_WINDOW:
        return _error_response(
            400,
            "capture_records=false requires the Pi correlation mode",
        )
    registration_token = payload.get("registration_token")
    if registration_token is not None and (
        not isinstance(registration_token, str)
        or not registration_token
        or len(registration_token) > 128
        or not registration_token.isascii()
    ):
        return _error_response(
            400,
            "registration_token must be a non-empty ASCII string up to 128 characters",
        )
    if registration_token is not None and correlation_mode != _PI_TURN_WINDOW:
        return _error_response(
            400,
            "registration_token requires the Pi correlation mode",
        )
    try:
        await _collector(request).register(
            request_id,
            correlation_mode=correlation_mode,
            capture_records=capture_records,
            registration_token=registration_token,
        )
    except Exception as error:
        return _error_response(409, str(error))
    return JSONResponse(
        {"request_id": request_id, "status": "ready"},
        status_code=201,
    )


async def stream(request: Request) -> Response:
    unauthorized = _authorize(request, request.app.state.control_token)
    if unauthorized is not None:
        return unauthorized
    request_id = RequestId(request.path_params["request_id"])
    registration_token = request.query_params.get("registration_token")
    if registration_token is not None and (
        not registration_token
        or len(registration_token) > 128
        or not registration_token.isascii()
    ):
        return _error_response(
            400,
            "registration_token must be a non-empty ASCII string up to 128 characters",
        )
    try:
        attached = await _collector(request).attach_stream(
            request_id,
            registration_token=registration_token,
        )
    except _StreamAlreadyAttached:
        return _error_response(409, "request_id already has an attached stream")
    if attached is None:
        return _error_response(404, "request_id is not registered")
    queue, token = attached

    async def records() -> AsyncIterator[bytes]:
        try:
            while True:
                try:
                    record = await queue.get()
                except _AtofQueueClosed as error:
                    if error.error is not None:
                        raise RuntimeError("ATOF stream terminated") from error.error
                    return
                yield (
                    json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n"
                ).encode()
        finally:
            await _collector(request).detach_stream(request_id, queue, token)

    return StreamingResponse(records(), media_type="application/x-ndjson")


async def deregister(request: Request) -> Response:
    unauthorized = _authorize(request, request.app.state.control_token)
    if unauthorized is not None:
        return unauthorized
    request_id = RequestId(request.path_params["request_id"])
    remove_queue = _query_bool(request, "remove_queue", default=False)
    if remove_queue is None:
        return _error_response(400, "remove_queue must be a boolean")
    pi_boundary = request.query_params.get("pi_boundary")
    if pi_boundary is not None and pi_boundary not in _PI_BOUNDARY_ACTIONS:
        return _error_response(
            400,
            "pi_boundary must be preserve, release, or wait",
        )
    registration_token = request.query_params.get("registration_token")
    if registration_token is not None and (
        not registration_token
        or len(registration_token) > 128
        or not registration_token.isascii()
    ):
        return _error_response(
            400,
            "registration_token must be a non-empty ASCII string up to 128 characters",
        )
    try:
        await _collector(request).deregister(
            request_id,
            remove_queue=remove_queue,
            pi_boundary=pi_boundary,
            registration_token=registration_token,
        )
    except RuntimeError as error:
        return _error_response(409, str(error))
    return Response(status_code=204)


async def atof(request: Request) -> Response:
    unauthorized = _authorize(request, request.app.state.publish_token)
    if unauthorized is not None:
        return unauthorized
    buffer = bytearray()
    try:
        async for chunk in request.stream():
            buffer.extend(chunk)
            while True:
                newline = buffer.find(b"\n")
                if newline < 0:
                    if len(buffer) > _MAX_RECORD_BYTES:
                        raise _RecordTooLarge
                    break
                if newline > _MAX_RECORD_BYTES:
                    raise _RecordTooLarge
                line = bytes(buffer[:newline])
                del buffer[: newline + 1]
                await _emit_atof_line(_collector(request), line)
        if buffer:
            if len(buffer) > _MAX_RECORD_BYTES:
                raise _RecordTooLarge
            await _emit_atof_line(_collector(request), bytes(buffer))
    except _RecordTooLarge:
        return _error_response(413, "ATOF record is too large")
    except ClientDisconnect:
        logger.info("ATOF publisher disconnected before completing the request")
    return Response(status_code=200)


async def _emit_atof_line(collector: AtofCollector, line: bytes) -> None:
    stripped = line.strip()
    if not stripped:
        return
    try:
        record = json.loads(stripped)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return
    if isinstance(record, dict):
        await collector.route(record, byte_size=len(stripped))


async def _request_json(request: Request) -> dict[str, Any] | None:
    try:
        payload = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _request_id(payload: dict[str, Any]) -> RequestId | None:
    value = payload.get("request_id")
    return RequestId(value) if isinstance(value, str) and value else None


def _query_bool(request: Request, name: str, *, default: bool) -> bool | None:
    value = request.query_params.get(name)
    if value is None:
        return default
    if value.lower() in {"1", "true"}:
        return True
    if value.lower() in {"0", "false"}:
        return False
    return None


def _error_response(status_code: int, detail: str) -> JSONResponse:
    return JSONResponse({"detail": detail}, status_code=status_code)


def create_app(
    collector: AtofCollector | None = None,
    *,
    publish_token: str | None = None,
    control_token: str | None = None,
    standalone: bool = False,
) -> Starlette:
    collector = collector or AtofCollector(standalone=standalone)

    @asynccontextmanager
    async def lifespan(_: Starlette) -> AsyncIterator[None]:
        try:
            yield
        finally:
            await collector.close()

    application = Starlette(
        routes=[
            Route("/healthz", healthz, methods=["GET"]),
            Route("/v1/register", register, methods=["POST"]),
            Route("/v1/stream/{request_id}", stream, methods=["GET"]),
            Route(
                "/v1/deregister-request/{request_id}",
                deregister,
                methods=["DELETE"],
            ),
            Route("/v1/atof", atof, methods=["POST"]),
        ],
        lifespan=lifespan,
    )
    application.state.collector = collector
    application.state.publish_token = publish_token
    application.state.control_token = control_token
    return application
