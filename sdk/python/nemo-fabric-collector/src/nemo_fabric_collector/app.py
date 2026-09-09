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
from dataclasses import dataclass
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
_QUEUE_PUT_TIMEOUT_SECONDS = 0.0

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

    async def register(self, request_id: RequestId) -> None:
        async with self.state_lock:
            if request_id in self.request_states:
                raise RuntimeError(
                    f"request_id {request_id!r} is already registered"
                )

            if self._standalone and self.request_uuids:
                raise RuntimeError(
                    "standalone collector already has a registered request"
                )

            self.request_uuids[request_id] = set()
            self.request_messages[request_id] = _AtofRecordQueue(
                maxsize=self._queue_maxsize,
                max_bytes=self._queue_max_bytes,
            )
            self.request_states[request_id] = _RequestState()

    async def attach_stream(
        self,
        request_id: RequestId,
    ) -> tuple[_AtofRecordQueue, object] | None:
        async with self.state_lock:
            state = self.request_states.get(request_id)
            queue = self.request_messages.get(request_id)
            if state is None or queue is None:
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
                state.phase = _SubscriptionPhase.CLOSED
                self.request_messages.pop(request_id, None)
                self.request_states.pop(request_id, None)
            elif state.phase is _SubscriptionPhase.STREAMING:
                state.phase = _SubscriptionPhase.DISCONNECTED

    async def deregister(self, request_id: RequestId, *, remove_queue: bool) -> None:
        async with self.state_lock:
            self._remove_routes(request_id)
            queue = self.request_messages.get(request_id)
            state = self.request_states.get(request_id)
            if queue is None or state is None:
                return

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

        if queue is None:
            return
        try:
            await queue.put(record, byte_size=byte_size)
        except (_AtofQueueClosed, _AtofQueueFull):
            # Preserve the successful publisher response for a partially
            # processed NDJSON payload rather than causing a retry that could
            # duplicate records already enqueued from that payload.
            pass

    async def close(self) -> None:
        async with self.state_lock:
            queues = tuple(self.request_messages.values())
            self.request_uuids.clear()
            self.uuid_to_request.clear()
            self.request_messages.clear()
            self.request_states.clear()
            for queue in queues:
                queue.close(
                    drain=False,
                    reason=_TerminationReason.COLLECTOR_SHUTDOWN,
                )

    def _route_request(self, record: dict[str, Any]) -> RequestId | None:
        if self._standalone:
            return next(iter(self.request_uuids))

        uuid = _record_uuid(record)
        if uuid is None:
            return None

        # In the current streaming.py implementation, there was specific handling
        # for Hermes turns. Ask Yuchen why this was needed.
        root_request_id = _root_request_id(record)
        if (
            root_request_id is not None
            and self._accepts_records(root_request_id)
        ):
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
    try:
        await _collector(request).register(request_id)
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
    try:
        attached = await _collector(request).attach_stream(request_id)
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
                    json.dumps(record, separators=(",", ":"), ensure_ascii=False)
                    + "\n"
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
    await _collector(request).deregister(request_id, remove_queue=remove_queue)
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
