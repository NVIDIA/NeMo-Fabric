# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Relay-backed streaming support for the NVIDIA NeMo Fabric Python SDK."""

from __future__ import annotations

import asyncio
import json
import warnings
from collections.abc import Coroutine
from contextlib import suppress
from typing import Any

from nemo_fabric.models import (
    FabricConfig,
    RelayAtofConfig,
    RelayAtofFileSinkConfig,
    RelayAtofStreamSinkConfig,
    RelayConfig,
    RelayObservabilityConfig,
)
from nemo_fabric.types import RunResult

_DRAIN_SECONDS = 0.25
_MAX_RECORD_BYTES = 1024 * 1024
_QUEUE_MAX_BYTES = 16 * 1024 * 1024
_QUEUE_MAXSIZE = 1024
_READ_SIZE = 64 * 1024
_STREAM_SINK_NAME = "nemo-fabric-stream"


class _RecordTooLarge(ValueError):
    pass


class _AtofRecordQueue:
    def __init__(self, *, maxsize: int, max_bytes: int) -> None:
        self._queue: asyncio.Queue[tuple[dict[str, Any], int]] = asyncio.Queue(
            maxsize=maxsize
        )
        self._max_bytes = max_bytes
        self._queued_bytes = 0
        self._space_available = asyncio.Event()
        self._space_available.set()

    def empty(self) -> bool:
        return self._queue.empty()

    def full(self) -> bool:
        return self._queue.full() or self._queued_bytes >= self._max_bytes

    async def put(
        self,
        record: dict[str, Any],
        *,
        byte_size: int | None = None,
    ) -> None:
        size = byte_size if byte_size is not None else _record_size(record)
        if size > self._max_bytes:
            raise _RecordTooLarge
        while self._queued_bytes + size > self._max_bytes:
            self._space_available.clear()
            await self._space_available.wait()
        self._queued_bytes += size
        try:
            await self._queue.put((record, size))
        except BaseException:
            self._queued_bytes -= size
            self._space_available.set()
            raise

    def put_nowait(
        self,
        record: dict[str, Any],
        *,
        byte_size: int | None = None,
    ) -> None:
        size = byte_size if byte_size is not None else _record_size(record)
        if size > self._max_bytes:
            raise _RecordTooLarge
        if self._queued_bytes + size > self._max_bytes:
            raise asyncio.QueueFull
        self._queue.put_nowait((record, size))
        self._queued_bytes += size

    async def get(self) -> dict[str, Any]:
        record, size = await self._queue.get()
        self._release(size)
        return record

    def get_nowait(self) -> dict[str, Any]:
        record, size = self._queue.get_nowait()
        self._release(size)
        return record

    def _release(self, size: int) -> None:
        self._queued_bytes -= size
        self._space_available.set()


class InvokeStream:
    """Async iterator of raw ATOF records for one runtime invocation.

    Consume the final normalized result separately with :meth:`result`. If
    iteration stops early, call :meth:`aclose` before starting another turn.
    """

    def __init__(
        self,
        invoke: Coroutine[Any, Any, RunResult],
        listener: _AtofStreamListener,
        *,
        request_id: str | None = None,
        turn_index: int | None = None,
    ) -> None:
        """lazydocs: ignore"""

        self._listener = listener
        self._closed = False
        self._finalized = False
        listener.begin_stream(request_id=request_id, turn_index=turn_index)
        try:
            self._task = asyncio.create_task(invoke)
        except BaseException:
            listener.end_stream()
            invoke.close()
            raise

    def __aiter__(self) -> InvokeStream:
        """Return this stream as its asynchronous iterator."""

        return self

    async def __anext__(self) -> dict[str, Any]:
        """Return the next raw ATOF record."""

        queue = self._listener.records
        while True:
            if self._closed:
                await self._finalize()
                raise StopAsyncIteration
            if not queue.empty():
                return queue.get_nowait()
            if self._task.done():
                try:
                    return await asyncio.wait_for(queue.get(), _DRAIN_SECONDS)
                except TimeoutError:
                    await self._finalize(warn_if_unavailable=True)
                    raise StopAsyncIteration from None

            getter = asyncio.create_task(queue.get())
            await asyncio.wait(
                {getter, self._task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if getter.done() and not getter.cancelled():
                return getter.result()
            getter.cancel()
            with suppress(asyncio.CancelledError):
                await getter

    async def result(self) -> RunResult:
        """Return the terminal normalized result without adding it to the stream."""

        return await asyncio.shield(self._task)

    async def aclose(self) -> None:
        """Stop iteration and drain this turn without cancelling the invocation."""

        self._closed = True
        await self._finalize()

    async def _finalize(self, *, warn_if_unavailable: bool = False) -> None:
        if self._finalized:
            return
        queue = self._listener.records
        while not self._task.done():
            getter = asyncio.create_task(queue.get())
            try:
                await asyncio.wait(
                    {getter, self._task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
            finally:
                if not getter.done():
                    getter.cancel()
                with suppress(asyncio.CancelledError):
                    await getter

        invocation_completed = False
        try:
            await asyncio.shield(self._task)
            invocation_completed = True
        except asyncio.CancelledError:
            if not self._task.cancelled():
                raise
        except Exception:
            pass

        loop = asyncio.get_running_loop()
        deadline = loop.time() + _DRAIN_SECONDS
        while True:
            while not queue.empty():
                queue.get_nowait()
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            try:
                await asyncio.wait_for(queue.get(), remaining)
            except TimeoutError:
                break
        self._listener.end_stream()
        self._finalized = True
        if invocation_completed and warn_if_unavailable:
            self._listener.warn_if_unavailable()


class _AtofStreamListener:
    """Receive chunked NDJSON ATOF records on an SDK-owned loopback endpoint."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        maxsize: int = _QUEUE_MAXSIZE,
        max_bytes: int = _QUEUE_MAX_BYTES,
        max_record_bytes: int = _MAX_RECORD_BYTES,
    ) -> None:
        self._host = host
        self._port = port
        self._queue = _AtofRecordQueue(maxsize=maxsize, max_bytes=max_bytes)
        self._max_record_bytes = min(max_record_bytes, max_bytes)
        self._server: asyncio.Server | None = None
        self._bound_port: int | None = None
        self._accepting = False
        self._request_id: str | None = None
        self._turn_index: int | None = None
        self._turn_root_uuid: str | None = None
        self._turn_scope_uuids: set[str] = set()
        self._saw_atof_data = False
        self._matched_turn_root = False
        self._active_atof_connections = 0
        self._saw_atof_connection = False
        self._lost_atof_connection = False
        self._warned_unconnected = False
        self._warned_uncorrelated = False
        self._warned_interrupted = False
        self._tasks: set[asyncio.Task[None]] = set()
        self._writers: set[asyncio.StreamWriter] = set()

    @property
    def url(self) -> str:
        """Return the listener endpoint after startup."""

        if self._bound_port is None:
            raise RuntimeError("ATOF stream listener is not started")
        return f"http://{self._host}:{self._bound_port}/atof"

    @property
    def records(self) -> _AtofRecordQueue:
        """Return the bounded record queue used by the active stream."""

        return self._queue

    async def start(self) -> _AtofStreamListener:
        """Bind the loopback HTTP server."""

        self._server = await asyncio.start_server(
            self._connected,
            self._host,
            self._port,
        )
        socket = self._server.sockets[0]
        self._bound_port = int(socket.getsockname()[1])
        return self

    def begin_stream(
        self,
        *,
        request_id: str | None = None,
        turn_index: int | None = None,
    ) -> None:
        """Route subsequent records to the active invocation queue."""

        if self._accepting:
            raise RuntimeError("ATOF stream listener already has an active consumer")
        while not self._queue.empty():
            self._queue.get_nowait()
        self._request_id = request_id
        self._turn_index = turn_index
        self._turn_root_uuid = None
        self._turn_scope_uuids.clear()
        self._saw_atof_data = False
        self._matched_turn_root = request_id is None and turn_index is None
        self._saw_atof_connection = self._active_atof_connections > 0
        self._lost_atof_connection = False
        self._accepting = True

    def end_stream(self) -> None:
        """Discard records until another streaming invocation begins."""

        self._accepting = False
        self._request_id = None
        self._turn_index = None
        self._turn_root_uuid = None
        self._turn_scope_uuids.clear()

    def warn_if_unavailable(self) -> None:
        """Warn once when Relay is unreachable or turn correlation fails."""

        if not self._saw_atof_connection or (
            self._lost_atof_connection
            and not self._saw_atof_data
            and self._active_atof_connections == 0
        ):
            if self._warned_unconnected:
                return
            self._warned_unconnected = True
            warnings.warn(
                "No Relay ATOF connection reached the SDK loopback listener. "
                "Relay-backed streaming yielded no records. Claude and Codex "
                "gateways must run in the same network namespace as the SDK to "
                "reach 127.0.0.1.",
                RuntimeWarning,
                stacklevel=3,
            )
            return
        if (
            self._saw_atof_data
            and not self._matched_turn_root
            and not self._warned_uncorrelated
        ):
            self._warned_uncorrelated = True
            warnings.warn(
                "Relay ATOF data reached the SDK listener, but no record matched "
                "the active Fabric turn. Relay-backed streaming yielded no "
                "records. Verify the Relay turn correlation metadata and record "
                "size limits.",
                RuntimeWarning,
                stacklevel=3,
            )
            return
        if (
            self._lost_atof_connection
            and self._matched_turn_root
            and self._active_atof_connections == 0
            and not self._warned_interrupted
        ):
            self._warned_interrupted = True
            warnings.warn(
                "The Relay ATOF connection closed during the active Fabric "
                "turn. Relay-backed streaming may be incomplete.",
                RuntimeWarning,
                stacklevel=3,
            )

    def _connected(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        task = asyncio.create_task(self._handle_client(reader, writer))
        self._tasks.add(task)
        task.add_done_callback(self._task_done)

    def _task_done(self, task: asyncio.Task[None]) -> None:
        self._tasks.discard(task)
        if not task.cancelled():
            task.exception()

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        self._writers.add(writer)
        is_atof_connection = False
        is_chunked = False
        chunked_body_completed = False
        try:
            request = await reader.readuntil(b"\r\n\r\n")
            request_line, *header_lines = request[:-4].split(b"\r\n")
            method, target, _ = request_line.decode("ascii").split(" ", 2)
            headers = _http_headers(header_lines)
            if method != "POST" or target.split("?", 1)[0] != "/atof":
                await _write_response(writer, 404, "Not Found")
                return
            is_atof_connection = True
            self._active_atof_connections += 1
            if self._accepting:
                self._saw_atof_connection = True
            if headers.get("expect", "").lower() == "100-continue":
                writer.write(b"HTTP/1.1 100 Continue\r\n\r\n")
                await writer.drain()

            buffer = bytearray()
            is_chunked = "chunked" in headers.get("transfer-encoding", "").lower()
            if is_chunked:
                await self._read_chunked(reader, buffer)
                chunked_body_completed = True
            elif "content-length" in headers:
                await self._read_sized(
                    reader,
                    buffer,
                    int(headers["content-length"]),
                )
            else:
                await _write_response(writer, 411, "Length Required")
                return
            await self._emit(buffer)
            await _write_response(writer, 200, "OK")
        except _RecordTooLarge:
            with suppress(ConnectionError):
                await _write_response(writer, 413, "Content Too Large")
        except (ValueError, UnicodeDecodeError, asyncio.IncompleteReadError):
            with suppress(ConnectionError):
                await _write_response(writer, 400, "Bad Request")
        except ConnectionError:
            pass
        except asyncio.CancelledError:
            raise
        finally:
            if is_atof_connection:
                self._active_atof_connections -= 1
                if is_chunked and not chunked_body_completed and self._accepting:
                    self._lost_atof_connection = True
            self._writers.discard(writer)
            writer.close()
            with suppress(ConnectionError):
                await writer.wait_closed()

    async def _read_chunked(
        self,
        reader: asyncio.StreamReader,
        buffer: bytearray,
    ) -> None:
        while True:
            size_line = await reader.readline()
            size = int(size_line.split(b";", 1)[0].strip(), 16)
            if size == 0:
                while True:
                    trailer = await reader.readline()
                    if trailer in (b"\r\n", b"\n"):
                        return
                    if trailer == b"":
                        raise ValueError("incomplete HTTP chunk trailers")
            remaining = size
            while remaining:
                chunk = await reader.readexactly(min(_READ_SIZE, remaining))
                remaining -= len(chunk)
                await self._feed(buffer, chunk)
            if await reader.readexactly(2) != b"\r\n":
                raise ValueError("invalid HTTP chunk terminator")

    async def _read_sized(
        self,
        reader: asyncio.StreamReader,
        buffer: bytearray,
        size: int,
    ) -> None:
        if size < 0:
            raise ValueError("negative HTTP content length")
        remaining = size
        while remaining:
            chunk = await reader.readexactly(min(_READ_SIZE, remaining))
            remaining -= len(chunk)
            await self._feed(buffer, chunk)

    async def _feed(self, buffer: bytearray, chunk: bytes) -> None:
        if chunk and self._accepting:
            self._saw_atof_data = True
        buffer.extend(chunk)
        while True:
            newline = buffer.find(b"\n")
            if newline < 0:
                if len(buffer) > self._max_record_bytes:
                    raise _RecordTooLarge
                return
            if newline > self._max_record_bytes:
                raise _RecordTooLarge
            line = bytes(buffer[:newline])
            del buffer[: newline + 1]
            await self._emit(line)

    async def _emit(self, line: bytes | bytearray) -> None:
        stripped = bytes(line).strip()
        if not stripped or not self._accepting:
            return
        if len(stripped) > self._max_record_bytes:
            raise _RecordTooLarge
        try:
            record = json.loads(stripped)
        except json.JSONDecodeError:
            return
        if isinstance(record, dict):
            if self._belongs_to_active_turn(record):
                await self._queue.put(record, byte_size=len(stripped))

    def _belongs_to_active_turn(self, record: dict[str, Any]) -> bool:
        if self._request_id is None and self._turn_index is None:
            return True

        uuid = record.get("uuid")
        if not isinstance(uuid, str):
            return False
        if self._turn_root_uuid is None:
            if not self._matches_turn_root(record):
                return False
            self._turn_root_uuid = uuid
            self._turn_scope_uuids.add(uuid)
            self._matched_turn_root = True
            return True

        if uuid in self._turn_scope_uuids:
            return True
        parent_uuid = record.get("parent_uuid")
        if (
            not isinstance(parent_uuid, str)
            or parent_uuid not in self._turn_scope_uuids
        ):
            return False
        if record.get("kind") == "scope" and record.get("scope_category") == "start":
            self._turn_scope_uuids.add(uuid)
        return True

    def _matches_turn_root(self, record: dict[str, Any]) -> bool:
        if record.get("kind") != "scope" or record.get("scope_category") != "start":
            return False
        metadata = record.get("metadata")
        if not isinstance(metadata, dict):
            return False
        if (
            self._request_id is not None
            and metadata.get("nemo_fabric_request_id") == self._request_id
        ):
            return True
        return (
            self._turn_index is not None
            and metadata.get("nemo_relay_scope_role") == "turn"
            and metadata.get("turn_index") == self._turn_index
        )

    async def close(self) -> None:
        """Stop accepting records and close active HTTP connections."""

        self._accepting = False
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        for writer in tuple(self._writers):
            writer.close()
        for task in tuple(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._writers.clear()
        self._bound_port = None


def _relay_enabled(config: FabricConfig) -> bool:
    telemetry = config.telemetry
    return telemetry is not None and "relay" in telemetry.providers


def _record_size(record: dict[str, Any]) -> int:
    return len(json.dumps(record, separators=(",", ":"), ensure_ascii=False).encode())


def _sink_name(
    sink: RelayAtofFileSinkConfig | RelayAtofStreamSinkConfig | dict[str, Any],
) -> str | None:
    name = sink.get("name") if isinstance(sink, dict) else getattr(sink, "name", None)
    return name if isinstance(name, str) else None


def _with_stream_sink(config: FabricConfig, url: str) -> FabricConfig:
    copied = config.model_copy(deep=True)
    if copied.relay is None:
        relay = RelayConfig()
    elif isinstance(copied.relay, RelayConfig):
        relay = copied.relay
    else:
        relay = RelayConfig.model_validate(copied.relay)

    if relay.observability is None:
        observability = RelayObservabilityConfig()
    elif isinstance(relay.observability, RelayObservabilityConfig):
        observability = relay.observability
    else:
        observability = RelayObservabilityConfig.model_validate(relay.observability)

    if observability.atof is None:
        atof = RelayAtofConfig()
    elif isinstance(observability.atof, RelayAtofConfig):
        atof = observability.atof
    else:
        atof = RelayAtofConfig.model_validate(observability.atof)

    if atof.enabled:
        sinks = [
            sink for sink in atof.sinks or () if _sink_name(sink) != _STREAM_SINK_NAME
        ]
    else:
        atof = RelayAtofConfig(enabled=True)
        sinks = []
    sinks.append(
        RelayAtofStreamSinkConfig(
            name=_STREAM_SINK_NAME,
            url=url,
            transport="ndjson",
        )
    )
    atof.sinks = sinks
    observability.atof = atof
    relay.observability = observability
    copied.relay = relay
    return copied


def _http_headers(lines: list[bytes]) -> dict[str, str]:
    headers: dict[str, str] = {}
    for line in lines:
        name, separator, value = line.partition(b":")
        if not separator:
            raise ValueError("invalid HTTP header")
        headers[name.decode("ascii").strip().lower()] = value.decode("ascii").strip()
    return headers


async def _write_response(
    writer: asyncio.StreamWriter,
    status: int,
    reason: str,
) -> None:
    body = reason.encode("ascii")
    writer.write(
        f"HTTP/1.1 {status} {reason}\r\n".encode("ascii")
        + f"Content-Length: {len(body)}\r\n".encode("ascii")
        + b"Connection: close\r\n"
        + b"Content-Type: text/plain\r\n\r\n"
        + body
    )
    await writer.drain()
