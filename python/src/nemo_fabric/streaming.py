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
    RelayAtofStreamSinkConfig,
    RelayConfig,
    RelayObservabilityConfig,
)
from nemo_fabric.types import RunResult

_DRAIN_SECONDS = 0.25
_QUEUE_MAXSIZE = 1024
_READ_SIZE = 64 * 1024
_STREAM_SINK_NAME = "nemo-fabric-stream"


class InvokeStream:
    """Async iterator of raw ATOF records for one runtime invocation.

    Consume the final normalized result separately with :meth:`result`. If
    iteration stops early, call :meth:`aclose` before starting another turn.
    """

    def __init__(
        self,
        invoke: Coroutine[Any, Any, RunResult],
        listener: _AtofStreamListener,
    ) -> None:
        """lazydocs: ignore"""

        self._listener = listener
        self._closed = False
        self._finalized = False
        listener.begin_stream()
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
                    await self._finalize(warn_if_unconnected=True)
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

    async def _finalize(self, *, warn_if_unconnected: bool = False) -> None:
        if self._finalized:
            return
        invocation_completed = False
        try:
            await asyncio.shield(self._task)
            invocation_completed = True
        except asyncio.CancelledError:
            if not self._task.cancelled():
                raise
        except Exception:
            pass

        queue = self._listener.records
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
        if invocation_completed and warn_if_unconnected:
            self._listener.warn_if_unconnected()


class _AtofStreamListener:
    """Receive chunked NDJSON ATOF records on an SDK-owned loopback endpoint."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        maxsize: int = _QUEUE_MAXSIZE,
    ) -> None:
        self._host = host
        self._port = port
        self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=maxsize)
        self._server: asyncio.Server | None = None
        self._bound_port: int | None = None
        self._accepting = False
        self._has_atof_connection = False
        self._warned_unconnected = False
        self._tasks: set[asyncio.Task[None]] = set()
        self._writers: set[asyncio.StreamWriter] = set()

    @property
    def url(self) -> str:
        """Return the listener endpoint after startup."""

        if self._bound_port is None:
            raise RuntimeError("ATOF stream listener is not started")
        return f"http://{self._host}:{self._bound_port}/atof"

    @property
    def records(self) -> asyncio.Queue[dict[str, Any]]:
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

    def begin_stream(self) -> None:
        """Route subsequent records to the active invocation queue."""

        if self._accepting:
            raise RuntimeError("ATOF stream listener already has an active consumer")
        while not self._queue.empty():
            self._queue.get_nowait()
        self._accepting = True

    def end_stream(self) -> None:
        """Discard records until another streaming invocation begins."""

        self._accepting = False

    def warn_if_unconnected(self) -> None:
        """Warn once when Relay has never reached this listener."""

        if self._has_atof_connection or self._warned_unconnected:
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
        try:
            request = await reader.readuntil(b"\r\n\r\n")
            request_line, *header_lines = request[:-4].split(b"\r\n")
            method, target, _ = request_line.decode("ascii").split(" ", 2)
            headers = _http_headers(header_lines)
            if method != "POST" or target.split("?", 1)[0] != "/atof":
                await _write_response(writer, 404, "Not Found")
                return
            self._has_atof_connection = True
            if headers.get("expect", "").lower() == "100-continue":
                writer.write(b"HTTP/1.1 100 Continue\r\n\r\n")
                await writer.drain()

            buffer = bytearray()
            if "chunked" in headers.get("transfer-encoding", "").lower():
                await self._read_chunked(reader, buffer)
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
        except (ValueError, UnicodeDecodeError, asyncio.IncompleteReadError):
            with suppress(ConnectionError):
                await _write_response(writer, 400, "Bad Request")
        except ConnectionError:
            pass
        except asyncio.CancelledError:
            raise
        finally:
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
                while await reader.readline() not in (b"\r\n", b"\n", b""):
                    pass
                return
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
        buffer.extend(chunk)
        while True:
            newline = buffer.find(b"\n")
            if newline < 0:
                return
            line = bytes(buffer[:newline])
            del buffer[: newline + 1]
            await self._emit(line)

    async def _emit(self, line: bytes | bytearray) -> None:
        stripped = bytes(line).strip()
        if not stripped or not self._accepting:
            return
        try:
            record = json.loads(stripped)
        except json.JSONDecodeError:
            return
        if isinstance(record, dict):
            await self._queue.put(record)

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


def _sink_name(sink: Any) -> str | None:
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
