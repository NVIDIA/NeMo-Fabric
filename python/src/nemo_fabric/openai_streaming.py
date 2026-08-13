# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapter-native OpenAI streaming for the NVIDIA NeMo Fabric Python SDK."""

from __future__ import annotations

import asyncio
import json
import secrets
from collections.abc import Awaitable, Callable
from contextlib import suppress
from typing import Any

from nemo_fabric.errors import FabricRuntimeError
from nemo_fabric.types import RunResult

_END = object()
_HOST = "127.0.0.1"
_OPENAI_STREAM_COMPLETION_TIMEOUT = 10.0
_OPENAI_STREAM_HEADER_TIMEOUT = 10.0
_MAX_PENDING_CONNECTIONS = 8
_MAX_RECORD_BYTES = 1024 * 1024
_QUEUE_MAX_BYTES = 16 * 1024 * 1024
_QUEUE_MAXSIZE = 1024
_READ_SIZE = 64 * 1024
_STREAM_PATH = "/openai-stream"
_UINT32_MAX = (1 << 32) - 1
_UINT64_MAX = (1 << 64) - 1


class _ProtocolError(ValueError):
    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


class _ChunkQueue:
    def __init__(self, *, maxsize: int, max_bytes: int) -> None:
        self._queue: asyncio.Queue[tuple[dict[str, Any] | object, int]] = (
            asyncio.Queue(maxsize=maxsize)
        )
        self._max_bytes = max_bytes
        self._queued_bytes = 0
        self._space_available = asyncio.Event()
        self._space_available.set()

    def empty(self) -> bool:
        return self._queue.empty()

    async def put(self, item: dict[str, Any] | object, size: int = 0) -> None:
        if size > self._max_bytes:
            raise _ProtocolError("OpenAI stream record exceeds the queue byte limit", 413)
        while self._queued_bytes + size > self._max_bytes:
            self._space_available.clear()
            await self._space_available.wait()
        self._queued_bytes += size
        try:
            await self._queue.put((item, size))
        except BaseException:
            self._queued_bytes -= size
            self._space_available.set()
            raise

    async def get(self) -> dict[str, Any] | object:
        item, size = await self._queue.get()
        self._queued_bytes -= size
        self._space_available.set()
        return item

    def get_nowait(self) -> dict[str, Any] | object:
        item, size = self._queue.get_nowait()
        self._queued_bytes -= size
        self._space_available.set()
        return item


class _OpenAIStreamListener:
    """Receive one authenticated chunked-NDJSON OpenAI event stream."""

    def __init__(
        self,
        *,
        runtime_id: str,
        request_id: str,
        port: int = 0,
        maxsize: int = _QUEUE_MAXSIZE,
        max_bytes: int = _QUEUE_MAX_BYTES,
        max_record_bytes: int = _MAX_RECORD_BYTES,
    ) -> None:
        self._runtime_id = runtime_id
        self._request_id = request_id
        self._port = port
        self._token = secrets.token_urlsafe(32)
        self._queue = _ChunkQueue(maxsize=maxsize, max_bytes=max_bytes)
        self._max_record_bytes = min(max_record_bytes, max_bytes)
        self._server: asyncio.Server | None = None
        self._bound_port: int | None = None
        self._invocation_id: str | None = None
        self._expected_sequence = 0
        self._connected = False
        self._connection_closed = False
        self._ended = False
        self._discard = False
        self._error: FabricRuntimeError | None = None
        self._completion = asyncio.Event()
        self._tasks: set[asyncio.Task[None]] = set()
        self._writers: set[asyncio.StreamWriter] = set()

    @property
    def transport(self) -> dict[str, Any]:
        if self._bound_port is None:
            raise RuntimeError("OpenAI stream listener is not started")
        return {"port": self._bound_port, "token": self._token}

    @property
    def invocation_id(self) -> str | None:
        return self._invocation_id

    @property
    def error(self) -> FabricRuntimeError | None:
        return self._error

    @property
    def records(self) -> _ChunkQueue:
        return self._queue

    @property
    def connection_closed(self) -> bool:
        return self._connection_closed

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._accept, _HOST, self._port)
        socket = self._server.sockets[0]
        self._bound_port = int(socket.getsockname()[1])

    def discard(self) -> None:
        self._discard = True

    async def wait_completed(self) -> None:
        await self._completion.wait()

    def fail(self, message: str) -> None:
        self._set_error(message)
        self._completion.set()

    def _accept(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        if len(self._tasks) >= _MAX_PENDING_CONNECTIONS:
            writer.close()
            return
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
        claimed = False
        try:
            request = await asyncio.wait_for(
                reader.readuntil(b"\r\n\r\n"),
                _OPENAI_STREAM_HEADER_TIMEOUT,
            )
            request_line, *header_lines = request[:-4].split(b"\r\n")
            method, target, _version = request_line.decode("ascii").split(" ", 2)
            headers = _http_headers(header_lines)
            if method != "POST" or target.split("?", 1)[0] != _STREAM_PATH:
                raise _ProtocolError("Unknown OpenAI stream endpoint", 404)
            authorization = headers.get("authorization", "")
            if not secrets.compare_digest(authorization, f"Bearer {self._token}"):
                raise _ProtocolError("Invalid OpenAI stream authorization", 401)
            if headers.get("content-type", "").split(";", 1)[0].strip() != (
                "application/x-ndjson"
            ):
                raise _ProtocolError("OpenAI stream must use NDJSON", 415)
            transfer_codings = [
                coding.strip().lower()
                for coding in headers.get("transfer-encoding", "").split(",")
                if coding.strip()
            ]
            if not transfer_codings or transfer_codings[-1] != "chunked":
                raise _ProtocolError("OpenAI stream must use chunked transfer encoding", 411)
            if headers.get("expect", "").lower() != "100-continue":
                raise _ProtocolError("OpenAI stream must use Expect: 100-continue", 417)
            # Authentication and request validation are connection-local. Claim the
            # invocation only after they succeed, with no await between check/set.
            if self._connected:
                raise _ProtocolError("OpenAI stream listener accepts one connection", 409)
            self._connected = True
            claimed = True
            writer.write(b"HTTP/1.1 100 Continue\r\n\r\n")
            await writer.drain()
            buffer = bytearray()
            await self._read_chunked(reader, buffer)
            if buffer.strip():
                await self._emit_line(bytes(buffer))
            if not self._ended:
                raise _ProtocolError("OpenAI stream ended without an end record")
            await _write_http_response(writer, 200, "OK")
        except _ProtocolError as error:
            if claimed:
                self._set_error(str(error))
            with suppress(ConnectionError):
                await _write_http_response(writer, error.status, "Rejected")
        except (
            UnicodeDecodeError,
            ValueError,
            RecursionError,
            asyncio.IncompleteReadError,
            asyncio.LimitOverrunError,
        ):
            if claimed:
                self._set_error("OpenAI stream contained malformed transport data")
            with suppress(ConnectionError):
                await _write_http_response(writer, 400, "Bad Request")
        except ConnectionError:
            if claimed:
                self._connection_closed = True
                self._set_error("OpenAI stream connection closed before completion")
        except asyncio.CancelledError:
            raise
        except Exception:
            if claimed:
                self._set_error("OpenAI stream transport failed while parsing records")
            with suppress(ConnectionError):
                await _write_http_response(writer, 400, "Bad Request")
        finally:
            if claimed and not self._ended and self._error is not None:
                await self._queue.put(_END)
            if claimed:
                self._completion.set()
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
            if not size_line:
                raise ConnectionError("OpenAI stream connection closed before completion")
            size = int(size_line.split(b";", 1)[0].strip(), 16)
            if size < 0:
                raise _ProtocolError("Invalid OpenAI stream chunk size")
            if size == 0:
                while True:
                    trailer = await reader.readline()
                    if trailer in (b"\r\n", b"\n"):
                        return
                    if trailer == b"":
                        raise _ProtocolError("Incomplete OpenAI stream trailers")
            remaining = size
            while remaining:
                chunk = await reader.readexactly(min(_READ_SIZE, remaining))
                remaining -= len(chunk)
                await self._feed(buffer, chunk)
            if await reader.readexactly(2) != b"\r\n":
                raise _ProtocolError("Invalid OpenAI stream chunk terminator")

    async def _feed(self, buffer: bytearray, chunk: bytes) -> None:
        buffer.extend(chunk)
        while True:
            newline = buffer.find(b"\n")
            if newline < 0:
                # A record exactly at the limit can have a trailing CR while
                # waiting for the LF half of its delimiter.
                pending_crlf = (
                    len(buffer) == self._max_record_bytes + 1
                    and buffer.endswith(b"\r")
                )
                if len(buffer) > self._max_record_bytes and not pending_crlf:
                    raise _ProtocolError(
                        f"OpenAI stream record exceeds {self._max_record_bytes} bytes",
                        413,
                    )
                return
            line = bytes(buffer[:newline])
            del buffer[: newline + 1]
            await self._emit_line(line)

    async def _emit_line(self, line: bytes) -> None:
        payload = line.removesuffix(b"\r")
        if len(payload) > self._max_record_bytes:
            raise _ProtocolError(
                f"OpenAI stream record exceeds {self._max_record_bytes} bytes",
                413,
            )
        stripped = payload.strip()
        if not stripped:
            return
        try:
            record = json.loads(stripped, parse_constant=_reject_json_constant)
        except (json.JSONDecodeError, RecursionError, ValueError) as error:
            raise _ProtocolError("OpenAI stream record is not valid JSON") from error
        if not isinstance(record, dict):
            raise _ProtocolError("OpenAI stream record must be a mapping")
        self._validate_identity(record)
        sequence = record.get("sequence")
        if (
            isinstance(sequence, bool)
            or not isinstance(sequence, int)
            or sequence != self._expected_sequence
        ):
            raise _ProtocolError("OpenAI stream record sequence is not monotonic")
        self._expected_sequence += 1
        record_type = record.get("type")
        if record_type == "chunk":
            if set(record) != {
                "type",
                "sequence",
                "runtime_id",
                "invocation_id",
                "request_id",
                "chunk",
            }:
                raise _ProtocolError("OpenAI stream chunk record shape is invalid")
            if self._ended:
                raise _ProtocolError("OpenAI stream emitted a chunk after end")
            chunk = _validate_openai_chunk(record.get("chunk"))
            if not self._discard:
                await self._queue.put(chunk, len(stripped))
            return
        if record_type == "end":
            if self._ended or set(record) != {
                "type",
                "sequence",
                "runtime_id",
                "invocation_id",
                "request_id",
            }:
                raise _ProtocolError("OpenAI stream end record is invalid")
            self._ended = True
            await self._queue.put(_END)
            return
        raise _ProtocolError("OpenAI stream record has an unknown type")

    def _validate_identity(self, record: dict[str, Any]) -> None:
        if record.get("runtime_id") != self._runtime_id:
            raise _ProtocolError("OpenAI stream runtime ID does not match")
        if record.get("request_id") != self._request_id:
            raise _ProtocolError("OpenAI stream request ID does not match")
        invocation_id = record.get("invocation_id")
        if not isinstance(invocation_id, str) or not invocation_id:
            raise _ProtocolError("OpenAI stream invocation ID is missing")
        if self._invocation_id is None:
            self._invocation_id = invocation_id
        elif invocation_id != self._invocation_id:
            raise _ProtocolError("OpenAI stream invocation ID changed")

    def _set_error(self, message: str) -> None:
        if self._error is None:
            self._error = FabricRuntimeError(
                message,
                stage="invoke",
                code="openai_stream_protocol_error",
            )

    async def close(self) -> None:
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


def _validate_openai_chunk(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("object") != "chat.completion.chunk":
        raise _ProtocolError(
            "OpenAI stream event must be a chat.completion.chunk mapping"
        )
    identifier = value.get("id")
    model = value.get("model")
    created = value.get("created")
    choices = value.get("choices")
    if not isinstance(identifier, str) or not identifier.strip():
        raise _ProtocolError(
            "OpenAI stream event id must be a non-empty string containing "
            "a non-whitespace character"
        )
    if not isinstance(model, str) or not model.strip():
        raise _ProtocolError(
            "OpenAI stream event model must be a non-empty string containing "
            "a non-whitespace character"
        )
    if (
        isinstance(created, bool)
        or not isinstance(created, int)
        or not 0 <= created <= _UINT64_MAX
    ):
        raise _ProtocolError(
            "OpenAI stream event created must be an unsigned 64-bit integer"
        )
    if not isinstance(choices, list):
        raise _ProtocolError("OpenAI stream event choices must be a list")
    for choice in choices:
        if not isinstance(choice, dict):
            raise _ProtocolError("OpenAI stream choices must be mappings")
        index = choice.get("index")
        delta = choice.get("delta")
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index <= _UINT32_MAX
        ):
            raise _ProtocolError(
                "OpenAI stream choice index must be an unsigned 32-bit integer"
            )
        if not isinstance(delta, dict):
            raise _ProtocolError("OpenAI stream choice delta must be a mapping")
        for name in ("content", "refusal", "role"):
            if name in delta and delta[name] is not None and not isinstance(
                delta[name], str
            ):
                raise _ProtocolError(
                    f"OpenAI stream choice delta {name} must be a string or null"
                )
        if "function_call" in delta and delta["function_call"] is not None:
            if not isinstance(delta["function_call"], dict):
                raise _ProtocolError(
                    "OpenAI stream choice delta function_call must be a mapping or null"
                )
        if "tool_calls" in delta and delta["tool_calls"] is not None:
            tool_calls = delta["tool_calls"]
            if not isinstance(tool_calls, list) or not all(
                isinstance(tool_call, dict) for tool_call in tool_calls
            ):
                raise _ProtocolError(
                    "OpenAI stream choice delta tool_calls must be a list of mappings or null"
                )
        if "finish_reason" in choice and choice["finish_reason"] is not None:
            if not isinstance(choice["finish_reason"], str):
                raise _ProtocolError(
                    "OpenAI stream choice finish_reason must be a string or null"
                )
        if "logprobs" in choice and choice["logprobs"] is not None:
            if not isinstance(choice["logprobs"], dict):
                raise _ProtocolError(
                    "OpenAI stream choice logprobs must be a mapping or null"
                )
    if "usage" in value and value["usage"] is not None:
        if not isinstance(value["usage"], dict):
            raise _ProtocolError("OpenAI stream event usage must be a mapping or null")
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value}")


class OpenAIInvokeStream:
    """Async iterator of OpenAI chat-completion chunks for one invocation.

    Await ``result()`` for the separate normalized terminal result. Call
    ``aclose()`` when iteration stops early; it drains the stream without
    cancelling the invocation.
    """

    def __init__(
        self,
        invoke: Callable[[dict[str, Any]], Awaitable[RunResult]],
        *,
        runtime_id: str,
        request_id: str,
        on_result: Callable[[RunResult], None] | None = None,
        on_protocol_failure: Callable[[], None] | None = None,
    ) -> None:
        """lazydocs: ignore"""

        self._listener = _OpenAIStreamListener(
            runtime_id=runtime_id,
            request_id=request_id,
        )
        self._closed = False
        self._finalized = False
        self._end_observed = False
        self._pending_item: dict[str, Any] | object | None = None
        self._accepted_result: RunResult | None = None
        self._on_result = on_result
        self._on_protocol_failure = on_protocol_failure
        self._protocol_failure_reported = False
        self._finalize_lock = asyncio.Lock()
        run = self._run(invoke)
        try:
            self._task = asyncio.create_task(run)
        except BaseException:
            run.close()
            raise

    async def _run(
        self,
        invoke: Callable[[dict[str, Any]], Awaitable[RunResult]],
    ) -> RunResult:
        await self._listener.start()
        return await invoke(self._listener.transport)

    def __aiter__(self) -> OpenAIInvokeStream:
        return self

    async def __anext__(self) -> dict[str, Any]:
        if self._closed:
            await self._finalize()
            raise StopAsyncIteration
        queue = self._listener.records
        while True:
            if self._end_observed:
                await self._finalize()
                self._raise_stream_error()
                raise StopAsyncIteration
            if self._pending_item is not None:
                item = self._pending_item
                self._pending_item = None
            elif not queue.empty():
                item = queue.get_nowait()
            elif self._task.done():
                if self._task.cancelled() or self._task.exception() is not None:
                    await self._finalize()
                    self._raise_stream_error()
                    raise StopAsyncIteration
                result = self._task.result()
                if (
                    result.status != "succeeded"
                    and self._listener.invocation_id is None
                ):
                    await self._finalize()
                    raise StopAsyncIteration
                getter = asyncio.create_task(queue.get())
                try:
                    await asyncio.wait(
                        {getter},
                        timeout=_OPENAI_STREAM_COMPLETION_TIMEOUT,
                    )
                except asyncio.CancelledError:
                    if not getter.done():
                        getter.cancel()
                    try:
                        self._pending_item = await getter
                    except asyncio.CancelledError:
                        pass
                    raise
                if getter.done() and not getter.cancelled():
                    item = getter.result()
                else:
                    getter.cancel()
                    with suppress(asyncio.CancelledError):
                        await getter
                    self._listener.fail(
                        "OpenAI stream did not establish and complete its event channel"
                    )
                    await self._finalize()
                    self._raise_stream_error()
                    raise StopAsyncIteration from None
            else:
                getter = asyncio.create_task(queue.get())
                try:
                    await asyncio.wait(
                        {getter, self._task},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                except asyncio.CancelledError:
                    if not getter.done():
                        getter.cancel()
                    try:
                        self._pending_item = await getter
                    except asyncio.CancelledError:
                        pass
                    raise
                if getter.done() and not getter.cancelled():
                    item = getter.result()
                else:
                    getter.cancel()
                    with suppress(asyncio.CancelledError):
                        await getter
                    continue
            if item is _END:
                self._end_observed = True
                await self._finalize()
                self._raise_stream_error()
                raise StopAsyncIteration
            return item  # type: ignore[return-value]

    async def result(self) -> RunResult:
        """Drain the stream and return its separate normalized terminal result."""

        await self._finalize()
        try:
            result = await asyncio.shield(self._task)
        except asyncio.CancelledError:
            if self._task.cancelled():
                self._raise_protocol_error()
            raise
        except Exception:
            self._raise_stream_error()
            raise
        self._raise_stream_error()
        return self._accepted_result if self._accepted_result is not None else result

    async def aclose(self) -> None:
        """Discard unread chunks and wait without cancelling the invocation."""

        self._closed = True
        await self._finalize()

    async def _finalize(self) -> None:
        async with self._finalize_lock:
            if self._finalized:
                return
            self._listener.discard()
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
            while not queue.empty():
                queue.get_nowait()

            result: RunResult | None = None
            try:
                result = await asyncio.shield(self._task)
            except asyncio.CancelledError:
                if not self._task.cancelled():
                    raise
            except Exception:
                pass

            if result is not None:
                if result.status == "succeeded" or self._listener.invocation_id is not None:
                    try:
                        await asyncio.wait_for(
                            self._listener.wait_completed(),
                            _OPENAI_STREAM_COMPLETION_TIMEOUT,
                        )
                    except TimeoutError:
                        self._listener.fail(
                            "OpenAI stream did not establish and complete its event channel"
                        )
                while not queue.empty():
                    queue.get_nowait()
                self._validate_and_accept_result(result)

            self._pending_item = None
            if self._listener.error is not None:
                self._report_protocol_failure()
            await self._listener.close()
            self._finalized = True

    def _validate_and_accept_result(self, result: RunResult) -> None:
        if self._accepted_result is not None:
            return
        if self._listener.error is None:
            invocation_id = self._listener.invocation_id
            if invocation_id is None:
                mismatched = result.status == "succeeded"
            else:
                mismatched = result.invocation_id != invocation_id
            if mismatched:
                self._listener._set_error(
                    "OpenAI stream invocation ID does not match its terminal result"
                )
        if self._listener.error is not None:
            self._report_protocol_failure()
            return
        self._accepted_result = result
        if self._on_result is not None:
            self._on_result(result)

    def _report_protocol_failure(self) -> None:
        if self._protocol_failure_reported:
            return
        self._protocol_failure_reported = True
        if self._on_protocol_failure is not None:
            self._on_protocol_failure()

    def _raise_protocol_error(self) -> None:
        error = self._listener.error
        if error is not None:
            self._report_protocol_failure()
            raise error

    def _raise_stream_error(self) -> None:
        if (
            self._listener.connection_closed
            and self._task.done()
            and not self._task.cancelled()
        ):
            invocation_error = self._task.exception()
            if invocation_error is not None:
                raise invocation_error
        self._raise_protocol_error()


def _http_headers(lines: list[bytes]) -> dict[str, str]:
    headers: dict[str, str] = {}
    for line in lines:
        name, value = line.decode("iso-8859-1").split(":", 1)
        headers[name.strip().lower()] = value.strip()
    return headers


async def _write_http_response(
    writer: asyncio.StreamWriter,
    status: int,
    reason: str,
) -> None:
    writer.write(
        f"HTTP/1.1 {status} {reason}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".encode(
            "ascii"
        )
    )
    await writer.drain()
