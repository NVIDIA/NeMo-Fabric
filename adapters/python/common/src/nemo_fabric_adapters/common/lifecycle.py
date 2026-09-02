# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wire protocol for persistent local adapter hosts."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import traceback
from collections.abc import Awaitable
from collections.abc import Callable
from collections.abc import Iterator
from collections.abc import Mapping
from contextlib import contextmanager
from contextlib import redirect_stdout
from contextlib import suppress
from dataclasses import dataclass
from typing import Any
from typing import Protocol
from typing import TextIO

from nemo_fabric_adapter_contract.models import AgentRunRequest
from nemo_fabric_adapter_contract.models import AgentRunResult
from nemo_fabric_adapter_contract.models import RuntimeContext


class AdapterRuntime(Protocol):
    """One adapter-owned runtime living for the complete host lifetime."""

    async def start(self, payload: dict[str, Any]) -> None:
        """Initialize runtime-owned SDK clients and resources."""

    async def invoke(
        self,
        request: AgentRunRequest,
        context: RuntimeContext,
    ) -> AgentRunResult:
        """Execute one invocation against the initialized runtime."""

    async def stop(self) -> None:
        """Release all resources owned by the runtime."""


RuntimeFactory = Callable[[], AdapterRuntime]
ConfigLoader = Callable[[Any], Any]
OpenAIChunkEmitter = Callable[[Mapping[str, Any]], Awaitable[None]]

_OPENAI_STREAM_CONNECT_TIMEOUT = 10.0
_OPENAI_STREAM_HOST = "127.0.0.1"
_OPENAI_STREAM_PATH = "/openai-stream"
_OPENAI_STREAM_PROTOCOL = "fabric.openai_stream/v1alpha1"
_OPENAI_STREAM_PROFILE = "openai.chat_completions.chunk/v1"
_OPENAI_STREAM_RECORD_LIMIT = 1024 * 1024
_UINT32_MAX = (1 << 32) - 1
_UINT64_MAX = (1 << 64) - 1


class LifecycleError(Exception):
    """Adapter-supplied lifecycle failure safe to return across the protocol."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        retryable: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable
        self.metadata = dict(metadata or {})


class _AdapterCallError(LifecycleError):
    """Failure raised while executing an adapter runtime method."""


async def _close_stream_writer(writer: asyncio.StreamWriter) -> None:
    writer.close()
    with suppress(OSError):
        await writer.wait_closed()


class _OpenAIStreamWriter:
    """Write correlated OpenAI chunks to one SDK-owned HTTP endpoint."""

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        sink: dict[str, Any],
    ) -> None:
        self._reader = reader
        self._writer = writer
        self._sink = sink
        self._sequence = 0
        self._finished = False
        self._write_lock = asyncio.Lock()

    @classmethod
    async def connect(
        cls,
        payload: dict[str, Any],
    ) -> tuple[_OpenAIStreamWriter, dict[str, Any]]:
        sink, adapter_payload = _validated_openai_stream_payload(payload)
        reader: asyncio.StreamReader | None = None
        writer: asyncio.StreamWriter | None = None
        connected = False
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(sink["host"], sink["port"]),
                _OPENAI_STREAM_CONNECT_TIMEOUT,
            )
            request = (
                f"POST {_OPENAI_STREAM_PATH} HTTP/1.1\r\n"
                f"Host: {_OPENAI_STREAM_HOST}:{sink['port']}\r\n"
                f"Authorization: Bearer {sink['token']}\r\n"
                "Content-Type: application/x-ndjson\r\n"
                "Transfer-Encoding: chunked\r\n"
                "Expect: 100-continue\r\n"
                "Connection: close\r\n\r\n"
            )
            writer.write(request.encode("ascii"))
            await writer.drain()
            status = await asyncio.wait_for(
                _read_http_response(reader),
                _OPENAI_STREAM_CONNECT_TIMEOUT,
            )
            if status != 100:
                raise LifecycleError(
                    "lifecycle_stream_transport_failed",
                    "OpenAI stream listener rejected the adapter connection",
                )
            connected = True
        except LifecycleError:
            raise
        except Exception as error:
            raise LifecycleError(
                "lifecycle_stream_transport_failed",
                "Adapter could not connect to the OpenAI stream listener",
            ) from error
        finally:
            if writer is not None and not connected:
                await _close_stream_writer(writer)
        assert reader is not None and writer is not None
        return cls(reader, writer, sink), adapter_payload

    async def emit(self, chunk: Mapping[str, Any]) -> None:
        if not isinstance(chunk, Mapping):
            raise LifecycleError(
                "lifecycle_invalid_openai_stream_event",
                "OpenAI stream events must be mappings",
            )
        event = _validated_openai_chunk(dict(chunk))
        async with self._write_lock:
            if self._finished:
                raise LifecycleError(
                    "lifecycle_stream_transport_failed",
                    "Adapter cannot emit after finishing the OpenAI event stream",
                )
            await self._write_record("chunk", chunk=event)

    async def finish(self) -> None:
        async with self._write_lock:
            if self._finished:
                return
            self._finished = True
            try:
                await self._write_record("end")
                self._writer.write(b"0\r\n\r\n")
                await self._writer.drain()
                status = await asyncio.wait_for(
                    _read_http_response(self._reader),
                    _OPENAI_STREAM_CONNECT_TIMEOUT,
                )
                if status != 200:
                    raise LifecycleError(
                        "lifecycle_stream_transport_failed",
                        "OpenAI stream listener rejected the event stream",
                    )
            except LifecycleError:
                raise
            except Exception as error:
                raise LifecycleError(
                    "lifecycle_stream_transport_failed",
                    "Adapter could not finish the OpenAI event stream",
                ) from error
            finally:
                await _close_stream_writer(self._writer)

    async def _write_record(
        self,
        record_type: str,
        *,
        chunk: dict[str, Any] | None = None,
    ) -> None:
        record = {
            "type": record_type,
            "sequence": self._sequence,
            "runtime_id": self._sink["runtime_id"],
            "invocation_id": self._sink["invocation_id"],
            "request_id": self._sink["request_id"],
        }
        if chunk is not None:
            record["chunk"] = chunk
        try:
            encoded = (
                json.dumps(
                    record,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode("utf-8")
                + b"\n"
            )
        except (TypeError, ValueError) as error:
            raise LifecycleError(
                "lifecycle_invalid_openai_stream_event",
                "OpenAI stream events must contain JSON-compatible values",
            ) from error
        if len(encoded) > _OPENAI_STREAM_RECORD_LIMIT:
            raise LifecycleError(
                "lifecycle_openai_stream_event_too_large",
                "OpenAI stream event exceeds the 1 MiB record limit",
            )
        self._writer.write(f"{len(encoded):X}\r\n".encode("ascii"))
        self._writer.write(encoded)
        self._writer.write(b"\r\n")
        try:
            await self._writer.drain()
        except Exception as error:
            raise LifecycleError(
                "lifecycle_stream_transport_failed",
                "Adapter lost the OpenAI stream listener connection",
            ) from error
        self._sequence += 1


async def _read_http_response(reader: asyncio.StreamReader) -> int:
    status_line = await reader.readline()
    try:
        _version, raw_status, _reason = status_line.decode("ascii").split(" ", 2)
        status = int(raw_status)
    except (UnicodeDecodeError, ValueError) as error:
        raise LifecycleError(
            "lifecycle_stream_transport_failed",
            "OpenAI stream listener returned an invalid HTTP response",
        ) from error
    while True:
        line = await reader.readline()
        if line in (b"\r\n", b"\n"):
            return status
        if not line:
            raise LifecycleError(
                "lifecycle_stream_transport_failed",
                "OpenAI stream listener closed an incomplete HTTP response",
            )


def _validated_openai_stream_payload(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    sink = payload.get("stream")
    context = payload.get("runtime_context")
    if not isinstance(sink, dict) or not isinstance(context, dict):
        raise LifecycleError(
            "lifecycle_invalid_stream_sink",
            "OpenAI stream invocation is missing its transport",
        )
    expected = {
        "protocol_version": _OPENAI_STREAM_PROTOCOL,
        "profile": _OPENAI_STREAM_PROFILE,
        "host": _OPENAI_STREAM_HOST,
    }
    if any(sink.get(key) != value for key, value in expected.items()):
        raise LifecycleError(
            "lifecycle_invalid_stream_sink",
            "OpenAI stream transport uses an unsupported protocol or endpoint",
        )
    port = sink.get("port")
    token = sink.get("token")
    if (
        isinstance(port, bool)
        or not isinstance(port, int)
        or not 0 < port <= 65535
        or not isinstance(token, str)
        or not token
        or "\r" in token
        or "\n" in token
    ):
        raise LifecycleError(
            "lifecycle_invalid_stream_sink",
            "OpenAI stream transport credentials are invalid",
        )
    for name in ("runtime_id", "invocation_id", "request_id"):
        if not isinstance(sink.get(name), str) or sink[name] != context.get(name):
            raise LifecycleError(
                "lifecycle_invalid_stream_sink",
                "OpenAI stream transport identity does not match the invocation",
            )
    adapter_payload = {key: value for key, value in payload.items() if key != "stream"}
    return sink, adapter_payload


def _validated_openai_chunk(value: dict[str, Any]) -> dict[str, Any]:
    def invalid(message: str) -> LifecycleError:
        return LifecycleError("lifecycle_invalid_openai_stream_event", message)

    if value.get("object") != "chat.completion.chunk":
        raise invalid("OpenAI stream events must use object 'chat.completion.chunk'")
    identifier = value.get("id")
    model = value.get("model")
    created = value.get("created")
    choices = value.get("choices")
    if not isinstance(identifier, str) or not identifier.strip():
        raise invalid(
            "OpenAI stream event id must be a non-empty string containing "
            "a non-whitespace character"
        )
    if not isinstance(model, str) or not model.strip():
        raise invalid(
            "OpenAI stream event model must be a non-empty string containing "
            "a non-whitespace character"
        )
    if (
        isinstance(created, bool)
        or not isinstance(created, int)
        or not 0 <= created <= _UINT64_MAX
    ):
        raise invalid("OpenAI stream event created must be an unsigned 64-bit integer")
    if not isinstance(choices, list):
        raise invalid("OpenAI stream event choices must be a list")
    for choice in choices:
        if not isinstance(choice, dict):
            raise invalid("OpenAI stream choices must be mappings")
        index = choice.get("index")
        delta = choice.get("delta")
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index <= _UINT32_MAX
        ):
            raise invalid(
                "OpenAI stream choice index must be an unsigned 32-bit integer"
            )
        if not isinstance(delta, dict):
            raise invalid("OpenAI stream choice delta must be a mapping")
        for name in ("content", "refusal", "role"):
            if (
                name in delta
                and delta[name] is not None
                and not isinstance(delta[name], str)
            ):
                raise invalid(
                    f"OpenAI stream choice delta {name} must be a string or null"
                )
        if "function_call" in delta and delta["function_call"] is not None:
            if not isinstance(delta["function_call"], dict):
                raise invalid(
                    "OpenAI stream choice delta function_call must be a mapping or null"
                )
        if "tool_calls" in delta and delta["tool_calls"] is not None:
            tool_calls = delta["tool_calls"]
            if not isinstance(tool_calls, list) or not all(
                isinstance(tool_call, dict) for tool_call in tool_calls
            ):
                raise invalid(
                    "OpenAI stream choice delta tool_calls must be a list of mappings or null"
                )
        if "finish_reason" in choice and choice["finish_reason"] is not None:
            if not isinstance(choice["finish_reason"], str):
                raise invalid(
                    "OpenAI stream choice finish_reason must be a string or null"
                )
        if "logprobs" in choice and choice["logprobs"] is not None:
            if not isinstance(choice["logprobs"], dict):
                raise invalid("OpenAI stream choice logprobs must be a mapping or null")
    if "usage" in value and value["usage"] is not None:
        if not isinstance(value["usage"], dict):
            raise invalid("OpenAI stream event usage must be a mapping or null")
    return value


@dataclass
class _HostState:
    runtime: AdapterRuntime | None = None
    runtime_id: str | None = None
    failed: bool = False

    def clear(self) -> None:
        self.runtime = None
        self.runtime_id = None
        self.failed = False


def _error(
    stage: str,
    code: str,
    message: str,
    *,
    retryable: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    error: dict[str, Any] = {
        "stage": stage,
        "code": code,
        "message": message,
        "retryable": retryable,
    }
    if metadata:
        error["metadata"] = dict(metadata)
    return error


def _response(
    operation: str,
    *,
    output: Any = None,
    error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    outcome = (
        {"status": "succeeded", "output": output}
        if error is None
        else {"status": "failed", "error": error}
    )
    return {
        "operation": operation,
        "outcome": outcome,
    }


def _runtime_id(operation: str, payload: dict[str, Any]) -> str | None:
    if operation in {"start", "invoke", "invoke_openai_stream"}:
        value = (payload.get("runtime_context") or {}).get("runtime_id")
    else:
        value = payload.get("runtime_id")
    return value if isinstance(value, str) and value else None


@contextmanager
def _invocation_environment(payload: dict[str, Any]) -> Iterator[None]:
    telemetry = (payload.get("runtime_context") or {}).get("telemetry") or {}
    overlay = telemetry.get("env") if isinstance(telemetry, dict) else None
    if not isinstance(overlay, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in overlay.items()
    ):
        overlay = {}
    previous = {key: os.environ.get(key) for key in overlay}
    os.environ.update(overlay)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


async def _adapter_call(operation: str, call: Callable[[], Awaitable[Any]]) -> Any:
    try:
        # Protocol stdout is reserved for exactly one JSON response per line.
        # Keep incidental adapter and library output as host diagnostics.
        with redirect_stdout(sys.stderr):
            return await call()
    except LifecycleError as error:
        raise _AdapterCallError(
            error.code,
            error.message,
            retryable=error.retryable,
            metadata=error.metadata,
        ) from error
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        raise _AdapterCallError(
            f"lifecycle_adapter_{operation}_failed",
            f"Adapter failed during lifecycle {operation}",
        ) from error


def _failure_response(operation: str, error: LifecycleError) -> dict[str, Any]:
    return _response(
        operation,
        error=_error(
            "invoke" if operation == "invoke_openai_stream" else operation,
            error.code,
            error.message,
            retryable=error.retryable,
            metadata=error.metadata,
        ),
    )


async def _stop_after_eof(runtime: AdapterRuntime) -> None:
    try:
        await _adapter_call("stop", runtime.stop)
    except LifecycleError:
        traceback.print_exc(file=sys.stderr)


def _validated_request(
    message: dict[str, Any], operation: str
) -> tuple[dict[str, Any], str]:
    if operation not in {"start", "invoke", "invoke_openai_stream", "stop"}:
        raise LifecycleError(
            "lifecycle_invalid_operation",
            "Unknown lifecycle operation",
        )
    payload = message.get("payload")
    if not isinstance(payload, dict):
        raise LifecycleError(
            "lifecycle_invalid_payload",
            "Lifecycle payload must be a mapping",
        )
    message_runtime_id = _runtime_id(operation, payload)
    if message_runtime_id is None:
        raise LifecycleError(
            "lifecycle_invalid_runtime",
            "Lifecycle payload is missing a runtime ID",
        )
    return payload, message_runtime_id


def _active_runtime(state: _HostState, message_runtime_id: str) -> AdapterRuntime:
    if state.runtime is None or state.runtime_id is None:
        raise LifecycleError(
            "lifecycle_not_started",
            "Lifecycle host has not started a runtime",
        )
    if message_runtime_id != state.runtime_id:
        raise LifecycleError(
            "lifecycle_runtime_mismatch",
            "Lifecycle payload does not match the active runtime",
        )
    return state.runtime


async def _handle_start(
    state: _HostState,
    runtime_factory: RuntimeFactory,
    payload: dict[str, Any],
    message_runtime_id: str,
    config_loader: ConfigLoader | None,
) -> dict[str, Any]:
    if state.runtime is not None:
        raise LifecycleError(
            "lifecycle_already_started",
            "Lifecycle host already owns a runtime",
        )
    if config_loader is not None:
        try:
            config = config_loader(payload.get("config"))
        except Exception as error:
            raise LifecycleError(
                "lifecycle_invalid_config",
                "Adapter config does not match its typed contract",
            ) from error
        payload = {**payload, "config": config}

    candidate = runtime_factory()
    try:
        await _adapter_call("start", lambda: candidate.start(payload))
    except LifecycleError:
        await _stop_after_eof(candidate)
        raise
    state.runtime = candidate
    state.runtime_id = message_runtime_id
    state.failed = False
    return _response("start")


async def _handle_invoke(
    state: _HostState,
    runtime: AdapterRuntime,
    payload: dict[str, Any],
) -> dict[str, Any]:
    if state.failed:
        raise LifecycleError(
            "lifecycle_runtime_failed",
            "Lifecycle runtime cannot accept another invocation",
        )
    request, context = _typed_invocation(payload)
    with _invocation_environment(payload):
        result = await _adapter_call(
            "invoke",
            lambda: runtime.invoke(request, context),
        )
    return _response("invoke", output=_typed_result(result))


async def _handle_invoke_openai_stream(
    state: _HostState,
    runtime: AdapterRuntime,
    payload: dict[str, Any],
) -> dict[str, Any]:
    if state.failed:
        raise LifecycleError(
            "lifecycle_runtime_failed",
            "Lifecycle runtime cannot accept another invocation",
        )
    invoke = getattr(runtime, "invoke_openai_stream", None)
    if not callable(invoke):
        raise LifecycleError(
            "lifecycle_openai_stream_unsupported",
            "Adapter runtime does not implement OpenAI streaming",
        )
    writer, adapter_payload = await _OpenAIStreamWriter.connect(payload)
    request, context = _typed_invocation(adapter_payload)
    adapter_error: BaseException | None = None
    output: Any = None
    try:
        with _invocation_environment(adapter_payload):
            output = await _adapter_call(
                "invoke_openai_stream",
                lambda: invoke(request, context, writer.emit),
            )
    except BaseException as error:
        adapter_error = error
    try:
        await writer.finish()
    except BaseException as finish_error:
        if adapter_error is not None:
            traceback.print_exception(finish_error, file=sys.stderr)
            raise adapter_error from finish_error
        raise
    if adapter_error is not None:
        raise adapter_error
    return _response("invoke_openai_stream", output=_typed_result(output))


def _typed_invocation(
    payload: dict[str, Any],
) -> tuple[AgentRunRequest, RuntimeContext]:
    try:
        request = AgentRunRequest.from_mapping(payload.get("request"))
        context = RuntimeContext.from_mapping(payload.get("runtime_context"))
    except Exception as error:
        raise LifecycleError(
            "lifecycle_invalid_request",
            "Invocation does not match the typed adapter contract",
        ) from error
    return request, context


def _typed_result(result: Any) -> dict[str, Any]:
    if not isinstance(result, AgentRunResult):
        raise LifecycleError(
            "lifecycle_invalid_response",
            "Adapter must return AgentRunResult",
        )
    try:
        return result.to_mapping()
    except Exception as error:
        raise LifecycleError(
            "lifecycle_invalid_response",
            "Adapter returned an invalid AgentRunResult",
        ) from error


async def _handle_stop(
    state: _HostState,
    runtime: AdapterRuntime,
) -> dict[str, Any]:
    try:
        await _adapter_call("stop", runtime.stop)
    finally:
        state.clear()
    return _response("stop")


async def _dispatch(
    state: _HostState,
    runtime_factory: RuntimeFactory,
    operation: str,
    payload: dict[str, Any],
    message_runtime_id: str,
    config_loader: ConfigLoader | None,
) -> dict[str, Any]:
    if operation == "start":
        return await _handle_start(
            state,
            runtime_factory,
            payload,
            message_runtime_id,
            config_loader,
        )
    runtime = _active_runtime(state, message_runtime_id)
    if operation == "invoke":
        return await _handle_invoke(state, runtime, payload)
    if operation == "invoke_openai_stream":
        return await _handle_invoke_openai_stream(state, runtime, payload)
    return await _handle_stop(state, runtime)


def _encode_response(
    state: _HostState,
    operation: str,
    response: dict[str, Any],
) -> str:
    try:
        return json.dumps(response, sort_keys=True)
    except (TypeError, ValueError):
        traceback.print_exc(file=sys.stderr)
        if (
            operation in {"invoke", "invoke_openai_stream"}
            and state.runtime is not None
        ):
            state.failed = True
        return json.dumps(
            _response(
                operation,
                error=_error(
                    "invoke" if operation == "invoke_openai_stream" else operation,
                    "lifecycle_invalid_response",
                    "Adapter returned a non-JSON lifecycle response",
                ),
            ),
            sort_keys=True,
        )


async def _serve(
    runtime_factory: RuntimeFactory,
    *,
    config_loader: ConfigLoader | None,
    input_stream: TextIO,
    output_stream: TextIO,
) -> None:
    state = _HostState()
    try:
        while True:
            # Keep this event loop alive while idle. Persistent SDK clients such
            # as ClaudeSDKClient own background tasks tied to this exact loop.
            line = await asyncio.to_thread(input_stream.readline)
            if not line:
                break

            operation = "start"
            should_stop = False
            try:
                message = json.loads(line)
                if not isinstance(message, dict):
                    raise TypeError("lifecycle request must be a mapping")
                raw_operation = message.get("operation")
                operation = raw_operation if isinstance(raw_operation, str) else "start"
                payload, message_runtime_id = _validated_request(message, operation)
                response = await _dispatch(
                    state,
                    runtime_factory,
                    operation,
                    payload,
                    message_runtime_id,
                    config_loader,
                )
                should_stop = operation == "stop"
            except LifecycleError as error:
                if (
                    operation in {"invoke", "invoke_openai_stream"}
                    and state.runtime is not None
                    and isinstance(error, _AdapterCallError)
                ):
                    state.failed = True
                response = _failure_response(operation, error)
                should_stop = should_stop or operation in {"start", "stop"}
            except Exception as error:
                traceback.print_exc(file=sys.stderr)
                if (
                    operation in {"invoke", "invoke_openai_stream"}
                    and state.runtime is not None
                ):
                    state.failed = True
                response = _response(
                    operation,
                    error=_error(
                        "invoke" if operation == "invoke_openai_stream" else operation,
                        "lifecycle_invalid_request",
                        "Invalid lifecycle request",
                    ),
                )

            encoded = _encode_response(state, operation, response)
            print(encoded, file=output_stream, flush=True)
            if should_stop:
                break
    finally:
        if state.runtime is not None:
            await _stop_after_eof(state.runtime)


def serve(
    runtime_factory: RuntimeFactory,
    *,
    config_loader: ConfigLoader | None = None,
    input_stream: TextIO = sys.stdin,
    output_stream: TextIO = sys.stdout,
) -> None:
    """Serve ordered lifecycle requests for exactly one Fabric runtime.

    ``config_loader`` decodes and validates the southbound start configuration.
    Contract-compliant adapters use ``AgentConfig.from_mapping`` so the runtime
    receives the canonical ``AgentConfig`` in ``payload["config"]``. The
    callable remains optional because normalizing the complete start payload is
    separate from the typed invocation boundary.
    """

    # Reserve process stdout for the protocol for the entire host lifetime,
    # including SDK background tasks running while the host is idle.
    with redirect_stdout(sys.stderr):
        asyncio.run(
            _serve(
                runtime_factory,
                config_loader=config_loader,
                input_stream=input_stream,
                output_stream=output_stream,
            )
        )
