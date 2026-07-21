# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wire protocol for persistent local adapter hosts."""

from __future__ import annotations

import asyncio
import copy
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
from dataclasses import dataclass
from typing import Any
from typing import Protocol
from typing import TextIO


class AdapterRuntime(Protocol):
    """One adapter-owned runtime living for the complete host lifetime."""

    async def start(self, payload: dict[str, Any]) -> None:
        """Initialize runtime-owned SDK clients and resources."""

    async def invoke(self, payload: dict[str, Any]) -> Any:
        """Execute one invocation against the initialized runtime."""

    async def stop(self) -> None:
        """Release all resources owned by the runtime."""


RuntimeFactory = Callable[[], AdapterRuntime]


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


@dataclass
class _HostState:
    runtime: AdapterRuntime | None = None
    runtime_id: str | None = None
    start_payload: dict[str, Any] | None = None
    failed: bool = False

    def clear(self) -> None:
        self.runtime = None
        self.runtime_id = None
        self.start_payload = None
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
    if operation in {"start", "invoke"}:
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
            operation,
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
    if operation not in {"start", "invoke", "stop"}:
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
) -> dict[str, Any]:
    if state.runtime is not None:
        raise LifecycleError(
            "lifecycle_already_started",
            "Lifecycle host already owns a runtime",
        )
    candidate = runtime_factory()
    retained_payload = copy.deepcopy(payload)
    try:
        await _adapter_call("start", lambda: candidate.start(payload))
    except LifecycleError:
        await _stop_after_eof(candidate)
        raise
    state.runtime = candidate
    state.runtime_id = message_runtime_id
    state.start_payload = retained_payload
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
    if state.start_payload is None:
        raise LifecycleError(
            "lifecycle_not_started",
            "Lifecycle host has no retained start payload",
        )
    invocation_payload = copy.deepcopy(state.start_payload)
    invocation_payload["runtime_context"] = payload.get("runtime_context")
    invocation_payload["request"] = payload.get("request")
    with _invocation_environment(payload):
        output = await _adapter_call(
            "invoke", lambda: runtime.invoke(invocation_payload)
        )
    return _response("invoke", output=output)


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
) -> dict[str, Any]:
    if operation == "start":
        return await _handle_start(
            state,
            runtime_factory,
            payload,
            message_runtime_id,
        )
    runtime = _active_runtime(state, message_runtime_id)
    if operation == "invoke":
        return await _handle_invoke(state, runtime, payload)
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
        if operation == "invoke" and state.runtime is not None:
            state.failed = True
        return json.dumps(
            _response(
                operation,
                error=_error(
                    operation,
                    "lifecycle_invalid_response",
                    "Adapter returned a non-JSON lifecycle response",
                ),
            ),
            sort_keys=True,
        )


async def _serve(
    runtime_factory: RuntimeFactory,
    *,
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
                )
                should_stop = operation == "stop"
            except LifecycleError as error:
                if (
                    operation == "invoke"
                    and state.runtime is not None
                    and isinstance(error, _AdapterCallError)
                ):
                    state.failed = True
                response = _failure_response(operation, error)
                should_stop = should_stop or operation in {"start", "stop"}
            except Exception as error:
                traceback.print_exc(file=sys.stderr)
                if operation == "invoke" and state.runtime is not None:
                    state.failed = True
                response = _response(
                    operation,
                    error=_error(
                        operation,
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
    input_stream: TextIO = sys.stdin,
    output_stream: TextIO = sys.stdout,
) -> None:
    """Serve ordered lifecycle requests for exactly one Fabric runtime."""

    # Reserve process stdout for the protocol for the entire host lifetime,
    # including SDK background tasks running while the host is idle.
    with redirect_stdout(sys.stderr):
        asyncio.run(
            _serve(
                runtime_factory,
                input_stream=input_stream,
                output_stream=output_stream,
            )
        )
