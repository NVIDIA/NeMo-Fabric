# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Versioned host protocol for persistent local adapter runtimes."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import traceback
from collections.abc import Callable
from collections.abc import Iterator
from collections.abc import Mapping
from collections.abc import Awaitable
from contextlib import contextmanager
from contextlib import redirect_stdout
from typing import Any
from typing import Protocol
from typing import TextIO


CONTRACT_VERSION = "fabric.adapter.lifecycle/v1alpha1"
CONTRACT_ENV = "FABRIC_ADAPTER_LIFECYCLE_CONTRACT"


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


def is_lifecycle_host(environ: Mapping[str, str] = os.environ) -> bool:
    """Return whether Fabric requested the persistent local-host protocol."""

    return environ.get(CONTRACT_ENV) == CONTRACT_VERSION


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
        "contract_version": CONTRACT_VERSION,
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
    except LifecycleError:
        raise
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        raise LifecycleError(
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


async def _serve(
    runtime_factory: RuntimeFactory,
    *,
    input_stream: TextIO,
    output_stream: TextIO,
) -> None:
    runtime: AdapterRuntime | None = None
    active_runtime_id: str | None = None
    runtime_failed = False
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
                if operation not in {"start", "invoke", "stop"}:
                    raise LifecycleError(
                        "lifecycle_invalid_operation",
                        "Unknown lifecycle operation",
                    )
                if message.get("contract_version") != CONTRACT_VERSION:
                    raise LifecycleError(
                        "lifecycle_contract_mismatch",
                        f"Expected lifecycle contract {CONTRACT_VERSION}",
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

                if operation == "start":
                    if runtime is not None:
                        raise LifecycleError(
                            "lifecycle_already_started",
                            "Lifecycle host already owns a runtime",
                        )
                    candidate = runtime_factory()
                    try:
                        await _adapter_call("start", lambda: candidate.start(payload))
                    except LifecycleError:
                        await _stop_after_eof(candidate)
                        raise
                    runtime = candidate
                    active_runtime_id = message_runtime_id
                    runtime_failed = False
                    response = _response(operation)
                else:
                    if runtime is None or active_runtime_id is None:
                        raise LifecycleError(
                            "lifecycle_not_started",
                            "Lifecycle host has not started a runtime",
                        )
                    if message_runtime_id != active_runtime_id:
                        raise LifecycleError(
                            "lifecycle_runtime_mismatch",
                            "Lifecycle payload does not match the active runtime",
                        )
                    if operation == "invoke":
                        if runtime_failed:
                            raise LifecycleError(
                                "lifecycle_runtime_failed",
                                "Lifecycle runtime cannot accept another invocation",
                            )
                        with _invocation_environment(payload):
                            output = await _adapter_call(
                                "invoke", lambda: runtime.invoke(payload)
                            )
                        response = _response(operation, output=output)
                    else:
                        try:
                            await _adapter_call("stop", runtime.stop)
                        finally:
                            runtime = None
                            active_runtime_id = None
                            runtime_failed = False
                            should_stop = True
                        response = _response(operation)
            except LifecycleError as error:
                if (
                    operation == "invoke"
                    and runtime is not None
                    and error.code == "lifecycle_adapter_invoke_failed"
                ):
                    runtime_failed = True
                response = _failure_response(operation, error)
                should_stop = should_stop or operation in {"start", "stop"}
            except Exception as error:
                traceback.print_exc(file=sys.stderr)
                if operation == "invoke" and runtime is not None:
                    runtime_failed = True
                response = _response(
                    operation,
                    error=_error(
                        operation,
                        "lifecycle_invalid_request",
                        "Invalid lifecycle request",
                    ),
                )

            try:
                encoded = json.dumps(response, sort_keys=True)
            except (TypeError, ValueError):
                traceback.print_exc(file=sys.stderr)
                if operation == "invoke" and runtime is not None:
                    runtime_failed = True
                encoded = json.dumps(
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
            print(encoded, file=output_stream, flush=True)
            if should_stop:
                break
    finally:
        if runtime is not None:
            await _stop_after_eof(runtime)


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
