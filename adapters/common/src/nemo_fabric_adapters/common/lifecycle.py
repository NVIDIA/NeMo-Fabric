# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Versioned lifecycle host for adapters that support persistent runtimes."""

from __future__ import annotations

import json
import os
import sys
import traceback
from collections.abc import Callable
from collections.abc import Iterator
from collections.abc import Mapping
from contextlib import contextmanager
from contextlib import redirect_stdout
from typing import Any
from typing import TextIO


CONTRACT_VERSION = "fabric.adapter.lifecycle/v1alpha1"
CONTRACT_ENV = "FABRIC_ADAPTER_LIFECYCLE_CONTRACT"

AdapterRun = Callable[[dict[str, Any]], dict[str, Any]]


def is_lifecycle_host(environ: Mapping[str, str]) -> bool:
    """Return whether Fabric requested the versioned lifecycle host protocol."""

    return CONTRACT_ENV in environ


def _error(stage: str, code: str, message: str) -> dict[str, Any]:
    return {
        "stage": stage,
        "code": code,
        "message": message,
        "retryable": False,
    }


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


def _runtime_id(message: dict[str, Any]) -> str | None:
    operation = message.get("operation")
    payload = message.get("payload") or {}
    if operation == "start":
        value = (payload.get("runtime") or {}).get("runtime_id")
    elif operation == "invoke":
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


def _handle_message(
    message: dict[str, Any],
    *,
    run: AdapterRun,
    active_runtime_id: str | None,
) -> tuple[dict[str, Any], str | None, bool]:
    operation = message.get("operation")
    if operation not in {"start", "invoke", "stop"}:
        return (
            _response(
                "start",
                error=_error(
                    "start", "lifecycle_invalid_operation", "Unknown lifecycle operation"
                ),
            ),
            active_runtime_id,
            False,
        )
    if message.get("contract_version") != CONTRACT_VERSION:
        return (
            _response(
                operation,
                error=_error(
                    operation,
                    "lifecycle_contract_mismatch",
                    f"Expected lifecycle contract {CONTRACT_VERSION}",
                ),
            ),
            active_runtime_id,
            False,
        )

    runtime_id = _runtime_id(message)
    if runtime_id is None:
        return (
            _response(
                operation,
                error=_error(
                    operation,
                    "lifecycle_invalid_runtime",
                    "Lifecycle payload is missing a runtime ID",
                ),
            ),
            active_runtime_id,
            False,
        )

    if operation == "start":
        if active_runtime_id is not None:
            return (
                _response(
                    operation,
                    error=_error(
                        operation,
                        "lifecycle_already_started",
                        "Lifecycle host already owns a runtime",
                    ),
                ),
                active_runtime_id,
                False,
            )
        return _response(operation), runtime_id, False

    if active_runtime_id != runtime_id:
        return (
            _response(
                operation,
                error=_error(
                    operation,
                    "lifecycle_runtime_mismatch",
                    "Lifecycle payload does not match the active runtime",
                ),
            ),
            active_runtime_id,
            False,
        )
    if operation == "invoke":
        payload = message.get("payload")
        if not isinstance(payload, dict):
            return (
                _response(
                    operation,
                    error=_error(
                        operation,
                        "lifecycle_invalid_payload",
                        "Invoke payload must be a mapping",
                    ),
                ),
                active_runtime_id,
                False,
            )
        # Protocol stdout is reserved for one JSON response per line. Preserve
        # incidental adapter/library output as diagnostics instead.
        try:
            with _invocation_environment(payload), redirect_stdout(sys.stderr):
                output = run(payload)
        except Exception:
            traceback.print_exc(file=sys.stderr)
            return (
                _response(
                    operation,
                    error=_error(
                        operation,
                        "lifecycle_adapter_failure",
                        "Adapter failed while processing the invocation",
                    ),
                ),
                active_runtime_id,
                False,
            )
        return _response(operation, output=output), active_runtime_id, False

    return _response(operation), None, True


def serve(
    run: AdapterRun,
    *,
    input_stream: TextIO = sys.stdin,
    output_stream: TextIO = sys.stdout,
) -> None:
    """Serve ordered lifecycle requests for exactly one Fabric runtime."""

    active_runtime_id: str | None = None
    for line in input_stream:
        try:
            message = json.loads(line)
            if not isinstance(message, dict):
                raise TypeError("lifecycle request must be a mapping")
            response, active_runtime_id, should_stop = _handle_message(
                message,
                run=run,
                active_runtime_id=active_runtime_id,
            )
        except Exception as error:  # Protocol boundary must retain diagnostics.
            print(f"Invalid lifecycle request: {error}", file=sys.stderr, flush=True)
            response = _response(
                "start",
                error=_error(
                    "start", "lifecycle_invalid_request", "Invalid lifecycle request"
                ),
            )
            should_stop = False
        print(json.dumps(response, sort_keys=True), file=output_stream, flush=True)
        if should_stop:
            break
