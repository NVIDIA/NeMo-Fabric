# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime lifecycle support for the Fabric Python SDK."""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from enum import Enum
from threading import Thread
from typing import Any

from pydantic import ValidationError

from nemo_fabric.errors import FabricConfigError, FabricError, FabricRuntimeError, FabricStateError
from nemo_fabric.models import RunRequest
from nemo_fabric.types import RunPlan, RunResult, RuntimeHandle


class RuntimeStatus(str, Enum):
    """Lifecycle state of a runtime.

    ``ACTIVE`` accepts invocations, ``STOPPED`` has released its runtime, and
    ``FAILED`` records a lifecycle failure that prevents further invocations
    but still permits cleanup.
    """

    ACTIVE = "active"
    STOPPED = "stopped"
    FAILED = "failed"


class Runtime:
    """One logical, stateful harness execution.

    Create runtimes with ``Fabric.start_runtime()`` rather than calling the
    constructor. A runtime serializes invocations and preserves adapter-owned
    harness state across turns. Use it as an asynchronous context manager to
    stop the runtime on exit.

    Runtime-scoped overrides are recursively merged with invocation overrides;
    invocation values win.
    """

    def __init__(
        self,
        *,
        client: Any,
        plan: RunPlan | Mapping[str, Any],
        runtime: RuntimeHandle | Mapping[str, Any],
        overrides: Mapping[str, Any] | None = None,
    ) -> None:
        """lazydocs: ignore"""

        self._plan = plan if isinstance(plan, RunPlan) else RunPlan.from_mapping(plan)
        self._runtime = (
            runtime if isinstance(runtime, RuntimeHandle) else RuntimeHandle.from_mapping(runtime)
        )
        self._client = client
        self._overrides = _json_mapping(overrides, "runtime overrides")
        self._messages: list[Any] = []
        self._invocations: list[dict[str, Any]] = []
        self._status = RuntimeStatus.ACTIVE
        self._current_task: asyncio.Task[Any] | None = None
        self._closing = False

    @property
    def status(self) -> RuntimeStatus:
        """Return the current ``ACTIVE``, ``STOPPED``, or ``FAILED`` state."""

        return self._status

    @property
    def messages(self) -> list[Any]:
        """Return a deep copy of the latest harness-provided message history."""

        return deepcopy(self._messages)

    @property
    def invocations(self) -> list[dict[str, Any]]:
        """Return copied request, runtime, and invocation IDs for completed turns."""

        return deepcopy(self._invocations)

    @property
    def handle(self) -> RuntimeHandle:
        """Return a detached snapshot of the runtime handle."""

        return RuntimeHandle.from_mapping(self._runtime.to_mapping())

    @property
    def runtime_id(self) -> str:
        """Return the unique identifier for this started runtime lifecycle."""

        return self._runtime.runtime_id

    async def invoke(
        self,
        *,
        input: Any = None,
        request: RunRequest | None = None,
    ) -> RunResult:
        """Run one turn on this runtime.

        ``input`` and ``request`` are mutually exclusive. Runtime overrides are
        merged below invocation overrides from ``RunRequest``. Concurrent turns
        on the same runtime are rejected.

        Args:
            input: JSON-compatible turn input.
            request: Complete validated ``RunRequest``.

        Returns:
            The normalized ``RunResult`` for this turn.

        Raises:
            FabricConfigError: If request fields conflict or are not
                JSON-compatible.
            FabricStateError: If the runtime is not active, is stopping, or is
                already running a turn.
            FabricNativeUnavailableError: If the native extension is missing.
            FabricRuntimeError: If native invocation fails before returning a
                normalized result.
        """

        if self._status is not RuntimeStatus.ACTIVE:
            raise FabricStateError(f"cannot invoke a {self._status.value} runtime")
        if self._closing:
            raise FabricStateError("cannot invoke while runtime shutdown is in progress")
        if self._current_task is not None:
            raise FabricStateError("runtime is already running an invocation")
        self._current_task = asyncio.current_task()
        try:
            payload = _run_request_payload(
                input=input,
                request=request,
            )
            merged = _merge_overrides(self._overrides, payload.get("overrides"))
            if merged:
                payload["overrides"] = merged
            else:
                payload.pop("overrides", None)
            native_result: dict[str, Any] | None = None
            try:
                native = self._client._require_native_module("invoke")

                def invoke() -> dict[str, Any]:
                    nonlocal native_result
                    native_result = json.loads(
                        native.invoke_runtime(
                            json.dumps(self._plan.to_mapping()),
                            json.dumps(self._runtime.to_mapping()),
                            json.dumps(payload),
                        )
                    )
                    return native_result

                result = await _call_blocking(invoke)
                typed_result = RunResult.from_mapping(result)
            except asyncio.CancelledError:
                if native_result is not None:
                    try:
                        self._absorb(RunResult.from_mapping(native_result))
                    except Exception:
                        pass
                stopped = False

                def stop_after_cancel() -> Any:
                    nonlocal stopped
                    result = json.loads(
                        native.stop_runtime(
                            json.dumps(self._plan.to_mapping()),
                            json.dumps(self._runtime.to_mapping()),
                        )
                    )
                    stopped = True
                    return result

                try:
                    await _call_blocking(stop_after_cancel)
                except asyncio.CancelledError:
                    self._status = RuntimeStatus.STOPPED if stopped else RuntimeStatus.FAILED
                    raise
                except Exception:
                    self._status = RuntimeStatus.FAILED
                else:
                    self._status = RuntimeStatus.STOPPED
                raise
            except FabricError:
                self._status = RuntimeStatus.FAILED
                raise
            except Exception as error:
                self._status = RuntimeStatus.FAILED
                raise FabricRuntimeError(str(error), stage="invoke") from error
            self._absorb(typed_result)
            return typed_result
        except FabricError:
            raise
        except Exception as error:
            raise FabricRuntimeError(str(error), stage="invoke") from error
        finally:
            self._current_task = None

    async def stop(self) -> None:
        """Destroy an idle runtime exactly once.

        Repeated calls after a successful stop are no-ops. A failed runtime may
        still be stopped so its resources are released.

        Raises:
            FabricStateError: If the runtime is already stopping or has an
                invocation in flight.
            FabricNativeUnavailableError: If the native extension is missing.
            FabricRuntimeError: If native runtime shutdown fails.
        """

        if self._status is RuntimeStatus.STOPPED:
            return
        if self._current_task is not None:
            raise FabricStateError("cannot stop while a turn is in flight")
        if self._closing:
            raise FabricStateError("runtime shutdown is already in progress")
        self._closing = True
        stopped = False
        try:
            native = self._client._require_native_module("stop")

            def stop() -> Any:
                nonlocal stopped
                result = json.loads(
                    native.stop_runtime(
                        json.dumps(self._plan.to_mapping()),
                        json.dumps(self._runtime.to_mapping()),
                    )
                )
                stopped = True
                return result

            await _call_blocking(stop)
        except asyncio.CancelledError:
            self._status = RuntimeStatus.STOPPED if stopped else RuntimeStatus.FAILED
            raise
        except FabricError:
            self._status = RuntimeStatus.FAILED
            raise
        except Exception as error:
            self._status = RuntimeStatus.FAILED
            raise FabricRuntimeError(str(error), stage="stop") from error
        else:
            self._status = RuntimeStatus.STOPPED
        finally:
            self._closing = False

    def _absorb(self, result: RunResult) -> None:
        self._invocations.append(
            {
                "request_id": result.request_id,
                "runtime_id": result.runtime_id,
                "invocation_id": result.invocation_id,
            }
        )
        output = result.output
        messages = output.get("messages") if isinstance(output, Mapping) else None
        if isinstance(messages, Sequence) and not isinstance(messages, (str, bytes)):
            self._messages = deepcopy(list(messages))

    async def __aenter__(self) -> "Runtime":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> None:
        try:
            await self.stop()
        except Exception as cleanup_error:
            if exc is None:
                raise
            exc.add_note(f"runtime cleanup failed: {cleanup_error}")


def _json_mapping(value: Mapping[str, Any] | None, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise FabricConfigError(f"{name} must be a JSON object")
    pending: list[Any] = [value]
    seen: set[int] = set()
    while pending:
        item = pending.pop()
        if isinstance(item, (Mapping, list, tuple)):
            identity = id(item)
            if identity in seen:
                continue
            seen.add(identity)
        if isinstance(item, Mapping):
            if any(not isinstance(key, str) for key in item):
                raise FabricConfigError(f"{name} keys must be strings")
            pending.extend(item.values())
        elif isinstance(item, (list, tuple)):
            pending.extend(item)
    try:
        return json.loads(json.dumps(dict(value), allow_nan=False))
    except (TypeError, ValueError) as error:
        raise FabricConfigError(f"{name} must contain JSON-compatible values") from error


def _merge_overrides(
    base: Mapping[str, Any] | None,
    extra: Mapping[str, Any] | None,
) -> dict[str, Any]:
    result = _json_mapping(base, "request overrides")
    for key, value in _json_mapping(extra, "request overrides").items():
        current = result.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            result[key] = _merge_overrides(current, value)
        else:
            result[key] = value
    return result


def _run_request_payload(
    *,
    input: Any,
    request: RunRequest | None,
) -> dict[str, Any]:
    if input is not None and request is not None:
        raise FabricConfigError("input and request are mutually exclusive")
    try:
        if request is not None:
            if not isinstance(request, RunRequest):
                raise FabricConfigError("request must be a RunRequest")
            payload = request.to_mapping()
        else:
            payload = RunRequest(input=input).to_mapping()
    except ValidationError as error:
        raise FabricConfigError(str(error)) from error
    return payload


async def _run_native_lifecycle(
    native: Any,
    plan: Mapping[str, Any],
    request: Mapping[str, Any],
) -> dict[str, Any]:
    def run() -> dict[str, Any]:
        plan_json = json.dumps(dict(plan))
        runtime = json.loads(native.start_runtime(plan_json))
        runtime_json = json.dumps(runtime)
        result: dict[str, Any] | None = None
        invoke_error: Exception | None = None
        try:
            try:
                result = json.loads(
                    native.invoke_runtime(plan_json, runtime_json, json.dumps(dict(request)))
                )
            except Exception as error:
                invoke_error = error
                raise
            return result
        finally:
            try:
                stop_events = json.loads(native.stop_runtime(plan_json, runtime_json))
            except Exception:
                if invoke_error is None:
                    raise
                stop_events = []
            if result is not None and isinstance(stop_events, list):
                result.setdefault("events", []).extend(stop_events)

    try:
        return await _call_blocking(run)
    except FabricError:
        raise
    except Exception as error:
        raise FabricRuntimeError(str(error), stage="run") from error


async def _call_blocking(func: Any) -> Any:
    worker: concurrent.futures.Future[Any] = concurrent.futures.Future()

    def run() -> None:
        try:
            result = func()
        except BaseException as error:
            worker.set_exception(error)
        else:
            worker.set_result(result)

    Thread(target=run, name="nemo-fabric-worker", daemon=True).start()
    try:
        while not worker.done():
            await asyncio.sleep(0.001)
        return worker.result()
    except asyncio.CancelledError as cancelled:
        while not worker.done():
            try:
                await asyncio.sleep(0.001)
            except asyncio.CancelledError:
                continue
        try:
            worker.result()
        except BaseException:
            pass
        raise cancelled
