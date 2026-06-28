# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Session lifecycle support for the Fabric Python SDK."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Mapping, Sequence
from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import Any

from nemo_fabric.errors import (
    FabricCapabilityError,
    FabricConfigError,
    FabricError,
    FabricRuntimeError,
    FabricStateError,
)
from nemo_fabric.types import (
    FabricEvent,
    RunPlan,
    RunRequest,
    RunResult,
    RuntimeHandle,
    RuntimeUpdate,
    RuntimeUpdateResult,
    SessionInfo,
)


class SessionStatus(str, Enum):
    """Lifecycle state of a session runtime."""

    ACTIVE = "active"
    STOPPED = "stopped"
    FAILED = "failed"


class Session:
    """One ordered multi-turn conversation over a Fabric runtime."""

    def __init__(
        self,
        *,
        client: Any,
        plan: RunPlan | Mapping[str, Any],
        runtime: RuntimeHandle | Mapping[str, Any],
        overrides: Mapping[str, Any] | None = None,
        session_id: str | None = None,
    ) -> None:
        self._plan = plan if isinstance(plan, RunPlan) else RunPlan.from_mapping(plan)
        _require_session_runtime(self._plan, "Session")
        self._runtime = (
            runtime if isinstance(runtime, RuntimeHandle) else RuntimeHandle.from_mapping(runtime)
        )
        self._client = client
        self._overrides = _json_mapping(overrides, "session overrides")
        self._session_id = session_id
        self._messages: list[Any] = []
        self._invocations: list[dict[str, Any]] = []
        self._status = SessionStatus.ACTIVE
        self._current_task: asyncio.Task[Any] | None = None
        self._closing = False

    @property
    def status(self) -> SessionStatus:
        return self._status

    @property
    def messages(self) -> list[Any]:
        return deepcopy(self._messages)

    @property
    def invocations(self) -> list[dict[str, Any]]:
        return deepcopy(self._invocations)

    @property
    def runtime(self) -> RuntimeHandle:
        return RuntimeHandle.from_mapping(self._runtime.to_mapping())

    @property
    def runtime_id(self) -> str:
        return self._runtime.runtime_id

    @property
    def session_id(self) -> str:
        return self._session_id or self.runtime_id

    @property
    def info(self) -> SessionInfo:
        return SessionInfo.from_mapping(
            {
                "session_id": self.session_id,
                "runtime_id": self.runtime_id,
                "agent_name": self._runtime.agent_name,
                "profiles": self._plan.profiles,
                "harness": self._runtime.harness,
                "adapter_id": self._runtime.adapter_id,
                "adapter_kind": self._runtime.adapter_kind,
                "status": self._status.value,
                "capabilities": self._plan.capabilities,
            }
        )

    async def invoke(
        self,
        *,
        input: Any = None,
        request: RunRequest | Mapping[str, Any] | None = None,
        request_id: str | None = None,
        context: Mapping[str, Any] | None = None,
        overrides: Mapping[str, Any] | None = None,
    ) -> RunResult:
        """Run one turn; turns are serialized for non-concurrent runtimes."""

        if self._status is not SessionStatus.ACTIVE:
            raise FabricStateError(f"cannot invoke a {self._status.value} session")
        if self._closing:
            raise FabricStateError("cannot invoke while session shutdown is in progress")
        if self._current_task is not None:
            raise FabricStateError("session is already running a turn")
        self._current_task = asyncio.current_task()
        try:
            payload = _run_request_payload(
                input=input,
                input_file=None,
                request=request,
                request_file=None,
                request_id=request_id,
                context=context,
                overrides=overrides,
            )
            payload["context"] = {
                **payload.get("context", {}),
                "session_id": self.session_id,
            }
            merged = _merge_overrides(self._overrides, payload.get("overrides"))
            if merged:
                payload["overrides"] = merged
            else:
                payload.pop("overrides", None)
            native = self._client._require_native_module("invoke")
            result = await _call_blocking(
                lambda: json.loads(
                    native.invoke_runtime(
                        json.dumps(self._plan.to_mapping()),
                        json.dumps(self._runtime.to_mapping()),
                        json.dumps(payload),
                    )
                )
            )
            typed_result = RunResult.from_mapping(result)
            self._absorb(typed_result)
            return typed_result
        except FabricError:
            raise
        except Exception as error:
            raise FabricRuntimeError(str(error), stage="invoke") from error
        finally:
            self._current_task = None

    async def stream(
        self,
        *,
        input: Any = None,
        request: RunRequest | Mapping[str, Any] | None = None,
        request_id: str | None = None,
        context: Mapping[str, Any] | None = None,
        overrides: Mapping[str, Any] | None = None,
    ) -> AsyncIterator[FabricEvent | RunResult]:
        """Yield buffered events followed by one terminal result."""

        result = await self.invoke(
            input=input,
            request=request,
            request_id=request_id,
            context=context,
            overrides=overrides,
        )
        for event in result.events:
            yield event
        yield result

    async def update(self, update: RuntimeUpdate) -> RuntimeUpdateResult:
        """Apply a capability-gated runtime update."""

        if not isinstance(update, RuntimeUpdate):
            raise FabricConfigError("update must be a RuntimeUpdate")
        if not self._plan.capabilities.updates:
            raise FabricCapabilityError(
                "runtime updates are not supported",
                stage="update",
                code="updates_not_supported",
            )
        raise FabricCapabilityError(
            "runtime update transport is not implemented",
            stage="update",
            code="updates_not_implemented",
        )

    async def cancel(self) -> None:
        """Cancel the current invocation when the runtime declares support."""

        if not self._plan.capabilities.cancellation:
            raise FabricCapabilityError(
                "runtime cancellation is not supported",
                stage="cancel",
                code="cancellation_not_supported",
            )
        raise FabricCapabilityError(
            "runtime cancellation transport is not implemented",
            stage="cancel",
            code="cancellation_not_implemented",
        )

    async def stop(self) -> None:
        """Destroy an idle runtime exactly once."""

        if self._status is SessionStatus.STOPPED:
            return
        if self._status is SessionStatus.FAILED:
            raise FabricStateError("cannot stop a failed session")
        if self._current_task is not None:
            raise FabricStateError("cannot stop while a turn is in flight")
        if self._closing:
            raise FabricStateError("session shutdown is already in progress")
        self._closing = True
        try:
            native = self._client._require_native_module("stop")
            await _call_blocking(
                lambda: json.loads(
                    native.stop_runtime(
                        json.dumps(self._plan.to_mapping()),
                        json.dumps(self._runtime.to_mapping()),
                    )
                )
            )
        except FabricError:
            self._status = SessionStatus.FAILED
            raise
        except Exception as error:
            self._status = SessionStatus.FAILED
            raise FabricRuntimeError(str(error), stage="stop") from error
        else:
            self._status = SessionStatus.STOPPED
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
            self._messages = [dict(message) for message in messages]

    async def __aenter__(self) -> "Session":
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        await self.stop()


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
    input_file: str | Path | None,
    request: RunRequest | Mapping[str, Any] | None,
    request_file: str | Path | None,
    request_id: str | None,
    context: Mapping[str, Any] | None,
    overrides: Mapping[str, Any] | None,
) -> dict[str, Any]:
    primary_sources = [
        input is not None,
        input_file is not None,
        request is not None,
        request_file is not None,
    ]
    if sum(primary_sources) > 1:
        raise FabricConfigError(
            "at most one input source is allowed: input, input_file, request, or request_file"
        )
    separate_fields = request_id is not None or context is not None or overrides is not None
    if (request is not None or request_file is not None) and separate_fields:
        raise FabricConfigError(
            "a complete request cannot be combined with separate request fields"
        )
    if request_file is not None:
        try:
            raw = json.loads(Path(request_file).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise FabricConfigError(f"failed to read request file: {error}") from error
        payload = RunRequest.from_mapping(raw).to_mapping()
    elif request is not None:
        payload = (
            request.to_mapping()
            if isinstance(request, RunRequest)
            else RunRequest.from_mapping(request).to_mapping()
        )
    elif input_file is not None:
        try:
            file_input = Path(input_file).read_text(encoding="utf-8")
        except OSError as error:
            raise FabricConfigError(f"failed to read input file: {error}") from error
        payload = RunRequest(
            input=file_input,
            request_id=request_id,
            context=context,
            overrides=overrides,
        ).to_mapping()
    else:
        payload = RunRequest(
            input=input,
            request_id=request_id,
            context=context,
            overrides=overrides,
        ).to_mapping()
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
    task = asyncio.create_task(asyncio.to_thread(func))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        try:
            await asyncio.shield(task)
        except Exception:
            pass
        raise


def _require_session_runtime(plan: RunPlan | Mapping[str, Any], method: str) -> None:
    typed_plan = plan if isinstance(plan, RunPlan) else RunPlan.from_mapping(plan)
    if not typed_plan.capabilities.session:
        raise FabricCapabilityError(
            f"{method} requires session capability",
            stage="start",
            code="session_not_supported",
        )
