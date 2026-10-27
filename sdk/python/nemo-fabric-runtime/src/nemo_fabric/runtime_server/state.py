# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Manage independently addressed SDK runtimes in one sandbox."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from nemo_fabric import (
    Fabric,
    FabricConfig,
    FabricConfigError,
    CheckRequest,
    EnvironmentHandle,
    RunRequest,
    Runtime,
    RuntimeStatus,
)
from nemo_fabric.runtime_server.protocol import (
    OPERATIONS,
    ProtocolError,
    envelope,
    validate_request,
)
from nemo_fabric.runtime_server.health import (
    check_response,
    runtime_report,
    server_report,
)

MAX_STOPPED_RUNTIME_RECORDS = 128


@dataclass
class OwnedRuntime:
    runtime: Runtime | None
    agent: str
    state: str = "running"
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    generation: int = 0


class RuntimeServer:
    """Route commands to independently managed runtimes owned by this server."""

    def __init__(
        self, *, directory: Path = Path("/sandbox"), fabric: Fabric | None = None
    ):
        self.directory = directory
        self.fabric = fabric if fabric is not None else Fabric()
        self.runtimes: dict[str, OwnedRuntime] = {}
        self.shutting_down = False
        # Sticky after any successful or uncertain start. The transport uses this
        # to retain its sandbox ownership marker after runtimes are stopped.
        self.has_used_runtime = False
        # An uncertain SDK start may have created an unaddressable runtime. Do not
        # issue another start until the enclosing sandbox is replaced.
        self.start_uncertain = False
        self._start_lock = asyncio.Lock()

    def _owned(self, request: dict[str, Any]) -> OwnedRuntime:
        owned = self.runtimes.get(request["runtime_id"])
        if owned is None:
            raise ProtocolError("runtime_unavailable")
        if owned.agent != request["agent"]:
            raise ProtocolError("wrong_agent")
        return owned

    async def _stop(self, owned: OwnedRuntime) -> None:
        owned.generation += 1
        owned.state = "unknown"
        runtime = owned.runtime
        if runtime is not None:
            await runtime.stop()
        owned.runtime = None
        owned.state = "stopped"
        runtime_id = runtime.runtime_id if runtime is not None else None
        if runtime_id is not None:
            self.runtimes.pop(runtime_id, None)
            self.runtimes[runtime_id] = owned
        stopped_ids = [
            runtime_id
            for runtime_id, record in self.runtimes.items()
            if record.state == "stopped"
        ]
        for runtime_id in stopped_ids[:-MAX_STOPPED_RUNTIME_RECORDS]:
            self.runtimes.pop(runtime_id, None)

    async def handle(self, request: Any) -> dict[str, Any]:
        started = time.monotonic()
        operation = (
            request.get("operation", "unknown")
            if isinstance(request, dict)
            else "unknown"
        )
        if operation not in OPERATIONS:
            operation = "unknown"
        result = None
        changed, effects, stage = False, "none", "request"
        try:
            validate_request(request)
            if operation == "serve":
                raise ProtocolError()
            if operation == "check":
                if "runtime_id" in request:
                    return await self._check_runtime(request, started)
                return envelope(
                    operation,
                    server_report(
                        "stopping" if self.shutting_down else "responding", started
                    ),
                    error=ProtocolError("server_stopping", "check")
                    if self.shutting_down
                    else None,
                )
            if self.shutting_down:
                raise ProtocolError("host_stopping")

            if operation == "start":
                async with self._start_lock:
                    if self.shutting_down:
                        raise ProtocolError("host_stopping")
                    if self.start_uncertain:
                        raise ProtocolError("start_uncertain", "start")
                    stage = "validation"
                    config = FabricConfig.model_validate(request["config"])
                    previously_used = self.has_used_runtime
                    # Keep the sandbox marker if this request is cancelled while
                    # native start is running. A known validation rejection has
                    # no effects and restores the prior marker state.
                    self.has_used_runtime = True
                    self.start_uncertain = True
                    changed, effects, stage = None, "unknown", "start"
                    try:
                        runtime = await self.fabric.start_runtime(
                            config, base_dir=self.directory
                        )
                    except FabricConfigError:
                        self.has_used_runtime = previously_used
                        self.start_uncertain = False
                        changed, effects = False, "none"
                        raise

                    if runtime.runtime_id in self.runtimes:
                        # A duplicate SDK identity cannot be safely routed to
                        # one of two runtime objects; keep the start outcome
                        # uncertain and retain sandbox ownership.
                        raise ProtocolError("runtime_id_conflict", "start")
                    owned = OwnedRuntime(runtime, request["agent"], "unknown")
                    self.runtimes[runtime.runtime_id] = owned
                    # The runtime is addressable; stop can clean it up.
                    self.start_uncertain = False
                    if runtime.status != RuntimeStatus.ACTIVE:
                        result = {"runtime_id": runtime.runtime_id}
                        raise ProtocolError("start_failed", "start")
                    owned.state = "running"
                    response = envelope(
                        operation, runtime.handle.to_mapping(), changed=True
                    )
                    self.start_uncertain = False
                    return response

            owned = self._owned(request)
            if operation == "stop" and owned.state != "stopped":
                owned.generation += 1
                owned.state = "stopping"
            async with owned.lock:
                if self.shutting_down:
                    raise ProtocolError("host_stopping")
                if operation == "stop":
                    if owned.state == "stopped":
                        return envelope(operation, result)
                    changed, effects, stage = None, "unknown", "stop"
                    await self._stop(owned)
                    return envelope(operation, result, changed=True)

                stage = "invoke"
                if (
                    owned.state != "running"
                    or owned.runtime is None
                    or owned.runtime.status != RuntimeStatus.ACTIVE
                ):
                    raise ProtocolError("runtime_unavailable", stage)
                stage = "validation"
                if "request" in request:
                    invoke_arguments = {
                        "request": RunRequest.model_validate(request["request"])
                    }
                else:
                    RunRequest(input=request["input"])
                    invoke_arguments = {"input": request["input"]}
                changed, effects, stage = None, "unknown", "invoke"
                result = (await owned.runtime.invoke(**invoke_arguments)).to_mapping()
                return envelope(operation, result, changed=None)
        except Exception as error:
            if not isinstance(error, ProtocolError):
                if isinstance(error, (FabricConfigError, ValidationError)):
                    code = (
                        "validation_unavailable"
                        if getattr(error, "code", None)
                        == "adapter_capability_unverified"
                        else "validation_failed"
                    )
                    error = ProtocolError(code, "validation")
                else:
                    error = ProtocolError(f"{stage}_failed", stage)
            return envelope(
                operation, result, changed=changed, error=error, effects=effects
            )

    async def _check_runtime(
        self, command: dict[str, Any], started: float
    ) -> dict[str, Any]:
        request = CheckRequest(
            scope="runtime",
            runtime_id=command["runtime_id"],
            level=command.get("level", "live"),
            timeout_seconds=command.get("timeout_seconds", 3.0),
        )
        owned = self.runtimes.get(request.runtime_id)
        if owned is None:
            return check_response(
                runtime_report(request, "runtime_unavailable", started)
            )
        if owned.agent != command["agent"]:
            raise ProtocolError("wrong_agent", "check")
        runtime = owned.runtime
        if (
            self.shutting_down
            or owned.state != "running"
            or runtime is None
            or runtime.status != RuntimeStatus.ACTIVE
        ):
            return check_response(
                runtime_report(request, "runtime_unavailable", started)
            )
        generation = owned.generation
        remaining = request.timeout_seconds - (time.monotonic() - started)
        if remaining <= 0:
            return check_response(runtime_report(request, "probe_timeout", started))
        try:
            report = await asyncio.wait_for(
                self.fabric.check(
                    environment=EnvironmentHandle.from_mapping(
                        runtime.handle.environment
                    ),
                    request=request.model_copy(update={"timeout_seconds": remaining}),
                ),
                remaining,
            )
            report = report._complete(started).to_mapping()
        except Exception:
            report = runtime_report(request, "probe_error", started)
        if (
            self.shutting_down
            or self.runtimes.get(request.runtime_id) is not owned
            or owned.runtime is not runtime
            or owned.generation != generation
        ):
            report = runtime_report(request, "observation_changed", started)
        elif time.monotonic() - started >= request.timeout_seconds:
            report = runtime_report(request, "probe_timeout", started)
        return check_response(report)

    async def shutdown(self) -> None:
        """Stop idle runtimes. The enclosing sandbox owns descendant cleanup."""
        self.shutting_down = True
        if self._start_lock.locked() or self.start_uncertain:
            raise ProtocolError("cleanup_unconfirmed", "stop")
        if any(owned.lock.locked() for owned in self.runtimes.values()):
            raise ProtocolError("runtime_busy", "stop")
        cleanup_failures: list[Exception] = []
        async with self._start_lock:
            for owned in tuple(self.runtimes.values()):
                async with owned.lock:
                    if owned.state != "stopped":
                        try:
                            await self._stop(owned)
                        except Exception as error:
                            cleanup_failures.append(error)
        if cleanup_failures:
            raise ProtocolError("cleanup_unconfirmed", "stop")
