# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One runtime lifecycle per sandbox, using the existing public SDK."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from nemo_fabric import Fabric, FabricConfig, FabricConfigError, Runtime, RuntimeStatus
from nemo_fabric.host.protocol import (
    OPERATIONS,
    ProtocolError,
    envelope,
    validate_request,
)


@dataclass
class OwnedRuntime:
    runtime: Runtime
    agent: str
    state: str = "running"


class RuntimeHost:
    """Serialize commands for one owned runtime. A stopped sandbox cannot be reused."""

    def __init__(
        self, *, directory: Path = Path("/sandbox"), fabric: Fabric | None = None
    ):
        self.directory = directory
        self.fabric = fabric if fabric is not None else Fabric()
        self.runtimes: dict[str, OwnedRuntime] = {}
        self.shutting_down = False
        # Sticky after a successful start or an uncertain SDK start. Ordinary SDK
        # stop cannot establish descendant cleanup, so the sandbox stays owned.
        self.lifecycle_used = False
        self._lock = asyncio.Lock()

    def _owned(self, request: dict[str, Any]) -> OwnedRuntime:
        owned = self.runtimes.get(request["runtime_id"])
        if owned is None:
            raise ProtocolError("runtime_unavailable")
        if owned.agent != request["agent"]:
            raise ProtocolError("wrong_agent")
        return owned

    async def _stop(self, owned: OwnedRuntime) -> None:
        owned.state = "unknown"
        await owned.runtime.stop()
        owned.state = "stopped"

    async def handle(self, request: Any) -> dict[str, Any]:
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
            if self.shutting_down:
                raise ProtocolError("host_stopping")
            if operation == "start" and self.lifecycle_used:
                raise ProtocolError("runtime_exists", "start")
            async with self._lock:
                if self.shutting_down:
                    raise ProtocolError("host_stopping")
                if operation == "start":
                    if self.lifecycle_used:
                        raise ProtocolError("runtime_exists", "start")
                    stage = "validation"
                    config = FabricConfig.model_validate(request["config"])
                    changed, effects, stage = None, "unknown", "start"
                    # Set ownership before awaiting native work, including when
                    # the requesting connection closes or start is cancelled.
                    self.lifecycle_used = True
                    try:
                        runtime = await self.fabric.start_runtime(
                            config, base_dir=self.directory
                        )
                    except FabricConfigError:
                        self.lifecycle_used = False
                        changed, effects = False, "none"
                        raise
                    self.runtimes[runtime.runtime_id] = OwnedRuntime(
                        runtime, request["agent"], "unknown"
                    )
                    if runtime.status != RuntimeStatus.ACTIVE:
                        raise ProtocolError("start_failed", "start")
                    self.runtimes[runtime.runtime_id].state = "running"
                    return envelope(
                        operation, {"runtime_id": runtime.runtime_id}, changed=True
                    )
                owned = self._owned(request)
                result = {"runtime_id": request["runtime_id"]}
                if operation == "stop":
                    if owned.state == "stopped":
                        return envelope(operation, result)
                    changed, effects, stage = None, "unknown", "stop"
                    await self._stop(owned)
                    return envelope(operation, result, changed=True)
                stage = "invoke"
                if (
                    owned.state != "running"
                    or owned.runtime.status != RuntimeStatus.ACTIVE
                ):
                    raise ProtocolError("runtime_unavailable", stage)
                changed, effects = None, "unknown"
                native = (
                    await owned.runtime.invoke(input=request["input"])
                ).to_mapping()
                result = {"runtime_id": request["runtime_id"], "fabric_result": native}
                if native.get("status") != "succeeded":
                    raise ProtocolError("invoke_failed", stage)
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

    async def shutdown(self) -> None:
        """Stop an idle runtime. The enclosing sandbox owns descendant cleanup."""
        self.shutting_down = True
        if self._lock.locked():
            raise ProtocolError("cleanup_unconfirmed", "stop")
        async with self._lock:
            for owned in self.runtimes.values():
                if owned.state != "stopped":
                    await self._stop(owned)
