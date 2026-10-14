# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal Fabric-ready adapter; it has no OpenShell dependency."""

import asyncio
import hashlib
import json
from pathlib import Path

from nemo_fabric_adapter_contract.models import (
    AgentArtifact,
    AgentConfig,
    AgentRunError,
    AgentRunResult,
    AgentRunStatus,
    RuntimeContext,
)
from nemo_fabric_adapters.common import lifecycle


class EchoRuntime:
    async def start(self, payload):
        if not isinstance(payload["config"], AgentConfig):
            raise ValueError("Echo requires a validated AgentConfig")
        self.context = RuntimeContext.from_mapping(payload["runtime_context"])
        self.inputs = []

    async def invoke(self, request, context):
        if context.runtime_id != self.context.runtime_id:
            raise ValueError("invocation must match the active runtime")
        barrier_path = request.context.get("parallel_barrier")
        both_runtimes_entered = None
        if barrier_path:
            barrier = Path(barrier_path)
            barrier.parent.mkdir(parents=True, exist_ok=True)
            runtime_key = hashlib.sha256(context.runtime_id.encode()).hexdigest()
            barrier.with_name(f"{barrier.name}.{runtime_key}.ready").touch()
            deadline = asyncio.get_running_loop().time() + 10
            while len(list(barrier.parent.glob(f"{barrier.name}.*.ready"))) < 2:
                if asyncio.get_running_loop().time() >= deadline:
                    break
                await asyncio.sleep(0.05)
            both_runtimes_entered = (
                len(list(barrier.parent.glob(f"{barrier.name}.*.ready"))) >= 2
            )
        self.inputs.append(request.input)
        output = {"first_input": self.inputs[0], "turn": len(self.inputs)}
        if both_runtimes_entered is not None:
            output["both_runtimes_entered"] = both_runtimes_entered
        artifacts = []
        if len(self.inputs) == 2:
            path = Path(context.artifacts.root) / "echo.json"
            path.write_text(json.dumps(output))
            artifacts = [
                AgentArtifact(
                    name="echo",
                    kind="receipt",
                    path="echo.json",
                    media_type="application/json",
                )
            ]
        if request.context.get("large_internal_artifact"):
            internal = Path(context.artifacts.root) / ".fabric" / "stdout.txt"
            internal.parent.mkdir(parents=True, exist_ok=True)
            internal.write_bytes(b"x" * (256 * 1024 + 1))
            artifacts.append(
                AgentArtifact(
                    name="internal-sdk-output",
                    kind="receipt",
                    path=".fabric/stdout.txt",
                    media_type="text/plain",
                )
            )
        if request.context.get("fail"):
            return AgentRunResult(
                status=AgentRunStatus.FAILED,
                output=request.context.get("partial_output"),
                error=AgentRunError(
                    code="echo_failed", message="Controlled fixture failure"
                ),
                artifacts=artifacts,
            )
        return AgentRunResult(
            status=AgentRunStatus.SUCCEEDED, output=output, artifacts=artifacts
        )

    async def stop(self):
        self.inputs = []


if __name__ == "__main__":
    lifecycle.serve(EchoRuntime, config_loader=AgentConfig.from_mapping)
