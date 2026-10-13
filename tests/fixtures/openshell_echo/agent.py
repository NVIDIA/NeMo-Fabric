# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal Fabric-ready adapter; it has no OpenShell dependency."""

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
        self.inputs.append(request.input)
        output = {"first_input": self.inputs[0], "turn": len(self.inputs)}
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
