# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a stateful echo agent through Fabric's ordinary adapter lifecycle.

The adapter keeps input history and declares a receipt on the second turn.
It contains no OpenShell integration; the environment provider supplies that.
"""

import json
from pathlib import Path

from nemo_fabric_adapter_contract.models import (
    AgentArtifact,
    AgentConfig,
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
            receipt = Path(context.artifacts.root) / "echo.json"
            receipt.write_text(json.dumps(output))
            artifacts = [
                AgentArtifact(
                    name="echo",
                    kind="receipt",
                    path="echo.json",
                    media_type="application/json",
                )
            ]
        return AgentRunResult(
            status=AgentRunStatus.SUCCEEDED, output=output, artifacts=artifacts
        )

    async def stop(self):
        self.inputs = []


if __name__ == "__main__":
    lifecycle.serve(EchoRuntime, config_loader=AgentConfig.from_mapping)
