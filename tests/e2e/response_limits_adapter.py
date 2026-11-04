# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Produce controlled complete outputs and binary artifacts through the real SDK."""

import hashlib
import os
from pathlib import Path

from nemo_fabric_adapter_contract.models import (
    AgentArtifact,
    AgentConfig,
    AgentRunResult,
    AgentRunStatus,
)
from nemo_fabric_adapters.common import lifecycle


class LimitRuntime:
    async def start(self, payload):
        pass

    async def invoke(self, request, context):
        artifacts = []
        digests = []
        for index, size in enumerate(request.context.get("artifact_sizes", [])):
            name = f"screenshot-{index}.png"
            path = Path(context.artifacts.root) / name
            content = os.urandom(size)
            path.write_bytes(content)
            digests.append(hashlib.sha256(content).hexdigest())
            artifacts.append(
                AgentArtifact(name=name, kind="file", path=name, media_type="image/png")
            )
        return AgentRunResult(
            status=AgentRunStatus.SUCCEEDED,
            output={
                "response": "x" * request.context.get("response_bytes", 16),
                "sha256": digests,
            },
            artifacts=artifacts,
        )

    async def stop(self):
        pass


if __name__ == "__main__":
    lifecycle.serve(LimitRuntime, config_loader=AgentConfig.from_mapping)
