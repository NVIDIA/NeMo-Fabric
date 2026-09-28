# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Credential-free adapter proving discovery, planning, and execution ownership."""

from nemo_fabric_adapter_contract.models import (
    AgentConfig,
    AgentRunResult,
    AgentRunStatus,
)
from nemo_fabric_adapters.common import lifecycle


class FixtureRuntime:
    async def start(self, payload):
        self.config = payload["config"]

    async def invoke(self, request, context):
        return AgentRunResult(
            status=AgentRunStatus.SUCCEEDED,
            output={
                "settings": self.config.harness.settings,
                "input": request.input,
            },
        )

    async def stop(self):
        pass


if __name__ == "__main__":
    lifecycle.serve(FixtureRuntime, config_loader=AgentConfig.from_mapping)
