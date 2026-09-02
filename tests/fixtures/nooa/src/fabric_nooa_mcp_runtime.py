# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run one deterministic MCP-backed NOOA adapter invocation in a subprocess."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from nemo_fabric_adapter_contract.models import AgentConfig
from nemo_fabric_adapter_contract.models import AgentRunRequest
from nemo_fabric_adapter_contract.models import RuntimeContext
from nemo_fabric_adapters.nooa.adapter import NooaRuntime


async def run(workspace: Path) -> dict[str, Any]:
    runtime_context = RuntimeContext.from_mapping(
        {
            "runtime_id": "runtime-subprocess",
            "invocation_id": "invocation-subprocess",
            "request_id": "request-subprocess",
            "environment": {
                "environment_id": "environment-subprocess",
                "provider": "local",
                "control_location": "in_env_control",
                "ownership": "caller_owned",
            },
            "artifacts": {},
        }
    )
    config = AgentConfig.from_mapping(
        {
            "mcp": {
                "servers": {
                    "calculator": {
                        "transport": "stdio",
                        "url": sys.executable,
                        "args": ["fixture-mcp-server"],
                    }
                }
            },
            "workflow": {
                "entrypoint": {
                    "kind": "interactive_agent_factory",
                    "ref": "fabric_nooa_test_target:create_agent",
                },
                "settings": {},
            },
        }
    )
    runtime = NooaRuntime()
    await runtime.start(
        {
            "agent_name": "interactive-subprocess",
            "base_dir": str(workspace),
            "config": config,
            "runtime_context": {
                "runtime_id": runtime_context.runtime_id,
                "environment": {
                    "workspace": str(workspace),
                    "artifacts": str(workspace / "artifacts"),
                },
            },
        }
    )
    try:
        result = await runtime.invoke(
            AgentRunRequest.from_mapping({"input": "hello MCP"}),
            runtime_context,
        )
        return result.to_mapping()
    finally:
        await runtime.stop()


if __name__ == "__main__":
    print(json.dumps(asyncio.run(run(Path(sys.argv[1])))))
