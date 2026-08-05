# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a NAT ReAct workflow with a portable calculator MCP server."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from nemo_fabric import Fabric
from nemo_fabric import FabricConfig
from nemo_fabric import HarnessConfig
from nemo_fabric import InstructionConfig
from nemo_fabric import InstructionsConfig
from nemo_fabric import MetadataConfig
from nemo_fabric import ModelConfig
from nemo_fabric import RuntimeConfig


def build_config() -> FabricConfig:
    """Build the portable calculator configuration."""

    config = FabricConfig(
        metadata=MetadataConfig(
            name="nat-calculator",
            description="Uses calculator tools exposed by an MCP server.",
        ),
        harness=HarnessConfig(
            adapter_id="nvidia.fabric.nat",
            resolution="preinstalled",
            settings={
                "workflow": {
                    "_type": "react_agent",
                    "llm_name": "default",
                    "tool_names": [],
                }
            },
        ),
        models={
            "default": ModelConfig(
                provider="nvidia",
                model="nvidia/nemotron-3-nano-30b-a3b",
                api_key_env="NVIDIA_API_KEY",
                temperature=0.0,
            )
        },
        instructions=InstructionsConfig(
            system=InstructionConfig(
                content="Use the calculator tools for arithmetic. Return a concise answer."
            )
        ),
        runtime=RuntimeConfig(input_schema="text", output_schema="message"),
    )
    config.add_mcp_server(
        "calculator",
        transport="streamable-http",
        url=os.environ.get("CALCULATOR_MCP_URL", "http://127.0.0.1:9901/mcp"),
        exposure="harness_native",
        blocked_tools=["divide"],
    )
    return config


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=Path.cwd())
    parser.add_argument("--input", default="What is 21 multiplied by 2?")
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()

    fabric = Fabric()
    config = build_config()
    output = (
        fabric.plan(config, base_dir=args.base_dir)
        if args.plan
        else await fabric.run(config, base_dir=args.base_dir, input=args.input)
    )
    print(json.dumps(output.to_mapping(), indent=2))


if __name__ == "__main__":
    asyncio.run(main())
