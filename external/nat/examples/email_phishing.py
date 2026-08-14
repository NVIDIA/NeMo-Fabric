# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a NeMo Agent Toolkit workflow with the installed email-phishing analyzer function."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from nemo_fabric import Fabric
from nemo_fabric import FabricConfig
from nemo_fabric import HarnessConfig
from nemo_fabric import InstructionConfig
from nemo_fabric import InstructionsConfig
from nemo_fabric import MetadataConfig
from nemo_fabric import ModelConfig
from nemo_fabric import RuntimeConfig
from nemo_fabric import ToolsConfig
from nemo_fabric import WorkflowConfig
from nemo_fabric import WorkflowEntrypointConfig


def build_config() -> FabricConfig:
    """Build the native NeMo Agent Toolkit email-phishing configuration."""

    tools = ToolsConfig()
    tools.add_definition(
        "email_phishing_analyzer",
        kind="function",
        ref="email_phishing_analyzer",
        settings={"llm": "default"},
    )
    tools.enabled = ["email_phishing_analyzer"]

    config = FabricConfig(
        metadata=MetadataConfig(
            name="nat-email-phishing-analyzer",
            description="Classifies an email with an installed NeMo Agent Toolkit function.",
        ),
        harness=HarnessConfig(
            adapter_id="nvidia.fabric.nat",
            resolution="preinstalled",
        ),
        workflow=WorkflowConfig(
            entrypoint=WorkflowEntrypointConfig(
                kind="factory",
                ref="fabric.agent.react",
            ),
            settings={
                "llm_name": "default",
                "use_native_tool_calling": True,
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
                content='State whether the email is "phishing" or "benign" and explain why.'
            )
        ),
        tools=tools,
        runtime=RuntimeConfig(input_schema="text", output_schema="message"),
    )
    return config


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=Path.cwd())
    parser.add_argument(
        "--input",
        default="Urgent: confirm your password at http://example.invalid today.",
    )
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
