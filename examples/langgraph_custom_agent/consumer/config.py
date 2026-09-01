# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Consumer-owned FabricConfig variants for the custom agent."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Literal

from nemo_fabric import FabricConfig
from nemo_fabric import DiscoveryConfig
from nemo_fabric import HarnessConfig
from nemo_fabric import InstructionConfig
from nemo_fabric import InstructionsConfig
from nemo_fabric import MetadataConfig
from nemo_fabric import ModelConfig
from nemo_fabric import RelayAtifConfig
from nemo_fabric import RelayAtofConfig
from nemo_fabric import RelayAtofFileSinkConfig
from nemo_fabric import RelayObservabilityConfig
from nemo_fabric import RuntimeConfig

ADAPTER_ID = "nvidia.fabric.example.langgraph.email-phishing"
PUBLIC_DEFAULT_MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
FRONTIER_DEFAULT_MODEL = "nvidia/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
PUBLIC_BASE_URL = "https://integrate.api.nvidia.com/v1"
URL_INSPECTOR_SERVER = (
    Path(__file__).parents[1] / "mcp" / "url_inspector.py"
).resolve()
DESCRIPTOR_DIRECTORY = (Path(__file__).parents[1] / "adapter").resolve()


def _config(*, model: str, api_key_env: str, base_url: str) -> FabricConfig:
    return FabricConfig(
        metadata=MetadataConfig(
            name="langgraph-email-phishing",
            description="Classifies email risk with a dedicated LangGraph agent.",
        ),
        discovery=DiscoveryConfig(local_paths=[DESCRIPTOR_DIRECTORY]),
        harness=HarnessConfig(
            adapter_id=ADAPTER_ID,
            resolution="preinstalled",
        ),
        models={
            "default": ModelConfig(
                provider="nvidia",
                model=model,
                api_key_env=api_key_env,
                base_url=base_url,
                temperature=0.0,
            )
        },
        instructions=InstructionsConfig(
            system=InstructionConfig(
                content=(
                    "Explain why the fixed classification follows from the detected "
                    "signals. Do not change the classification."
                )
            )
        ),
        runtime=RuntimeConfig(input_schema="text", output_schema="message"),
    )


def public_config(model: str = PUBLIC_DEFAULT_MODEL) -> FabricConfig:
    """Use the public NVIDIA API Catalog endpoint."""

    return _config(
        model=model,
        api_key_env="NVIDIA_API_KEY",
        base_url=PUBLIC_BASE_URL,
    )


def frontier_config(model: str = FRONTIER_DEFAULT_MODEL) -> FabricConfig:
    """Use an internal NVIDIA Frontier OpenAI-compatible endpoint."""

    base_url = os.environ.get("NVIDIA_FRONTIER_BASE_URL")
    if not base_url:
        raise RuntimeError("NVIDIA_FRONTIER_BASE_URL is required for frontier testing")
    return _config(
        model=model,
        api_key_env="NVIDIA_FRONTIER_API_KEY",
        base_url=base_url,
    )


def with_system_instruction(
    base: FabricConfig,
    content: str,
    *,
    mode: Literal["replace", "append"] = "replace",
) -> FabricConfig:
    """Return an independent config with a different normalized instruction."""

    config = base.model_copy(deep=True)
    config.instructions = InstructionsConfig(
        system=InstructionConfig(content=content, mode=mode)
    )
    return config


def with_temperature(base: FabricConfig, temperature: float) -> FabricConfig:
    """Return an independent config with a different model temperature."""

    config = base.model_copy(deep=True)
    config.models["default"].temperature = temperature
    return config


def with_url_inspector_mcp(base: FabricConfig) -> FabricConfig:
    """Return an independent config with the example's stdio MCP server."""

    config = base.model_copy(deep=True)
    config.add_mcp_server(
        "url-inspector",
        transport="stdio",
        url=os.environ.get("ADAPTER_PYTHON", sys.executable),
        args=[str(URL_INSPECTOR_SERVER)],
        exposure="harness_native",
        allowed_tools=["inspect_url"],
    )
    return config


def with_relay(base: FabricConfig) -> FabricConfig:
    """Return an independent config with Relay ATOF and ATIF enabled."""

    config = base.model_copy(deep=True)
    config.runtime.artifacts = "./artifacts"
    config.enable_relay(
        output_dir="./artifacts/relay",
        observability=RelayObservabilityConfig(
            atof=RelayAtofConfig(
                enabled=True,
                sinks=[
                    RelayAtofFileSinkConfig(
                        output_directory="./artifacts/relay",
                        filename="events.atof.jsonl",
                        mode="append",
                    )
                ],
            ),
            atif=RelayAtifConfig(
                enabled=True,
                output_directory="./artifacts/relay",
                filename_template="trajectory-{session_id}.atif.json",
                agent_name="langgraph-email-phishing",
                agent_version="fabric-sdk-example",
            ),
        ),
    )
    return config
