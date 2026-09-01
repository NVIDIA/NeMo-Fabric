# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Descriptor and planning contract for the LangGraph custom-agent example."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from nemo_fabric import Fabric
from nemo_fabric import FabricConfig
from nemo_fabric import DiscoveryConfig
from nemo_fabric import HarnessConfig
from nemo_fabric import InstructionConfig
from nemo_fabric import InstructionsConfig
from nemo_fabric import MetadataConfig
from nemo_fabric import ModelConfig

from examples.langgraph_custom_agent.consumer.config import URL_INSPECTOR_SERVER
from examples.langgraph_custom_agent.consumer.config import public_config
from examples.langgraph_custom_agent.consumer.config import with_relay
from examples.langgraph_custom_agent.consumer.config import with_url_inspector_mcp

ROOT = Path(__file__).parents[3]
ADAPTER_ID = "nvidia.fabric.example.langgraph.email-phishing"
DESCRIPTOR = (
    ROOT
    / "examples"
    / "langgraph_custom_agent"
    / "adapter"
    / "email-phishing.fabric-adapter.json"
)


def test_descriptor_freezes_the_custom_agent_contract_surface():
    descriptor = json.loads(DESCRIPTOR.read_text(encoding="utf-8"))

    assert descriptor == {
        "contract_version": "fabric.adapter/v1alpha2",
        "adapter_id": ADAPTER_ID,
        "adapter_kind": "python",
        "runner": {"module": "examples.langgraph_custom_agent.adapter.runtime"},
        "requirements": {},
        "config": {
            "accepts": [
                "models",
                "models.base_url",
                "models.temperature",
                "instructions.system",
                "mcp",
                "mcp.tool_filters",
            ],
            "system_instruction_modes": ["replace", "append"],
        },
        "telemetry": {
            "providers": {
                "relay": {
                    "outputs": ["atif"],
                    "integration_modes": ["sdk"],
                }
            }
        },
        "capabilities": {
            "cancellation": False,
            "service": False,
            "streaming": False,
            "updates": False,
        },
    }


def test_plan_projects_only_the_advertised_agent_config(tmp_path: Path):
    config = FabricConfig(
        metadata=MetadataConfig(name="langgraph-email-phishing"),
        harness=HarnessConfig(
            adapter_id=ADAPTER_ID,
            resolution="preinstalled",
        ),
        discovery=DiscoveryConfig(local_paths=[DESCRIPTOR.parent]),
        models={
            "default": ModelConfig(
                provider="nvidia",
                model="nvidia/test-model",
                api_key_env="NVIDIA_API_KEY",
                base_url="https://integrate.api.nvidia.com/v1",
                temperature=0.2,
            )
        },
        instructions=InstructionsConfig(
            system=InstructionConfig(content="Explain the email risk assessment.")
        ),
    )

    plan = Fabric().plan(config, base_dir=tmp_path)

    assert plan.adapter.adapter_id == ADAPTER_ID
    assert plan.to_mapping()["agent_config"] == {
        "models": {
            "default": {
                "provider": "nvidia",
                "model": "nvidia/test-model",
                "api_key_env": "NVIDIA_API_KEY",
                "temperature": 0.2,
                "base_url": "https://integrate.api.nvidia.com/v1",
            }
        },
        "instructions": {
            "system": {
                "content": "Explain the email risk assessment.",
                "mode": "replace",
            }
        },
    }


def test_plan_accepts_only_the_verified_relay_output(tmp_path: Path):
    plan = Fabric().plan(with_relay(public_config()), base_dir=tmp_path).to_mapping()

    assert plan["telemetry_plan"]["relay_enabled"] is True
    assert plan["telemetry_plan"]["providers"] == ["relay"]
    assert plan["telemetry_plan"]["adapter_outputs"] == ["atif"]


def test_plan_projects_optional_stdio_mcp_to_agent_config(tmp_path: Path):
    plan = (
        Fabric()
        .plan(
            with_url_inspector_mcp(public_config()),
            base_dir=tmp_path,
        )
        .to_mapping()
    )

    assert plan["agent_config"]["mcp"] == {
        "servers": {
            "url-inspector": {
                "transport": "stdio",
                "url": os.environ.get("ADAPTER_PYTHON", sys.executable),
                "args": [str(URL_INSPECTOR_SERVER)],
                "allowed_tools": ["inspect_url"],
            }
        }
    }
