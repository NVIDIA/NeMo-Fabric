# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Planning and requirements coverage for the standard Strands ACP harness."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from nemo_fabric import (
    DiscoveryConfig,
    Fabric,
    FabricConfig,
    FabricConfigError,
    HarnessConfig,
    InstructionConfig,
    InstructionsConfig,
    MetadataConfig,
    ModelConfig,
)

ROOT = Path(__file__).resolve().parents[2]
DESCRIPTOR = ROOT / "adapters/typescript/strands/strands.fabric-adapter.json"


@pytest.fixture(name="strands_config")
def strands_config_fixture():
    return FabricConfig(
        metadata=MetadataConfig(name="strands-test"),
        discovery=DiscoveryConfig(local_paths=[DESCRIPTOR]),
        harness=HarnessConfig(adapter_id="nvidia.fabric.strands"),
        models={
            "default": ModelConfig(
                provider="openai",
                model="test-model",
                base_url="http://localhost:8000/v1",
            )
        },
        instructions=InstructionsConfig(
            system=InstructionConfig(mode="append", content="Review carefully.")
        ),
    )


def test_strands_plans_models_instructions_and_mcp(strands_config):
    strands_config.add_mcp_server(
        "probe", transport="stdio", url="/usr/bin/node", args=["probe.mjs"]
    )
    plan = Fabric().plan(strands_config, base_dir=ROOT)
    projected = plan.to_mapping()["agent_config"]
    assert projected["models"]["default"]["base_url"] == "http://localhost:8000/v1"
    assert projected["instructions"]["system"] == {
        "mode": "append",
        "content": "Review carefully.",
    }
    assert projected["mcp"]["servers"]["probe"]["transport"] == "stdio"
    descriptor = plan.adapter_descriptor["descriptor"]
    assert descriptor == {**json.loads(DESCRIPTOR.read_text()), "telemetry": {}}
    assert descriptor["capabilities"] == {
        "metadata": {"health_checks": ["live"]},
        "streaming": False,
        "cancellation": False,
        "updates": False,
        "service": False,
    }


@pytest.mark.parametrize("provider", ["bedrock", "nvidia"])
def test_strands_plans_each_supported_provider(strands_config, provider):
    strands_config.models["default"] = ModelConfig(
        provider=provider,
        model="test-model",
        base_url="https://integrate.api.nvidia.com/v1"
        if provider == "nvidia"
        else None,
        api_key_env="NVIDIA_API_KEY" if provider == "nvidia" else None,
    )
    assert (
        Fabric()
        .plan(strands_config, base_dir=ROOT)
        .to_mapping()["agent_config"]["models"]["default"]["provider"]
        == provider
    )


@pytest.mark.parametrize(
    "settings",
    [{"unknown": True}, {"builtin_tools": ["subagent"]}, {"max_invocations": 0}],
)
def test_strands_rejects_unsupported_settings(strands_config, settings):
    strands_config.harness.settings = settings
    with pytest.raises(FabricConfigError):
        Fabric().plan(strands_config, base_dir=ROOT)


def test_strands_rejects_replacement_and_unsupported_model_fields(strands_config):
    strands_config.instructions.system.mode = "replace"
    with pytest.raises(FabricConfigError, match="replace"):
        Fabric().plan(strands_config, base_dir=ROOT)
    strands_config.instructions.system.mode = "append"
    strands_config.models["default"].temperature = 0.3
    with pytest.raises(FabricConfigError):
        Fabric().plan(strands_config, base_dir=ROOT)


async def test_strands_doctor_reports_missing_binaries(strands_config):
    os.environ["PATH"] = ""
    report = await Fabric().doctor(strands_config, base_dir=ROOT)
    assert report.status == "fail"
    failures = [
        c.message
        for c in report.checks
        if c.name == "requirement.binary" and c.status == "fail"
    ]
    assert any("node" in message for message in failures)
    assert any("strands" in message for message in failures)
