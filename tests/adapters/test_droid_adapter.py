# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused planning and packaging tests for the Factory Droid SDK adapter."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from nemo_fabric import DiscoveryConfig
from nemo_fabric import Fabric
from nemo_fabric import FabricConfig
from nemo_fabric import FabricConfigError
from nemo_fabric import HarnessConfig
from nemo_fabric import InstructionConfig
from nemo_fabric import InstructionsConfig
from nemo_fabric import MetadataConfig
from nemo_fabric import ModelConfig
from nemo_fabric import RuntimeConfig
from nemo_fabric import ToolsConfig

ROOT = Path(__file__).resolve().parents[2]
ADAPTER_ROOT = ROOT / "adapters" / "typescript" / "droid"
DESCRIPTOR = ADAPTER_ROOT / "droid.fabric-adapter.json"


def config(*, settings=None, provider="factory", base_url=None) -> FabricConfig:
    return FabricConfig(
        metadata=MetadataConfig(name="droid-adapter-test"),
        harness=HarnessConfig(
            adapter_id="nvidia.fabric.droid",
            settings={} if settings is None else settings,
        ),
        discovery=DiscoveryConfig(local_paths=[DESCRIPTOR]),
        models={
            "default": ModelConfig(
                provider=provider,
                model="auto",
                api_key_env="FACTORY_API_KEY",
                base_url=base_url,
            )
        },
    )


def test_droid_descriptor_declares_only_the_supported_surface():
    descriptor = json.loads(DESCRIPTOR.read_text(encoding="utf-8"))

    assert descriptor["contract_version"] == "fabric.adapter/v1alpha2"
    assert descriptor["adapter_id"] == "nvidia.fabric.droid"
    assert descriptor["runner"] == {"command": "node", "script": "dist/cli.js"}
    assert descriptor["requirements"] == {"binaries": ["node", "droid"]}
    assert descriptor["config"] == {
        "accepts": [
            "models",
            "instructions.system",
            "tools.enabled",
            "tools.blocked",
            "mcp",
            "skills",
        ],
        "system_instruction_modes": ["replace", "append"],
    }
    assert descriptor["settings_schema"] == {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }
    assert descriptor["model_schema"]["properties"]["provider"] == {
        "const": "factory"
    }
    assert descriptor["model_schema"]["additionalProperties"] is False
    assert "telemetry" not in descriptor
    assert descriptor["capabilities"] == {
        "streaming": False,
        "cancellation": False,
        "updates": False,
        "service": False,
    }


def test_droid_descriptor_plans_all_advertised_normalized_mappings():
    adapter_config = config()
    adapter_config.instructions = InstructionsConfig(
        system=InstructionConfig(content="Keep the native prompt.", mode="append")
    )
    adapter_config.tools = ToolsConfig(enabled=["Read"], blocked=["Execute"])
    adapter_config.add_mcp_server(
        "local",
        transport="stdio",
        url="node",
        args=["server.js"],
        env={"MODE": "test"},
    )
    adapter_config.add_skill_path("examples/code_review_agent/skills/code-review")

    plan = Fabric().plan(adapter_config, base_dir=ROOT)
    agent = plan.to_mapping()["agent_config"]

    assert plan.adapter_descriptor["descriptor"]["adapter_id"] == "nvidia.fabric.droid"
    assert agent["models"]["default"] == {
        "provider": "factory",
        "model": "auto",
        "api_key_env": "FACTORY_API_KEY",
    }
    assert agent["instructions"] == {
        "system": {"content": "Keep the native prompt.", "mode": "append"}
    }
    assert agent["tools"] == {"enabled": ["Read"], "blocked": ["Execute"]}
    assert agent["mcp"]["servers"]["local"] == {
        "transport": "stdio",
        "url": "node",
        "args": ["server.js"],
        "env": {"MODE": "test"},
    }
    assert agent["skills"] == {
        "paths": [
            str((ROOT / "examples/code_review_agent/skills/code-review").resolve())
        ]
    }


@pytest.mark.parametrize(
    "adapter_config",
    [
        config(settings={"unknown": True}),
        config(provider="openai"),
        config(base_url="https://example.test/v1"),
    ],
)
def test_droid_descriptor_rejects_unknown_settings_and_unsupported_model_fields(
    adapter_config,
):
    with pytest.raises(FabricConfigError):
        Fabric().plan(adapter_config, base_dir=ROOT)


def test_droid_descriptor_rejects_unsupported_runtime_and_tool_definitions():
    limited = config()
    limited.runtime = RuntimeConfig(max_turns=2)
    with pytest.raises(FabricConfigError, match="runtime.max_turns"):
        Fabric().plan(limited, base_dir=ROOT)

    custom_tool = config()
    custom_tool.add_tool_definition("custom", kind="module", ref="tool.js")
    with pytest.raises(FabricConfigError, match="tools.definitions"):
        Fabric().plan(custom_tool, base_dir=ROOT)


async def test_droid_descriptor_doctor_reports_node_and_droid(tmp_path, monkeypatch):
    for name in ("node", "droid"):
        binary = tmp_path / (f"{name}.exe" if os.name == "nt" else name)
        binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        if os.name != "nt":
            binary.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))

    report = await Fabric().doctor(config(), base_dir=ROOT)

    assert report.status in {"pass", "warn"}
    checks = {
        check.message
        for check in report.checks
        if check.name == "requirement.binary" and check.status == "pass"
    }
    assert any("node" in message for message in checks)
    assert any("droid" in message for message in checks)


async def test_droid_descriptor_doctor_reports_missing_binaries(monkeypatch):
    monkeypatch.setenv("PATH", "")

    report = await Fabric().doctor(config(), base_dir=ROOT)

    assert report.status == "fail"
