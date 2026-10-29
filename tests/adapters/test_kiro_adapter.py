# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused planning tests for the Kiro ACP adapter."""

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
from nemo_fabric import ToolsConfig


ROOT = Path(__file__).resolve().parents[2]
DESCRIPTOR = ROOT / "adapters" / "typescript" / "kiro" / "kiro.fabric-adapter.json"


def config(*, settings=None, base_url=None, provider="kiro") -> FabricConfig:
    return FabricConfig(
        metadata=MetadataConfig(name="kiro-adapter-test"),
        discovery=DiscoveryConfig(local_paths=[DESCRIPTOR]),
        harness=HarnessConfig(
            adapter_id="nvidia.fabric.kiro",
            settings={} if settings is None else settings,
        ),
        models={
            "default": ModelConfig(
                provider=provider,
                model="claude-sonnet",
                api_key_env="KIRO_API_KEY",
                base_url=base_url,
            )
        },
    )


def test_kiro_descriptor_declares_only_the_supported_surface():
    descriptor = json.loads(DESCRIPTOR.read_text(encoding="utf-8"))

    assert descriptor["contract_version"] == "fabric.adapter/v1alpha2"
    assert descriptor["adapter_id"] == "nvidia.fabric.kiro"
    assert descriptor["adapter_kind"] == "process"
    assert descriptor["runner"] == {"command": "node", "script": "dist/cli.js"}
    assert descriptor["requirements"] == {"binaries": ["node", "kiro-cli"]}
    assert descriptor["config"] == {"accepts": ["models", "models.base_url", "mcp"]}
    assert descriptor["model_schema"]["properties"]["provider"] == {
        "enum": ["kiro", "openai"]
    }
    assert descriptor["model_schema"]["properties"]["api_key_env"] == {
        "type": "string",
        "pattern": "^[A-Za-z_][A-Za-z0-9_]*$",
    }
    native, byom = descriptor["model_schema"]["oneOf"]
    assert native["properties"] == {
        "provider": {"const": "kiro"},
        "api_key_env": {"const": "KIRO_API_KEY"},
        "base_url": False,
    }
    assert byom == {
        "properties": {"provider": {"const": "openai"}},
        "required": ["base_url"],
    }
    assert (
        descriptor["settings_schema"]["properties"]["experimental_byom"]["default"]
        is False
    )
    assert (
        descriptor["settings_schema"]["properties"]["experimental_byom"]["type"]
        == "boolean"
    )
    assert descriptor["model_schema"]["additionalProperties"] is False
    assert descriptor["settings_schema"]["additionalProperties"] is False
    assert descriptor["capabilities"] == {
        "metadata": {"health_checks": ["live"]},
        "streaming": False,
        "cancellation": False,
        "updates": False,
        "service": False,
    }


def test_kiro_descriptor_plans_model_settings_and_mcp():
    adapter_config = config(
        settings={"mode_id": "plan", "effort": "high", "permission_mode": "deny"}
    )
    adapter_config.add_mcp_server(
        "local",
        transport="stdio",
        url="/usr/bin/node",
        args=["server.mjs"],
        env={"MODE": "test"},
    )

    plan = Fabric().plan(adapter_config, base_dir=ROOT)
    agent = plan.to_mapping()["agent_config"]

    assert plan.adapter_descriptor["descriptor"]["adapter_id"] == "nvidia.fabric.kiro"
    assert agent["models"]["default"] == {
        "provider": "kiro",
        "model": "claude-sonnet",
        "api_key_env": "KIRO_API_KEY",
    }
    assert agent["harness"]["settings"] == {
        "mode_id": "plan",
        "effort": "high",
        "permission_mode": "deny",
    }
    assert agent["mcp"]["servers"]["local"] == {
        "transport": "stdio",
        "url": "/usr/bin/node",
        "args": ["server.mjs"],
        "env": {"MODE": "test"},
    }


@pytest.mark.parametrize(
    "adapter_config",
    [
        config(settings={"unknown": True}),
        config(base_url="https://example.test/v1"),
        config(settings={"experimental_byom": True}, provider="openai"),
        config(settings={"experimental_byom": "true"}),
    ],
)
def test_kiro_descriptor_rejects_unknown_settings_and_model_endpoint(adapter_config):
    with pytest.raises(FabricConfigError):
        Fabric().plan(adapter_config, base_dir=ROOT)


def test_kiro_descriptor_rejects_unadvertised_normalized_fields():
    instructed = config()
    instructed.instructions = InstructionsConfig(
        system=InstructionConfig(content="unsupported")
    )
    with pytest.raises(FabricConfigError, match="instructions.system"):
        Fabric().plan(instructed, base_dir=ROOT)

    tools = config()
    tools.tools = ToolsConfig(enabled=["read"])
    with pytest.raises(FabricConfigError, match="tools.enabled"):
        Fabric().plan(tools, base_dir=ROOT)

    skills = config()
    skills.add_skill_path("tests/fixtures/default")
    with pytest.raises(FabricConfigError, match="skills"):
        Fabric().plan(skills, base_dir=ROOT)


async def test_kiro_descriptor_doctor_reports_node_and_kiro_cli(tmp_path, monkeypatch):
    for name in ("node", "kiro-cli"):
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
    assert any("kiro-cli" in message for message in checks)


async def test_kiro_descriptor_doctor_reports_missing_binaries(monkeypatch):
    monkeypatch.setenv("PATH", "")

    report = await Fabric().doctor(config(), base_dir=ROOT)

    assert report.status == "fail"
    checks = {
        check.message
        for check in report.checks
        if check.name == "requirement.binary" and check.status == "fail"
    }
    assert any("node" in message for message in checks)
    assert any("kiro-cli" in message for message in checks)


def test_kiro_descriptor_plans_explicit_experimental_openai_bridge():
    adapter_config = config(
        settings={"experimental_byom": True, "permission_mode": "deny"},
        provider="openai",
        base_url="https://example.test/v1",
    )
    adapter_config.models["default"].api_key_env = "LOCAL_MODEL_KEY"

    plan = Fabric().plan(adapter_config, base_dir=ROOT)
    agent = plan.to_mapping()["agent_config"]

    assert agent["models"]["default"] == {
        "provider": "openai",
        "model": "claude-sonnet",
        "api_key_env": "LOCAL_MODEL_KEY",
        "base_url": "https://example.test/v1",
    }
    assert agent["harness"]["settings"] == {
        "experimental_byom": True,
        "permission_mode": "deny",
    }
    # The descriptor validates each block independently. The direct adapter
    # enforces the cross-block requirement for the explicit BYOM opt-in.
