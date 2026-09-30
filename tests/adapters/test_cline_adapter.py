# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused planning and packaging tests for the Cline SDK adapter."""

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
ADAPTER_ROOT = ROOT / "adapters" / "typescript" / "cline"
DESCRIPTOR = ADAPTER_ROOT / "cline.fabric-adapter.json"


def config(*, settings=None, temperature=None) -> FabricConfig:
    return FabricConfig(
        metadata=MetadataConfig(name="cline-adapter-test"),
        harness=HarnessConfig(
            adapter_id="nvidia.fabric.cline",
            settings={} if settings is None else settings,
        ),
        discovery=DiscoveryConfig(local_paths=[DESCRIPTOR]),
        models={
            "default": ModelConfig(
                provider="nvidia",
                model="nvidia/test-model",
                api_key_env="NVIDIA_API_KEY",
                base_url="https://integrate.api.nvidia.com/v1",
                temperature=temperature,
            )
        },
    )


def test_cline_descriptor_declares_only_the_supported_surface():
    descriptor = json.loads(DESCRIPTOR.read_text(encoding="utf-8"))

    assert descriptor["contract_version"] == "fabric.adapter/v1alpha2"
    assert descriptor["adapter_id"] == "nvidia.fabric.cline"
    assert descriptor["adapter_kind"] == "process"
    assert descriptor["runner"] == {"command": "node", "script": "dist/cli.js"}
    assert descriptor["requirements"] == {"binaries": ["node"]}
    assert descriptor["config"] == {
        "accepts": [
            "models",
            "models.base_url",
            "instructions.system",
            "tools.enabled",
            "tools.blocked",
            "mcp",
            "skills",
        ],
        "system_instruction_modes": ["replace"],
    }
    assert descriptor["settings_schema"] == {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }
    assert descriptor["model_schema"]["required"] == [
        "provider",
        "model",
        "api_key_env",
    ]
    assert descriptor["model_schema"]["additionalProperties"] is False
    assert (
        descriptor["extension_schemas"]["run_result"]["additionalProperties"] is False
    )
    assert descriptor["extension_schemas"]["usage"]["additionalProperties"] is False
    assert "telemetry" not in descriptor
    assert descriptor["capabilities"] == {
        "streaming": False,
        "cancellation": False,
        "updates": False,
        "service": False,
    }


def test_cline_descriptor_plans_all_advertised_normalized_mappings():
    adapter_config = config()
    adapter_config.instructions = InstructionsConfig(
        system=InstructionConfig(
            content="Use the configured instructions.", mode="replace"
        )
    )
    adapter_config.tools = ToolsConfig(
        enabled=["read_files", "skills"], blocked=["run_commands"]
    )
    adapter_config.add_skill_path("tests/fixtures/default")
    adapter_config.add_mcp_server(
        "local",
        transport="stdio",
        url="node",
        args=["server.js"],
        env={"MODE": "test"},
    )

    plan = Fabric().plan(adapter_config, base_dir=ROOT)
    agent = plan.to_mapping()["agent_config"]

    assert plan.adapter_descriptor["descriptor"]["adapter_id"] == "nvidia.fabric.cline"
    assert agent["models"]["default"] == {
        "provider": "nvidia",
        "model": "nvidia/test-model",
        "api_key_env": "NVIDIA_API_KEY",
        "base_url": "https://integrate.api.nvidia.com/v1",
    }
    assert agent["instructions"] == {
        "system": {"content": "Use the configured instructions.", "mode": "replace"}
    }
    assert agent["tools"] == {
        "enabled": ["read_files", "skills"],
        "blocked": ["run_commands"],
    }
    assert agent["skills"] == {
        "paths": [str((ROOT / "tests/fixtures/default").resolve())]
    }
    assert agent["mcp"]["servers"]["local"] == {
        "transport": "stdio",
        "url": "node",
        "args": ["server.js"],
        "env": {"MODE": "test"},
    }


@pytest.mark.parametrize(
    "adapter_config",
    [
        config(settings={"unknown": True}),
        config(temperature=0.2),
    ],
)
def test_cline_descriptor_rejects_unknown_settings_and_unsupported_model_fields(
    adapter_config,
):
    with pytest.raises(FabricConfigError):
        Fabric().plan(adapter_config, base_dir=ROOT)


def test_cline_descriptor_rejects_unsupported_instruction_runtime_and_tool_fields():
    appended = config()
    appended.instructions = InstructionsConfig(
        system=InstructionConfig(content="append", mode="append")
    )
    with pytest.raises(FabricConfigError, match="instructions.system.mode"):
        Fabric().plan(appended, base_dir=ROOT)

    limited = config()
    limited.runtime = RuntimeConfig(max_turns=2)
    with pytest.raises(FabricConfigError, match="runtime.max_turns"):
        Fabric().plan(limited, base_dir=ROOT)

    custom_tool = config()
    custom_tool.add_tool_definition("custom", kind="module", ref="tool.js")
    with pytest.raises(FabricConfigError, match="tools.definitions"):
        Fabric().plan(custom_tool, base_dir=ROOT)


async def test_cline_descriptor_doctor_reports_node(tmp_path, monkeypatch):
    node = tmp_path / ("node.exe" if os.name == "nt" else "node")
    node.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    if os.name != "nt":
        node.chmod(0o755)
    monkeypatch.setenv(
        "PATH", f"{tmp_path}{os.pathsep}{os.environ.get('PATH', '')}"
    )

    report = await Fabric().doctor(config(), base_dir=ROOT)

    assert report.status in {"pass", "warn"}
    assert any(
        check.name == "requirement.binary"
        and "node" in check.message
        and check.status == "pass"
        for check in report.checks
    )


async def test_cline_descriptor_doctor_reports_missing_node(monkeypatch):
    monkeypatch.setenv("PATH", "")

    report = await Fabric().doctor(config(), base_dir=ROOT)

    assert report.status == "fail"
    assert any(
        check.name == "requirement.binary"
        and "node" in check.message
        and check.status == "fail"
        for check in report.checks
    )
