# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused planning tests for the OpenCode v2 adapter."""

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
from nemo_fabric import McpConfig
from nemo_fabric import McpServerConfig
from nemo_fabric import ModelConfig
from nemo_fabric import SkillConfig

ROOT = Path(__file__).resolve().parents[2]
DESCRIPTOR = ROOT / "adapters/typescript/opencode/opencode.fabric-adapter.json"


def config(
    *,
    api_key_env: str | None = "TEST_API_KEY",
    base_url: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
    system_instruction: InstructionConfig | None = None,
    skill_paths: list[str] | None = None,
    mcp_server: McpServerConfig | None = None,
) -> FabricConfig:
    return FabricConfig(
        metadata=MetadataConfig(name="opencode-adapter-test"),
        harness=HarnessConfig(adapter_id="nvidia.fabric.opencode"),
        discovery=DiscoveryConfig(local_paths=[DESCRIPTOR]),
        models={
            "default": ModelConfig(
                provider="nvidia",
                model="nvidia/nemotron-3.5-lightning-30b-a3b",
                api_key_env=api_key_env,
                base_url=base_url,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
        },
        instructions=(InstructionsConfig(system=system_instruction) if system_instruction else None),
        skills=SkillConfig(paths=skill_paths) if skill_paths else None,
        mcp=McpConfig(servers={"test": mcp_server}) if mcp_server else None,
    )


def test_opencode_descriptor_declares_the_minimum_supported_surface():
    descriptor = json.loads(DESCRIPTOR.read_text(encoding="utf-8"))

    assert descriptor == {
        "contract_version": "fabric.adapter/v1alpha2",
        "adapter_id": "nvidia.fabric.opencode",
        "adapter_kind": "process",
        "runner": {"command": "bun", "script": "dist/cli.js"},
        "requirements": {"binaries": ["bun"]},
        "config": {
            "accepts": [
                "models",
                "models.base_url",
                "models.temperature",
                "models.top_p",
                "instructions.system",
                "skills",
                "mcp",
            ],
            "system_instruction_modes": ["replace"],
        },
        "model_schema": {
            "type": "object",
            "properties": {
                "provider": {"type": "string", "minLength": 1},
                "model": {"type": "string", "minLength": 1},
                "api_key_env": {
                    "type": "string",
                    "minLength": 1,
                    "pattern": "^[A-Za-z_][A-Za-z0-9_]*$",
                },
                "base_url": {
                    "type": ["string", "null"],
                    "format": "uri",
                    "pattern": "^https?://",
                },
                "temperature": {"type": "number"},
                "top_p": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": ["provider", "model", "api_key_env"],
            "additionalProperties": False,
            "allOf": [
                {
                    "if": {
                        "anyOf": [
                            {"required": ["temperature"]},
                            {"required": ["top_p"]},
                        ]
                    },
                    "then": {
                        "properties": {
                            "base_url": {
                                "type": "string",
                                "format": "uri",
                                "pattern": "^https?://",
                            }
                        },
                        "required": ["base_url"],
                    },
                }
            ],
        },
        "capabilities": {
            "streaming": False,
            "cancellation": False,
            "updates": False,
            "service": False,
        },
    }


def test_opencode_descriptor_plans_and_projects_the_selected_model():
    plan = Fabric().plan(config(), base_dir=ROOT)

    assert plan.adapter_descriptor["descriptor"]["adapter_id"] == "nvidia.fabric.opencode"
    assert plan.agent_config == {
        "models": {
            "default": {
                "provider": "nvidia",
                "model": "nvidia/nemotron-3.5-lightning-30b-a3b",
                "api_key_env": "TEST_API_KEY",
            }
        }
    }


def test_opencode_model_schema_requires_a_credential_name():
    with pytest.raises(FabricConfigError, match="api_key_env"):
        Fabric().plan(config(api_key_env=None), base_dir=ROOT)


def test_opencode_model_schema_requires_a_portable_credential_name():
    with pytest.raises(FabricConfigError, match="api_key_env"):
        Fabric().plan(config(api_key_env="MODEL-API-KEY"), base_dir=ROOT)


def test_opencode_descriptor_plans_an_openai_compatible_model_endpoint():
    plan = Fabric().plan(config(base_url="https://example.test/v1"), base_dir=ROOT)

    assert plan.agent_config["models"]["default"]["base_url"] == "https://example.test/v1"


def test_opencode_descriptor_projects_sampling_for_an_openai_compatible_model_endpoint():
    plan = Fabric().plan(
        config(base_url="https://example.test/v1", temperature=0.25, top_p=0.8),
        base_dir=ROOT,
    )

    assert plan.agent_config["models"]["default"] == {
        "provider": "nvidia",
        "model": "nvidia/nemotron-3.5-lightning-30b-a3b",
        "api_key_env": "TEST_API_KEY",
        "base_url": "https://example.test/v1",
        "temperature": 0.25,
        "top_p": 0.8,
    }


@pytest.mark.parametrize("field, value", [("temperature", 0.25), ("top_p", 0.8)])
def test_opencode_descriptor_rejects_sampling_without_an_openai_compatible_endpoint(
    field: str, value: float
):
    with pytest.raises(FabricConfigError, match="base_url"):
        Fabric().plan(config(**{field: value}), base_dir=ROOT)


def test_opencode_descriptor_rejects_unsupported_max_tokens():
    with pytest.raises(FabricConfigError, match="max_tokens"):
        Fabric().plan(config(max_tokens=128), base_dir=ROOT)


def test_opencode_descriptor_projects_replace_system_instruction():
    plan = Fabric().plan(
        config(system_instruction=InstructionConfig(content="Fabric system instruction", mode="replace")),
        base_dir=ROOT,
    )

    assert plan.agent_config["instructions"] == {
        "system": {"content": "Fabric system instruction", "mode": "replace"}
    }


def test_opencode_descriptor_rejects_append_system_instruction():
    with pytest.raises(FabricConfigError, match="append"):
        Fabric().plan(
            config(system_instruction=InstructionConfig(content="Append this", mode="append")),
            base_dir=ROOT,
        )


def test_opencode_descriptor_projects_configured_skill_paths():
    plan = Fabric().plan(config(skill_paths=["skills/review"]), base_dir=ROOT)

    assert plan.agent_config["skills"] == {"paths": [str(ROOT / "skills" / "review")]}


def test_opencode_descriptor_projects_a_native_streamable_http_mcp_server():
    plan = Fabric().plan(
        config(
            mcp_server=McpServerConfig(
                transport="streamable-http",
                url="https://mcp.example.test",
                custom_headers={"X-Fabric-MCP": "configured"},
            )
        ),
        base_dir=ROOT,
    )

    assert plan.agent_config["mcp"] == {
        "servers": {
            "test": {
                "transport": "streamable-http",
                "url": "https://mcp.example.test",
                "custom_headers": {"X-Fabric-MCP": "configured"},
            }
        }
    }


def test_opencode_descriptor_rejects_mcp_tool_filters():
    with pytest.raises(FabricConfigError, match="allowed_tools"):
        Fabric().plan(
            config(
                mcp_server=McpServerConfig(
                    transport="streamable-http",
                    url="https://mcp.example.test",
                    allowed_tools=["search"],
                )
            ),
            base_dir=ROOT,
        )


def test_opencode_descriptor_rejects_a_non_http_model_endpoint():
    with pytest.raises(FabricConfigError, match="base_url"):
        Fabric().plan(config(base_url="ftp://example.test/v1"), base_dir=ROOT)


@pytest.mark.anyio
async def test_opencode_descriptor_doctor_requires_bun(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    bun = tmp_path / ("bun.exe" if os.name == "nt" else "bun")
    if os.name == "nt":
        bun.write_bytes(b"")
    else:
        bun.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        bun.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ.get('PATH', '')}")

    report = await Fabric().doctor(config(), base_dir=ROOT)

    assert report.status in {"pass", "warn"}
    assert any(
        check.name == "requirement.binary" and "bun" in check.message and check.status == "pass"
        for check in report.checks
    )


@pytest.mark.anyio
async def test_opencode_descriptor_doctor_reports_a_missing_bun(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("PATH", "")

    report = await Fabric().doctor(config(), base_dir=ROOT)

    assert report.status == "fail"
    assert any(
        check.name == "requirement.binary" and "bun" in check.message and check.status == "fail"
        for check in report.checks
    )
