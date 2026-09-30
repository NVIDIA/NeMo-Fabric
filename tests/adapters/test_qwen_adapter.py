# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Planning and local-runtime tests for the direct Qwen Code adapter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from nemo_fabric import DiscoveryConfig
from nemo_fabric import EnvironmentConfig
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
from nemo_fabric import RuntimeConfig
from nemo_fabric import SkillConfig
from nemo_fabric import ToolsConfig
from examples.code_review_agent.config import qwen_config as review_qwen_config

ROOT = Path(__file__).resolve().parents[2]
DESCRIPTOR = ROOT / "adapters/typescript/qwen/qwen.fabric-adapter.json"


def qwen_config(
    workspace: Path,
    *,
    base_url: str | None = None,
    provider: str = "openai",
    system_mode: str = "append",
) -> FabricConfig:
    return FabricConfig(
        metadata=MetadataConfig(name="qwen-adapter-test"),
        discovery=DiscoveryConfig(local_paths=[DESCRIPTOR]),
        harness=HarnessConfig(
            adapter_id="nvidia.fabric.qwen",
            settings={"permission_mode": "default"},
        ),
        models={
            "default": ModelConfig(
                provider=provider,
                model="fabric-test-model",
                api_key_env="QWEN_E2E_KEY",
                base_url=base_url,
            )
        },
        instructions=InstructionsConfig(
            system=InstructionConfig(content="Answer concisely.", mode=system_mode)
        ),
        environment=EnvironmentConfig(
            provider="local", workspace=workspace, env={"QWEN_E2E_KEY": "test-key"}
        ),
        runtime=RuntimeConfig(input_schema="text", output_schema="message"),
    )


def test_qwen_descriptor_claims_only_verified_fields():
    descriptor = json.loads(DESCRIPTOR.read_text(encoding="utf-8"))
    assert descriptor["adapter_id"] == "nvidia.fabric.qwen"
    assert descriptor["requirements"] == {"binaries": ["node"]}
    assert descriptor["config"] == {
        "accepts": [
            "models",
            "models.base_url",
            "models.temperature",
            "models.top_p",
            "instructions.system",
            "tools.blocked",
            "skills",
            "mcp",
            "mcp.tool_filters",
        ],
        "system_instruction_modes": ["replace", "append"],
    }
    assert descriptor["capabilities"] == {
        "streaming": False,
        "cancellation": False,
        "updates": False,
        "service": False,
    }


def test_qwen_plans_and_projects_config(tmp_path: Path):
    config = qwen_config(tmp_path)
    config.skills = SkillConfig(paths=["examples/code_review_agent/skills/code-review"])
    config.tools = ToolsConfig(blocked=["Bash"])
    config.mcp = McpConfig(
        servers={
            "probe": McpServerConfig(
                transport="stdio",
                url="node",
                args=["server.mjs"],
                allowed_tools=["echo"],
                blocked_tools=["hidden"],
            )
        }
    )
    plan = Fabric().plan(config, base_dir=ROOT)

    assert plan.adapter_descriptor["descriptor"]["adapter_id"] == "nvidia.fabric.qwen"
    assert plan.agent_config["models"]["default"] == {
        "provider": "openai",
        "model": "fabric-test-model",
        "api_key_env": "QWEN_E2E_KEY",
    }
    assert plan.agent_config["instructions"]["system"]["mode"] == "append"
    assert plan.agent_config["tools"]["blocked"] == ["Bash"]
    assert plan.agent_config["skills"]["paths"]
    assert plan.agent_config["mcp"] == {
        "servers": {
            "probe": {
                "transport": "stdio",
                "url": "node",
                "args": ["server.mjs"],
                "allowed_tools": ["echo"],
                "blocked_tools": ["hidden"],
            }
        }
    }


def test_qwen_rejects_unclaimed_normalized_config(tmp_path: Path):
    config = qwen_config(tmp_path)
    config.tools = ToolsConfig(enabled=[])
    with pytest.raises(FabricConfigError, match="tools.enabled"):
        Fabric().plan(config, base_dir=ROOT)


def test_qwen_rejects_unsupported_provider(tmp_path: Path):
    with pytest.raises(FabricConfigError, match="provider"):
        Fabric().plan(qwen_config(tmp_path, provider="anthropic"), base_dir=ROOT)


def test_code_review_variant_uses_qwen_native_tool_names():
    config = review_qwen_config()
    assert config.tools.blocked == [
        "exec",
        "run_shell_command",
        "edit",
        "write_file",
        "notebook_edit",
    ]
