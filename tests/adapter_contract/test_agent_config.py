# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the typed southbound adapter configuration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from nemo_fabric_adapter_contract.models import AgentConfig
from nemo_fabric_adapter_contract.models import AgentConfigBlock
from nemo_fabric_adapter_contract.models import AgentHarnessConfig
from nemo_fabric_adapter_contract.models import AgentInstructionConfig
from nemo_fabric_adapter_contract.models import AgentInstructionsConfig
from nemo_fabric_adapter_contract.models import AgentMcpConfig
from nemo_fabric_adapter_contract.models import AgentMcpServerConfig
from nemo_fabric_adapter_contract.models import AgentModelConfig
from nemo_fabric_adapter_contract.models import AgentRuntimeConfig
from nemo_fabric_adapter_contract.models import AgentSkillConfig
from nemo_fabric_adapter_contract.models import AgentToolDefinition
from nemo_fabric_adapter_contract.models import AgentToolsConfig
from nemo_fabric_adapter_contract.models import AgentWorkflowConfig
from nemo_fabric_adapter_contract.models import AgentWorkflowEntrypointConfig
from nemo_fabric_adapter_contract.models import extension_schema
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import ValidationError


class _TypedExtensions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workflow_type: str
    retries: int = 0


AGENT_CONFIG_BLOCKS = (
    AgentConfig,
    AgentConfigBlock,
    AgentHarnessConfig,
    AgentInstructionConfig,
    AgentInstructionsConfig,
    AgentMcpConfig,
    AgentMcpServerConfig,
    AgentModelConfig,
    AgentRuntimeConfig,
    AgentSkillConfig,
    AgentToolDefinition,
    AgentToolsConfig,
    AgentWorkflowConfig,
    AgentWorkflowEntrypointConfig,
)


@pytest.mark.parametrize("block_type", [AgentConfigBlock, AgentConfig])
def test_agent_config_blocks_set_mapping_extensions(block_type: type[AgentConfigBlock]):
    raw: dict[str, Any] = {"profile": {"enabled": True}}

    block = block_type().set_extensions(raw)
    raw["profile"]["enabled"] = False

    assert block.extensions == {"profile": {"enabled": True}}
    assert block.to_mapping()["extensions"] == {"profile": {"enabled": True}}


def test_agent_config_block_accepts_typed_extensions():
    extensions = _TypedExtensions(workflow_type="react_agent", retries=2)

    config = AgentConfig().set_extensions(extensions)

    assert config.to_mapping() == {
        "extensions": {
            "workflow_type": "react_agent",
            "retries": 2,
        },
    }


def test_agent_config_block_omits_empty_extensions():
    assert AgentConfig().to_mapping() == {}


def test_agent_config_blocks_reject_implicit_and_non_json_extensions():
    with pytest.raises(ValidationError, match="extra_forbidden"):
        AgentConfig(implicit_extension=True)  # type: ignore[call-arg]

    with pytest.raises(ValidationError, match="valid JSON value"):
        AgentConfig().set_extensions({"unsupported": object()})


@pytest.mark.parametrize("model", AGENT_CONFIG_BLOCKS)
def test_agent_config_schema_exposes_explicit_extensions_on_every_block(
    model: type[AgentConfigBlock],
):
    schema = model.model_json_schema()

    assert schema["additionalProperties"] is False
    assert "extensions" in schema["properties"]


def test_extension_schema_uses_typed_pydantic_model():
    schema = extension_schema(_TypedExtensions)

    assert schema["additionalProperties"] is False
    assert schema["required"] == ["workflow_type"]


def test_agent_config_model_tracks_rust_schema_root_fields():
    rust_schema = json.loads(
        Path("schemas/agent-config.schema.json").read_text(encoding="utf-8")
    )
    pydantic_schema = AgentConfig.model_json_schema()

    assert rust_schema["additionalProperties"] is False
    assert set(pydantic_schema["properties"]) == set(rust_schema["properties"])
