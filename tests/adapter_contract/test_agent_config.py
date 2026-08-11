# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the typed southbound adapter configuration."""

from __future__ import annotations

import json
from dataclasses import fields
from dataclasses import is_dataclass
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
from nemo_fabric_adapter_contract.codec import ContractValidationError
from nemo_fabric_adapter_contract.pydantic_support import extension_schema
from nemo_fabric_adapter_contract.pydantic_support import set_pydantic_extensions
from nemo_fabric_adapter_contract.pydantic_support import type_adapter
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import ValidationError


ROOT = Path(__file__).resolve().parents[2]


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

    config = set_pydantic_extensions(AgentConfig(), extensions)

    assert config.to_mapping() == {
        "extensions": {
            "workflow_type": "react_agent",
            "retries": 2,
        },
    }


def test_agent_config_block_omits_empty_extensions():
    assert AgentConfig().to_mapping() == {}


def test_agent_config_blocks_reject_implicit_and_non_json_extensions():
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        AgentConfig(implicit_extension=True)  # type: ignore[call-arg]

    with pytest.raises(ContractValidationError, match="valid JSON value"):
        AgentConfig().set_extensions({"unsupported": object()})


@pytest.mark.parametrize(
    ("payload", "path"),
    [
        ({"extensions": []}, "extensions"),
        ({"harness": {"settings": "not-an-object"}}, "harness.settings"),
    ],
)
def test_agent_config_rejects_non_object_json_mappings(payload, path):
    with pytest.raises(
        ContractValidationError,
        match=rf"{path}: must be a JSON object",
    ):
        AgentConfig.from_mapping(payload)


@pytest.mark.parametrize("model", AGENT_CONFIG_BLOCKS)
def test_agent_config_schema_exposes_explicit_extensions_on_every_block(
    model: type[AgentConfigBlock],
):
    schema = type_adapter(model).json_schema()

    assert is_dataclass(model)
    assert schema["additionalProperties"] is False
    assert "extensions" in schema["properties"]


def test_extension_schema_uses_typed_pydantic_model():
    schema = extension_schema(_TypedExtensions)

    assert schema["additionalProperties"] is False
    assert schema["required"] == ["workflow_type"]


def test_optional_pydantic_adapter_reuses_canonical_dataclass():
    adapter = type_adapter(AgentConfig)

    config = adapter.validate_python({"harness": {"settings": {"profile": "pydantic"}}})

    assert is_dataclass(config)
    assert config.to_mapping() == {"harness": {"settings": {"profile": "pydantic"}}}
    with pytest.raises(ValidationError, match="unexpected_keyword_argument"):
        adapter.validate_python({"unknown": True})


def test_agent_config_from_mapping_reports_nested_field_path():
    with pytest.raises(
        ContractValidationError,
        match=r"models\.default: missing required field 'model'",
    ):
        AgentConfig.from_mapping(
            {
                "models": {
                    "default": {
                        "provider": "nvidia",
                    }
                }
            }
        )


def test_agent_model_config_rejects_float_overflow():
    with pytest.raises(
        ContractValidationError,
        match="temperature: must be a finite number",
    ):
        AgentModelConfig(
            provider="nvidia",
            model="test-model",
            temperature=10**1000,
        )


def test_agent_mcp_server_config_preserves_http_authentication():
    server = AgentMcpServerConfig.from_mapping(
        {
            "transport": "streamable-http",
            "url": "https://mcp.example.test/mcp",
            "authentication": {
                "type": "oauth2",
                "client_id": "fabric-client",
            },
            "custom_headers": {"X-Tenant": "fabric"},
        }
    )

    assert server.to_mapping() == {
        "transport": "streamable-http",
        "url": "https://mcp.example.test/mcp",
        "authentication": {
            "type": "oauth2",
            "client_id": "fabric-client",
        },
        "custom_headers": {"X-Tenant": "fabric"},
    }


def test_agent_config_model_tracks_rust_schema_root_fields():
    rust_schema = json.loads(
        (ROOT / "schemas/adapter-contract/agent-config.schema.json").read_text(
            encoding="utf-8"
        )
    )
    dataclass_fields = {item.name for item in fields(AgentConfig)}

    assert rust_schema["additionalProperties"] is False
    assert dataclass_fields == set(rust_schema["properties"])


def test_agent_config_dataclasses_track_rust_schema_block_fields():
    rust_schema = json.loads(
        (ROOT / "schemas/adapter-contract/agent-config.schema.json").read_text(
            encoding="utf-8"
        )
    )
    blocks = {
        model.__name__: model
        for model in AGENT_CONFIG_BLOCKS
        if model not in {AgentConfig, AgentConfigBlock}
    }

    assert set(blocks) == set(rust_schema["$defs"]).difference({"InstructionMode"})
    for name, model in blocks.items():
        assert {item.name for item in fields(model)} == set(
            rust_schema["$defs"][name]["properties"]
        )
