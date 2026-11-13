# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static adapter requirements agree between planning and host metadata admission."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
from nemo_fabric import DiscoveryConfig
from nemo_fabric import Fabric
from nemo_fabric import FabricConfig
from nemo_fabric import FabricConfigError
from nemo_fabric import HarnessConfig
from nemo_fabric import MetadataConfig
from nemo_fabric import ModelConfig
from nemo_fabric import ToolsConfig
from nemo_fabric import WorkflowConfig
from nemo_fabric import inspect_adapter
from nemo_fabric_adapter_catalog import get_adapter_descriptor
from nemo_fabric_adapter_catalog import get_target_descriptor


@pytest.fixture(name="admission_config")
def admission_config_fixture():
    return FabricConfig(
        metadata=MetadataConfig(name="conditional-admission"),
        discovery=DiscoveryConfig(
            local_paths=[str(Path(__file__).resolve().parents[2] / "adapters")]
        ),
        harness=HarnessConfig(adapter_id="nvidia.fabric.langchain.deepagents"),
    )


@pytest.mark.parametrize(
    "adapter_id,settings,tools,model",
    [
        ("nvidia.fabric.nooa", {}, None, None),
        (
            "nvidia.fabric.langchain.deepagents",
            {"deepagents": {"backend": {"type": "local_shell"}}},
            None,
            None,
        ),
        (
            "nvidia.fabric.langchain.deepagents",
            {"deepagents": {"backend": {"type": "local_shell"}}},
            {"blocked": ["read_file"]},
            None,
        ),
        (
            "nvidia.fabric.cline",
            {},
            None,
            {
                "provider": "openai",
                "model": "test-model",
                "api_key_env": "TEST_API_KEY",
            },
        ),
        (
            "nvidia.fabric.pi",
            {},
            None,
            {
                "provider": "openai",
                "model": "test-model",
                "api_key_env": "TEST_API_KEY",
                "cost": {"input": 0},
            },
        ),
    ],
)
def test_invalid_static_requirements_fail_plan_and_host_admission(
    admission_config, adapter_id, settings, tools, model, tmp_path
):
    admission_config.harness = HarnessConfig(adapter_id=adapter_id, settings=settings)
    admission_config.tools = ToolsConfig(**tools) if tools is not None else None
    if model is not None:
        admission_config.models = {"default": ModelConfig(**model)}

    with pytest.raises(FabricConfigError):
        inspect_adapter(admission_config, get_adapter_descriptor(adapter_id))
    with pytest.raises(FabricConfigError):
        Fabric().plan(admission_config, base_dir=tmp_path)


@pytest.mark.parametrize(
    "tools",
    [
        {"enabled": []},
        {"enabled": ["execute", "read_file"]},
        {"blocked": ["execute"]},
    ],
)
def test_local_shell_preserves_explicit_policy(admission_config, tools, tmp_path):
    admission_config.harness.settings = {
        "deepagents": {"backend": {"type": "local_shell"}}
    }
    admission_config.tools = ToolsConfig(**tools)
    before = deepcopy(admission_config.to_mapping())
    inspect_adapter(
        admission_config,
        get_adapter_descriptor(admission_config.harness.adapter_id),
    )
    plan = Fabric().plan(admission_config, base_dir=tmp_path)
    assert admission_config.to_mapping() == before
    assert ToolsConfig.from_mapping(plan["config"]["tools"]) == admission_config.tools
    assert plan["agent_config"]["tools"] == before["tools"]


def test_registered_nooa_workflow_is_accepted(admission_config, tmp_path):
    admission_config.harness = HarnessConfig(adapter_id="nvidia.fabric.nooa")
    admission_config.workflow = WorkflowConfig(target_id="nvidia.nooa.coding-agent")
    inspect_adapter(
        admission_config,
        get_adapter_descriptor("nvidia.fabric.nooa"),
        target_descriptor=get_target_descriptor("nvidia.nooa.coding-agent"),
    )
    assert Fabric().plan(
        admission_config, base_dir=tmp_path
    ).config.workflow.target_id == ("nvidia.nooa.coding-agent")


@pytest.mark.parametrize("provider", ["nvidia", "openai-native"])
def test_cline_supported_provider_is_accepted(admission_config, provider, tmp_path):
    admission_config.harness = HarnessConfig(adapter_id="nvidia.fabric.cline")
    admission_config.models = {
        "default": ModelConfig(
            provider=provider,
            model="test-model",
            api_key_env="TEST_API_KEY",
            base_url="https://models.example/v1",
        )
    }
    inspect_adapter(admission_config, get_adapter_descriptor("nvidia.fabric.cline"))
    Fabric().plan(admission_config, base_dir=tmp_path)


@pytest.mark.parametrize("custom", [False, True])
def test_pi_native_overlay_and_complete_custom_model_are_accepted(
    admission_config, custom, tmp_path
):
    admission_config.harness = HarnessConfig(adapter_id="nvidia.fabric.pi")
    metadata = (
        {
            "api": "openai-completions",
            "context_window": 32768,
            "max_tokens": 4096,
            "input": ["text"],
            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
        }
        if custom
        else {}
    )
    admission_config.models = {
        "default": ModelConfig(
            provider="custom" if custom else "openai",
            model="custom-model" if custom else "gpt-4o",
            api_key_env="TEST_API_KEY",
            **({"base_url": "https://models.example/v1"} if custom else {}),
            **metadata,
        )
    }
    inspect_adapter(admission_config, get_adapter_descriptor("nvidia.fabric.pi"))
    Fabric().plan(admission_config, base_dir=tmp_path)


@pytest.mark.parametrize(
    "schema",
    [
        {"type": 7},
        {"type": "string"},
        {"type": "object", "$ref": "https://schemas.example/remote.json"},
    ],
)
def test_invalid_cross_field_schema_fails_metadata_loading(admission_config, schema):
    descriptor = get_adapter_descriptor(admission_config.harness.adapter_id)
    descriptor["config_schema"] = schema
    with pytest.raises(FabricConfigError, match="config_schema"):
        inspect_adapter(admission_config, descriptor)


def test_cross_field_error_does_not_disclose_config_values(admission_config, tmp_path):
    secret = "private-configuration-value"
    admission_config.harness.settings = {
        "deepagents": {"backend": {"type": "local_shell"}}
    }
    admission_config.metadata.description = secret
    descriptor = get_adapter_descriptor(admission_config.harness.adapter_id)
    descriptor["config_schema"] = {"type": "object", "not": {}}
    with pytest.raises(FabricConfigError) as error:
        inspect_adapter(admission_config, descriptor)
    assert secret not in str(error.value)
