# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused planning tests for the Pi SDK adapter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from nemo_fabric import DiscoveryConfig
from nemo_fabric import Fabric
from nemo_fabric import FabricConfig
from nemo_fabric import FabricConfigError
from nemo_fabric import HarnessConfig
from nemo_fabric import MetadataConfig
from nemo_fabric import ModelConfig

ROOT = Path(__file__).resolve().parents[2]
DESCRIPTOR = ROOT / "adapters/typescript/pi/pi.fabric-adapter.json"


def config(*, api_key_env: str | None = "TEST_API_KEY") -> FabricConfig:
    return FabricConfig(
        metadata=MetadataConfig(name="pi-adapter-test"),
        harness=HarnessConfig(adapter_id="nvidia.fabric.pi"),
        discovery=DiscoveryConfig(local_paths=[DESCRIPTOR]),
        models={
            "default": ModelConfig(
                provider="openai",
                model="gpt-4.1-mini",
                api_key_env=api_key_env,
            )
        },
    )


def test_pi_descriptor_declares_the_supported_surface():
    descriptor = json.loads(DESCRIPTOR.read_text(encoding="utf-8"))

    assert descriptor["contract_version"] == "fabric.adapter/v1alpha2"
    assert descriptor["adapter_id"] == "nvidia.fabric.pi"
    assert descriptor["adapter_kind"] == "process"
    assert descriptor["runner"] == {"command": "node", "script": "dist/cli.js"}
    assert descriptor["config"]["accepts"] == [
        "models",
        "models.base_url",
        "instructions.system",
        "tools.definitions",
        "tools.enabled",
        "tools.blocked",
        "skills",
    ]
    assert descriptor["capabilities"] == {
        "streaming": False,
        "cancellation": False,
        "updates": False,
        "service": False,
    }

    assert descriptor["tool_definition_schema"] == {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "kind": {"const": "module"},
            "ref": {
                "type": "string",
                "minLength": 1,
                "pattern": "^[^#]+(?:#[A-Za-z_$][A-Za-z0-9_$]*)?$",
            },
            "settings": {"type": "object"},
        },
        "required": ["kind", "ref"],
        "additionalProperties": False,
    }


def test_pi_descriptor_validates_module_tool_definitions():
    adapter_config = config()
    adapter_config.add_tool_definition(
        "review_context",
        kind="module",
        ref="tools/review-context.ts#createTool",
        settings={"format": "brief"},
    )

    plan = Fabric().plan(adapter_config, base_dir=ROOT)

    assert plan.agent_config["tools"] == {
        "definitions": {
            "review_context": {
                "kind": "module",
                "ref": "tools/review-context.ts#createTool",
                "settings": {"format": "brief"},
            }
        }
    }


def test_pi_descriptor_plans_and_projects_the_selected_model():
    plan = Fabric().plan(config(), base_dir=ROOT)

    assert plan.adapter_descriptor["descriptor"]["adapter_id"] == "nvidia.fabric.pi"
    assert plan.agent_config == {
        "models": {
            "default": {
                "provider": "openai",
                "model": "gpt-4.1-mini",
                "api_key_env": "TEST_API_KEY",
            }
        }
    }


def test_pi_model_schema_requires_a_credential_name():
    with pytest.raises(FabricConfigError, match="api_key_env"):
        Fabric().plan(config(api_key_env=None), base_dir=ROOT)
