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
from nemo_fabric import RelayAtifConfig
from nemo_fabric import RelayAtofConfig
from nemo_fabric import RelayAtofFileSinkConfig
from nemo_fabric import RelayAtofStreamSinkConfig
from nemo_fabric import RelayObservabilityConfig

ROOT = Path(__file__).resolve().parents[2]
DESCRIPTOR = ROOT / "adapters/typescript/pi/pi.fabric-adapter.json"
RELAY_PLUGIN_FIXTURE = ROOT / "tests/fixtures/pi-relay-core-plugin.json"


def config(
    *,
    api_key_env: str | None = "TEST_API_KEY",
    descriptor: Path = DESCRIPTOR,
    adapter_id: str = "nvidia.fabric.pi",
) -> FabricConfig:
    return FabricConfig(
        metadata=MetadataConfig(name="pi-adapter-test"),
        harness=HarnessConfig(adapter_id=adapter_id),
        discovery=DiscoveryConfig(local_paths=[descriptor]),
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
    assert descriptor["config"]["system_instruction_modes"] == ["replace"]
    assert descriptor["settings_schema"]["properties"]["relay_extension_path"] == {
        "type": "string",
        "minLength": 1,
        "description": (
            "Absolute path or environment.workspace-relative path to the "
            "NeMo Relay 0.9 Pi extension"
        ),
    }
    assert descriptor["extension_schemas"]["run_result"] == {
        "type": "object",
        "properties": {"pi_turn_started": {"type": "boolean"}},
        "required": ["pi_turn_started"],
        "additionalProperties": False,
    }
    assert descriptor["telemetry"] == {
        "providers": {
            "relay": {"outputs": ["atif", "otel", "openinference"]},
        }
    }
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


def test_pi_descriptor_plans_relay_telemetry():
    adapter_config = config()
    adapter_config.enable_relay(output_dir="./artifacts/relay")

    plan = Fabric().plan(adapter_config, base_dir=ROOT)

    assert plan.telemetry_plan == {
        "providers": ["relay"],
        "relay_enabled": True,
        "relay_output_dir": "./artifacts/relay",
        "adapter_outputs": ["atif", "openinference", "otel"],
    }


def test_pi_relay_plugin_fixture_is_core_authored():
    adapter_config = config()
    adapter_config.enable_relay(
        observability=RelayObservabilityConfig(
            atof=RelayAtofConfig(
                enabled=True,
                sinks=[
                    RelayAtofFileSinkConfig(
                        output_directory="relay-atof", mode="overwrite"
                    ),
                    RelayAtofStreamSinkConfig(
                        name="nemo-fabric-stream",
                        url="http://127.0.0.1:4319/atof",
                        transport="ndjson",
                        header_env={"authorization": "RELAY_AUTHORIZATION"},
                    ),
                ],
            ),
            atif=RelayAtifConfig(enabled=True),
        )
    )

    plan = Fabric().plan(adapter_config, base_dir=ROOT)

    assert plan.telemetry_plan["relay_config"] == json.loads(
        RELAY_PLUGIN_FIXTURE.read_text(encoding="utf-8")
    )


def test_pi_relay_atif_preserves_model_name_override():
    adapter_config = config()
    adapter_config.enable_relay(
        observability=RelayObservabilityConfig(
            atif=RelayAtifConfig(enabled=True, model_name="trajectory-model")
        )
    )

    plan = Fabric().plan(adapter_config, base_dir=ROOT)

    relay_config = plan.telemetry_plan["relay_config"]
    assert (
        relay_config["components"][0]["config"]["atif"]["model_name"]
        == "trajectory-model"
    )


def test_pi_descriptor_rejects_relay_when_support_is_not_declared(tmp_path: Path):
    unsupported = json.loads(DESCRIPTOR.read_text(encoding="utf-8"))
    del unsupported["telemetry"]
    unsupported["adapter_id"] = "test.fabric.pi-no-relay"
    descriptor = tmp_path / "pi.fabric-adapter.json"
    descriptor.write_text(json.dumps(unsupported), encoding="utf-8")
    adapter_config = config(
        descriptor=descriptor,
        adapter_id="test.fabric.pi-no-relay",
    )
    adapter_config.enable_relay(output_dir="./artifacts/relay")

    with pytest.raises(
        FabricConfigError,
        match=r"adapter `test\.fabric\.pi-no-relay` does not support `telemetry\.providers` value `relay`",
    ):
        Fabric().plan(adapter_config, base_dir=ROOT)


def test_pi_model_schema_requires_a_credential_name():
    with pytest.raises(FabricConfigError, match="api_key_env"):
        Fabric().plan(config(api_key_env=None), base_dir=ROOT)
