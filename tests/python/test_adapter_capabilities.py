# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import AsyncMock
from unittest.mock import Mock

import pytest
from nemo_fabric import Fabric
from nemo_fabric import FabricConfig
from nemo_fabric import FabricConfigError
from nemo_fabric import HarnessConfig
from nemo_fabric import MetadataConfig
from nemo_fabric import inspect_adapter
from nemo_fabric.capabilities import _adapter_descriptor_sha256
from nemo_fabric.capabilities import _verify_adapter_descriptor
from nemo_fabric_adapter_catalog import get_adapter_descriptor


@pytest.fixture(name="custom_descriptor")
def custom_descriptor_fixture():
    return {
        "contract_version": "fabric.adapter/v1alpha2",
        "adapter_id": "example.fabric.custom",
        "adapter_kind": "python",
        "runner": {"module": "not_installed.adapter"},
        "config": {"accepts": ["skills", "mcp"]},
        "telemetry": {"providers": {"relay": {"outputs": ["atif"]}}},
    }


@pytest.fixture(name="custom_config")
def custom_config_fixture():
    return FabricConfig(
        metadata=MetadataConfig(name="inspection"),
        harness=HarnessConfig(adapter_id="example.fabric.custom"),
    )


@pytest.mark.parametrize(
    "adapter_id,skills,mcp",
    [
        ("nvidia.fabric.claude", True, True),
        ("nvidia.fabric.codex", True, True),
        ("nvidia.fabric.hermes", True, True),
        ("nvidia.fabric.langchain.deepagents", True, True),
        ("nvidia.fabric.mini-swe-agent", False, False),
        ("nvidia.fabric.openclaw", True, True),
        ("nvidia.fabric.openhands", True, True),
        ("nvidia.fabric.remote-agent", False, False),
        ("nvidia.fabric.cline", True, True),
        ("nvidia.fabric.kilo", True, True),
        ("nvidia.fabric.opencode", True, True),
        ("nvidia.fabric.pi", True, True),
        ("nvidia.fabric.qwen", True, True),
    ],
)
def test_official_profiles_follow_catalog_metadata(adapter_id, skills, mcp):
    descriptor = get_adapter_descriptor(adapter_id)
    settings = (
        {"base_url": "http://localhost:12345"}
        if adapter_id == "nvidia.fabric.remote-agent"
        else {}
    )
    config = FabricConfig(
        metadata=MetadataConfig(name="inspection"),
        harness=HarnessConfig(adapter_id=adapter_id, settings=settings),
    )
    profile = inspect_adapter(config, descriptor)
    assert (profile.skills, profile.mcp, profile.atif) == (skills, mcp, False)
    assert len(profile.descriptor_sha256) == 64


def test_inspection_does_not_load_sdk_or_mutate_task_discovery(
    custom_descriptor, custom_config, monkeypatch
):
    from nemo_fabric import DiscoveryConfig

    custom_config.discovery = DiscoveryConfig(local_paths=["/task/only/descriptors"])
    import os
    import nemo_fabric.client as client

    os.environ["ADAPTER_PYTHON"] = "/task/only/nonexistent-python"
    monkeypatch.setattr(
        client._native,
        "plan_config",
        Mock(side_effect=AssertionError("execution discovery must not run")),
    )
    before = custom_config.to_mapping()
    monkeypatch.setitem(sys.modules, "not_installed", None)
    profile = inspect_adapter(custom_config, custom_descriptor)
    assert profile.skills and profile.mcp
    assert custom_config.to_mapping() == before
    with pytest.raises(FrozenInstanceError):
        profile.skills = False


@pytest.mark.parametrize(
    "enabled,expected", [(None, False), (False, False), (True, True)]
)
def test_atif_requires_selected_provider_and_enabled_output(
    custom_descriptor, custom_config, enabled, expected
):
    if enabled is not None:
        custom_config.enable_relay(observability={"atif": {"enabled": enabled}})
    assert inspect_adapter(custom_config, custom_descriptor).atif is expected


def test_atif_does_not_union_unselected_provider_outputs(
    custom_descriptor, custom_config
):
    from nemo_fabric import TelemetryConfig

    custom_descriptor["telemetry"]["providers"]["native"] = {"outputs": ["otel"]}
    custom_config.telemetry = TelemetryConfig(providers={"native": {}})
    assert inspect_adapter(custom_config, custom_descriptor).atif is False


def test_ambiguous_relay_components_do_not_confirm_atif(
    custom_descriptor, custom_config
):
    from nemo_fabric import RelayComponentConfig

    custom_config.enable_relay(observability={"atif": {"enabled": True}})
    custom_config.relay.components.append(
        RelayComponentConfig(kind="observability", config={"atif": {"enabled": False}})
    )
    with pytest.raises(FabricConfigError, match="Cannot confirm requested Relay ATIF"):
        inspect_adapter(custom_config, custom_descriptor)


def test_relay_atif_request_requires_declared_output(custom_descriptor, custom_config):
    custom_descriptor["telemetry"]["providers"]["relay"]["outputs"] = ["otel"]
    custom_config.enable_relay(observability={"atif": {"enabled": True}})
    with pytest.raises(FabricConfigError, match="Cannot confirm requested Relay ATIF"):
        inspect_adapter(custom_config, custom_descriptor)


@pytest.mark.parametrize("atif", [[], {"enabled": "true"}, {"enabled": 1}])
def test_malformed_generic_relay_atif_fails_admission(
    custom_descriptor, custom_config, atif
):
    from nemo_fabric import RelayComponentConfig

    custom_config.enable_relay()
    custom_config.relay.components.append(
        RelayComponentConfig(kind="observability", config={"atif": atif})
    )
    with pytest.raises(FabricConfigError, match="Relay ATIF admission requires"):
        inspect_adapter(custom_config, custom_descriptor)


def test_native_atif_requires_selected_declared_provider(
    custom_descriptor, custom_config
):
    from nemo_fabric import TelemetryConfig

    custom_descriptor["telemetry"]["providers"]["native"] = {"outputs": ["atif"]}
    assert not inspect_adapter(custom_config, custom_descriptor).atif
    custom_config.telemetry = TelemetryConfig(providers={"native": {}})
    assert inspect_adapter(custom_config, custom_descriptor).atif


def test_inspection_requires_typed_config(custom_descriptor):
    with pytest.raises(FabricConfigError, match="typed FabricConfig"):
        inspect_adapter({}, custom_descriptor)


@pytest.mark.parametrize("feature", ["skills", "mcp", "telemetry"])
def test_missing_metadata_rejects_unconfirmed_requested_features(
    custom_config, feature
):
    if feature == "skills":
        custom_config.add_skill_path("/task/skills/example")
    elif feature == "mcp":
        custom_config.add_mcp_server("tools", transport="stdio", url="tool-server")
    else:
        custom_config.enable_relay()
    with pytest.raises(
        FabricConfigError, match="host descriptor metadata is unavailable"
    ):
        inspect_adapter(custom_config, None)


def test_missing_or_incomplete_metadata_makes_conservative_claims(
    custom_config, custom_descriptor
):
    missing = inspect_adapter(custom_config, None)
    assert (missing.skills, missing.mcp, missing.atif, missing.descriptor_sha256) == (
        False,
        False,
        False,
        None,
    )
    del custom_descriptor["config"]
    del custom_descriptor["telemetry"]
    incomplete = inspect_adapter(custom_config, custom_descriptor)
    assert (incomplete.skills, incomplete.mcp, incomplete.atif) == (False, False, False)


@pytest.mark.parametrize("feature", ["skills", "mcp", "telemetry"])
def test_known_unsupported_features_fail_during_inspection(
    custom_descriptor, custom_config, feature
):
    custom_descriptor["config"]["accepts"] = []
    custom_descriptor["telemetry"]["providers"] = {}
    if feature == "skills":
        custom_config.add_skill_path("/task/skills/example")
    elif feature == "mcp":
        custom_config.add_mcp_server("tools", transport="stdio", url="tool-server")
    else:
        custom_config.enable_relay()
    with pytest.raises(FabricConfigError):
        inspect_adapter(custom_config, custom_descriptor)


def test_descriptor_fingerprint_ignores_origin_but_detects_contract_drift(
    tmp_path, custom_descriptor, custom_config
):
    from nemo_fabric import DiscoveryConfig

    profile = inspect_adapter(custom_config, custom_descriptor)
    path = tmp_path / "custom.fabric-adapter.json"
    path.write_text(json.dumps(custom_descriptor), encoding="utf-8")
    custom_config.discovery = DiscoveryConfig(local_paths=[path])
    plan = Fabric().plan(custom_config, base_dir=tmp_path)
    assert profile.descriptor_sha256 == _adapter_descriptor_sha256(plan)
    _verify_adapter_descriptor(plan, profile.descriptor_sha256)
    custom_descriptor["config"]["accepts"].remove("mcp")
    path.write_text(json.dumps(custom_descriptor), encoding="utf-8")
    changed = Fabric().plan(custom_config, base_dir=tmp_path)
    with pytest.raises(
        FabricConfigError, match="Host/task adapter descriptor mismatch"
    ):
        _verify_adapter_descriptor(changed, profile.descriptor_sha256)


@pytest.mark.parametrize("method", ["run", "start_runtime"])
async def test_sdk_rejects_drift_before_native_start(
    tmp_path, custom_descriptor, custom_config, monkeypatch, method
):
    import nemo_fabric.client as client
    from nemo_fabric import DiscoveryConfig

    path = tmp_path / "custom.fabric-adapter.json"
    path.write_text(json.dumps(custom_descriptor), encoding="utf-8")
    custom_config.discovery = DiscoveryConfig(local_paths=[path])
    start = Mock(side_effect=AssertionError("native start must not run"))
    native = Mock(spec=client._native)
    native.plan_config = client._native.plan_config
    native.start_runtime = start
    monkeypatch.setattr(client, "_native", native)
    lifecycle = AsyncMock(side_effect=AssertionError("lifecycle must not run"))
    monkeypatch.setattr(client, "_run_native_lifecycle", lifecycle)
    with pytest.raises(
        FabricConfigError, match="Host/task adapter descriptor mismatch"
    ):
        await getattr(Fabric(), method)(
            custom_config, base_dir=tmp_path, expected_descriptor_sha256="0" * 64
        )
    start.assert_not_called()
    lifecycle.assert_not_called()


@pytest.mark.usefixtures("requires_harbor")
def test_external_harbor_adapter_uses_profile_without_bridge_changes(
    tmp_path, custom_descriptor
):
    from nemo_fabric.integrations.harbor import FabricAgent

    path = tmp_path / "custom.fabric-adapter.json"
    path.write_text(json.dumps(custom_descriptor), encoding="utf-8")
    agent = FabricAgent(
        logs_dir=tmp_path,
        fabric_adapter_id=custom_descriptor["adapter_id"],
        fabric_adapter_descriptor=path,
        fabric_discovery_paths=["/task/installed/descriptor.json"],
        skills_dir="/task/skills",
    )
    assert agent._capability_profile.skills
    payload = agent._build_spec("use the skill")
    assert (
        payload.adapter_descriptor_sha256 == agent._capability_profile.descriptor_sha256
    )
    assert payload.config.discovery.local_paths == ["/task/installed/descriptor.json"]


@pytest.mark.usefixtures("requires_harbor")
def test_harbor_capabilities_are_per_instance(tmp_path, custom_descriptor, monkeypatch):
    from pydantic import create_model
    from nemo_fabric.integrations.harbor import fabric_agent

    capabilities = create_model(
        "MockCurrentHarborCapabilities",
        __base__=fabric_agent.AgentCapabilities,
        skills=(bool, False),
        mcp_servers=(bool, False),
    )
    monkeypatch.setattr(fabric_agent, "AgentCapabilities", capabilities)
    path = tmp_path / "custom.fabric-adapter.json"
    path.write_text(json.dumps(custom_descriptor), encoding="utf-8")
    enabled = fabric_agent.FabricAgent(
        logs_dir=tmp_path,
        fabric_adapter_id=custom_descriptor["adapter_id"],
        fabric_adapter_descriptor=path,
        fabric_telemetry="relay",
    )
    conservative = fabric_agent.FabricAgent(
        logs_dir=tmp_path, fabric_adapter_id="example.fabric.unknown"
    )
    assert enabled.capabilities.skills and enabled.capabilities.mcp_servers
    assert enabled.capabilities.atif
    assert not conservative.capabilities.skills
    assert not conservative.capabilities.mcp_servers
    assert not conservative.capabilities.atif
    assert not fabric_agent.FabricAgent.capabilities.atif


@pytest.mark.parametrize("descriptor", [{}, {"adapter_id": "example.other"}, []])
def test_invalid_metadata_fails_clearly(custom_config, descriptor):
    with pytest.raises(FabricConfigError, match="Host descriptor does not match"):
        inspect_adapter(custom_config, descriptor)


@pytest.mark.usefixtures("requires_harbor")
def test_harbor_rejects_unsupported_mcp_before_setup(tmp_path):
    from harbor.models.task.config import MCPServerConfig
    from nemo_fabric.integrations.harbor import FabricAgent

    with pytest.raises(FabricConfigError, match="mcp"):
        FabricAgent(
            logs_dir=tmp_path,
            fabric_adapter_id="nvidia.fabric.mini-swe-agent",
            mcp_servers=[
                MCPServerConfig(name="tools", transport="stdio", command="tool-server")
            ],
        )


@pytest.mark.usefixtures("requires_harbor")
@pytest.mark.parametrize(
    "adapter_id", ["nvidia.fabric.mini-swe-agent", "example.fabric.unknown"]
)
def test_harbor_rejects_unconfirmed_skill_collection_before_setup(tmp_path, adapter_id):
    from nemo_fabric.integrations.harbor import FabricAgent

    with pytest.raises(FabricConfigError, match="skills"):
        FabricAgent(
            logs_dir=tmp_path,
            fabric_adapter_id=adapter_id,
            skills_dir="/task/only/skills",
        )


@pytest.mark.usefixtures("requires_harbor")
async def test_real_harbor_runner_matches_custom_task_descriptor(
    hermes_shim_agent_dir, tmp_path
):
    from nemo_fabric import RunRequest
    from nemo_fabric.integrations.harbor import runner
    from nemo_fabric.integrations.harbor.models import FabricRunPayload
    from _utils.configs import hermes_shim_config

    config = hermes_shim_config()
    plan = Fabric().plan(config, base_dir=hermes_shim_agent_dir)
    descriptor = plan.to_mapping()["adapter_descriptor"]["descriptor"]
    profile = inspect_adapter(config, descriptor)
    payload = FabricRunPayload(
        config=config,
        config_base_dir=str(hermes_shim_agent_dir),
        logs_dir=str(tmp_path),
        request=RunRequest(input="hello custom descriptor"),
        adapter_descriptor_sha256=profile.descriptor_sha256,
    )
    result = await runner.run(payload)
    assert result.status == "succeeded"
    assert result.adapter_id == descriptor["adapter_id"]
    payload.adapter_descriptor_sha256 = "0" * 64
    with pytest.raises(
        FabricConfigError, match="Host/task adapter descriptor mismatch"
    ):
        await runner.run(payload)
