# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
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
    profile = inspect_adapter(config, descriptor).capability_profile
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
    profile = inspect_adapter(custom_config, custom_descriptor).capability_profile
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
    assert (
        inspect_adapter(custom_config, custom_descriptor).capability_profile.atif
        is expected
    )


def test_atif_does_not_union_unselected_provider_outputs(
    custom_descriptor, custom_config
):
    from nemo_fabric import TelemetryConfig

    custom_descriptor["telemetry"]["providers"]["native"] = {"outputs": ["otel"]}
    custom_config.telemetry = TelemetryConfig(providers={"native": {}})
    assert (
        inspect_adapter(custom_config, custom_descriptor).capability_profile.atif
        is False
    )


def test_ambiguous_relay_components_do_not_confirm_atif(
    custom_descriptor, custom_config
):
    from nemo_fabric import RelayComponentConfig

    custom_config.enable_relay(observability={"atif": {"enabled": True}})
    custom_config.relay.components.append(
        RelayComponentConfig(kind="observability", config={"atif": {"enabled": False}})
    )
    report = inspect_adapter(custom_config, custom_descriptor)
    assert report.capability_profile is None
    assert any(
        "Cannot confirm requested Relay ATIF" in issue.reason for issue in report.issues
    )


def test_relay_atif_request_requires_declared_output(custom_descriptor, custom_config):
    custom_descriptor["telemetry"]["providers"]["relay"]["outputs"] = ["otel"]
    custom_config.enable_relay(observability={"atif": {"enabled": True}})
    report = inspect_adapter(custom_config, custom_descriptor)
    assert report.capability_profile is None
    assert any(
        "Cannot confirm requested Relay ATIF" in issue.reason for issue in report.issues
    )


@pytest.mark.parametrize("component_style", ["first_class", "generic"])
@pytest.mark.parametrize(
    "outputs,enabled,ambiguous,compatible",
    [
        (["otel"], True, False, False),
        (["atif"], True, False, True),
        (["otel"], False, False, True),
        (["atif"], True, True, False),
    ],
)
def test_relay_atif_inspection_matches_planning(
    custom_config,
    custom_descriptor,
    tmp_path,
    component_style,
    outputs,
    enabled,
    ambiguous,
    compatible,
):
    from nemo_fabric import DiscoveryConfig, RelayComponentConfig

    custom_descriptor["telemetry"]["providers"]["relay"]["outputs"] = outputs
    if component_style == "first_class":
        custom_config.enable_relay(observability={"atif": {"enabled": enabled}})
    else:
        custom_config.enable_relay()
        custom_config.relay.components.append(
            RelayComponentConfig(
                kind="observability", config={"atif": {"enabled": enabled}}
            )
        )
    if ambiguous:
        custom_config.relay.components.append(
            RelayComponentConfig(
                kind="observability", config={"atif": {"enabled": False}}
            )
        )
    path = tmp_path / "custom.fabric-adapter.json"
    path.write_text(json.dumps(custom_descriptor))
    custom_config.discovery = DiscoveryConfig(local_paths=[path])

    report = inspect_adapter(custom_config, custom_descriptor)
    if compatible:
        assert report.issues == []
        assert report.capability_profile.atif is enabled
        Fabric().plan(custom_config, base_dir=tmp_path)
    else:
        assert report.capability_profile is None
        assert len(report.issues) == 1
        issue = report.issues[0]
        assert issue.field == "telemetry.providers.relay"
        with pytest.raises(FabricConfigError) as planning_error:
            Fabric().plan(custom_config, base_dir=tmp_path)
        with pytest.raises(FabricConfigError) as inspection_error:
            inspect_adapter(custom_config, custom_descriptor, diagnostics=False)
        assert str(planning_error.value) == str(inspection_error.value)
        assert issue.field in str(planning_error.value)
        assert issue.reason in str(planning_error.value)


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
    assert not inspect_adapter(custom_config, custom_descriptor).capability_profile.atif
    custom_config.telemetry = TelemetryConfig(providers={"native": {}})
    assert inspect_adapter(custom_config, custom_descriptor).capability_profile.atif


def test_inspection_requires_typed_config(custom_descriptor):
    with pytest.raises(FabricConfigError, match="typed FabricConfig"):
        inspect_adapter({}, custom_descriptor)


@pytest.mark.parametrize(
    "feature",
    [
        "skills",
        "skill_directories",
        "skill_duplicates",
        "bundled_skills",
        "mcp",
        "telemetry",
    ],
)
def test_missing_metadata_rejects_unconfirmed_requested_features(
    custom_config, feature
):
    from nemo_fabric import BundledSkills
    from nemo_fabric import SkillConfig
    from nemo_fabric import SkillDirectory

    if feature == "skills":
        custom_config.add_skill_path("/task/skills/example")
    elif feature == "skill_directories":
        custom_config.skills = SkillConfig(
            directories=[SkillDirectory(folder="/task/skills")]
        )
    elif feature == "skill_duplicates":
        custom_config.skills = SkillConfig(directories=[], duplicates="reject")
    elif feature == "bundled_skills":
        custom_config.skills = SkillConfig(bundled=BundledSkills(enabled=False))
    elif feature == "mcp":
        custom_config.add_mcp_server("tools", transport="stdio", url="tool-server")
    else:
        custom_config.enable_relay()
    report = inspect_adapter(custom_config, None)
    assert report.capability_profile is None
    assert "host descriptor metadata is unavailable" in report.issues[0].reason


def test_missing_metadata_accepts_empty_skill_directories(custom_config):
    from nemo_fabric import SkillConfig

    # An empty list asks for no external skills, so core needs no descriptor.
    custom_config.skills = SkillConfig(directories=[])

    profile = inspect_adapter(custom_config, None).capability_profile

    assert (profile.adapter_id, profile.skills) == ("example.fabric.custom", False)


def test_missing_or_incomplete_metadata_makes_conservative_claims(
    custom_config, custom_descriptor
):
    missing = inspect_adapter(custom_config, None).capability_profile
    assert (missing.skills, missing.mcp, missing.atif, missing.descriptor_sha256) == (
        False,
        False,
        False,
        None,
    )
    del custom_descriptor["config"]
    del custom_descriptor["telemetry"]
    incomplete = inspect_adapter(custom_config, custom_descriptor).capability_profile
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
    report = inspect_adapter(custom_config, custom_descriptor)
    assert report.issues and report.capability_profile is None


def test_descriptor_fingerprint_ignores_origin_but_detects_contract_drift(
    tmp_path, custom_descriptor, custom_config
):
    from nemo_fabric import DiscoveryConfig

    profile = inspect_adapter(custom_config, custom_descriptor).capability_profile
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
    profile = inspect_adapter(config, descriptor).capability_profile
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


@pytest.fixture(name="workflow_metadata")
def workflow_metadata_fixture(custom_config, custom_descriptor):
    from nemo_fabric import WorkflowConfig

    custom_descriptor["target_types"] = ["workflow"]
    custom_config.workflow = WorkflowConfig(
        target_id="example.custom.workflow", settings={"mode": "safe"}
    )
    target = {
        "contract_version": "fabric.adapter/v1alpha2",
        "type": "workflow",
        "id": "example.custom.workflow",
        "adapter_id": custom_descriptor["adapter_id"],
        "spec": {
            "entrypoint": {
                "kind": "module_function",
                "ref": "not_installed.workflow:run",
            },
            "settings_schema": {
                "type": "object",
                "properties": {"mode": {"enum": ["safe", "other"]}},
                "additionalProperties": False,
            },
        },
    }
    return custom_config, custom_descriptor, target


def test_workflow_inspection_uses_supplied_metadata_without_sdk_or_task_discovery(
    workflow_metadata, monkeypatch
):
    from nemo_fabric import DiscoveryConfig
    import nemo_fabric.client as client

    config, descriptor, target = workflow_metadata
    config.discovery = DiscoveryConfig(local_paths=["/task-only/missing-descriptors"])
    monkeypatch.setitem(sys.modules, "not_installed", None)
    monkeypatch.setattr(
        client._native,
        "plan_config",
        Mock(side_effect=AssertionError("execution discovery must not run")),
    )
    before = config.to_mapping()
    profile = inspect_adapter(
        config, descriptor, target_descriptor=target
    ).capability_profile
    assert profile.skills and profile.mcp
    assert len(profile.descriptor_sha256) == 64
    assert config.to_mapping() == before
    config.harness = None
    assert (
        inspect_adapter(config, descriptor, target_descriptor=target).capability_profile
        == profile
    )


@pytest.mark.parametrize("missing", ["adapter", "target"])
def test_workflow_inspection_requires_both_metadata_records(workflow_metadata, missing):
    config, descriptor, target = workflow_metadata
    with pytest.raises(FabricConfigError, match="requires adapter and target metadata"):
        inspect_adapter(
            config,
            None if missing == "adapter" else descriptor,
            target_descriptor=None if missing == "target" else target,
        )


def test_target_metadata_without_workflow_is_rejected(custom_config, custom_descriptor):
    with pytest.raises(FabricConfigError, match="requires a configured workflow"):
        inspect_adapter(custom_config, custom_descriptor, target_descriptor={})


@pytest.mark.parametrize(
    "invalid", ["target_id", "adapter_id", "type", "target_support", "settings"]
)
def test_workflow_inspection_fails_closed_and_sanitizes_invalid_metadata(
    workflow_metadata, invalid
):
    config, descriptor, target = workflow_metadata
    secret = "private-model-setting-do-not-disclose"
    if invalid == "target_id":
        target["id"] = "example.wrong.workflow"
    elif invalid == "adapter_id":
        target["adapter_id"] = "example.other.adapter"
    elif invalid == "type":
        target["type"] = "unsupported"
    elif invalid == "target_support":
        descriptor["target_types"] = []
    else:
        config.workflow.settings["mode"] = secret
    if invalid in {"settings", "target_support"}:
        report = inspect_adapter(config, descriptor, target_descriptor=target)
        assert report.issues and report.capability_profile is None
        assert secret not in repr(report.issues)
    else:
        with pytest.raises(FabricConfigError) as caught:
            inspect_adapter(config, descriptor, target_descriptor=target)
        assert secret not in str(caught.value)
        assert any(
            field in str(caught.value)
            for field in ["workflow.target_id", "target_types", "target descriptor"]
        )


def test_workflow_fingerprint_matches_execution_plan_and_includes_target(
    workflow_metadata, tmp_path
):
    from nemo_fabric import DiscoveryConfig

    config, descriptor, target = workflow_metadata
    profile = inspect_adapter(
        config, descriptor, target_descriptor=target
    ).capability_profile
    adapter_path = tmp_path / "custom.fabric-adapter.json"
    target_path = tmp_path / "custom.fabric-target.json"
    adapter_path.write_text(json.dumps(descriptor))
    target_path.write_text(json.dumps(target))
    config.discovery = DiscoveryConfig(local_paths=[adapter_path, target_path])
    plan = Fabric().plan(config, base_dir=tmp_path)
    mapping = plan.to_mapping()
    canonical = json.dumps(
        {
            "adapter": mapping["adapter_descriptor"]["descriptor"],
            "target": mapping["adapter_target_descriptor"]["descriptor"],
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    assert profile.descriptor_sha256 == hashlib.sha256(canonical.encode()).hexdigest()
    assert _adapter_descriptor_sha256(plan) == profile.descriptor_sha256
    _verify_adapter_descriptor(plan, profile.descriptor_sha256)


def test_standalone_fingerprint_keeps_original_byte_contract(
    custom_config, custom_descriptor, tmp_path
):
    from nemo_fabric import DiscoveryConfig

    path = tmp_path / "custom.fabric-adapter.json"
    path.write_text(json.dumps(custom_descriptor))
    custom_config.discovery = DiscoveryConfig(local_paths=[path])
    plan = Fabric().plan(custom_config, base_dir=tmp_path)
    original = json.dumps(
        plan.to_mapping()["adapter_descriptor"]["descriptor"],
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    assert (
        _adapter_descriptor_sha256(plan)
        == hashlib.sha256(original.encode()).hexdigest()
    )


@pytest.mark.parametrize("method", ["run", "start_runtime"])
async def test_sdk_rejects_target_drift_before_native_start(
    workflow_metadata, tmp_path, monkeypatch, method
):
    from nemo_fabric import DiscoveryConfig
    import nemo_fabric.client as client

    config, descriptor, target = workflow_metadata
    profile = inspect_adapter(
        config, descriptor, target_descriptor=target
    ).capability_profile
    adapter_path = tmp_path / "custom.fabric-adapter.json"
    target_path = tmp_path / "custom.fabric-target.json"
    adapter_path.write_text(json.dumps(descriptor))
    target["spec"]["entrypoint"]["ref"] = "not_installed.changed:run"
    target_path.write_text(json.dumps(target))
    config.discovery = DiscoveryConfig(local_paths=[adapter_path, target_path])
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
            config,
            base_dir=tmp_path,
            expected_descriptor_sha256=profile.descriptor_sha256,
        )
    start.assert_not_called()
    lifecycle.assert_not_called()


def test_inspection_collects_independent_failures_without_files(
    custom_config, custom_descriptor, monkeypatch, tmp_path
):
    import tempfile

    custom_descriptor["config"]["accepts"] = ["models"]
    config = FabricConfig.from_mapping(
        {
            **custom_config.to_mapping(),
            "models": {
                "default": {
                    "provider": "openai",
                    "model": "private-model",
                    "temperature": 0.5,
                    "max_tokens": 20,
                }
            },
            "tools": {"enabled": ["private-tool"]},
            "mcp": {
                "servers": {
                    "private-server": {"transport": "stdio", "url": "private-command"}
                }
            },
        }
    )
    monkeypatch.setattr(
        tempfile, "TemporaryDirectory", Mock(side_effect=AssertionError("no files"))
    )
    import builtins
    import os

    os.environ["TMPDIR"] = str(tmp_path)
    before = set(tmp_path.iterdir())
    with monkeypatch.context() as no_files:
        no_files.setattr(builtins, "open", Mock(side_effect=AssertionError("no files")))
        no_files.setattr(os, "open", Mock(side_effect=AssertionError("no files")))
        report = inspect_adapter(config, custom_descriptor)
    assert set(tmp_path.iterdir()) == before
    assert report.capability_profile is None
    fields = [issue.field for issue in report.issues]
    assert fields == sorted(fields)
    assert {
        "models.default.temperature",
        "models.default.max_tokens",
        "tools.enabled",
    } <= set(fields)
    assert any(field.startswith("mcp") for field in fields)
    assert all(
        issue.adapter_id == custom_descriptor["adapter_id"] for issue in report.issues
    )
    assert "private-model" not in repr(report.issues)
    assert "private-command" not in repr(report.issues)
    assert "private-tool" not in repr(report.issues)
    assert inspect_adapter(config, custom_descriptor) == report


@pytest.mark.parametrize(
    "schema_field",
    ["settings_schema", "model_schema", "tool_definition_schema", "config_schema"],
)
def test_inspection_rejects_invalid_schema(
    custom_config, custom_descriptor, schema_field
):
    custom_descriptor[schema_field] = {"type": "invalid-private-value"}
    with pytest.raises(FabricConfigError, match="invalid adapter descriptor") as error:
        inspect_adapter(custom_config, custom_descriptor)
    assert schema_field in str(error.value)
    assert "invalid-private-value" in str(error.value)


def test_inspection_success_returns_profile(custom_config, custom_descriptor):
    report = inspect_adapter(custom_config, custom_descriptor)
    assert report.issues == []
    assert report.capability_profile is not None
    assert report == inspect_adapter(custom_config, custom_descriptor)


@pytest.mark.parametrize("invalid", ["target_id", "adapter_id", "schema"])
def test_workflow_malformed_metadata_raises(workflow_metadata, invalid):
    config, descriptor, target = workflow_metadata
    if invalid == "target_id":
        target["id"] = "other.workflow"
    elif invalid == "adapter_id":
        target["adapter_id"] = "other.adapter"
    else:
        target["spec"]["settings_schema"] = {"type": "invalid-private-value"}
    with pytest.raises(FabricConfigError):
        inspect_adapter(config, descriptor, target_descriptor=target)


@pytest.mark.parametrize("failure", ["provider", "atif"])
def test_telemetry_compatibility_returns_issues(
    custom_config, custom_descriptor, failure
):
    custom_config.enable_relay(observability={"atif": {"enabled": True}})
    if failure == "provider":
        custom_descriptor["telemetry"]["providers"] = {}
    else:
        custom_descriptor["telemetry"]["providers"]["relay"]["outputs"] = ["otel"]
    report = inspect_adapter(custom_config, custom_descriptor)
    assert report.capability_profile is None
    assert report.issues
    assert all(issue.field.startswith("telemetry.providers") for issue in report.issues)


@pytest.mark.parametrize("artifact_source", [None, "runtime", "environment"])
def test_keep_state_artifact_requirement_matches_planning(
    custom_config, custom_descriptor, tmp_path, artifact_source
):
    from nemo_fabric import DiscoveryConfig, EnvironmentConfig

    custom_config.runtime.keep_state = True
    custom_config.environment = EnvironmentConfig(provider="local", workspace=tmp_path)
    custom_descriptor["config"]["accepts"].append("runtime.keep_state")
    path = tmp_path / "custom.fabric-adapter.json"
    path.write_text(json.dumps(custom_descriptor))
    custom_config.discovery = DiscoveryConfig(local_paths=[path])
    if artifact_source is not None:
        getattr(custom_config, artifact_source).artifacts = tmp_path / "artifacts"
        assert inspect_adapter(custom_config, custom_descriptor).issues == []
        Fabric().plan(custom_config, base_dir=tmp_path)
    else:
        with pytest.raises(FabricConfigError) as inspection_error:
            inspect_adapter(custom_config, custom_descriptor)
        with pytest.raises(FabricConfigError) as planning_error:
            Fabric().plan(custom_config, base_dir=tmp_path)
        assert str(inspection_error.value) == str(planning_error.value)
        assert "runtime.keep_state" in str(inspection_error.value)
        assert "requires an artifact directory" in str(inspection_error.value)


@pytest.mark.parametrize("unsupported_tools", [False, True])
@pytest.mark.parametrize("atif", [[], {"enabled": "invalid"}, {"enabled": 1}])
def test_malformed_relay_atif_is_not_hidden_by_compatibility_issues(
    custom_config, custom_descriptor, unsupported_tools, atif
):
    from nemo_fabric import RelayComponentConfig, ToolsConfig

    custom_config.enable_relay()
    custom_config.relay.components.append(
        RelayComponentConfig(kind="observability", config={"atif": atif})
    )
    if unsupported_tools:
        custom_config.tools = ToolsConfig(enabled=["private-tool"])
    with pytest.raises(FabricConfigError, match="Relay ATIF admission requires"):
        inspect_adapter(custom_config, custom_descriptor)


def test_inspection_defaults_to_reports_with_a_rejection_escape_hatch(
    custom_config, custom_descriptor
):
    from nemo_fabric import ToolsConfig

    custom_config.tools = ToolsConfig(enabled=["private-tool"])
    report = inspect_adapter(custom_config, custom_descriptor)
    assert report.issues and report.capability_profile is None
    assert report == inspect_adapter(custom_config, custom_descriptor, diagnostics=True)
    with pytest.raises(FabricConfigError, match="tools.enabled"):
        inspect_adapter(custom_config, custom_descriptor, diagnostics=False)


@pytest.mark.parametrize("structured_support", [False, True])
def test_skill_directory_inspection_matches_planning(
    custom_config, custom_descriptor, tmp_path, structured_support
):
    from nemo_fabric import DiscoveryConfig, SkillConfig, SkillDirectory

    custom_config.skills = SkillConfig(
        directories=[SkillDirectory(folder="/task-only/skills")]
    )
    if structured_support:
        custom_descriptor["skills"] = get_adapter_descriptor("nvidia.fabric.hermes")[
            "skills"
        ]
    path = tmp_path / "custom.fabric-adapter.json"
    path.write_text(json.dumps(custom_descriptor))
    custom_config.discovery = DiscoveryConfig(local_paths=[path])
    report = inspect_adapter(custom_config, custom_descriptor)
    if structured_support:
        assert report.issues == []
        Fabric().plan(custom_config, base_dir=tmp_path)
    else:
        assert report.capability_profile is None
        assert any(issue.field.startswith("skills") for issue in report.issues)
        with pytest.raises(FabricConfigError, match="skills"):
            Fabric().plan(custom_config, base_dir=tmp_path)


@pytest.mark.parametrize("diagnostics", [True, False])
def test_missing_descriptor_runs_keep_state_validation(
    custom_config, tmp_path, diagnostics
):
    from nemo_fabric import DiscoveryConfig

    descriptor = get_adapter_descriptor("nvidia.fabric.hermes")
    custom_config.harness.adapter_id = descriptor["adapter_id"]
    custom_config.runtime.keep_state = True
    path = tmp_path / "hermes.fabric-adapter.json"
    path.write_text(json.dumps(descriptor))
    custom_config.discovery = DiscoveryConfig(local_paths=[path])
    with pytest.raises(FabricConfigError) as inspected:
        inspect_adapter(custom_config, None, diagnostics=diagnostics)
    with pytest.raises(FabricConfigError) as planned:
        Fabric().plan(custom_config, base_dir=tmp_path)
    assert str(inspected.value) == str(planned.value)
    assert "runtime.keep_state" in str(inspected.value)


@pytest.mark.parametrize("descriptor_present", [False, True])
@pytest.mark.parametrize("diagnostics", [False, True])
def test_inspection_preserves_configuration_error_fields(
    custom_config, custom_descriptor, descriptor_present, diagnostics
):
    # Bypass authoring-time validation to exercise the native validator.
    custom_config.runtime = custom_config.runtime.model_copy(
        update={"timeout_seconds": 0}
    )
    with pytest.raises(FabricConfigError, match="runtime.timeout_seconds"):
        inspect_adapter(
            custom_config,
            custom_descriptor if descriptor_present else None,
            diagnostics=diagnostics,
        )


@pytest.mark.parametrize("diagnostics", [False, True])
def test_inspection_preserves_descriptor_contract_errors(
    custom_config, custom_descriptor, diagnostics
):
    custom_descriptor["contract_version"] = "fabric.adapter/unsupported"
    with pytest.raises(FabricConfigError) as error:
        inspect_adapter(custom_config, custom_descriptor, diagnostics=diagnostics)
    assert "contract_version" in str(error.value)
    assert "fabric.adapter/unsupported" in str(error.value)
    assert "invalid adapter descriptor or embedded schema" not in str(error.value)


def test_missing_metadata_returns_one_combined_issue(custom_config):
    custom_config.add_skill_path("/task/skills/example")
    custom_config.add_mcp_server("tools", transport="stdio", url="tool-server")
    custom_config.enable_relay()
    report = inspect_adapter(custom_config, None)
    assert report.capability_profile is None
    assert len(report.issues) == 1
    assert report.issues[0].field == "descriptor"
    with pytest.raises(
        FabricConfigError, match="host descriptor metadata is unavailable"
    ):
        inspect_adapter(custom_config, None, diagnostics=False)


def test_catalog_and_file_metadata_return_the_same_issues(tmp_path):
    from nemo_fabric import DiscoveryConfig, ModelConfig

    descriptor = get_adapter_descriptor("nvidia.fabric.claude")
    config = FabricConfig(
        metadata=MetadataConfig(name="parity"),
        harness=HarnessConfig(adapter_id=descriptor["adapter_id"]),
        models={
            "default": ModelConfig(
                provider="openai",
                model="private-model",
                temperature=0.5,
                max_tokens=100,
            )
        },
    )
    path = tmp_path / "claude.fabric-adapter.json"
    path.write_text(json.dumps(descriptor))
    from_file = json.loads(path.read_text())
    catalog_report = inspect_adapter(config, descriptor)
    file_report = inspect_adapter(config, from_file)
    assert catalog_report.issues and catalog_report == file_report
    assert {"models.default.temperature", "models.default.max_tokens"} <= {
        issue.field for issue in file_report.issues
    }
    config.discovery = DiscoveryConfig(local_paths=[path])
    with pytest.raises(FabricConfigError):
        Fabric().plan(config, base_dir=tmp_path)


@pytest.mark.parametrize(
    "category,field",
    [
        ("settings", "harness.settings.a"),
        ("definitions", "tools.definitions.a.description"),
        ("extensions", "extensions.a"),
    ],
)
def test_inspection_retains_first_error_boundaries(
    custom_config, custom_descriptor, category, field
):
    from nemo_fabric import ToolDefinitionConfig, ToolsConfig

    if category == "settings":
        custom_descriptor["settings_schema"] = {
            "type": "object",
            "properties": {name: {"type": "string"} for name in ["a", "z"]},
        }
        custom_config.harness.settings = {"z": True, "a": True}
    elif category == "definitions":
        custom_descriptor["config"]["accepts"].append("tools.definitions")
        custom_descriptor["tool_definition_schema"] = {
            "type": "object",
            "required": ["description"],
        }
        custom_config.tools = ToolsConfig(
            definitions={
                name: ToolDefinitionConfig(kind="function", ref=name)
                for name in ["z", "a"]
            }
        )
    else:
        custom_descriptor["extension_schemas"] = {
            "agent_config": {
                "type": "object",
                "properties": {name: {"type": "string"} for name in ["a", "z"]},
            }
        }
        for name in ["z", "a"]:
            setattr(custom_config, name, True)
    report = inspect_adapter(custom_config, custom_descriptor)
    assert report.capability_profile is None
    assert [issue.field for issue in report.issues] == [field]
    with pytest.raises(FabricConfigError, match=field):
        inspect_adapter(custom_config, custom_descriptor, diagnostics=False)


def test_unsupported_target_type_is_reported_by_default(workflow_metadata):
    config, descriptor, target = workflow_metadata
    descriptor["target_types"] = []
    report = inspect_adapter(config, descriptor, target_descriptor=target)
    assert report.capability_profile is None
    assert [issue.field for issue in report.issues] == ["workflow.target_id"]
    with pytest.raises(FabricConfigError, match="workflow.target_id"):
        inspect_adapter(config, descriptor, target_descriptor=target, diagnostics=False)
