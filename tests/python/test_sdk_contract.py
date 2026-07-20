# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the public Python SDK request/result contract."""

from __future__ import annotations

import json
from inspect import signature
from pathlib import Path
from typing import Any
from typing import get_overloads

import nemo_fabric
import nemo_fabric.errors as fabric_errors
import pytest
from nemo_fabric import AdapterInfo
from nemo_fabric import DoctorReport
from nemo_fabric import EnvironmentConfig
from nemo_fabric import Fabric
from nemo_fabric import FabricCapabilityError
from nemo_fabric import FabricConfig
from nemo_fabric import FabricConfigError
from nemo_fabric import FabricError
from nemo_fabric import FabricNativeUnavailableError
from nemo_fabric import FabricRuntimeError
from nemo_fabric import FabricStateError
from nemo_fabric import HarnessConfig
from nemo_fabric import McpConfig
from nemo_fabric import MetadataConfig
from nemo_fabric import RelayAtifConfig
from nemo_fabric import RelayAtofConfig
from nemo_fabric import RelayComponentConfig
from nemo_fabric import RelayConfigPolicy
from nemo_fabric import RelayObservabilityConfig
from nemo_fabric import RunOutput
from nemo_fabric import RunPlan
from nemo_fabric import RunRequest
from nemo_fabric import RunResult
from nemo_fabric import Runtime
from nemo_fabric import RuntimeCapabilities
from nemo_fabric import RuntimeConfig
from nemo_fabric import RuntimeHandle
from nemo_fabric import SkillConfig
from nemo_fabric import TelemetryConfig
from nemo_fabric import ToolsConfig
from nemo_fabric.types import _FabricConfigSnapshot
from nemo_fabric.types import _ToolsConfig
from pydantic import ValidationError


def test_public_contract_has_no_unreleased_aliases():
    assert list(signature(Fabric).parameters) == []
    assert not hasattr(Fabric, "__aenter__")
    assert not hasattr(Fabric, "__aexit__")
    assert not hasattr(RunRequest, "from_text")
    assert not hasattr(nemo_fabric, "RunRequestModel")
    for name in ("plan_config", "run_config", "doctor_config", "start", "start_config"):
        assert not hasattr(Fabric, name)

    assert not hasattr(Fabric, "resolve")
    for name in ("plan", "doctor", "run", "start_runtime"):
        assert not get_overloads(getattr(Fabric, name)), name

    assert not hasattr(fabric_errors, "FabricCliError")


def test_typed_config_validates_required_fields_and_preserves_extensions():
    raw = {
        "schema_version": "fabric.agent/v1alpha1",
        "metadata": {"name": "demo", "owner": "sdk"},
        "harness": {"adapter_id": "test.fabric.shim", "future": True},
        "runtime": {},
        "future_top_level": {"enabled": True},
    }

    config = FabricConfig.from_mapping(raw)
    raw["metadata"]["name"] = "mutated"

    assert isinstance(config.metadata, MetadataConfig)
    assert config.environment is None
    assert config.metadata.name == "demo"
    assert config.metadata.description is None
    assert "transport" not in config.runtime.to_mapping()
    assert config.metadata.extra_fields == {"owner": "sdk"}
    assert config.harness.extra_fields == {"future": True}
    assert config.extra_fields == {"future_top_level": {"enabled": True}}
    assert config.to_mapping()["future_top_level"] == {"enabled": True}
    assert "models" not in config.to_mapping()

    runtime = RuntimeConfig(input_schema="http")
    config.runtime = runtime
    config.future_runtime = {"enabled": True}
    assert isinstance(config.runtime, RuntimeConfig)
    assert config.extra_fields["future_runtime"] == {"enabled": True}

    with pytest.raises(ValidationError, match="metadata"):
        FabricConfig.from_mapping({"harness": {"adapter_id": "test.fabric.shim"}})
    with pytest.raises(ValidationError, match="adapter_id"):
        HarnessConfig(adapter_id="")
    with pytest.raises(ValidationError, match="settings"):
        HarnessConfig(
            adapter_id="test.fabric.shim",
            settings=[],  # type: ignore[arg-type]
        )
    with pytest.raises(ValidationError, match="settings"):
        EnvironmentConfig(settings=[])  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="runtime"):
        FabricConfig(
            metadata=MetadataConfig(name="demo"),
            harness=HarnessConfig(adapter_id="test.fabric.shim"),
            runtime=[],  # type: ignore[arg-type]
        )
    with pytest.raises(ValidationError, match="models"):
        FabricConfig(
            metadata=MetadataConfig(name="demo"),
            harness=HarnessConfig(adapter_id="test.fabric.shim"),
            models=[],  # type: ignore[arg-type]
        )


def test_typed_config_authoring_helpers_emit_schema_shape():
    config = FabricConfig(
        metadata=MetadataConfig(name="demo"),
        harness=HarnessConfig(adapter_id="test.fabric.shim"),
        models={
            "default": {
                "provider": "test",
                "model": "test-model",
            }
        },
    )

    config.add_skill_path("./skills/review").add_skill_path("./skills/review")
    config.add_mcp_server(
        "github",
        transport="streamable-http",
        url="${GITHUB_MCP_URL}",
        exposure="fabric_managed",
    )
    config.enable_relay(
        project="fabric-tests",
        output_dir="./artifacts/relay",
    )
    config.block_tools("browser", "shell", "browser")

    assert isinstance(config.mcp, McpConfig)
    assert isinstance(config.skills, SkillConfig)
    assert isinstance(config.telemetry, TelemetryConfig)
    assert isinstance(config.tools, ToolsConfig)

    assert config.to_mapping()["tools"] == {"blocked": ["browser", "shell"]}
    assert config.to_mapping()["skills"] == {"paths": ["./skills/review"]}
    assert config.to_mapping()["mcp"] == {
        "servers": {
            "github": {
                "transport": "streamable-http",
                "url": "${GITHUB_MCP_URL}",
                "exposure": "fabric_managed",
            }
        }
    }
    assert config.to_mapping()["telemetry"] == {
        "providers": {"relay": {}},
    }
    assert config.to_mapping()["relay"] == {
        "project": "fabric-tests",
        "output_dir": "./artifacts/relay",
        "components": [],
    }

    config.remove_mcp_server("github").remove_mcp_server("missing")
    config.remove_skill_path("./skills/review").remove_skill_path("./skills/missing")
    assert config.mcp is None
    assert config.skills is None
    assert "mcp" not in config.to_mapping()
    assert "skills" not in config.to_mapping()

    with pytest.raises(ValidationError, match="exposure"):
        config.add_mcp_server(
            "bad",
            transport="streamable-http",
            url="http://example.invalid",
            exposure="sideways",
        )
    with pytest.raises(ValidationError, match="providers"):
        TelemetryConfig(providers={"sideways": {}})


def test_typed_tools_config_serializes_blocked_policy():
    config = FabricConfig(
        metadata=MetadataConfig(name="demo"),
        harness=HarnessConfig(adapter_id="test.fabric.shim"),
        tools=ToolsConfig(blocked=["browser"]),
    )

    config.block_tools("shell", "browser")

    assert config.to_mapping()["tools"] == {"blocked": ["browser", "shell"]}


def test_run_plan_config_block_tools_emits_canonical_shape():
    config = _FabricConfigSnapshot.from_mapping(_plan()["config"])

    config.block_tools("browser", "shell", "browser")

    assert config.to_mapping()["tools"] == {"blocked": ["browser", "shell"]}


def test_run_plan_tools_config_rejects_scalar_blocked_value():
    with pytest.raises(FabricConfigError, match="tools blocked"):
        _ToolsConfig(blocked="browser")  # type: ignore[arg-type]


def test_fabric_config_authors_first_class_relay_observability():
    config = _fabric_config()

    config.enable_relay(
        output_dir="./artifacts/relay",
        observability=RelayObservabilityConfig(
            atof=RelayAtofConfig(
                enabled=True,
                output_directory="./artifacts/relay",
                filename="events.atof.jsonl",
                mode="overwrite",
            ),
            atif=RelayAtifConfig(
                enabled=True,
                output_directory="./artifacts/relay",
                filename_template="trajectory-{session_id}.atif.json",
                agent_name="fabric-tests",
            ),
        ),
        components=[
            RelayComponentConfig(kind="switchyard", config={"route": "canary"}),
        ],
        policy=RelayConfigPolicy(unknown_component="error"),
    )

    assert config.to_mapping()["telemetry"] == {
        "providers": {"relay": {}},
    }
    assert config.to_mapping()["relay"] == {
        "output_dir": "./artifacts/relay",
        "observability": {
            "version": 1,
            "atof": {
                "enabled": True,
                "output_directory": "./artifacts/relay",
                "filename": "events.atof.jsonl",
                "mode": "overwrite",
            },
            "atif": {
                "enabled": True,
                "agent_name": "fabric-tests",
                "model_name": "unknown",
                "output_directory": "./artifacts/relay",
                "filename_template": "trajectory-{session_id}.atif.json",
            },
        },
        "components": [
            {
                "kind": "switchyard",
                "enabled": True,
                "config": {"route": "canary"},
            },
        ],
        "policy": {
            "unknown_component": "error",
            "unknown_field": "warn",
            "unsupported_value": "error",
        },
    }


def test_fabric_config_enable_relay_preserves_omitted_fields():
    config = _fabric_config()

    config.enable_relay(
        project="fabric-tests",
        output_dir="./artifacts/relay",
        observability={"atif": {"enabled": True}},
        components=[{"kind": "switchyard"}],
    )
    initial = config.to_mapping()["relay"]
    config.enable_relay(policy={"unknown_component": "error"})

    relay = config.to_mapping()["relay"]
    assert relay["project"] == initial["project"]
    assert relay["output_dir"] == initial["output_dir"]
    assert relay["observability"] == initial["observability"]
    assert relay["components"] == initial["components"]
    assert relay["policy"]["unknown_component"] == "error"

    config.enable_relay(components=[])
    assert config.to_mapping()["relay"]["components"] == []


def test_telemetry_config_enable_native_preserves_existing_config():
    telemetry = TelemetryConfig()

    telemetry.enable_native(config={"components": [{"kind": "observability"}]})
    telemetry.enable_native()

    assert telemetry.to_mapping()["providers"]["native"]["config"] == {
        "components": [{"kind": "observability"}],
    }


def test_config_emits_schema_shape_and_validates():
    config = FabricConfig(
        metadata={"name": "demo", "owner": "sdk"},
        harness={"adapter_id": "test.fabric.shim", "future": True},
        models={
            "default": {
                "provider": "test",
                "model": "test-model",
                "temperature": 0.0,
            }
        },
        future_top_level={"enabled": True},
    )
    config.add_skill_path("./skills/review")
    config.add_mcp_server(
        "github",
        transport="streamable-http",
        url="${GITHUB_MCP_URL}",
        exposure="fabric_managed",
    )
    config.enable_relay(project="fabric-tests", output_dir="./artifacts/relay")

    emitted = config.to_mapping()

    assert emitted["schema_version"] == "fabric.agent/v1alpha1"
    assert emitted["metadata"]["owner"] == "sdk"
    assert emitted["harness"]["future"] is True
    assert emitted["runtime"] == {}
    assert emitted["models"]["default"]["model"] == "test-model"
    assert emitted["skills"] == {"paths": ["./skills/review"]}
    assert emitted["mcp"]["servers"]["github"]["exposure"] == "fabric_managed"
    assert emitted["telemetry"]["providers"] == {"relay": {}}
    assert emitted["relay"] == {
        "project": "fabric-tests",
        "output_dir": "./artifacts/relay",
        "components": [],
    }
    assert config.extra_fields == {"future_top_level": {"enabled": True}}

    normalized = FabricConfig.model_validate(config)
    assert normalized.to_mapping()["future_top_level"] == {"enabled": True}

    with pytest.raises(ValidationError):
        FabricConfig(metadata={"name": "missing-harness"})  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        config.add_mcp_server(
            "bad",
            transport="streamable-http",
            url="http://example.invalid",
            exposure="sideways",  # type: ignore[arg-type]
        )


def test_agent_model_tracks_rust_schema_top_level_fields():
    schema = json.loads(Path("schemas/agent.schema.json").read_text(encoding="utf-8"))
    pydantic_schema = FabricConfig.model_json_schema()

    assert set(pydantic_schema["properties"]).issuperset(schema["properties"])
    assert set(pydantic_schema["required"]) == {"metadata", "harness"}
    assert set(schema["required"]) == {"schema_version", "metadata", "harness", "runtime"}


def test_environment_model_defines_extension_field_ownership():
    properties = EnvironmentConfig.model_json_schema()["properties"]

    assert "environment provider" in properties["settings"]["description"]
    assert "without Fabric semantics" in properties["metadata"]["description"]
    assert "existing environment" in properties["connection"]["description"]
    assert "environment teardown" in properties["ownership"]["description"]
    assert "outside or inside" in properties["control_location"]["description"]


def test_inspection_models_are_typed_read_only_mappings():
    plan = RunPlan.from_mapping(
        {
            "agent_name": "demo",
            "base_dir": ".",
            "config": {
                "metadata": {"name": "demo"},
                "harness": {"adapter_id": "test.fabric.shim"},
                "runtime": {"input_schema": "chat"},
            },
            "adapter_descriptor": {
                "descriptor": {
                    "adapter_id": "test.fabric.shim",
                    "harness": "hermes",
                    "adapter_kind": "python",
                    "future": "value",
                }
            },
            "capabilities": {
                "service": False,
                "streaming": False,
                "updates": False,
                "cancellation": False,
                "future_capability": "declared",
            },
        }
    )

    assert isinstance(plan.adapter, AdapterInfo)
    assert isinstance(plan.capabilities, RuntimeCapabilities)
    assert plan.base_dir == Path(".")
    assert plan.adapter.harness == "hermes"
    assert "harness_type" not in plan.adapter
    assert plan.adapter.extra_fields["future"] == "value"
    assert plan.capabilities.extra_fields["future_capability"] == "declared"
    resolved = plan.to_mapping()
    plan.config.metadata.name = "mutated"
    assert plan.to_mapping() == resolved
    with pytest.raises(TypeError):
        plan["agent_name"] = "mutated"  # type: ignore[index]


def test_run_plan_config_rejects_removed_profiles_and_missing_base_dir():
    with pytest.raises(FabricConfigError, match="profiles are no longer supported"):
        _FabricConfigSnapshot.from_mapping(
            {
                "metadata": {"name": "demo"},
                "harness": {"adapter_id": "test.fabric.shim"},
                "profiles": {"review": {}},
            }
        )

    with pytest.raises(FabricConfigError, match="base_dir is required"):
        RunPlan.from_mapping(
            {
                "agent_name": "demo",
                "config": {
                    "metadata": {"name": "demo"},
                    "harness": {"adapter_id": "test.fabric.shim"},
                },
                "adapter": {
                    "adapter_id": "test.fabric.shim",
                    "harness": "shim",
                    "adapter_kind": "python",
                },
                "capabilities": {},
            }
        )


def test_runtime_handle_distinguishes_contract_and_extension_fields():
    handle = RuntimeHandle.from_mapping(
        {
            "runtime_id": "runtime-1",
            "runtime_binding": "binding-1",
            "agent_name": "demo",
            "harness": "hermes",
            "adapter_kind": "python",
            "adapter_id": "test.fabric.shim",
            "environment": {
                "environment_id": "environment-1",
                "provider": "local",
                "control_location": "external_control",
                "ownership": "caller_owned",
            },
            "future_handle_field": "value",
        }
    )

    assert handle.extra_fields == {"future_handle_field": "value"}


@pytest.mark.parametrize(
    "field",
    (
        "runtime_id",
        "runtime_binding",
        "agent_name",
        "harness",
        "adapter_kind",
        "environment",
    ),
)
def test_runtime_handle_requires_native_contract_fields(field):
    raw = _runtime()
    del raw[field]

    with pytest.raises(FabricConfigError, match=field.replace("_", " ")):
        RuntimeHandle.from_mapping(raw)


def test_run_plan_config_enable_relay_preserves_existing_relay_fields():
    config = _FabricConfigSnapshot.from_mapping(_plan()["config"])

    config.enable_relay(
        output_dir="./artifacts/relay",
        observability={"atif": {"enabled": True}},
    )
    config.enable_relay(policy={"unknown_component": "error"})

    assert config.to_mapping()["relay"] == {
        "output_dir": "./artifacts/relay",
        "observability": {"atif": {"enabled": True}},
        "policy": {"unknown_component": "error"},
    }


def test_run_plan_config_enable_native_preserves_existing_native_config():
    config = _FabricConfigSnapshot.from_mapping(_plan()["config"])

    config.telemetry.enable_native(config={"components": [{"kind": "observability"}]})
    config.telemetry.enable_native()

    assert config.to_mapping()["telemetry"]["providers"]["native"]["config"] == {
        "components": [{"kind": "observability"}],
    }


def test_runtime_capabilities_reject_non_boolean_values():
    with pytest.raises(FabricConfigError, match="streaming capability"):
        RuntimeCapabilities.from_mapping({"streaming": "false"})


def test_doctor_report_and_errors_expose_typed_contract_fields():
    report = DoctorReport.from_mapping(
        {
            "agent_name": "demo",
            "status": "warn",
            "checks": [
                {
                    "name": "runtime.adapter",
                    "status": "warn",
                    "message": "not implemented",
                }
            ],
        }
    )
    error = FabricRuntimeError(
        "invoke failed",
        stage="invoke",
        code="adapter_failed",
        retryable=True,
        details={"adapter_id": "test.fabric.shim"},
    )

    assert report.checks[0].name == "runtime.adapter"
    assert error.stage == "invoke"
    assert error.code == "adapter_failed"
    assert error.retryable is True
    assert error.details == {"adapter_id": "test.fabric.shim"}


def _plan() -> dict[str, Any]:
    config = {
        "metadata": {"name": "demo"},
        "harness": {"adapter_id": "test.fabric.shim"},
        "runtime": {
            "input_schema": "chat",
            "output_schema": "message",
        },
    }
    return {
        "agent_name": "demo",
        "base_dir": ".",
        "config": config,
        "adapter_descriptor": {
            "descriptor": {
                "adapter_kind": "python",
                "adapter_id": "test.fabric.shim",
                "harness": "hermes",
            }
        },
        "capabilities": {
            "service": False,
            "streaming": False,
            "updates": False,
            "cancellation": False,
        },
    }


def _runtime() -> dict[str, Any]:
    return {
        "runtime_id": "runtime-1",
        "runtime_binding": "fabric-runtime-binding-test",
        "agent_name": "demo",
        "harness": "hermes",
        "adapter_kind": "python",
        "adapter_id": "test.fabric.shim",
        "environment": {
            "environment_id": "environment-1",
            "provider": "local",
            "control_location": "external_control",
            "ownership": "caller_owned",
        },
    }


def _run_result(**updates: Any) -> dict[str, Any]:
    result = {
        "agent_name": "demo",
        "harness": "hermes",
        "adapter_kind": "python",
        "adapter_id": "test.fabric.shim",
        "runtime_id": "runtime-1",
        "invocation_id": "invocation-1",
        "request_id": "request-1",
        "status": "succeeded",
        "output": None,
        "artifacts": {"artifacts": []},
        "events": [],
    }
    result.update(updates)
    return result


def _fabric_config() -> FabricConfig:
    return FabricConfig(
        metadata=MetadataConfig(name="demo"),
        harness=HarnessConfig(adapter_id="test.fabric.shim"),
        runtime=RuntimeConfig(),
    )


class NativeRecorder:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.config_base_dir_calls: list[str | None] = []
        self.stopped = 0
        self.fail_invoke = False

    def plan_config(
        self,
        config_json: str,
        base_dir: str | None = None,
    ) -> str:
        assert json.loads(config_json)["metadata"]["name"] == "demo"
        self.config_base_dir_calls.append(base_dir)
        return json.dumps(_plan())

    def start_runtime(self, plan_json: str) -> str:
        assert json.loads(plan_json)["agent_name"] == "demo"
        return json.dumps(_runtime())

    def invoke_runtime(self, plan_json: str, runtime_json: str, request_json: str) -> str:
        if self.fail_invoke:
            raise RuntimeError("native invoke failed")
        request = json.loads(request_json)
        self.requests.append(request)
        return json.dumps(
            {
                "agent_name": "demo",
                "harness": "hermes",
                "adapter_kind": "python",
                "adapter_id": "test.fabric.shim",
                "runtime_id": json.loads(runtime_json)["runtime_id"],
                "invocation_id": "invocation-1",
                "request_id": request["request_id"],
                "status": "failed" if request["input"] == "fail" else "succeeded",
                "output": {"received": request["input"]},
                "error": {
                    "stage": "invoke",
                    "code": "adapter_failed",
                    "message": "adapter failed",
                    "retryable": False,
                }
                if request["input"] == "fail"
                else None,
                "artifacts": {"artifacts": []},
                "events": [
                    {
                        "event_id": "event-1",
                        "timestamp_millis": 1,
                        "kind": "invocation_end",
                        "message": "completed",
                    }
                ],
            }
        )

    def stop_runtime(self, plan_json: str, runtime_json: str) -> str:
        self.stopped += 1
        return json.dumps([])


class NativeClient(Fabric):
    def __init__(self, native: NativeRecorder) -> None:
        super().__init__()
        self.native = native

    def _native_module(self) -> NativeRecorder:
        return self.native

    def _require_native_module(self, method: str) -> NativeRecorder:
        return self.native


def test_run_request_is_validated_and_json_safe():
    context = {"run_id": "run-1", "labels": ["sdk"]}
    overrides = {"temperature": 0, "limits": {"turns": 1}}
    request = RunRequest(
        input={"messages": [{"role": "user", "content": "hello"}]},
        request_id="request-1",
        context=context,
        overrides=overrides,
    )
    context["labels"].append("mutated")
    overrides["limits"]["turns"] = 2

    assert request.request_id == "request-1"
    assert request.to_mapping()["input"] == {"messages": [{"role": "user", "content": "hello"}]}
    assert request.to_mapping()["context"] == {"run_id": "run-1", "labels": ["sdk"]}
    assert request.to_mapping()["overrides"] == {
        "temperature": 0,
        "limits": {"turns": 1},
    }

    copied = request.to_mapping()
    copied["context"]["run_id"] = "changed"
    assert request.to_mapping()["context"] == {"run_id": "run-1", "labels": ["sdk"]}


def test_run_request_from_mapping_copies_and_validates_context():
    raw = {
        "input": "hello",
        "request_id": "request-1",
        "context": {"job_id": "job-1"},
    }

    request = RunRequest.from_mapping(raw)
    raw["context"]["job_id"] = "mutated"

    assert request.input == "hello"
    assert request.context == {"job_id": "job-1"}

    with pytest.raises(ValidationError, match="request context"):
        RunRequest.from_mapping({"input": "bad", "context": "not-a-mapping"})


def test_run_request_constructor_validates_context_and_overrides():
    with pytest.raises(ValidationError, match="request context"):
        RunRequest(input="bad", context="not-a-mapping")  # type: ignore[arg-type]

    with pytest.raises(ValidationError, match="request overrides"):
        RunRequest(input="bad", overrides="not-a-mapping")  # type: ignore[arg-type]

    with pytest.raises(ValidationError, match="request context"):
        RunRequest(input="bad", context=[])  # type: ignore[arg-type]

    with pytest.raises(ValidationError, match="JSON-compatible"):
        RunRequest(input="bad", future_request=object())

    with pytest.raises(ValidationError, match="finite"):
        RunRequest(input=float("nan"))


def test_run_request_constructor_generates_request_metadata():
    request = RunRequest(input="hello")

    assert request.input == "hello"
    assert request.request_id.startswith("request-")
    assert request.context == {}


def test_run_request_preserves_extension_fields():
    request = RunRequest(
        input={"messages": [{"role": "user", "content": "hello"}]},
        request_id="request-1",
        context={"job_id": "job-1"},
        future_request={"enabled": True},
    )

    assert request.to_mapping()["input"] == {"messages": [{"role": "user", "content": "hello"}]}
    assert request.context == {"job_id": "job-1"}
    assert request.extra_fields["future_request"] == {"enabled": True}


@pytest.mark.parametrize("value", [{}, []])
def test_run_request_preserves_empty_structured_input(value):
    assert RunRequest(input=value).to_mapping()["input"] == value


def test_run_request_defaults_missing_input_to_empty_text():
    assert RunRequest().to_mapping()["input"] == ""


def test_run_result_wraps_nested_error_and_keeps_mapping_access():
    result = RunResult.from_mapping(
        _run_result(
            status="failed",
            output={},
            error={
                "stage": "invoke",
                "code": "adapter_failed",
                "message": "adapter failed",
                "retryable": False,
            },
            events=[{"kind": "log", "message": "hello"}],
        )
    )

    assert result["status"] == "failed"
    assert result.status == "failed"
    assert result.error.code == "adapter_failed"
    assert result.error["stage"] == "invoke"
    assert result.artifacts.artifacts == ()
    assert result.events[0].kind == "log"
    assert result.to_dict()["error"]["code"] == "adapter_failed"


def test_run_result_exposes_detached_json_values():
    result = RunResult.from_mapping(
        _run_result(
            output={"plugins": ["observability/nemo_relay"]},
            metadata={"labels": ["sdk"]},
            future={"values": [1]},
        )
    )

    output = result.output
    metadata = result.metadata
    future = result.extra_fields["future"]

    assert output["plugins"] == ["observability/nemo_relay"]
    assert metadata["labels"] == ["sdk"]
    assert future["values"] == [1]

    output["plugins"].append("mutated")
    metadata["labels"].append("mutated")
    future["values"].append(2)

    assert result.output.to_mapping() == {"plugins": ["observability/nemo_relay"]}
    assert result.metadata == {"labels": ["sdk"]}
    assert result.extra_fields["future"] == {"values": [1]}


def test_run_output_exposes_response_and_preserves_extensions():
    output = RunOutput.from_mapping(
        {
            "response": "hello",
            "thread_id": "abc",
        }
    )

    assert output.response == "hello"
    assert output["response"] == "hello"
    assert output["thread_id"] == "abc"
    assert output.to_mapping() == {
        "response": "hello",
        "thread_id": "abc",
    }


def test_run_result_wraps_object_output_as_run_output():
    result = RunResult.from_mapping(_run_result(output={"response": "hello", "usage": {"tokens": 1}}))

    assert isinstance(result.output, RunOutput)
    assert result.output.response == "hello"
    assert result.output["response"] == "hello"
    assert result.output["usage"] == {"tokens": 1}
    assert result.to_mapping()["output"] == {
        "response": "hello",
        "usage": {"tokens": 1},
    }


def test_run_output_omits_missing_response_from_mapping():
    output = RunOutput.from_mapping({"thread_id": "abc"})

    assert output.response is None
    assert "response" not in output.to_mapping()
    assert output.to_mapping() == {"thread_id": "abc"}


def test_run_output_preserves_explicit_null_response():
    output = RunOutput.from_mapping({"response": None})

    assert output.response is None
    assert output.to_mapping() == {"response": None}


def test_run_output_preserves_non_string_response_without_raising():
    output = RunOutput.from_mapping({"response": {"text": "hello"}})

    assert output.response == {"text": "hello"}
    assert output["response"] == {"text": "hello"}
    assert output.to_mapping() == {"response": {"text": "hello"}}


def test_run_result_preserves_structured_response_from_core_valid_output():
    result = RunResult.from_mapping(_run_result(output={"response": {"text": "hello"}, "usage": {"tokens": 1}}))

    assert isinstance(result.output, RunOutput)
    assert result.output.response == {"text": "hello"}
    assert result.output["response"] == {"text": "hello"}
    assert result.to_mapping()["output"] == {
        "response": {"text": "hello"},
        "usage": {"tokens": 1},
    }


def test_run_result_preserves_non_object_output():
    result = RunResult.from_mapping(_run_result(output="hello"))

    assert result.output == "hello"
    assert result.to_mapping()["output"] == "hello"


def test_run_result_normalizes_core_telemetry_reference():
    result = RunResult.from_mapping(
        _run_result(
            telemetry={
                "relay_enabled": True,
                "metadata": {
                    "relay_output_dir": "/tmp/relay",
                    "trace_id": "trace-1",
                },
            },
        )
    )

    assert result.telemetry[0].provider == "relay"
    assert result.telemetry[0].kind == "trace"
    assert result.telemetry[0].uri == "/tmp/relay"
    assert result.telemetry[0].trace_id == "trace-1"


def test_run_result_preserves_native_telemetry_provider():
    result = RunResult.from_mapping(
        _run_result(
            telemetry={
                "relay_enabled": False,
                "metadata": {"telemetry_providers": ["native"]},
            },
        )
    )

    assert result.telemetry[0].provider == "native"
    assert result.telemetry[0].metadata["relay_enabled"] is False


@pytest.mark.parametrize(
    "field",
    (
        "agent_name",
        "harness",
        "adapter_kind",
        "runtime_id",
        "invocation_id",
        "request_id",
        "status",
    ),
)
def test_run_result_requires_schema_identity_fields(field):
    raw = _run_result()
    del raw[field]

    with pytest.raises(FabricConfigError, match=field.replace("_", " ")):
        RunResult.from_mapping(raw)


async def test_run_accepts_full_run_request():
    native = NativeRecorder()
    client = NativeClient(native)

    result = await client.run(
        _fabric_config(),
        request=RunRequest(
            input="hello",
            request_id="request-4",
            context={"job_id": "job-4"},
            overrides={"request": True},
        ),
    )

    assert isinstance(result, RunResult)
    assert result.status == "succeeded"
    assert native.requests[0] == {
        "input": "hello",
        "request_id": "request-4",
        "context": {"job_id": "job-4"},
        "overrides": {"request": True},
    }


async def test_typed_source_accepts_run_request_and_returns_result():
    native = NativeRecorder()
    client = NativeClient(native)
    result = await client.run(
        _fabric_config(),
        request=RunRequest(
            input="hello",
            request_id="request-1",
            context={"job_id": "job-1"},
            overrides={"max_iterations": 1},
        ),
    )

    assert isinstance(result, RunResult)
    assert result.status == "succeeded"
    assert result["request_id"] == "request-1"
    assert native.requests[0] == {
        "input": "hello",
        "request_id": "request-1",
        "context": {"job_id": "job-1"},
        "overrides": {"max_iterations": 1},
    }


async def test_native_runtime_errors_use_typed_exception_and_stop_runtime():
    native = NativeRecorder()
    native.fail_invoke = True
    client = NativeClient(native)

    with pytest.raises(FabricRuntimeError, match="native invoke failed") as error:
        await client.run(_fabric_config(), input="hello")

    assert isinstance(error.value, FabricError)
    assert isinstance(error.value.__cause__, RuntimeError)
    assert native.stopped == 1


def test_public_sdk_exceptions_share_a_common_base():
    assert issubclass(FabricConfigError, FabricError)
    assert issubclass(FabricRuntimeError, FabricError)
    assert issubclass(FabricStateError, FabricError)
    assert issubclass(FabricCapabilityError, FabricError)
    assert issubclass(FabricNativeUnavailableError, FabricError)


async def test_runtime_invoke_accepts_run_request():
    native = NativeRecorder()
    runtime = Runtime(
        client=NativeClient(native),
        plan=_plan(),
        runtime=_runtime(),
        overrides={"runtime": True, "limits": {"runtime": 1}},
    )

    result = await runtime.invoke(
        request=RunRequest(
            input="hello",
            request_id="request-2",
            context={"job_id": "job-2"},
            overrides={"request": True, "limits": {"request": 1}},
        ),
    )

    assert isinstance(result, RunResult)
    assert result.request_id == "request-2"
    assert native.requests[0] == {
        "input": "hello",
        "request_id": "request-2",
        "context": {"job_id": "job-2"},
        "overrides": {
            "runtime": True,
            "request": True,
            "limits": {"runtime": 1, "request": 1},
        },
    }


async def test_runtime_handle_is_typed_and_detached():
    runtime = Runtime(
        client=NativeClient(NativeRecorder()),
        plan=RunPlan.from_mapping(_plan()),
        runtime=_runtime(),
    )

    assert isinstance(runtime.handle, RuntimeHandle)
    assert runtime.handle.harness == "hermes"
    assert runtime.handle.adapter_id == "test.fabric.shim"
    assert runtime.handle is not runtime.handle


async def test_run_rejects_multiple_primary_input_sources():
    client = NativeClient(NativeRecorder())

    with pytest.raises(FabricConfigError, match="mutually exclusive"):
        await client.run(
            _fabric_config(),
            input="hello",
            request={"input": "request"},
        )


async def test_run_rejects_raw_mapping_request():
    client = NativeClient(NativeRecorder())

    with pytest.raises(FabricConfigError, match="request must be a RunRequest"):
        await client.run(
            _fabric_config(),
            request={"input": "request"},  # type: ignore[arg-type]
        )


async def test_unified_agent_source_dispatches_fabric_config_to_runtime_path():
    native = NativeRecorder()
    client = NativeClient(native)

    result = await client.run(
        _fabric_config(),
        request=RunRequest(input="hello", request_id="request-5"),
    )

    assert result.request_id == "request-5"
    assert native.requests[0]["input"] == "hello"


async def test_lifecycle_methods_reject_raw_mapping_agent_source():
    native = NativeRecorder()
    client = NativeClient(native)

    with pytest.raises(FabricConfigError, match="FabricConfig.from_mapping"):
        await client.run({"metadata": {"name": "demo"}}, input="hello")

    assert native.requests == []


def test_config_methods_accept_real_pydantic_models_and_reject_lookalikes():
    class ModelDumpLike:
        def model_dump(self, *, mode: str, exclude_none: bool) -> dict[str, Any]:
            return {"metadata": {"name": "demo"}}

    native = NativeRecorder()
    client = NativeClient(native)

    with pytest.raises(FabricConfigError, match="FabricConfig.from_mapping"):
        client.plan({"metadata": {"name": "demo"}})

    with pytest.raises(FabricConfigError, match="FabricConfig"):
        client.plan(ModelDumpLike())

    config = FabricConfig(
        metadata={"name": "demo"},
        harness={"adapter_id": "test.fabric.shim"},
    )
    client.plan(config, base_dir=".")

    assert native.config_base_dir_calls == ["."]


def test_fabric_config_constructors_emit_schema_shaped_mappings():
    config = FabricConfig(
        metadata=MetadataConfig(name="demo"),
        harness=HarnessConfig(
            adapter_id="test.fabric.shim",
            resolution="preinstalled",
            settings={"workspace": "./ws"},
        ),
        runtime=RuntimeConfig(
            input_schema="chat",
            output_schema="message",
        ),
    )
    copied = config.to_mapping()
    copied["harness"]["settings"]["workspace"] = "mutated"

    assert config.schema_version == "fabric.agent/v1alpha1"
    assert config.metadata.to_mapping() == {"name": "demo"}
    assert config.harness.adapter_id == "test.fabric.shim"
    assert config.runtime.input_schema == "chat"
    assert config.harness.settings["workspace"] == "./ws"

    client = NativeClient(NativeRecorder())
    client.plan(config)


async def test_start_runtime_returns_runtime_with_typed_handle():
    runtime = await NativeClient(NativeRecorder()).start_runtime(_fabric_config())

    assert runtime.runtime_id == "runtime-1"
    assert isinstance(runtime.handle, RuntimeHandle)


async def test_runtime_state_errors_use_sdk_error_hierarchy():
    runtime = Runtime(
        client=NativeClient(NativeRecorder()),
        plan=_plan(),
        runtime=_runtime(),
    )
    await runtime.stop()

    with pytest.raises(FabricStateError, match="cannot invoke a stopped runtime"):
        await runtime.invoke(input="hello")
