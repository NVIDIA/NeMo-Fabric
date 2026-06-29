# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the public Python SDK request/result contract."""

from __future__ import annotations

import json
from inspect import signature
from typing import Any, get_overloads

import pytest

import nemo_fabric.errors as fabric_errors

from nemo_fabric import (
    AdapterInfo,
    DoctorReport,
    EffectiveConfig,
    EnvironmentConfig,
    FabricClient,
    FabricCapabilityError,
    FabricConfig,
    FabricConfigError,
    FabricError,
    FabricProfileConfig,
    FabricNativeUnavailableError,
    FabricRuntimeError,
    FabricStateError,
    HarnessConfig,
    MetadataConfig,
    RunPlan,
    RunRequest,
    RunResult,
    RuntimeCapabilities,
    RuntimeConfig,
    RuntimeHandle,
    RuntimeUpdate,
    Session,
    SessionInfo,
)


def test_public_contract_has_no_unreleased_aliases():
    assert list(signature(FabricClient).parameters) == []
    assert not hasattr(RunRequest, "from_text")
    for name in ("plan_config", "run_config", "doctor_config", "start", "start_config"):
        assert not hasattr(FabricClient, name)

    for name in ("resolve", "plan", "doctor", "run", "start_session", "start_service"):
        assert len(get_overloads(getattr(FabricClient, name))) == 2, name

    assert not hasattr(fabric_errors, "FabricCliError")


def test_typed_config_validates_required_fields_and_preserves_extensions():
    raw = {
        "schema_version": "fabric.agent/v1alpha1",
        "metadata": {"name": "demo", "owner": "sdk"},
        "harness": {"adapter_id": "test.fabric.shim", "future": True},
        "runtime": {"mode": "session"},
        "future_top_level": {"enabled": True},
    }

    config = FabricConfig.from_mapping(raw)
    raw["metadata"]["name"] = "mutated"

    assert isinstance(config.metadata, MetadataConfig)
    assert config.environment is None
    assert config.metadata.name == "demo"
    assert config.metadata.description is None
    assert config.runtime.transport is None
    assert "transport" not in config.runtime.to_mapping()
    assert config.metadata.extra_fields == {"owner": "sdk"}
    assert config.harness.extra_fields == {"future": True}
    assert config.extra_fields == {"future_top_level": {"enabled": True}}
    assert config.to_mapping()["future_top_level"] == {"enabled": True}
    assert "models" not in config.to_mapping()

    runtime = RuntimeConfig(mode="service")
    config.runtime = runtime
    config["future_runtime"] = {"enabled": True}
    assert isinstance(config.runtime, RuntimeConfig)
    assert config.extra_fields["future_runtime"] == {"enabled": True}

    with pytest.raises(TypeError):
        FabricConfig(  # type: ignore[call-arg]
            metadata=MetadataConfig(name="demo"),
            harness=HarnessConfig(adapter_id="test.fabric.shim"),
            unexpected=True,
        )
    with pytest.raises(FabricConfigError, match="metadata"):
        FabricConfig.from_mapping({"harness": {"adapter_id": "test.fabric.shim"}})
    with pytest.raises(FabricConfigError, match="adapter_id"):
        HarnessConfig(adapter_id="")
    with pytest.raises(FabricConfigError, match="runtime mode"):
        RuntimeConfig(mode="invalid")
    with pytest.raises(FabricConfigError, match="harness settings"):
        HarnessConfig(
            adapter_id="test.fabric.shim",
            settings=[],  # type: ignore[arg-type]
        )
    with pytest.raises(FabricConfigError, match="extra_fields"):
        MetadataConfig(name="demo", extra_fields=[])  # type: ignore[arg-type]
    with pytest.raises(FabricConfigError, match="environment settings"):
        EnvironmentConfig(settings=[])  # type: ignore[arg-type]
    with pytest.raises(FabricConfigError, match="runtime must be"):
        FabricConfig(
            metadata=MetadataConfig(name="demo"),
            harness=HarnessConfig(adapter_id="test.fabric.shim"),
            runtime=[],  # type: ignore[arg-type]
        )
    with pytest.raises(FabricConfigError, match="models"):
        FabricConfig(
            metadata=MetadataConfig(name="demo"),
            harness=HarnessConfig(adapter_id="test.fabric.shim"),
            models=[],  # type: ignore[arg-type]
        )


def test_typed_profile_preserves_partial_overlay_sections():
    profile = FabricProfileConfig.from_mapping(
        {
            "name": "session",
            "harness": {"settings": {"timeout_seconds": 30}},
            "runtime": {"mode": "session"},
        }
    )

    assert profile.to_mapping()["harness"] == {
        "settings": {"timeout_seconds": 30}
    }
    assert profile.to_mapping()["runtime"] == {"mode": "session"}


def test_inspection_models_are_typed_read_only_mappings():
    plan = RunPlan.from_mapping(
        {
            "agent_name": "demo",
            "profiles": ["runtime", "telemetry"],
            "effective_config": {
                "agent_name": "demo",
                "profiles": ["runtime", "telemetry"],
                "agent_root": ".",
                "config_path": "agent.yaml",
                "config_root": ".",
                "config": {
                    "metadata": {"name": "demo"},
                    "harness": {"adapter_id": "test.fabric.shim"},
                    "runtime": {"mode": "session"},
                },
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
                "session": True,
                "service": False,
                "streaming": False,
                "updates": False,
                "cancellation": False,
                "concurrent_invocations": False,
                "future_capability": "declared",
            },
        }
    )

    assert isinstance(plan.effective_config, EffectiveConfig)
    assert isinstance(plan.adapter, AdapterInfo)
    assert isinstance(plan.capabilities, RuntimeCapabilities)
    assert plan.profiles == ("runtime", "telemetry")
    assert plan.adapter.harness == "hermes"
    assert "harness_type" not in plan.adapter
    assert plan.adapter.extra_fields["future"] == "value"
    assert plan.capabilities.extra_fields["future_capability"] == "declared"
    resolved = plan.to_mapping()
    plan.effective_config.config.metadata.name = "mutated"
    assert plan.to_mapping() == resolved
    with pytest.raises(TypeError):
        plan["agent_name"] = "mutated"  # type: ignore[index]


def test_runtime_handle_distinguishes_contract_and_extension_fields():
    handle = RuntimeHandle.from_mapping(
        {
            "runtime_id": "runtime-1",
            "runtime_binding": "binding-1",
            "agent_name": "demo",
            "harness": "hermes",
            "mode": "session",
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
        "mode",
        "adapter_kind",
        "environment",
    ),
)
def test_runtime_handle_requires_native_contract_fields(field):
    raw = _runtime()
    del raw[field]

    with pytest.raises(FabricConfigError, match=field.replace("_", " ")):
        RuntimeHandle.from_mapping(raw)


@pytest.mark.parametrize(
    ("model", "payload"),
    (
        (EffectiveConfig, {"config": {}}),
        (DoctorReport, {}),
        (RunResult, {}),
        (SessionInfo, {}),
    ),
)
def test_snapshot_models_require_profiles(model, payload):
    with pytest.raises(FabricConfigError, match="profiles is required"):
        model.from_mapping(payload)


def test_run_plan_requires_profiles():
    raw = _plan()
    del raw["profiles"]

    with pytest.raises(FabricConfigError, match="RunPlan profiles is required"):
        RunPlan.from_mapping(raw)


def test_runtime_capabilities_reject_non_boolean_values():
    with pytest.raises(FabricConfigError, match="session capability"):
        RuntimeCapabilities.from_mapping({"session": "false"})


def test_doctor_report_and_errors_expose_typed_contract_fields():
    report = DoctorReport.from_mapping(
        {
            "agent_name": "demo",
            "profiles": [],
            "status": "warn",
            "checks": [
                {
                    "name": "runtime.mode",
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

    assert report.checks[0].name == "runtime.mode"
    assert error.stage == "invoke"
    assert error.code == "adapter_failed"
    assert error.retryable is True
    assert error.details == {"adapter_id": "test.fabric.shim"}


def _plan() -> dict[str, Any]:
    config = {
        "metadata": {"name": "demo"},
        "harness": {"adapter_id": "test.fabric.shim"},
        "runtime": {
            "mode": "session",
            "transport": "library",
            "input_schema": "chat",
            "output_schema": "message",
        },
    }
    return {
        "agent_name": "demo",
        "profiles": ["typed"],
        "effective_config": {
            "agent_name": "demo",
            "profiles": ["typed"],
            "agent_root": ".",
            "config_path": "agent.yaml",
            "config_root": ".",
            "config": config,
        },
        "config": config,
        "adapter_descriptor": {
            "descriptor": {
                "adapter_kind": "python",
                "adapter_id": "test.fabric.shim",
                "harness": "hermes",
            }
        },
        "capabilities": {
            "session": True,
            "service": False,
            "streaming": False,
            "updates": False,
            "cancellation": False,
            "concurrent_invocations": False,
        },
    }


def _runtime() -> dict[str, Any]:
    return {
        "runtime_id": "runtime-1",
        "runtime_binding": "fabric-runtime-binding-test",
        "agent_name": "demo",
        "harness": "hermes",
        "mode": "session",
        "adapter_kind": "python",
        "adapter_id": "test.fabric.shim",
        "environment": {
            "environment_id": "environment-1",
            "provider": "local",
            "control_location": "external_control",
            "ownership": "caller_owned",
        },
    }


def _fabric_config() -> FabricConfig:
    return FabricConfig(
        metadata=MetadataConfig(name="demo"),
        harness=HarnessConfig(adapter_id="test.fabric.shim"),
        runtime=RuntimeConfig(mode="session"),
    )


class NativeRecorder:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.path_profile_calls: list[Any] = []
        self.stopped = 0
        self.fail_invoke = False

    def plan(self, path: str, profile: Any = None) -> str:
        assert path == "agent"
        self.path_profile_calls.append(profile)
        return json.dumps(_plan())

    def inspect(self, path: str, profile: Any = None) -> str:
        assert path == "agent"
        self.path_profile_calls.append(profile)
        return json.dumps(_plan()["effective_config"])

    def resolve_config(
        self,
        config_json: str,
        profiles_json: str | None = None,
        base_dir: str | None = None,
    ) -> str:
        assert json.loads(config_json)["metadata"]["name"] == "demo"
        return json.dumps(_plan()["effective_config"])

    def plan_config(
        self,
        config_json: str,
        profiles_json: str | None = None,
        base_dir: str | None = None,
    ) -> str:
        assert json.loads(config_json)["metadata"]["name"] == "demo"
        return json.dumps(_plan())

    def start_runtime(self, plan_json: str) -> str:
        assert json.loads(plan_json)["agent_name"] == "demo"
        return json.dumps(_runtime())

    def invoke_runtime(
        self, plan_json: str, runtime_json: str, request_json: str
    ) -> str:
        if self.fail_invoke:
            raise RuntimeError("native invoke failed")
        request = json.loads(request_json)
        self.requests.append(request)
        return json.dumps(
            {
                "agent_name": "demo",
                "profiles": ["typed"],
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


class NativeClient(FabricClient):
    def __init__(self, native: NativeRecorder) -> None:
        super().__init__()
        self.native = native

    def _native_module(self) -> NativeRecorder:
        return self.native

    def _require_native_module(self, method: str) -> NativeRecorder:
        return self.native


def test_run_request_is_mapping_compatible_and_json_safe():
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

    assert request["request_id"] == "request-1"
    assert request.request_id == "request-1"
    assert request.to_mapping()["input"] == {
        "messages": [{"role": "user", "content": "hello"}]
    }
    assert request.to_mapping()["context"] == {"run_id": "run-1", "labels": ["sdk"]}
    assert request.to_mapping()["overrides"] == {
        "temperature": 0,
        "limits": {"turns": 1},
    }

    copied = request.to_dict()
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

    with pytest.raises(FabricConfigError, match="request context"):
        RunRequest.from_mapping({"input": "bad", "context": "not-a-mapping"})


def test_run_request_constructor_validates_context_and_overrides():
    with pytest.raises(FabricConfigError, match="request context"):
        RunRequest(input="bad", context="not-a-mapping")  # type: ignore[arg-type]

    with pytest.raises(FabricConfigError, match="request overrides"):
        RunRequest(input="bad", overrides="not-a-mapping")  # type: ignore[arg-type]

    with pytest.raises(FabricConfigError, match="request context"):
        RunRequest(input="bad", context=[])  # type: ignore[arg-type]

    with pytest.raises(FabricConfigError, match="request extra_fields"):
        RunRequest(input="bad", extra_fields=[])  # type: ignore[arg-type]

    with pytest.raises(FabricConfigError, match="finite"):
        RunRequest(input=float("nan"))


def test_run_request_constructor_generates_request_metadata():
    request = RunRequest(input="hello")

    assert request.input == "hello"
    assert request.request_id.startswith("request-")
    assert request.context == {}


def test_run_result_wraps_nested_error_and_keeps_mapping_access():
    result = RunResult.from_mapping(
        {
            "profiles": [],
            "request_id": "request-1",
            "status": "failed",
            "output": {},
            "error": {
                "stage": "invoke",
                "code": "adapter_failed",
                "message": "adapter failed",
                "retryable": False,
            },
            "artifacts": {"artifacts": []},
            "events": [{"kind": "log", "message": "hello"}],
        }
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
        {
            "profiles": [],
            "status": "succeeded",
            "output": {"plugins": ["observability/nemo_relay"]},
            "metadata": {"labels": ["sdk"]},
            "artifacts": {"artifacts": []},
            "events": [],
            "future": {"values": [1]},
        }
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

    assert result.output == {"plugins": ["observability/nemo_relay"]}
    assert result.metadata == {"labels": ["sdk"]}
    assert result.extra_fields["future"] == {"values": [1]}


def test_run_result_normalizes_core_telemetry_reference():
    result = RunResult.from_mapping(
        {
            "profiles": [],
            "status": "succeeded",
            "output": None,
            "artifacts": {"artifacts": []},
            "events": [],
            "telemetry": {
                "relay_enabled": True,
                "metadata": {
                    "relay_output_dir": "/tmp/relay",
                    "trace_id": "trace-1",
                },
            },
        }
    )

    assert result.telemetry[0].provider == "relay"
    assert result.telemetry[0].kind == "trace"
    assert result.telemetry[0].uri == "/tmp/relay"
    assert result.telemetry[0].trace_id == "trace-1"


async def test_run_accepts_full_run_request_on_native_path():
    native = NativeRecorder()
    client = NativeClient(native)

    with pytest.raises(FabricConfigError, match="complete request"):
        await client.run(
            "agent",
            request=RunRequest(input="hello"),
            context={"turn_id": "turn-4"},
        )

    result = await client.run(
        "agent",
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


async def test_typed_source_accepts_granular_request_fields_and_returns_result():
    native = NativeRecorder()
    client = NativeClient(native)
    result = await client.run(
        _fabric_config(),
        input="hello",
        request_id="request-1",
        context={"job_id": "job-1"},
        overrides={"max_iterations": 1},
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


async def test_invalid_request_context_raises_config_error():
    native = NativeRecorder()
    client = NativeClient(native)

    with pytest.raises(FabricConfigError, match="request context"):
        await client.run(
            _fabric_config(),
            input="hello",
            context="not-a-mapping",  # type: ignore[arg-type]
        )

    assert native.requests == []


async def test_native_runtime_errors_use_typed_exception_and_stop_runtime():
    native = NativeRecorder()
    native.fail_invoke = True
    client = NativeClient(native)

    with pytest.raises(FabricRuntimeError, match="native invoke failed") as error:
        await client.run(_fabric_config(), input="hello")

    assert isinstance(error.value, FabricError)
    assert isinstance(error.value.__cause__, RuntimeError)
    assert native.stopped == 1


async def test_start_service_reports_capability_failure_contract():
    client = NativeClient(NativeRecorder())

    with pytest.raises(FabricCapabilityError) as caught:
        await client.start_service("agent", service_id="service-1")

    assert caught.value.stage == "start"
    assert caught.value.code == "service_not_supported"
    assert caught.value.details == {"service": False, "service_id": "service-1"}


def test_public_sdk_exceptions_share_a_common_base():
    assert issubclass(FabricConfigError, FabricError)
    assert issubclass(FabricRuntimeError, FabricError)
    assert issubclass(FabricStateError, FabricError)
    assert issubclass(FabricCapabilityError, FabricError)
    assert issubclass(FabricNativeUnavailableError, FabricError)


async def test_session_invoke_accepts_run_request_and_turn_fields():
    native = NativeRecorder()
    session = Session(
        client=NativeClient(native),
        plan=_plan(),
        runtime=_runtime(),
        overrides={"session": True, "limits": {"session": 1}},
        session_id="session-1",
    )

    result = await session.invoke(
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
        "context": {
            "job_id": "job-2",
            "session_id": "session-1",
        },
        "overrides": {
            "session": True,
            "request": True,
            "limits": {"session": 1, "request": 1},
        },
    }

    with pytest.raises(FabricConfigError, match="complete request"):
        await session.invoke(
            request=RunRequest(input="hello"),
            context={"turn_id": "turn-1"},
        )


async def test_session_info_stream_and_capability_errors_are_typed():
    session = Session(
        client=NativeClient(NativeRecorder()),
        plan=RunPlan.from_mapping(_plan()),
        runtime=_runtime(),
        session_id="session-1",
    )

    assert isinstance(session.info, SessionInfo)
    assert session.info.profiles == ("typed",)
    assert session.info.harness == "hermes"
    assert session.info.adapter_id == "test.fabric.shim"

    streamed = [item async for item in session.stream(input="hello")]
    assert streamed[0].kind == "invocation_end"
    assert isinstance(streamed[-1], RunResult)

    with pytest.raises(FabricCapabilityError, match="cancellation"):
        await session.cancel()
    assert session.info.status == "active"

    with pytest.raises(FabricCapabilityError, match="updates"):
        await session.update(RuntimeUpdate.from_mapping({"overrides": {"x": 1}}))


async def test_run_rejects_multiple_primary_input_sources():
    client = NativeClient(NativeRecorder())

    with pytest.raises(FabricConfigError, match="at most one input source"):
        await client.run(
            _fabric_config(),
            input="hello",
            request={"input": "request"},
        )


async def test_unified_agent_source_dispatches_fabric_config_to_runtime_path():
    native = NativeRecorder()
    client = NativeClient(native)

    result = await client.run(
        _fabric_config(),
        input="hello",
        request_id="request-5",
    )

    assert result.request_id == "request-5"
    assert native.requests[0]["input"] == "hello"


async def test_lifecycle_methods_reject_raw_mapping_agent_source():
    native = NativeRecorder()
    client = NativeClient(native)

    with pytest.raises(FabricConfigError, match="FabricConfig.from_mapping"):
        await client.run({"metadata": {"name": "demo"}}, input="hello")

    assert native.requests == []


def test_config_methods_reject_raw_mappings_and_pydantic_like_objects():
    class ModelDumpLike:
        def model_dump(self, *, mode: str, exclude_none: bool) -> dict[str, Any]:
            return {"metadata": {"name": "demo"}}

    client = NativeClient(NativeRecorder())

    with pytest.raises(FabricConfigError, match="FabricConfig.from_mapping"):
        client.plan({"metadata": {"name": "demo"}})

    with pytest.raises(FabricConfigError, match="FabricConfig"):
        client.plan(ModelDumpLike())


def test_profile_configs_require_explicit_profile_config_conversion():
    client = NativeClient(NativeRecorder())

    with pytest.raises(FabricConfigError, match="FabricProfileConfig values"):
        client.plan(_fabric_config(), profiles="typed_relay")  # type: ignore[arg-type]

    with pytest.raises(FabricConfigError, match="FabricProfileConfig.from_mapping"):
        client.plan(
            _fabric_config(),
            profiles=[{"name": "typed_relay"}],
        )


def test_path_source_accepts_single_profile_name():
    native = NativeRecorder()

    NativeClient(native).plan("agent", profiles="hermes_session")

    assert native.path_profile_calls == [["hermes_session"]]


def test_fabric_config_constructors_emit_schema_shaped_mappings():
    config = FabricConfig(
        metadata=MetadataConfig(name="demo"),
        harness=HarnessConfig(
            adapter_id="test.fabric.shim",
            resolution="preinstalled",
            settings={"workspace": "./ws"},
        ),
        runtime=RuntimeConfig(
            mode="oneshot",
            transport="cli",
            input_schema="chat",
            output_schema="message",
        ),
    )
    copied = config.to_mapping()
    copied["harness"]["settings"]["workspace"] = "mutated"

    assert config["schema_version"] == "fabric.agent/v1alpha1"
    assert config["metadata"] == {"name": "demo"}
    assert config["harness"]["adapter_id"] == "test.fabric.shim"
    assert config["runtime"]["mode"] == "oneshot"
    assert config["harness"]["settings"]["workspace"] == "./ws"

    profile = FabricProfileConfig.from_mapping({"name": "typed_relay"})
    assert profile.to_mapping() == {
        "schema_version": "fabric.profile/v1alpha1",
        "name": "typed_relay",
    }


def test_resolve_accepts_path_and_fabric_config_sources():
    client = NativeClient(NativeRecorder())

    path_config = client.resolve("agent")
    typed_config = client.resolve(_fabric_config())

    assert path_config["config"]["runtime"]["mode"] == "session"
    assert typed_config["config"]["runtime"]["mode"] == "session"


async def test_start_session_alias_returns_session_and_info_includes_session_id():
    session = await NativeClient(NativeRecorder()).start_session(
        "agent",
        session_id="session-1",
    )

    assert session.session_id == "session-1"
    assert session.info["session_id"] == "session-1"


async def test_session_state_errors_use_sdk_error_hierarchy():
    session = Session(
        client=NativeClient(NativeRecorder()),
        plan=_plan(),
        runtime=_runtime(),
    )
    await session.stop()

    with pytest.raises(FabricStateError, match="cannot invoke a stopped session"):
        await session.invoke(input="hello")
