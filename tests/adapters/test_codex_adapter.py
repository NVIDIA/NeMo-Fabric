# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from nemo_fabric import Fabric
from nemo_fabric_adapters.codex import adapter
from openai_codex import AsyncCodex, AsyncThread, AsyncTurnHandle
from openai_codex.types import TurnStatus


@pytest.fixture(name="codex_payload")
def codex_payload_fixture(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return {
        "agent_name": "codex-test",
        "base_dir": str(tmp_path),
        "config": {
                "harness": {
                    "adapter_id": "nvidia.fabric.codex",
                    "settings": {
                        "sandbox": "workspace-write",
                        "config_overrides": {
                            "features.web_search": False,
                            "model_reasoning_effort": "high",
                        },
                    },
                },
                "models": {
                    "default": {
                        "provider": "openai",
                        "model": "openai/gpt-5.4",
                    }
                },
                "runtime": {},
        },
        "runtime_context": {
            "runtime_id": "runtime-1",
            "invocation_id": "invocation-1",
            "request_id": "request-1",
            "environment": {"workspace": str(workspace)},
            "artifacts": {"root": str(tmp_path / "artifacts")},
        },
        "request": {"input": "Inspect the change."},
    }


def successful_result(response="done"):
    return SimpleNamespace(
        id="turn-1",
        status=TurnStatus.completed,
        error=None,
        started_at=100,
        completed_at=101,
        duration_ms=1000,
        final_response=response,
        items=[
            {
                "id": "item-1",
                "type": "agentMessage",
                "phase": "final_answer",
                "text": response,
            }
        ],
        usage={"total": {"inputTokens": 10, "outputTokens": 3}},
    )


def mock_turn_handle(result=None):
    mock_handle = MagicMock(spec=AsyncTurnHandle)
    outcome = successful_result() if result is None else result
    if isinstance(outcome, BaseException):
        mock_handle.run.side_effect = outcome
    else:
        mock_handle.run.return_value = outcome
    mock_handle.interrupted = False

    async def mark_interrupted():
        mock_handle.interrupted = True

    mock_handle.interrupt.side_effect = mark_interrupted
    return mock_handle


def mock_thread(thread_id, result=None):
    mock_sdk_thread = MagicMock(spec=AsyncThread)
    mock_sdk_thread.id = thread_id
    mock_sdk_thread.handle = mock_turn_handle(result)
    mock_sdk_thread.turn.return_value = mock_sdk_thread.handle
    return mock_sdk_thread


@pytest.fixture(name="mock_codex")
def mock_codex_fixture(monkeypatch):
    mock_codex = MagicMock(spec=AsyncCodex)
    mock_codex.instances = []
    mock_codex.next_thread_id = "thread-123"
    mock_codex.next_result = None
    mock_codex.next_thread = None
    mock_codex.resume_thread_id = None

    def build_client(*, config):
        mock_client = MagicMock(spec=AsyncCodex)
        mock_client.config = config
        mock_client.closed = False
        mock_client.thread = None

        async def close():
            mock_client.closed = True

        async def thread_start(**_kwargs):
            mock_client.thread = (
                mock_codex.next_thread
                if mock_codex.next_thread is not None
                else mock_thread(mock_codex.next_thread_id, mock_codex.next_result)
            )
            return mock_client.thread

        async def thread_resume(thread_id, **_kwargs):
            resumed_thread_id = mock_codex.resume_thread_id or thread_id
            mock_client.thread = mock_thread(
                resumed_thread_id, mock_codex.next_result
            )
            return mock_client.thread

        mock_client.close.side_effect = close
        mock_client.thread_start.side_effect = thread_start
        mock_client.thread_resume.side_effect = thread_resume
        mock_codex.instances.append(mock_client)
        return mock_client

    mock_codex.side_effect = build_client
    monkeypatch.setattr(adapter, "AsyncCodex", mock_codex)
    return mock_codex


def test_sdk_oneshot_uses_native_thread_and_turn_contract(
    codex_payload, mock_codex, tmp_path
):
    os.environ["CODEX_HOME"] = str(tmp_path / "codex-home")
    os.environ["CODEX_INTERNAL_ORIGINATOR_OVERRIDE"] = "parent-codex"
    os.environ["FABRIC_UNRELATED_SECRET"] = "do-not-forward"
    codex_payload["config"]["harness"]["settings"]["env"] = {
        "CODEX_EXPLICIT": "forward-me"
    }

    output = adapter.run(codex_payload)

    assert output["completed"] is True
    assert output["adapter"] == "sdk"
    assert output["mode"] == "codex_sdk_runtime"
    assert output["thread_id"] == "thread-123"
    assert output["turn_id"] == "turn-1"
    assert output["response"] == "done"
    assert output["events"][0]["type"] == "agentMessage"
    assert "command" not in output
    assert "returncode" not in output

    client = mock_codex.instances[0]
    assert client.closed is True
    assert client.config.codex_bin is None
    assert client.config.launch_args_override is None
    assert client.config.cwd == str(
        Path(codex_payload["runtime_context"]["environment"]["workspace"])
    )
    assert client.config.env["CODEX_HOME"] == str(tmp_path / "codex-home")
    assert client.config.env["CODEX_EXPLICIT"] == "forward-me"
    assert (
        client.config.env["CODEX_INTERNAL_ORIGINATOR_OVERRIDE"]
        == "codex_python_sdk"
    )
    assert client.config.env["FABRIC_UNRELATED_SECRET"] == ""
    start = client.thread_start.await_args.kwargs
    assert start["model"] == "gpt-5.4"
    assert start["model_provider"] == "openai"
    assert start["sandbox"] == adapter.Sandbox.workspace_write
    assert start["config"] == {
        "features": {"web_search": False},
        "model_reasoning_effort": "high",
    }
    client.thread.turn.assert_awaited_once_with(
        "Inspect the change.", effort=None, output_schema=None
    )


def test_sdk_can_use_an_explicit_codex_runtime(codex_payload, mock_codex, tmp_path):
    codex_bin = tmp_path / "bin" / "codex"
    codex_bin.parent.mkdir()
    codex_bin.touch()
    codex_payload["config"]["harness"]["settings"][
        "codex_bin"
    ] = str(codex_bin)

    output = adapter.run(codex_payload)

    assert output["completed"] is True
    assert mock_codex.instances[0].config.codex_bin == str(codex_bin)


@pytest.mark.parametrize("codex_bin", ["bin/codex", "~/bin/codex"])
def test_sdk_resolves_relative_codex_runtime_from_base_dir(
    codex_payload, codex_bin
):
    codex_payload["config"]["harness"]["settings"][
        "codex_bin"
    ] = codex_bin

    config = adapter.sdk_config(codex_payload, relay=None)

    base_dir = Path(codex_payload["base_dir"])
    assert config.codex_bin == str((base_dir / codex_bin).resolve())


def test_sdk_keeps_absolute_codex_runtime_path(codex_payload, tmp_path):
    codex_bin = tmp_path / "bin" / ".." / "codex"
    codex_payload["config"]["harness"]["settings"][
        "codex_bin"
    ] = str(codex_bin)

    config = adapter.sdk_config(codex_payload, relay=None)

    assert config.codex_bin == str(codex_bin)


def test_runtime_resumes_sdk_thread_across_invocations(
    codex_payload, mock_codex
):
    first = adapter.run(codex_payload)
    codex_payload["runtime_context"]["invocation_id"] = "invocation-2"
    codex_payload["request"]["input"] = "Continue."
    second = adapter.run(codex_payload)

    assert first["thread_id"] == second["thread_id"] == "thread-123"
    mock_codex.instances[0].thread_start.assert_awaited_once()
    assert mock_codex.instances[1].thread_resume.await_args.args[0] == "thread-123"
    assert mock_codex.instances[1].thread.turn.await_args.args[0] == "Continue."
    state = json.loads(
        adapter.runtime_state_path(codex_payload, "runtime-1").read_text(
            encoding="utf-8"
        )
    )
    assert state == {
        "runtime_id": "runtime-1",
        "codex_thread_id": "thread-123",
    }


def test_runtime_rejects_corrupt_thread_state(codex_payload):
    state_path = adapter.runtime_state_path(codex_payload, "runtime-1")
    state_path.parent.mkdir(parents=True)
    state_path.write_text("{", encoding="utf-8")

    output = adapter.run(codex_payload)

    assert output["error"]["code"] == "codex_invalid_runtime_state"


def test_failed_sdk_turn_is_normalized_and_transport_is_closed(
    codex_payload, mock_codex
):
    mock_codex.next_result = RuntimeError("model request failed")

    output = adapter.run(codex_payload)

    assert output["error"] == {
        "code": "codex_turn_failed",
        "message": "model request failed",
        "retryable": False,
    }
    assert mock_codex.instances[0].closed is True
    assert not adapter.runtime_state_path(codex_payload, "runtime-1").exists()


def test_incomplete_sdk_turn_is_failed_without_persisting_thread(
    codex_payload, mock_codex
):
    result = successful_result(response=None)
    mock_codex.next_result = result

    output = adapter.run(codex_payload)

    assert output["error"]["code"] == "codex_turn_incomplete"
    assert output["turn_status"] == "completed"
    assert not adapter.runtime_state_path(codex_payload, "runtime-1").exists()


def test_selected_model_rejects_unsupported_provider(codex_payload, mock_codex):
    model = codex_payload["config"]["models"]["default"]
    model["provider"] = "anthropic"

    output = adapter.run(codex_payload)

    assert output["error"]["code"] == "codex_invalid_configuration"
    assert "provider must be openai or nvidia" in output["error"]["message"]
    mock_codex.assert_not_called()


def test_nvidia_provider_uses_responses_api_and_nvidia_credential(
    codex_payload, mock_codex
):
    model = codex_payload["config"]["models"]["default"]
    model.update(
        {
            "provider": "nvidia",
            "model": "openai/gpt-oss-120b",
            "api_key_env": "NVIDIA_API_KEY",
            "settings": {"base_url": "https://nvidia.example/v1/"},
        }
    )
    os.environ["NVIDIA_API_KEY"] = "nvidia-secret"

    output = adapter.run(codex_payload)

    assert output["completed"] is True
    client = mock_codex.instances[0]
    assert client.config.env["NVIDIA_API_KEY"] == "nvidia-secret"
    start = client.thread_start.await_args.kwargs
    assert start["model"] == "openai/gpt-oss-120b"
    assert start["model_provider"] == "nvidia"
    assert client.config.env["CODEX_HOME"].endswith("/.fabric/codex/nvidia-home")
    assert start["config"]["features"] == {"web_search": False}
    assert start["config"]["model_providers"] == {
        "nvidia": {
            "name": "NVIDIA",
            "base_url": "https://nvidia.example/v1",
            "env_key": "NVIDIA_API_KEY",
            "wire_api": "responses",
        }
    }


def test_nvidia_provider_requires_credential(codex_payload, mock_codex):
    model = codex_payload["config"]["models"]["default"]
    model.update(
        {
            "provider": "nvidia",
            "model": "openai/gpt-oss-120b",
            "api_key_env": "NVIDIA_API_KEY",
        }
    )
    os.environ.pop("NVIDIA_API_KEY", None)

    output = adapter.run(codex_payload)

    assert output["error"]["code"] == "codex_invalid_configuration"
    assert "NVIDIA_API_KEY is required" in output["error"]["message"]
    assert not (adapter.state_dir(codex_payload) / "nvidia-home").exists()
    mock_codex.assert_not_called()


def test_nvidia_provider_requires_endpoint(codex_payload, mock_codex):
    model = codex_payload["config"]["models"]["default"]
    model.update(
        {
            "provider": "nvidia",
            "model": "openai/gpt-oss-120b",
            "api_key_env": "NVIDIA_API_KEY",
        }
    )
    model.pop("settings", None)
    os.environ["NVIDIA_API_KEY"] = "nvidia-secret"
    os.environ.pop("NVIDIA_FRONTIER_BASE_URL", None)

    output = adapter.run(codex_payload)

    assert output["error"]["code"] == "codex_invalid_configuration"
    assert "NVIDIA_FRONTIER_BASE_URL" in output["error"]["message"]
    assert not (adapter.state_dir(codex_payload) / "nvidia-home").exists()
    mock_codex.assert_not_called()


def test_resume_rejects_changed_sdk_thread_identity(
    codex_payload, mock_codex
):
    adapter.save_thread_id(codex_payload, "runtime-1", "thread-persisted")
    mock_codex.resume_thread_id = "thread-replaced"

    output = adapter.run(codex_payload)

    assert output["error"]["code"] == "codex_thread_mismatch"
    assert mock_codex.instances[0].closed is True


def test_relay_uses_gateway_and_request_scoped_sdk_config(
    codex_payload, mock_codex, monkeypatch, tmp_path
):
    codex_payload["telemetry_plan"] = {
        "providers": ["relay"],
        "relay_enabled": True,
    }
    relay_config_path = tmp_path / "relay-config" / "config.toml"
    executable = tmp_path / "bin" / "nemo-relay"
    gateway = adapter.relay_gateway.RelayGatewayLaunch(
        executable=executable,
        config_path=relay_config_path,
        bind="127.0.0.1:43210",
        url="http://127.0.0.1:43210",
        log_path=relay_config_path.parent / "gateway.log",
    )
    relay = adapter.CodexRelaySettings(
        gateway=gateway,
        plugin_config={"version": 1, "components": []},
    )
    process = MagicMock()
    start_gateway = MagicMock(return_value=process)
    stop_gateway = MagicMock()
    monkeypatch.setattr(adapter, "prepare_codex_relay", MagicMock(return_value=relay))
    monkeypatch.setattr(
        adapter.relay_gateway, "start_relay_gateway", start_gateway
    )
    monkeypatch.setattr(adapter.relay_gateway, "stop_relay_gateway", stop_gateway)
    os.environ["FABRIC_RELAY_CONFIG_PATH"] = str(tmp_path / "relay.json")

    output = adapter.run(codex_payload)

    client = mock_codex.instances[0]
    start = client.thread_start.await_args.kwargs
    config = start["config"]
    assert start["model_provider"] == "openai"
    assert client.config.env["NEMO_RELAY_GATEWAY_URL"] == gateway.url
    assert config["bypass_hook_trust"] is True
    assert config["features"]["hooks"] is True
    assert config["features"]["multi_agent_v2"]["enabled"] is False
    assert config["features"]["web_search"] is False
    assert config["openai_base_url"] == gateway.url
    assert "model_providers" not in config
    assert config["hooks"]["SessionStart"][0]["hooks"][0] == {
        "type": "command",
        "command": f"{executable} hook-forward codex",
        "timeout": 30,
    }
    assert output["relay_runtime"] == {
        "enabled": True,
        "emitter": "codex-sdk/nemo-relay",
        "config_path": str(tmp_path / "relay.json"),
        "gateway_config_path": str(relay_config_path),
        "gateway_url": gateway.url,
        "gateway_log_path": str(gateway.log_path),
    }
    assert output["relay_artifacts"] == []
    start_gateway.assert_called_once_with(
        launch=gateway,
        cwd=Path(codex_payload["runtime_context"]["environment"]["workspace"]),
    )
    stop_gateway.assert_called_once_with(process)


def test_prepare_relay_reuses_one_resolved_executable(
    codex_payload, monkeypatch, tmp_path
):
    codex_payload["telemetry_plan"] = {
        "providers": ["relay"],
        "relay_enabled": True,
    }
    executable = tmp_path / "nemo-relay"
    config_path = tmp_path / "relay-config" / "config.toml"
    plugin_path = config_path.parent / "plugins.toml"
    resolve = MagicMock(return_value=executable)
    contract = MagicMock(
        return_value=adapter.relay_gateway.RelayCliContract(
            version=(0, 6, 0), observability_version=2
        )
    )
    write = MagicMock(return_value=(config_path, plugin_path))
    monkeypatch.setattr(adapter.relay_gateway, "resolve_relay_command", resolve)
    monkeypatch.setattr(
        adapter.relay_gateway, "relay_cli_contract", contract
    )
    monkeypatch.setattr(adapter.relay_gateway, "find_available_tcp_port", lambda: 43210)
    monkeypatch.setattr(
        adapter.common_utils,
        "load_relay_plugin_config",
        MagicMock(return_value={"version": 1, "components": []}),
    )
    monkeypatch.setattr(adapter.common_utils, "write_relay_configs", write)

    relay = adapter.prepare_codex_relay(codex_payload)

    assert relay is not None
    assert relay.gateway.executable == executable
    assert relay.gateway.url == "http://127.0.0.1:43210"
    resolve.assert_called_once_with(
        Path(codex_payload["base_dir"]).resolve(),
        "nemo-relay",
    )
    contract.assert_called_once_with(executable)
    write.assert_called_once_with(
        relay_config={},
        plugin_config={"version": 1, "components": []},
        observability_version=2,
    )


@pytest.mark.usefixtures("mock_codex")
def test_relay_cleanup_failure_changes_success_to_failure(
    codex_payload, monkeypatch, tmp_path
):
    gateway = adapter.relay_gateway.RelayGatewayLaunch(
        executable=tmp_path / "nemo-relay",
        config_path=tmp_path / "relay" / "config.toml",
        bind="127.0.0.1:43210",
        url="http://127.0.0.1:43210",
        log_path=tmp_path / "relay" / "gateway.log",
    )
    relay = adapter.CodexRelaySettings(
        gateway=gateway,
        plugin_config={"version": 1, "components": []},
    )
    monkeypatch.setattr(adapter, "prepare_codex_relay", lambda _: relay)
    monkeypatch.setattr(
        adapter.relay_gateway, "start_relay_gateway", lambda **_: MagicMock()
    )
    monkeypatch.setattr(
        adapter.relay_gateway,
        "stop_relay_gateway",
        MagicMock(side_effect=adapter.relay_gateway.RelayGatewayError("stuck")),
    )

    output = adapter.run(codex_payload)

    assert output["failed"] is True
    assert output["completed"] is False
    assert output["error"]["code"] == "codex_relay_stop_failed"
    assert output["relay_runtime"]["cleanup_error"] == output["error"]


def test_native_sdk_controls_and_telemetry_are_request_scoped(
    codex_payload, mock_codex
):
    settings = codex_payload["config"]["harness"]["settings"]
    settings.update(
        {
            "personality": "pragmatic",
            "reasoning_effort": "xhigh",
            "service_name": "fabric-codex-test",
            "output_schema": {
                "type": "object",
                "properties": {"summary": {"type": "string"}},
                "required": ["summary"],
            },
        }
    )
    codex_payload["telemetry_plan"] = {
        "providers": ["native"],
        "relay_enabled": False,
        "native_config": {
            "components": [
                {
                    "kind": "observability",
                    "enabled": True,
                    "config": {
                        "opentelemetry": {
                            "enabled": True,
                            "endpoint": "http://localhost:4318/v1/traces",
                            "transport": "http_binary",
                            "resource_attributes": {
                                "deployment.environment": "test"
                            },
                        }
                    },
                }
            ]
        },
    }

    output = adapter.run(codex_payload)

    assert output["failed"] is False
    client = mock_codex.instances[0]
    start = client.thread_start.await_args.kwargs
    assert start["personality"] == adapter.Personality.pragmatic
    assert start["service_name"] == "fabric-codex-test"
    assert start["config"]["otel"] == {
        "environment": "test",
        "trace_exporter": {
            "otlp-http": {
                "endpoint": "http://localhost:4318/v1/traces",
                "protocol": "binary",
            }
        },
    }
    turn = client.thread.turn.await_args.kwargs
    assert turn["effort"] == adapter.ReasoningEffort.xhigh
    assert turn["output_schema"]["required"] == ["summary"]


def test_timeout_interrupts_native_turn_and_closes_sdk(
    codex_payload, mock_codex
):
    mock_blocking_thread = mock_thread("thread-timeout")

    async def block():
        await asyncio.sleep(60)

    mock_blocking_thread.handle.run.side_effect = block
    mock_codex.next_thread = mock_blocking_thread
    codex_payload["config"]["harness"]["settings"][
        "timeout_seconds"
    ] = 0.01

    output = adapter.run(codex_payload)

    client = mock_codex.instances[0]
    assert output["error"]["code"] == "codex_timed_out"
    assert client.thread.handle.interrupted is True
    assert client.closed is True


@pytest.mark.parametrize(
    "setting", ["codex_command", "codex_args", "codex_profile", "skip_git_repo_check"]
)
def test_cli_only_settings_are_rejected(codex_payload, setting):
    codex_payload["config"]["harness"]["settings"][setting] = (
        "legacy"
    )

    output = adapter.run(codex_payload)

    assert output["error"]["code"] == "codex_invalid_configuration"
    assert setting in output["error"]["message"]


def test_adapter_rejects_structured_input(codex_payload):
    codex_payload["request"]["input"] = {
        "messages": [{"role": "user", "content": "Inspect the change."}]
    }

    output = adapter.run(codex_payload)

    assert output["error"]["code"] == "codex_invalid_request"


def test_descriptor_has_no_codex_binary_requirement():
    descriptor = json.loads(
        (
            Path(__file__).parents[2] / "adapters" / "codex" / "fabric-adapter.json"
        ).read_text(encoding="utf-8")
    )

    assert descriptor["adapter_id"] == "nvidia.fabric.codex"
    assert descriptor["runner"] == {
        "module": "nemo_fabric_adapters.codex.adapter",
        "callable": "run",
    }
    assert "requirements" not in descriptor


def test_codex_config_resolves_sdk_adapter():
    from examples.code_review_agent import BASE_DIR, codex_config

    plan = Fabric().plan(codex_config(), base_dir=BASE_DIR)

    assert plan.adapter.adapter_id == "nvidia.fabric.codex"
    assert plan.adapter.harness == "codex"
    assert plan.config.runtime.input_schema == "text"
    assert plan.config.harness.settings["reasoning_effort"] == "high"
    unsupported = plan["capability_plan"]["unsupported"]
    assert not unsupported.get("skill_paths")
    assert not unsupported.get("mcp_servers")


def test_environment_does_not_mutate_parent(codex_payload):
    os.environ["FABRIC_UNRELATED_SECRET"] = "parent-value"

    child = adapter.child_environment(codex_payload)

    assert child["FABRIC_UNRELATED_SECRET"] == ""
    assert os.environ["FABRIC_UNRELATED_SECRET"] == "parent-value"


def test_environment_preserves_runtime_telemetry_env(codex_payload):
    codex_payload["runtime_context"]["telemetry"] = {
        "env": {
            "FABRIC_RELAY_ENABLED": "true",
            "FABRIC_RELAY_CONFIG_PATH": "/tmp/relay.json",
            "CODEX_EXPLICIT": "telemetry",
        }
    }
    codex_payload["config"]["harness"]["settings"]["env"] = {
        "CODEX_EXPLICIT": "configured"
    }
    os.environ["FABRIC_RELAY_CONFIG_PATH"] = "/tmp/parent-relay.json"

    child = adapter.child_environment(codex_payload)

    assert child["FABRIC_RELAY_ENABLED"] == "true"
    assert child["FABRIC_RELAY_CONFIG_PATH"] == "/tmp/relay.json"
    assert child["CODEX_EXPLICIT"] == "configured"


@pytest.mark.parametrize(
    "telemetry_env",
    [
        [],
        {1: "value"},
        {"OTEL_EXPORTER_OTLP_ENDPOINT": 4318},
    ],
)
def test_environment_rejects_non_string_runtime_telemetry_env(
    codex_payload, telemetry_env
):
    codex_payload["runtime_context"]["telemetry"] = {"env": telemetry_env}

    with pytest.raises(
        adapter.AdapterInputError,
        match=r"runtime_context\.telemetry\.env must contain strings",
    ):
        adapter.child_environment(codex_payload)


@pytest.mark.parametrize("telemetry", [[], "invalid"])
def test_environment_rejects_non_mapping_runtime_telemetry(
    codex_payload, telemetry
):
    codex_payload["runtime_context"]["telemetry"] = telemetry

    with pytest.raises(
        adapter.AdapterInputError,
        match=r"runtime_context\.telemetry must be a mapping",
    ):
        adapter.child_environment(codex_payload)
