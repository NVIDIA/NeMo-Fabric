# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
import tomllib
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from nemo_fabric import Fabric
from nemo_fabric import FabricConfig
from nemo_fabric_adapters.codex_cli import adapter


@pytest.fixture(name="codex_payload")
def codex_payload_fixture(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    (codex_home / "team.toml").write_text("", encoding="utf-8")
    os.environ["CODEX_HOME"] = str(codex_home)
    return {
        "effective_config": {
            "agent_name": "codex-test",
            "config_root": str(tmp_path),
            "config": {
                "harness": {
                    "adapter_id": "nvidia.fabric.codex.cli",
                    "settings": {
                        "sandbox": "read-only",
                        "codex_profile": "team",
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


def codex_jsonl(thread_id, response, *, usage=None):
    events = [
        {"type": "thread.started", "thread_id": thread_id},
        {"type": "turn.started"},
        {
            "type": "item.completed",
            "item": {"id": "item-1", "type": "agent_message", "text": response},
        },
        {
            "type": "turn.completed",
            "usage": usage
            or {
                "input_tokens": 10,
                "cached_input_tokens": 2,
                "output_tokens": 3,
            },
        },
    ]
    return "\n".join(json.dumps(event) for event in events) + "\n"


def write_mock_codex(path):
    path.write_text(
        """#!/usr/bin/env python3
import json
import sys

args = sys.argv[1:]
thread_id = args[args.index("resume") + 1] if "resume" in args else "thread-fake"
prompt = sys.stdin.read().strip()
events = [
    {"type": "thread.started", "thread_id": thread_id},
    {"type": "turn.started"},
    {
        "type": "item.completed",
        "item": {
            "id": "item-1",
            "type": "agent_message",
            "text": f"{thread_id}:{prompt}",
        },
    },
    {
        "type": "turn.completed",
        "usage": {"input_tokens": 1, "cached_input_tokens": 0, "output_tokens": 1},
    },
]
for event in events:
    print(json.dumps(event))
""",
        encoding="utf-8",
    )
    path.chmod(0o755)


def fabric_config(tmp_path, mock_codex):
    return FabricConfig.from_mapping(
        {
            "schema_version": "fabric.agent/v1alpha1",
            "metadata": {"name": "codex-runtime-test"},
            "harness": {
                "adapter_id": "nvidia.fabric.codex.cli",
                "resolution": "preinstalled",
                "settings": {
                    "codex_command": str(mock_codex),
                    "sandbox": "read-only",
                    "skip_git_repo_check": True,
                },
            },
            "runtime": {
                "artifacts": str(tmp_path / "artifacts"),
            },
            "environment": {
                "provider": "local",
                "workspace": str(tmp_path),
                "artifacts": str(tmp_path / "artifacts"),
            },
        }
    )


def test_oneshot_command_uses_fabric_overrides_and_codex_owned_auth(
    codex_payload,
    tmp_path,
):
    codex_settings = adapter.write_config_files(codex_payload)

    command = adapter.build_command(
        codex_payload,
        codex_settings=codex_settings,
    )

    exec_index = command.index("exec")
    assert command[0] == "codex"
    assert command[1:exec_index] == []
    assert command[exec_index : exec_index + 2] == ["exec", "--json"]
    assert "--ephemeral" not in command
    assert ["--sandbox", "read-only"] == command[exec_index + 2 : exec_index + 4]
    assert ["--profile", "fabric-runtime-1"] == command[exec_index + 4 : exec_index + 6]
    assert "--dangerously-bypass-hook-trust" not in command
    assert ["--model", "gpt-5.4"] == command[-3:-1]
    assert command[-1] == "-"
    assert tomllib.loads(
        codex_settings.codex_profile_path.read_text(encoding="utf-8")
    ) == {
        "features": {"web_search": False},
        "model_reasoning_effort": "high",
    }


def test_configured_codex_profile_is_base_for_generated_profile(codex_payload):
    codex_home = Path(os.environ["CODEX_HOME"])
    source_profile = codex_home / "team.toml"
    source_profile.write_text(
        """approval_policy = "never"
model_reasoning_effort = "medium"

[features]
web_search = true
shell_snapshot = true
""",
        encoding="utf-8",
    )

    codex_settings = adapter.write_config_files(codex_payload)

    assert codex_settings.codex_profile_name == "fabric-runtime-1"
    assert tomllib.loads(
        codex_settings.codex_profile_path.read_text(encoding="utf-8")
    ) == {
        "approval_policy": "never",
        "model_reasoning_effort": "high",
        "features": {
            "web_search": False,
            "shell_snapshot": True,
        },
    }
    assert tomllib.loads(source_profile.read_text(encoding="utf-8")) == {
        "approval_policy": "never",
        "model_reasoning_effort": "medium",
        "features": {
            "web_search": True,
            "shell_snapshot": True,
        },
    }


def test_relative_codex_command_resolves_from_config_root(codex_payload):
    settings = codex_payload["effective_config"]["config"]["harness"]["settings"]
    settings["codex_command"] = "./tools/codex"
    settings["config_overrides"] = {}

    command = adapter.build_command(
        codex_payload,
        codex_settings=adapter.write_config_files(codex_payload),
    )

    config_root = Path(codex_payload["effective_config"]["config_root"])
    assert command[0] == str(config_root / "tools" / "codex")


def test_codex_home_uses_environment(tmp_path):
    os.environ["CODEX_HOME"] = str(tmp_path / "custom-codex-home")

    name, path = adapter.get_codex_profile_path(
        {"runtime_context": {"runtime_id": "runtime-1"}}
    )

    assert name == "fabric-runtime-1"
    assert path == tmp_path / "custom-codex-home" / "fabric-runtime-1.config.toml"


def test_codex_home_defaults_to_user_codex_directory():
    os.environ.pop("CODEX_HOME", None)

    name, path = adapter.get_codex_profile_path(
        {"runtime_context": {"runtime_id": "runtime-1"}}
    )

    assert name == "fabric-runtime-1"
    assert path == Path.home() / ".codex" / "fabric-runtime-1.config.toml"


def test_relay_routes_codex_through_standalone_gateway(
    codex_payload,
    monkeypatch,
    tmp_path,
):
    codex_payload["telemetry_plan"] = {
        "providers": ["relay"],
        "relay_enabled": True,
    }
    mock_find_port = MagicMock(return_value=43210)
    monkeypatch.setattr(
        adapter.relay_gateway, "find_available_tcp_port", mock_find_port
    )
    relay_executable = tmp_path / "bin" / "nemo-relay"
    relay_executable.parent.mkdir()
    relay_executable.touch()
    monkeypatch.setattr(
        adapter.relay_gateway,
        "resolve_relay_command",
        MagicMock(return_value=relay_executable),
    )
    relay_plugin_config = {"version": 1, "components": []}
    relay_config_path = tmp_path / "relay-config" / "config.toml"
    mock_load_config = MagicMock(return_value=relay_plugin_config)
    mock_write_config = MagicMock(
        return_value=(relay_config_path, tmp_path / "plugins.toml")
    )
    monkeypatch.setattr(
        adapter.common_utils,
        "load_relay_plugin_config",
        mock_load_config,
    )
    monkeypatch.setattr(
        adapter.common_utils,
        "write_relay_configs",
        mock_write_config,
    )
    monkeypatch.setattr(
        adapter.relay_gateway,
        "relay_cli_observability_version",
        MagicMock(return_value=2),
    )
    codex_settings = adapter.write_config_files(codex_payload)

    command = adapter.build_command(
        codex_payload,
        codex_settings=codex_settings,
    )

    assert command[0] == "codex"
    assert "nemo-relay" not in command
    assert "--dangerously-bypass-hook-trust" in command
    assert not any(value.startswith("hooks.") for value in command)
    assert "--config" not in command
    assert command[command.index("--profile") + 1] == "fabric-runtime-1"

    config = tomllib.loads(
        codex_settings.codex_profile_path.read_text(encoding="utf-8")
    )
    assert config["model_provider"] == "nemo-relay-openai"
    assert config["model_providers"]["nemo-relay-openai"] == {
        "name": "NeMo Relay OpenAI",
        "base_url": "http://127.0.0.1:43210",
        "wire_api": "responses",
        "requires_openai_auth": True,
        "supports_websockets": False,
    }
    assert config["features"]["hooks"] is True
    assert config["features"]["web_search"] is False
    assert config["model_reasoning_effort"] == "high"
    assert config["hooks"]["SessionStart"][0]["hooks"][0] == {
        "type": "command",
        "command": f"{relay_executable} hook-forward codex",
        "timeout": 30,
    }
    assert "UserPromptExpansion" in config["hooks"]
    mock_load_config.assert_called_once_with(codex_payload)
    mock_write_config.assert_called_once_with(
        relay_config={"agents": {"codex": {"command": "codex"}}},
        plugin_config=relay_plugin_config,
        observability_version=2,
    )


@pytest.mark.parametrize(
    ("transport", "expected_exporter", "expected_protocol"),
    [
        ("http_binary", "otlp-http", "binary"),
        ("grpc", "otlp-grpc", "grpc"),
    ],
)
def test_native_otel_profile_writes_codex_telemetry_config(
    codex_payload,
    tmp_path,
    transport,
    expected_exporter,
    expected_protocol,
):
    from examples.code_review_agent import codex_cli_config
    from examples.code_review_agent import with_native_otel

    config = codex_payload["effective_config"]["config"]
    typed = with_native_otel(codex_cli_config())
    assert typed.telemetry is not None
    native_config = typed.telemetry.to_mapping()["providers"]["native"]["config"]
    native_config["components"][0]["config"]["opentelemetry"]["transport"] = transport
    codex_payload["telemetry_plan"] = {
        "providers": ["native"],
        "relay_enabled": False,
        "native_config": native_config,
    }
    config["harness"]["settings"]["config_overrides"] = {}
    codex_settings = adapter.write_config_files(codex_payload)

    command = adapter.build_command(
        codex_payload,
        codex_settings=codex_settings,
    )

    assert command[command.index("--profile") + 1] == "fabric-runtime-1"
    assert "--dangerously-bypass-hook-trust" not in command
    assert tomllib.loads(
        codex_settings.codex_profile_path.read_text(encoding="utf-8")
    ) == {
        "otel": {
            "environment": "dev",
            "trace_exporter": {
                expected_exporter: {
                    "endpoint": "http://localhost:4318/v1/traces",
                    "protocol": expected_protocol,
                }
            },
        }
    }


def test_run_codex_configures_relay(codex_payload, monkeypatch, tmp_path):
    relay_plugin_config = {"version": 1, "components": []}
    relay_config_path = tmp_path / "relay-config" / "config.toml"
    mock_gateway = MagicMock()
    gateway_host = "127.0.0.1:43210"
    gateway_url = "http://127.0.0.1:43210"
    mock_start_gateway = MagicMock(return_value=mock_gateway)
    mock_stop_gateway = MagicMock()
    codex_home = tmp_path / "codex-home"
    profile_name = "fabric-runtime-1"
    profile_path = codex_home / f"{profile_name}.config.toml"
    gateway = adapter.relay_gateway.RelayGatewayLaunch(
        executable=tmp_path / "nemo-relay",
        config_path=relay_config_path,
        bind=gateway_host,
        url=gateway_url,
        log_path=relay_config_path.parent / "gateway.log",
    )
    codex_settings = adapter.CodexSettings(
        telemetry_provider="relay",
        codex_profile_name=profile_name,
        codex_profile_path=profile_path,
        relay=adapter.CodexRelaySettings(
            gateway=gateway,
            plugin_config=relay_plugin_config,
        ),
    )
    mock_write_config_files = MagicMock(return_value=codex_settings)
    mock_run = MagicMock(
        return_value=subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=codex_jsonl("thread-123", "done"),
            stderr="",
        )
    )
    os.environ["FABRIC_RELAY_ENABLED"] = "true"
    os.environ.pop("CODEX_HOME", None)
    os.environ["FABRIC_RELAY_CONFIG_PATH"] = str(tmp_path / "relay-config.json")
    monkeypatch.setattr(
        adapter.relay_gateway, "start_relay_gateway", mock_start_gateway
    )
    monkeypatch.setattr(adapter.relay_gateway, "stop_relay_gateway", mock_stop_gateway)
    monkeypatch.setattr(
        adapter,
        "write_config_files",
        mock_write_config_files,
    )
    monkeypatch.setattr(adapter.subprocess, "run", mock_run)

    result = adapter.run_codex(codex_payload)

    command = mock_run.call_args.args[0]
    assert result["relay_runtime"] == {
        "enabled": True,
        "config_path": os.environ["FABRIC_RELAY_CONFIG_PATH"],
        "emitter": "nemo-relay",
        "gateway_config_path": str(relay_config_path),
        "gateway_log_path": str(gateway.log_path),
    }
    assert result["relay_artifacts"] == []
    assert command[0] == "codex"
    assert "nemo-relay" not in command
    assert command[command.index("--profile") + 1] == profile_name
    assert mock_run.call_args.kwargs["env"]["NEMO_RELAY_GATEWAY_URL"] == gateway_url
    assert "CODEX_HOME" not in mock_run.call_args.kwargs["env"]
    assert not profile_path.exists()
    mock_write_config_files.assert_called_once_with(codex_payload)
    mock_start_gateway.assert_called_once_with(
        launch=gateway,
        cwd=Path(codex_payload["runtime_context"]["environment"]["workspace"]),
    )
    mock_stop_gateway.assert_called_once_with(mock_gateway)


def test_reported_command_redacts_secret_config_overrides():

    command = ["codex", "exec", "--config", 'provider.api_key="secret"', "-"]

    assert adapter.redact_command(command)[-2] == "<redacted>"
    assert command[-2] == 'provider.api_key="secret"'


def test_config_override_values_use_tomli_writer():

    assert adapter.toml_value("café") == '"café"'
    encoded = adapter.toml_value([1, "two"])
    assert tomllib.loads(f"value = {encoded}")["value"] == [1, "two"]
    with pytest.raises(ValueError, match="scalar or array"):
        adapter.toml_value({"nested": True})


@pytest.mark.parametrize("value", [[float("nan")], [1, [float("inf")]]])
def test_config_override_values_reject_nested_non_finite_numbers(value):

    with pytest.raises(ValueError, match="finite numbers"):
        adapter.toml_value(value)


def test_runtime_reuses_codex_thread_across_invocations(
    codex_payload, monkeypatch, tmp_path
):
    mock_run = MagicMock(
        side_effect=[
            subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout=codex_jsonl("thread-123", "first response"),
                stderr="",
            ),
            subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout=codex_jsonl("thread-123", "second response"),
                stderr="",
            ),
        ]
    )
    monkeypatch.setattr(adapter.subprocess, "run", mock_run)
    os.environ.pop("OPENAI_API_KEY", None)
    os.environ["CODEX_HOME"] = str(tmp_path / "codex-home")
    os.environ["FABRIC_UNRELATED_SECRET"] = "do-not-forward"
    codex_payload["effective_config"]["config"]["harness"]["settings"]["env"] = {
        "CODEX_EXPLICIT": "forward-me"
    }

    first = adapter.run_codex(codex_payload)
    codex_payload["runtime_context"]["invocation_id"] = "invocation-2"
    codex_payload["request"]["input"] = "Continue."
    second = adapter.run_codex(codex_payload)

    first_command = mock_run.call_args_list[0].args[0]
    second_command = mock_run.call_args_list[1].args[0]
    assert "--ephemeral" not in first_command
    assert "resume" not in first_command
    assert second_command[-3:] == ["resume", "thread-123", "-"]
    assert first["response"] == "first response"
    assert second["response"] == "second response"
    assert second["thread_id"] == "thread-123"
    assert second["usage"]["cached_input_tokens"] == 2
    child_env = mock_run.call_args_list[0].kwargs["env"]
    assert child_env["CODEX_HOME"] == str(tmp_path / "codex-home")
    assert child_env["CODEX_EXPLICIT"] == "forward-me"
    assert "OPENAI_API_KEY" not in child_env
    assert mock_run.call_args_list[0].kwargs["timeout"] == 1800

    state_path = adapter.runtime_state_path(codex_payload, "runtime-1")
    assert json.loads(state_path.read_text(encoding="utf-8")) == {
        "runtime_id": "runtime-1",
        "thread_id": "thread-123",
    }


def test_runtime_rejects_corrupt_codex_thread_state(codex_payload):
    state_path = adapter.runtime_state_path(codex_payload, "runtime-1")
    state_path.parent.mkdir(parents=True)
    state_path.write_text("{", encoding="utf-8")

    with pytest.raises(RuntimeError, match="invalid Codex runtime state"):
        adapter.load_thread_id(codex_payload, "runtime-1")


def test_runtime_persists_codex_thread_state(codex_payload, monkeypatch):
    mock_run = MagicMock(
        return_value=subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=codex_jsonl("thread-ephemeral", "done"),
            stderr="",
        )
    )
    monkeypatch.setattr(adapter.subprocess, "run", mock_run)

    output = adapter.run_codex(codex_payload)

    assert output["thread_id"] == "thread-ephemeral"
    assert output["response"] == "done"
    assert "events" not in output
    assert "stdout" not in output
    assert "stderr" not in output
    assert (Path(output["state_dir"]) / "runtimes").exists()


def test_adapter_rejects_structured_input_until_chat_is_supported(codex_payload):
    codex_payload["request"]["input"] = {
        "messages": [{"role": "user", "content": "Inspect the change."}]
    }

    with pytest.raises(ValueError, match="requires text input"):
        adapter.request_to_prompt(codex_payload)


@pytest.mark.parametrize("env", [[], "CODEX_FLAG=1"])
def test_adapter_rejects_non_mapping_env(codex_payload, env):
    settings = codex_payload["effective_config"]["config"]["harness"]["settings"]
    settings["env"] = env

    with pytest.raises(ValueError, match="env must be a mapping"):
        adapter.build_env(codex_payload)


@pytest.mark.parametrize(
    ("error", "message", "returncode"),
    [
        (FileNotFoundError("codex not found"), "codex not found", 127),
        (
            subprocess.TimeoutExpired(["codex"], 1800),
            "timed out after 1800 seconds",
            124,
        ),
    ],
)
def test_process_launch_failures_return_structured_results(
    codex_payload, monkeypatch, error, message, returncode
):
    monkeypatch.setattr(adapter.subprocess, "run", MagicMock(side_effect=error))

    output = adapter.run_codex(codex_payload)

    assert output["failed"] is True
    assert output["returncode"] == returncode
    assert message in output["error"]
    assert "stdout" not in output
    assert "stderr" not in output


def test_thread_mismatch_preserves_process_error(codex_payload, monkeypatch):
    adapter.save_thread_id(codex_payload, "runtime-1", "thread-persisted")
    monkeypatch.setattr(
        adapter.subprocess,
        "run",
        MagicMock(
            return_value=subprocess.CompletedProcess(
                args=[],
                returncode=1,
                stdout=codex_jsonl("thread-unexpected", "failed response"),
                stderr="Codex process failed",
            )
        ),
    )

    output = adapter.run_codex(codex_payload)

    assert output["failed"] is True
    assert output["error"] == "Codex process failed"


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), "30"])
def test_adapter_rejects_invalid_timeout(codex_payload, timeout):
    settings = codex_payload["effective_config"]["config"]["harness"]["settings"]
    settings["timeout_seconds"] = timeout

    with pytest.raises(ValueError, match="timeout_seconds"):
        adapter.run_codex(codex_payload)


def test_runtime_fails_if_codex_does_not_return_thread_identity(
    codex_payload, monkeypatch
):
    mock_run = MagicMock(
        return_value=subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps(
                {
                    "type": "item.completed",
                    "item": {"type": "agent_message", "text": "untracked"},
                }
            ),
            stderr="",
        )
    )
    monkeypatch.setattr(adapter.subprocess, "run", mock_run)

    output = adapter.run_codex(codex_payload)

    assert output["failed"] is True
    assert "thread identity" in output["error"]


def test_successful_process_without_final_response_is_failed(
    codex_payload, monkeypatch
):
    mock_run = MagicMock(
        return_value=subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="\n".join(
                [
                    json.dumps({"type": "thread.started", "thread_id": "thread-123"}),
                    json.dumps({"type": "turn.completed", "usage": {}}),
                ]
            ),
            stderr="",
        )
    )
    monkeypatch.setattr(adapter.subprocess, "run", mock_run)

    output = adapter.run_codex(codex_payload)

    assert output["failed"] is True
    assert "final agent message" in output["error"]


async def test_fabric_runtime_invokes_codex_then_resumes(tmp_path):
    mock_codex = tmp_path / "codex"
    write_mock_codex(mock_codex)
    config = fabric_config(tmp_path, mock_codex)

    async with await Fabric().start_runtime(
        config,
        base_dir=tmp_path,
    ) as runtime:
        first = await runtime.invoke(input="first")
        second = await runtime.invoke(input="second")

    assert first.runtime_id == second.runtime_id
    assert first.output["response"] == "thread-fake:first"
    assert second.output["response"] == "thread-fake:second"
    assert first.output["thread_id"] == second.output["thread_id"] == "thread-fake"
    assert "resume" not in first.output["command"]
    assert second.output["command"][-3:] == ["resume", "thread-fake", "-"]


async def test_fabric_oneshot_uses_cached_codex_auth(tmp_path):
    mock_codex = tmp_path / "codex"
    write_mock_codex(mock_codex)
    config = fabric_config(tmp_path, mock_codex)
    os.environ.pop("OPENAI_API_KEY", None)

    client = Fabric()
    report = await client.doctor(config, base_dir=tmp_path)
    result = await client.run(
        config,
        base_dir=tmp_path,
        input="inspect",
    )

    assert report.status == "pass"
    assert any(
        check.name == "requirement.binary" and "codex_command" in check.message
        for check in report.checks
    )
    assert not any(check.name == "requirement.env" for check in report.checks)
    assert result.output["response"] == "thread-fake:inspect"
    assert "--ephemeral" not in result.output["command"]


def test_codex_profile_resolves_runtime_adapter():
    from examples.code_review_agent import BASE_DIR
    from examples.code_review_agent import codex_cli_config

    plan = Fabric().plan(
        codex_cli_config(),
        base_dir=BASE_DIR,
    )

    assert plan.adapter.adapter_id == "nvidia.fabric.codex.cli"
    assert plan.adapter.harness == "codex"
    assert "mode" not in plan.effective_config.config.runtime
    assert plan.effective_config.config.runtime.input_schema == "text"
    settings = plan.effective_config.config.harness.settings
    assert settings["config_overrides"]["model_reasoning_effort"] == "high"
    unsupported = plan["capability_plan"]["unsupported"]
    assert not unsupported.get("skill_paths")
    assert not unsupported.get("mcp_servers")
