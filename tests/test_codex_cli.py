# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, call

import pytest
import yaml

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib

from nemo_fabric import Fabric, FabricConfig

ROOT = Path(__file__).resolve().parents[1]
ADAPTER_PATH = (
    ROOT
    / "adapters"
    / "codex-cli"
    / "src"
    / "nemo_fabric_adapters"
    / "codex_cli"
    / "adapter.py"
)


def load_codex_adapter():
    spec = importlib.util.spec_from_file_location("fabric_codex_adapter", ADAPTER_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
                "runtime": {"transport": "cli"},
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
                "transport": "cli",
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
    adapter = load_codex_adapter()
    codex_settings = adapter.write_config_files(codex_payload)

    command = adapter.build_command(
        codex_payload,
        codex_settings=codex_settings,
    )

    exec_index = command.index("exec")
    assert command[0] == "codex"
    assert command[1:exec_index] == []
    assert command[exec_index : exec_index + 3] == ["exec", "--json", "--ephemeral"]
    assert ["--sandbox", "read-only"] == command[exec_index + 3 : exec_index + 5]
    assert ["--profile", "fabric-runtime-1"] == command[exec_index + 5 : exec_index + 7]
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
    adapter = load_codex_adapter()
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
    adapter = load_codex_adapter()
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
    adapter = load_codex_adapter()
    os.environ["CODEX_HOME"] = str(tmp_path / "custom-codex-home")

    name, path = adapter.get_codex_profile_path(
        {"runtime_context": {"runtime_id": "runtime-1"}}
    )

    assert name == "fabric-runtime-1"
    assert path == tmp_path / "custom-codex-home" / "fabric-runtime-1.config.toml"


def test_codex_home_defaults_to_user_codex_directory():
    adapter = load_codex_adapter()
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
    adapter = load_codex_adapter()
    os.environ["FABRIC_RELAY_ENABLED"] = "true"
    mock_find_port = MagicMock(return_value=43210)
    monkeypatch.setattr(adapter, "find_available_tcp_port", mock_find_port)
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
        "command": "nemo-relay hook-forward codex",
        "timeout": 30,
    }
    mock_load_config.assert_called_once_with(codex_payload)
    mock_write_config.assert_called_once_with(
        relay_config={"agents": {"codex": {"command": "codex"}}},
        plugin_config=relay_plugin_config,
    )


def test_native_otel_profile_writes_codex_telemetry_config(codex_payload, tmp_path):
    adapter = load_codex_adapter()
    profile = yaml.safe_load(
        (ROOT / "examples/code-review-agent/profiles/native-otel.yaml").read_text(
            encoding="utf-8"
        )
    )
    config = codex_payload["effective_config"]["config"]
    config["telemetry"] = profile["telemetry"]
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
                "otlp-http": {
                    "endpoint": "http://localhost:4318/v1/traces",
                    "protocol": "binary",
                }
            },
        }
    }


def test_run_codex_configures_relay(codex_payload, monkeypatch, tmp_path):
    adapter = load_codex_adapter()
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
    codex_settings = adapter.CodexSettings(
        telemetry_provider="relay",
        relay_enabled=True,
        codex_profile_name=profile_name,
        codex_profile_path=profile_path,
        relay_gateway_host=gateway_host,
        relay_gateway_url=gateway_url,
        relay_gateway_port=43210,
        relay_config_path=relay_config_path,
        relay_plugin_config=relay_plugin_config,
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
    monkeypatch.setattr(adapter, "start_relay_gateway", mock_start_gateway)
    monkeypatch.setattr(adapter, "stop_relay_gateway", mock_stop_gateway)
    monkeypatch.setattr(adapter.time, "sleep", MagicMock())
    monkeypatch.setattr(
        adapter,
        "write_config_files",
        mock_write_config_files,
    )
    monkeypatch.setattr(adapter.subprocess, "run", mock_run)

    result = adapter.run_codex(codex_payload)

    command = mock_run.call_args.args[0]
    assert "relay_runtime" in result
    assert "relay_artifacts" in result
    assert command[0] == "codex"
    assert "nemo-relay" not in command
    assert command[command.index("--profile") + 1] == profile_name
    assert mock_run.call_args.kwargs["env"]["NEMO_RELAY_GATEWAY_URL"] == gateway_url
    assert "CODEX_HOME" not in mock_run.call_args.kwargs["env"]
    assert not profile_path.exists()
    mock_write_config_files.assert_called_once_with(codex_payload)
    mock_start_gateway.assert_called_once_with(
        codex_payload,
        Path(codex_payload["runtime_context"]["environment"]["workspace"]),
        codex_settings,
    )
    mock_stop_gateway.assert_called_once_with(mock_gateway)


def test_start_relay_gateway_waits_for_health_and_starts_process_group(
    codex_payload,
    monkeypatch,
    tmp_path,
):
    adapter = load_codex_adapter()
    relay_config_path = tmp_path / "relay-config" / "config.toml"
    relay_config_path.parent.mkdir()
    relay_config_path.write_text("[agents.codex]\ncommand = \"codex\"\n", encoding="utf-8")
    mock_process = MagicMock()
    mock_popen = MagicMock(return_value=mock_process)
    mock_wait = MagicMock()
    gateway_host = "127.0.0.1:43210"
    gateway_url = f"http://{gateway_host}"
    codex_settings = adapter.CodexSettings(
        telemetry_provider="relay",
        relay_enabled=True,
        codex_profile_name="fabric-runtime-1",
        codex_profile_path=tmp_path / "fabric-runtime-1.config.toml",
        relay_gateway_host=gateway_host,
        relay_gateway_url=gateway_url,
        relay_gateway_port=43210,
        relay_config_path=relay_config_path,
        relay_plugin_config={"version": 1, "components": []},
    )
    monkeypatch.setattr(adapter, "wait_for_relay_gateway", mock_wait)
    monkeypatch.setattr(adapter.subprocess, "Popen", mock_popen)

    process = adapter.start_relay_gateway(
        codex_payload,
        tmp_path,
        codex_settings,
    )

    assert process is mock_process
    assert mock_popen.call_args.args[0] == [
        "nemo-relay",
        "--config",
        str(relay_config_path),
        "--bind",
        gateway_host,
    ]
    assert mock_popen.call_args.kwargs["start_new_session"] is True
    mock_wait.assert_called_once_with(mock_process, f"{gateway_url}/healthz")


def test_wait_for_relay_gateway_times_out():
    adapter = load_codex_adapter()
    mock_process = MagicMock()
    mock_process.poll.return_value = None
    health_url = "http://127.0.0.1:43210/healthz"

    with pytest.raises(RuntimeError, match="gateway did not become ready"):
        adapter.wait_for_relay_gateway(mock_process, health_url, timeout=0)


def test_stop_relay_gateway_terminates_process():
    adapter = load_codex_adapter()
    mock_process = MagicMock()
    mock_process.poll.return_value = None

    adapter.stop_relay_gateway(mock_process)

    mock_process.terminate.assert_called_once_with()
    mock_process.wait.assert_called_once_with(timeout=5)


def test_stop_relay_gateway_kills_process_after_timeout():
    adapter = load_codex_adapter()
    mock_process = MagicMock()
    mock_process.poll.return_value = None
    mock_process.wait.side_effect = [subprocess.TimeoutExpired("nemo-relay", 5), None]

    adapter.stop_relay_gateway(mock_process)

    mock_process.terminate.assert_called_once_with()
    mock_process.kill.assert_called_once_with()
    assert mock_process.wait.call_args_list == [call(timeout=5), call(timeout=5)]


def test_reported_command_redacts_secret_config_overrides():
    adapter = load_codex_adapter()

    command = ["codex", "exec", "--config", 'provider.api_key="secret"', "-"]

    assert adapter.redact_command(command)[-2] == "<redacted>"
    assert command[-2] == 'provider.api_key="secret"'


def test_config_override_values_use_tomli_writer():
    adapter = load_codex_adapter()

    assert adapter.toml_value("café") == '"café"'
    encoded = adapter.toml_value([1, "two"])
    assert tomllib.loads(f"value = {encoded}")["value"] == [1, "two"]
    with pytest.raises(ValueError, match="scalar or array"):
        adapter.toml_value({"nested": True})


@pytest.mark.parametrize("value", [[float("nan")], [1, [float("inf")]]])
def test_config_override_values_reject_nested_non_finite_numbers(value):
    adapter = load_codex_adapter()

    with pytest.raises(ValueError, match="finite numbers"):
        adapter.toml_value(value)


def test_session_reuses_codex_thread_across_invocations(codex_payload, monkeypatch, tmp_path):
    adapter = load_codex_adapter()
    codex_payload["runtime_context"]["session_id"] = "review-session"
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
    assert second["session_id"] == "review-session"
    assert second["thread_id"] == "thread-123"
    assert second["usage"]["cached_input_tokens"] == 2
    child_env = mock_run.call_args_list[0].kwargs["env"]
    assert child_env["CODEX_HOME"] == str(tmp_path / "codex-home")
    assert child_env["CODEX_EXPLICIT"] == "forward-me"
    assert "OPENAI_API_KEY" not in child_env
    assert "FABRIC_UNRELATED_SECRET" not in child_env
    assert mock_run.call_args_list[0].kwargs["timeout"] == 1800

    state_path = adapter.session_state_path(codex_payload, "review-session")
    assert json.loads(state_path.read_text(encoding="utf-8")) == {
        "session_id": "review-session",
        "thread_id": "thread-123",
    }


def test_oneshot_does_not_persist_codex_thread(codex_payload, monkeypatch):
    adapter = load_codex_adapter()
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
    assert not (Path(output["state_dir"]) / "sessions").exists()


def test_adapter_rejects_structured_input_until_chat_is_supported(codex_payload):
    adapter = load_codex_adapter()
    codex_payload["request"]["input"] = {
        "messages": [{"role": "user", "content": "Inspect the change."}]
    }

    with pytest.raises(ValueError, match="requires text input"):
        adapter.request_to_prompt(codex_payload)


@pytest.mark.parametrize("env", [[], "CODEX_FLAG=1"])
def test_adapter_rejects_non_mapping_env(codex_payload, env):
    adapter = load_codex_adapter()
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
    adapter = load_codex_adapter()
    monkeypatch.setattr(adapter.subprocess, "run", MagicMock(side_effect=error))

    output = adapter.run_codex(codex_payload)

    assert output["failed"] is True
    assert output["returncode"] == returncode
    assert message in output["error"]
    assert "stdout" not in output
    assert "stderr" not in output


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), "30"])
def test_adapter_rejects_invalid_timeout(codex_payload, timeout):
    adapter = load_codex_adapter()
    settings = codex_payload["effective_config"]["config"]["harness"]["settings"]
    settings["timeout_seconds"] = timeout

    with pytest.raises(ValueError, match="timeout_seconds"):
        adapter.run_codex(codex_payload)


def test_session_fails_if_codex_does_not_return_thread_identity(
    codex_payload, monkeypatch
):
    adapter = load_codex_adapter()
    codex_payload["runtime_context"]["session_id"] = "review-session"
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


def test_successful_process_without_final_response_is_failed(codex_payload, monkeypatch):
    adapter = load_codex_adapter()
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


async def test_fabric_session_invokes_codex_then_resumes(tmp_path):
    mock_codex = tmp_path / "codex"
    write_mock_codex(mock_codex)
    config = fabric_config(tmp_path, mock_codex)

    async with await Fabric().start_session(
        config,
        base_dir=tmp_path,
        session_id="fabric-session",
    ) as session:
        assert session.session_id == "fabric-session"
        first = await session.invoke(input="first")
        second = await session.invoke(input="second")

    assert first.runtime_id == second.runtime_id
    assert first.output["response"] == "thread-fake:first"
    assert second.output["response"] == "thread-fake:second"
    assert first.output["thread_id"] == second.output["thread_id"] == "thread-fake"
    assert "resume" not in first.output["command"]
    assert second.output["command"][-3:] == ["resume", "thread-fake", "-"]


async def test_fabric_oneshot_is_ephemeral_and_uses_cached_codex_auth(tmp_path):
    mock_codex = tmp_path / "codex"
    write_mock_codex(mock_codex)
    config = fabric_config(tmp_path, mock_codex)
    os.environ.pop("OPENAI_API_KEY", None)

    async with Fabric() as client:
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
    assert "--ephemeral" in result.output["command"]
    assert result.output["session_id"] is None


@pytest.mark.parametrize(
    "profile",
    [
        "codex_cli",
        "codex_cli_session",
    ],
)
def test_codex_profiles_resolve_session_capability(profile):
    plan = Fabric().plan(
        ROOT / "examples" / "code-review-agent",
        profiles=[profile],
    )

    assert plan.adapter.adapter_id == "nvidia.fabric.codex.cli"
    assert plan.adapter.harness == "codex"
    assert "mode" not in plan.effective_config.config.runtime
    assert plan.effective_config.config.runtime.input_schema == "text"
    assert plan.capabilities.session is True
    settings = plan.effective_config.config.harness.settings
    assert settings["config_overrides"]["model_reasoning_effort"] == "high"
    unsupported = plan["capability_plan"]["unsupported"]
    assert not unsupported.get("skill_paths")
    assert not unsupported.get("mcp_servers")
