# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import os
import tomllib
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from claude_agent_sdk import AssistantMessage
from claude_agent_sdk import ClaudeSDKError
from claude_agent_sdk import CLIConnectionError
from claude_agent_sdk import CLIJSONDecodeError
from claude_agent_sdk import CLINotFoundError
from claude_agent_sdk import ProcessError
from claude_agent_sdk import ResultMessage
from claude_agent_sdk import SystemMessage
from claude_agent_sdk import TextBlock
from claude_agent_sdk._errors import MessageParseError
from nemo_fabric_adapters.claude import adapter

ROOT = Path(__file__).resolve().parents[2]
ANTHROPIC_AUTH_ENV_NAMES = {
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_CONFIG_DIR",
    "ANTHROPIC_FEDERATION_RULE_ID",
    "ANTHROPIC_IDENTITY_TOKEN",
    "ANTHROPIC_IDENTITY_TOKEN_FILE",
    "ANTHROPIC_ORGANIZATION_ID",
    "ANTHROPIC_PROFILE",
    "ANTHROPIC_SERVICE_ACCOUNT_ID",
    "ANTHROPIC_WORKSPACE_ID",
}


def test_claude_descriptor_is_narrow_and_versioned():
    descriptor_path = ROOT / "adapters" / "claude" / "fabric-adapter.json"
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))

    assert descriptor == {
        "contract_version": "fabric.adapter/v1alpha1",
        "adapter_id": "nvidia.fabric.claude",
        "harness": "claude",
        "adapter_kind": "python",
        "runner": {
            "module": "nemo_fabric_adapters.claude.adapter",
            "callable": "run",
        },
        "config": {
            "accepts": [
                "models",
                "tools",
                "tools.blocked",
                "mcp",
                "skills",
                "telemetry",
            ]
        },
        "telemetry": {
            "providers": {
                "relay": {
                    "outputs": ["atif", "otel", "openinference"],
                    "integration_modes": ["hooks", "gateway"],
                }
            }
        },
    }


@pytest.fixture(name="claude_payload")
def claude_payload_fixture(tmp_path) -> dict[str, Any]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    skill_path = tmp_path / "skills" / "review"
    skill_path.mkdir(parents=True)
    (skill_path / "SKILL.md").write_text("# Review\n", encoding="utf-8")
    return {
        "agent_name": "claude-test",
        "base_dir": str(tmp_path),
        "config": {
                "harness": {
                    "adapter_id": "nvidia.fabric.claude",
                    "settings": {
                        "system_prompt": "Review carefully.",
                        "allowed_tools": ["Read"],
                        "permission_mode": "dontAsk",
                        "max_turns": 4,
                        "max_budget_usd": 1.5,
                        "setting_sources": [],
                        "timeout_seconds": 30,
                        "env": {"ANTHROPIC_API_KEY": "configured-secret"},
                    },
                },
                "models": {
                    "default": {
                        "provider": "anthropic",
                        "model": "anthropic/claude-test-model",
                        "api_key_env": "ANTHROPIC_API_KEY",
                    }
                },
                "tools": {"blocked": ["Bash"]},
        },
        "runtime_context": {
            "runtime_id": "runtime-claude-1",
            "invocation_id": "invocation-1",
            "environment": {"workspace": str(workspace)},
            "artifacts": {"root": str(tmp_path / "artifacts"), "artifacts": []},
        },
        "request": {"request_id": "request-1", "input": "Inspect the patch"},
        "capability_plan": {
            "native": {
                "tools_configured": True,
                "mcp_servers": {
                    "repo": {
                        "transport": "stdio",
                        "url": "repo-mcp --root .",
                        "exposure": "harness_native",
                    },
                    "docs": {
                        "transport": "streamable-http",
                        "url": "https://mcp.example.test",
                        "exposure": "harness_native",
                    },
                },
                "skill_paths": [str(skill_path)],
            }
        },
    }


def test_build_options_maps_normalized_capabilities_and_claude_settings(claude_payload):
    options = adapter.build_options(claude_payload, resume="claude-session")

    assert options.resume == "claude-session"
    assert options.cwd == Path(
        claude_payload["runtime_context"]["environment"]["workspace"]
    )
    assert options.model == "claude-test-model"
    assert options.system_prompt == "Review carefully."
    assert options.tools is None
    assert options.allowed_tools == ["Read"]
    assert options.disallowed_tools == ["Bash"]
    assert options.permission_mode == "dontAsk"
    assert options.max_turns == 4
    assert options.max_budget_usd == 1.5
    assert options.setting_sources == []
    assert options.skills == "all"
    assert len(options.plugins) == 1
    plugin_path = Path(options.plugins[0]["path"])
    assert options.plugins[0]["type"] == "local"
    assert json.loads((plugin_path / ".claude-plugin" / "plugin.json").read_text()) == {
        "description": "Skills provided by NeMo Fabric",
        "name": "nemo-fabric-skills",
        "version": "1.0.0",
    }
    assert (plugin_path / "skills" / "review" / "SKILL.md").read_text() == "# Review\n"
    assert options.strict_mcp_config is True
    assert options.mcp_servers == {
        "docs": {"type": "http", "url": "https://mcp.example.test"},
        "repo": {"type": "stdio", "command": "repo-mcp", "args": ["--root", "."]},
    }
    assert "NEMO_RELAY_GATEWAY_URL" not in options.env
    assert "ANTHROPIC_BASE_URL" not in options.env


@pytest.fixture(name="relay_payload")
def relay_payload_fixture(claude_payload, tmp_path) -> dict[str, Any]:
    relay_intent_path = tmp_path / "relay-config.json"
    relay_intent_path.write_text(
        json.dumps(
            {
                "relay": {
                    "config": {
                        "atof": {"enabled": True},
                        "atif": {"enabled": True},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    os.environ["FABRIC_RELAY_CONFIG_PATH"] = str(relay_intent_path)
    claude_payload["telemetry_plan"] = {
        "providers": ["relay"],
        "relay_enabled": True,
    }
    return claude_payload


def test_prepare_claude_relay_writes_gateway_config_and_complete_hook_plugin(
    relay_payload, monkeypatch, tmp_path
):
    executable = tmp_path / "bin" / "nemo-relay"
    executable.parent.mkdir()
    executable.touch()
    monkeypatch.setattr(
        adapter.relay_gateway,
        "resolve_relay_command",
        MagicMock(return_value=executable),
    )
    monkeypatch.setattr(
        adapter.relay_gateway,
        "find_available_tcp_port",
        MagicMock(return_value=43210),
    )
    monkeypatch.setattr(
        adapter.relay_gateway,
        "relay_cli_contract",
        MagicMock(
            return_value=adapter.relay_gateway.RelayCliContract(
                version=(0, 6, 0), observability_version=2
            )
        ),
    )

    relay = adapter.prepare_claude_relay(relay_payload)

    assert relay is not None
    assert relay.gateway.executable == executable
    assert relay.gateway.bind == "127.0.0.1:43210"
    assert relay.gateway.url == "http://127.0.0.1:43210"
    assert relay.gateway.log_path == relay.gateway.config_path.parent / "gateway.log"
    with relay.gateway.config_path.open("rb") as stream:
        assert tomllib.load(stream) == {"agents": {"claude": {"command": "claude"}}}
    with (relay.gateway.config_path.parent / "plugins.toml").open("rb") as stream:
        plugin_config = tomllib.load(stream)
    assert plugin_config["components"][0]["kind"] == "observability"

    manifest = json.loads(
        (relay.plugin_path / ".claude-plugin" / "plugin.json").read_text(
            encoding="utf-8"
        )
    )
    hooks = json.loads(
        (relay.plugin_path / "hooks" / "hooks.json").read_text(encoding="utf-8")
    )["hooks"]
    assert manifest["name"] == "nemo-fabric-relay"
    assert set(hooks) == {
        "SessionStart",
        "UserPromptSubmit",
        "UserPromptExpansion",
        "PreToolUse",
        "PostToolUse",
        "PostToolUseFailure",
        "PermissionRequest",
        "SubagentStart",
        "SubagentStop",
        "Notification",
        "Stop",
        "PreCompact",
        "PostCompact",
        "SessionEnd",
    }
    assert hooks["SessionStart"][0] == {
        "hooks": [
            {
                "type": "command",
                "command": f"{executable} hook-forward claude",
                "timeout": 30,
            }
        ]
    }
    assert hooks["PermissionRequest"][0]["matcher"] == "*"


def test_build_options_adds_relay_plugin_and_gateway_environment(
    relay_payload, monkeypatch, tmp_path
):
    executable = tmp_path / "nemo-relay"
    executable.touch()
    monkeypatch.setattr(
        adapter.relay_gateway,
        "resolve_relay_command",
        MagicMock(return_value=executable),
    )
    monkeypatch.setattr(
        adapter.relay_gateway,
        "find_available_tcp_port",
        MagicMock(return_value=43210),
    )
    monkeypatch.setattr(
        adapter.relay_gateway,
        "relay_cli_contract",
        MagicMock(
            return_value=adapter.relay_gateway.RelayCliContract(
                version=(0, 6, 0), observability_version=2
            )
        ),
    )
    relay = adapter.prepare_claude_relay(relay_payload)

    options = adapter.build_options(relay_payload, resume=None, relay=relay)

    assert options.env["NEMO_RELAY_GATEWAY_URL"] == relay.gateway.url
    assert options.env["ANTHROPIC_BASE_URL"] == relay.gateway.url
    assert len(options.plugins) == 2
    assert Path(options.plugins[1]["path"]) == relay.plugin_path
    assert (
        Path(options.plugins[0]["path"]) / "skills" / "review" / "SKILL.md"
    ).exists()


def test_build_options_does_not_enable_skills_for_relay_plugin_alone(
    relay_payload, tmp_path
):
    relay_payload["capability_plan"]["native"]["skill_paths"] = []
    relay = adapter.ClaudeRelaySettings(
        gateway=adapter.relay_gateway.RelayGatewayLaunch(
            executable=tmp_path / "nemo-relay",
            config_path=tmp_path / "relay-config" / "config.toml",
            bind="127.0.0.1:43210",
            url="http://127.0.0.1:43210",
            log_path=tmp_path / "relay-config" / "gateway.log",
        ),
        plugin_config={"version": 1, "components": []},
        plugin_path=tmp_path / "relay-plugin",
    )

    options = adapter.build_options(relay_payload, resume=None, relay=relay)

    assert options.tools is None
    assert options.skills is None
    assert options.plugins == [{"type": "local", "path": str(relay.plugin_path)}]


def test_build_options_maps_blocked_tools_to_disallowed_tools(claude_payload):
    claude_payload["config"]["tools"] = {
        "blocked": ["Bash", "WebFetch"]
    }

    options = adapter.build_options(claude_payload, resume=None)

    assert options.tools is None
    assert options.disallowed_tools == ["Bash", "WebFetch"]


@pytest.mark.parametrize(
    ("name", "normalized_field"),
    [
        ("model_name", "FabricConfig.models"),
        ("cwd", "FabricConfig.environment.workspace"),
        ("tools", "FabricConfig.tools"),
        ("disallowed_tools", "FabricConfig.tools.blocked"),
        ("mcp_servers", "FabricConfig.mcp"),
        ("skills", "FabricConfig.skills"),
    ],
)
def test_build_options_rejects_normalized_capabilities_in_harness_settings(
    claude_payload, name, normalized_field
):
    claude_payload["config"]["harness"]["settings"][name] = []

    with pytest.raises(
        adapter.AdapterConfigError, match=normalized_field.replace(".", r"\.")
    ):
        adapter.build_options(claude_payload, resume=None)


def test_build_options_rejects_skill_path_without_skill_manifest(claude_payload):
    skill_path = Path(claude_payload["capability_plan"]["native"]["skill_paths"][0])
    (skill_path / "SKILL.md").unlink()

    with pytest.raises(adapter.AdapterConfigError, match="SKILL.md"):
        adapter.build_options(claude_payload, resume=None)


def test_build_options_maps_nvidia_provider_to_claude_gateway_environment(
    claude_payload,
):
    model = claude_payload["config"]["models"]["default"]
    model.update(
        {
            "provider": "nvidia",
            "model": "aws/anthropic/claude-opus-4-5",
            "api_key_env": "NVIDIA_API_KEY",
            "settings": {"base_url": "https://nvidia.example/"},
        }
    )
    os.environ["NVIDIA_API_KEY"] = "nvidia-secret"

    options = adapter.build_options(claude_payload, resume=None)

    assert options.model == "aws/anthropic/claude-opus-4-5"
    assert options.env["ANTHROPIC_BASE_URL"] == "https://nvidia.example"
    assert options.env["ANTHROPIC_AUTH_TOKEN"] == "nvidia-secret"
    assert options.env["ANTHROPIC_API_KEY"] == ""


def test_build_options_defaults_nvidia_provider_endpoint_and_credential(
    claude_payload,
):
    model = claude_payload["config"]["models"]["default"]
    model.update(
        {
            "provider": "nvidia",
            "model": "aws/anthropic/claude-opus-4-5",
        }
    )
    model.pop("api_key_env")
    claude_payload["config"]["harness"]["settings"].pop("env")
    os.environ["NVIDIA_API_KEY"] = "nvidia-secret"

    options = adapter.build_options(claude_payload, resume=None)

    assert options.env["ANTHROPIC_BASE_URL"] == adapter.NVIDIA_ANTHROPIC_BASE_URL
    assert options.env["ANTHROPIC_AUTH_TOKEN"] == "nvidia-secret"


def test_build_options_requires_nvidia_provider_credential(claude_payload):
    model = claude_payload["config"]["models"]["default"]
    model.update(
        {
            "provider": "nvidia",
            "model": "aws/anthropic/claude-opus-4-5",
            "api_key_env": "NVIDIA_API_KEY",
        }
    )
    os.environ.pop("NVIDIA_API_KEY", None)

    with pytest.raises(adapter.AdapterConfigError, match="NVIDIA_API_KEY is required"):
        adapter.build_options(claude_payload, resume=None)


def test_selected_model_rejects_unsupported_provider(claude_payload):
    model = claude_payload["config"]["models"]["default"]
    model["provider"] = "openai"

    with pytest.raises(
        adapter.AdapterConfigError, match="provider must be anthropic or nvidia"
    ):
        adapter.selected_model(claude_payload)


def test_state_round_trip_is_keyed_by_fabric_runtime(claude_payload):
    runtime_id = adapter.runtime_id(claude_payload)
    adapter.save_claude_session_id(claude_payload, runtime_id, "claude-session")

    assert (
        adapter.load_claude_session_id(claude_payload, runtime_id) == "claude-session"
    )
    state_path = adapter.runtime_state_path(claude_payload, runtime_id)
    assert state_path.parent.name == "runtimes"
    assert runtime_id not in state_path.name


def test_state_loader_rejects_non_object_json(claude_payload):
    runtime_id = adapter.runtime_id(claude_payload)
    state_path = adapter.runtime_state_path(claude_payload, runtime_id)
    state_path.parent.mkdir(parents=True)
    state_path.write_text("[]", encoding="utf-8")

    with pytest.raises(adapter.AdapterStateError, match="runtime state is invalid"):
        adapter.load_claude_session_id(claude_payload, runtime_id)


def test_normalize_result_exposes_session_usage_cost_and_buffered_events(
    claude_payload,
):
    messages = [
        SystemMessage(subtype="init", data={"session_id": "claude-session"}),
        AssistantMessage(
            content=[TextBlock(text="done")],
            model="claude-test-model",
            usage={"input_tokens": 10, "output_tokens": 3},
            session_id="claude-session",
        ),
    ]
    result = ResultMessage(
        subtype="success",
        duration_ms=100,
        duration_api_ms=80,
        is_error=False,
        num_turns=1,
        session_id="claude-session",
        total_cost_usd=0.02,
        usage={"input_tokens": 10, "output_tokens": 3},
        result="done",
    )

    output = adapter.normalize_result(claude_payload, messages, result)

    assert output["response"] == "done"
    assert output["session_id"] == "claude-session"
    assert output["usage"] == {"input_tokens": 10, "output_tokens": 3}
    assert output["cost_usd"] == 0.02
    assert output["duration_ms"] == 100
    assert [event["type"] for event in output["events"]] == [
        "SystemMessage",
        "AssistantMessage",
    ]


async def test_run_claude_resumes_and_persists_session(claude_payload, monkeypatch):
    captured = []

    async def query_result(*, prompt, options):
        captured.append((prompt, options.resume, dict(options.env), dict(os.environ)))
        yield AssistantMessage(
            content=[TextBlock(text="done")], model="claude-test-model"
        )
        yield ResultMessage(
            subtype="success",
            duration_ms=100,
            duration_api_ms=80,
            is_error=False,
            num_turns=1,
            session_id="claude-session",
            total_cost_usd=0.02,
            usage={"input_tokens": 1, "output_tokens": 1},
            result="done",
        )

    mock_query = MagicMock(side_effect=query_result)
    monkeypatch.setattr(adapter, "query", mock_query)
    monkeypatch.setenv("FABRIC_UNRELATED_SECRET", "do-not-forward")

    first = await adapter.run_claude(claude_payload)
    claude_payload["runtime_context"]["invocation_id"] = "invocation-2"
    second = await adapter.run_claude(claude_payload)

    assert first["failed"] is False
    assert second["failed"] is False
    assert [entry[0] for entry in captured] == [
        "Inspect the patch",
        "Inspect the patch",
    ]
    assert [entry[1] for entry in captured] == [None, "claude-session"]
    assert all(entry[2]["FABRIC_UNRELATED_SECRET"] == "" for entry in captured)
    assert all(
        entry[2]["ANTHROPIC_API_KEY"] == "configured-secret" for entry in captured
    )
    assert all(
        entry[3]["FABRIC_UNRELATED_SECRET"] == "do-not-forward" for entry in captured
    )
    assert os.environ["FABRIC_UNRELATED_SECRET"] == "do-not-forward"


async def test_run_claude_supervises_relay_and_reports_artifacts(
    relay_payload, monkeypatch, tmp_path
):
    executable = tmp_path / "nemo-relay"
    executable.touch()
    relay = adapter.ClaudeRelaySettings(
        gateway=adapter.relay_gateway.RelayGatewayLaunch(
            executable=executable,
            config_path=tmp_path / "relay-config" / "config.toml",
            bind="127.0.0.1:43210",
            url="http://127.0.0.1:43210",
            log_path=tmp_path / "relay-config" / "gateway.log",
        ),
        plugin_config={
            "version": 1,
            "components": [
                {
                    "kind": "observability",
                    "enabled": True,
                    "config": {
                        "atif": {
                            "enabled": True,
                            "output_directory": str(tmp_path / "atif"),
                        }
                    },
                }
            ],
        },
        plugin_path=tmp_path / "relay-plugin",
    )
    relay.plugin_path.mkdir()
    relay.gateway.log_path.parent.mkdir()
    relay.gateway.log_path.write_text("gateway started\n", encoding="utf-8")
    atif_path = tmp_path / "atif" / "trajectory-session.atif.json"
    atif_path.parent.mkdir()
    atif_path.write_text("{}", encoding="utf-8")
    process = MagicMock()
    mock_start = MagicMock(return_value=process)
    mock_stop = MagicMock()
    monkeypatch.setattr(adapter, "prepare_claude_relay", MagicMock(return_value=relay))
    monkeypatch.setattr(adapter.relay_gateway, "start_relay_gateway", mock_start)
    monkeypatch.setattr(adapter.relay_gateway, "stop_relay_gateway", mock_stop)

    async def query_result(*, prompt, options):
        assert options.env["ANTHROPIC_BASE_URL"] == relay.gateway.url
        assert Path(options.plugins[-1]["path"]) == relay.plugin_path
        yield ResultMessage(
            subtype="success",
            duration_ms=10,
            duration_api_ms=8,
            is_error=False,
            num_turns=1,
            session_id="claude-session",
            total_cost_usd=0.01,
            usage={"input_tokens": 1, "output_tokens": 1},
            result="done",
        )

    monkeypatch.setattr(adapter, "query", MagicMock(side_effect=query_result))

    output = await adapter.run_claude(relay_payload)

    assert output["relay_runtime"] == {
        "enabled": True,
        "emitter": "claude-agent-sdk/nemo-relay",
        "config_path": os.environ["FABRIC_RELAY_CONFIG_PATH"],
        "gateway_config_path": str(relay.gateway.config_path),
        "gateway_url": relay.gateway.url,
        "gateway_log_path": str(relay.gateway.log_path),
    }
    assert output["relay_artifacts"] == [{"kind": "atif", "path": str(atif_path)}]
    mock_start.assert_called_once_with(
        launch=relay.gateway,
        cwd=Path(relay_payload["runtime_context"]["environment"]["workspace"]),
    )
    mock_stop.assert_called_once_with(process)
    assert not relay.plugin_path.exists()


async def test_run_claude_preserves_result_when_relay_stop_fails(
    relay_payload, monkeypatch, tmp_path
):
    relay = adapter.ClaudeRelaySettings(
        gateway=adapter.relay_gateway.RelayGatewayLaunch(
            executable=tmp_path / "nemo-relay",
            config_path=tmp_path / "relay-config" / "config.toml",
            bind="127.0.0.1:43210",
            url="http://127.0.0.1:43210",
            log_path=tmp_path / "relay-config" / "gateway.log",
        ),
        plugin_config={"version": 1, "components": []},
        plugin_path=tmp_path / "relay-plugin",
    )
    relay.plugin_path.mkdir()
    monkeypatch.setattr(adapter, "prepare_claude_relay", MagicMock(return_value=relay))
    monkeypatch.setattr(
        adapter.relay_gateway,
        "start_relay_gateway",
        MagicMock(return_value=MagicMock()),
    )
    monkeypatch.setattr(
        adapter.relay_gateway,
        "stop_relay_gateway",
        MagicMock(
            side_effect=adapter.relay_gateway.RelayGatewayError("raw stop failure")
        ),
    )

    async def query_result(*, prompt, options):
        yield ResultMessage(
            subtype="success",
            duration_ms=10,
            duration_api_ms=8,
            is_error=False,
            num_turns=1,
            session_id="claude-session",
            total_cost_usd=0.01,
            usage={"input_tokens": 1, "output_tokens": 1},
            result="done",
        )

    monkeypatch.setattr(adapter, "query", MagicMock(side_effect=query_result))

    output = await adapter.run_claude(relay_payload)

    assert output["response"] == "done"
    assert output["completed"] is False
    assert output["failed"] is True
    assert output["error"] == {
        "code": "claude_relay_stop_failed",
        "message": "NeMo Relay gateway failed to stop",
        "retryable": False,
        "metadata": {"gateway_log_path": str(relay.gateway.log_path)},
    }
    assert output["relay_runtime"]["cleanup_error"] == output["error"]
    assert "raw stop failure" not in json.dumps(output)
    assert not relay.plugin_path.exists()


async def test_run_claude_preserves_result_when_relay_plugin_cleanup_fails(
    relay_payload, monkeypatch, tmp_path
):
    relay = adapter.ClaudeRelaySettings(
        gateway=adapter.relay_gateway.RelayGatewayLaunch(
            executable=tmp_path / "nemo-relay",
            config_path=tmp_path / "relay-config" / "config.toml",
            bind="127.0.0.1:43210",
            url="http://127.0.0.1:43210",
            log_path=tmp_path / "relay-config" / "gateway.log",
        ),
        plugin_config={"version": 1, "components": []},
        plugin_path=tmp_path / "relay-plugin",
    )
    relay.plugin_path.mkdir()
    process = MagicMock()
    mock_stop = MagicMock()
    mock_rmtree = MagicMock(side_effect=OSError("raw plugin cleanup failure"))
    monkeypatch.setattr(adapter, "prepare_claude_relay", MagicMock(return_value=relay))
    monkeypatch.setattr(
        adapter.relay_gateway,
        "start_relay_gateway",
        MagicMock(return_value=process),
    )
    monkeypatch.setattr(adapter.relay_gateway, "stop_relay_gateway", mock_stop)
    monkeypatch.setattr(adapter.shutil, "rmtree", mock_rmtree)

    async def query_result(**_):
        yield ResultMessage(
            subtype="success",
            duration_ms=10,
            duration_api_ms=8,
            is_error=False,
            num_turns=1,
            session_id="claude-session",
            total_cost_usd=0.01,
            usage={"input_tokens": 1, "output_tokens": 1},
            result="done",
        )

    monkeypatch.setattr(adapter, "query", MagicMock(side_effect=query_result))

    output = await adapter.run_claude(relay_payload)

    assert output["response"] == "done"
    assert output["completed"] is False
    assert output["failed"] is True
    assert output["error"] == {
        "code": "claude_relay_cleanup_failed",
        "message": "Claude Relay hook configuration could not be removed",
        "retryable": False,
    }
    assert output["relay_runtime"]["cleanup_error"] == output["error"]
    assert "raw plugin cleanup failure" not in json.dumps(output)
    mock_stop.assert_called_once_with(process)
    mock_rmtree.assert_called_once_with(relay.plugin_path)
    assert relay.plugin_path.exists()


@pytest.mark.parametrize(
    "failure", [ClaudeSDKError("sdk failed"), asyncio.CancelledError()]
)
async def test_run_claude_stops_relay_on_sdk_failure_or_cancellation(
    relay_payload, monkeypatch, tmp_path, failure
):
    relay = adapter.ClaudeRelaySettings(
        gateway=adapter.relay_gateway.RelayGatewayLaunch(
            executable=tmp_path / "nemo-relay",
            config_path=tmp_path / "relay-config" / "config.toml",
            bind="127.0.0.1:43210",
            url="http://127.0.0.1:43210",
            log_path=tmp_path / "relay-config" / "gateway.log",
        ),
        plugin_config={"version": 1, "components": []},
        plugin_path=tmp_path / "relay-plugin",
    )
    relay.plugin_path.mkdir()
    process = MagicMock()
    mock_stop = MagicMock()
    monkeypatch.setattr(adapter, "prepare_claude_relay", MagicMock(return_value=relay))
    monkeypatch.setattr(
        adapter.relay_gateway,
        "start_relay_gateway",
        MagicMock(return_value=process),
    )
    monkeypatch.setattr(adapter.relay_gateway, "stop_relay_gateway", mock_stop)

    async def query_failure(*, prompt, options):
        raise failure
        yield

    monkeypatch.setattr(adapter, "query", MagicMock(side_effect=query_failure))

    if isinstance(failure, asyncio.CancelledError):
        with pytest.raises(asyncio.CancelledError):
            await adapter.run_claude(relay_payload)
    else:
        output = await adapter.run_claude(relay_payload)
        assert output["error"]["code"] == "claude_failed"
        assert output["relay_runtime"]["enabled"] is True

    mock_stop.assert_called_once_with(process)
    assert not relay.plugin_path.exists()


@pytest.mark.parametrize(
    ("subtype", "is_error"),
    [("success", True), ("error_max_budget_usd", False)],
)
async def test_run_claude_preserves_failed_result_when_sdk_stream_raises(
    claude_payload,
    monkeypatch,
    caplog,
    subtype,
    is_error,
):
    async def query_error_result(**_):
        yield ResultMessage(
            subtype=subtype,
            duration_ms=10,
            duration_api_ms=8,
            is_error=is_error,
            num_turns=1,
            session_id="claude-session",
            result="Not logged in",
        )
        raise RuntimeError("raw SDK stream error")

    monkeypatch.setattr(adapter, "query", MagicMock(side_effect=query_error_result))

    output = await adapter.run_claude(claude_payload)

    assert output["response"] == "Not logged in"
    assert output["error"] == {
        "code": "claude_result_failed",
        "message": "Claude returned an error result",
        "retryable": False,
        "metadata": {"subtype": subtype},
    }
    assert "raw SDK stream error" not in json.dumps(output)
    assert "raw SDK stream error" in caplog.text


def test_run_reports_relay_start_failure_without_raw_diagnostic(
    relay_payload, monkeypatch, tmp_path
):
    executable = tmp_path / "nemo-relay"
    executable.touch()
    relay = adapter.ClaudeRelaySettings(
        gateway=adapter.relay_gateway.RelayGatewayLaunch(
            executable=executable,
            config_path=tmp_path / "relay-config" / "config.toml",
            bind="127.0.0.1:43210",
            url="http://127.0.0.1:43210",
            log_path=tmp_path / "relay-config" / "gateway.log",
        ),
        plugin_config={"version": 1, "components": []},
        plugin_path=tmp_path / "relay-plugin",
    )
    relay.plugin_path.mkdir()
    monkeypatch.setattr(adapter, "prepare_claude_relay", MagicMock(return_value=relay))
    monkeypatch.setattr(
        adapter.relay_gateway,
        "start_relay_gateway",
        MagicMock(
            side_effect=adapter.relay_gateway.RelayGatewayError(
                "raw gateway failure with secret"
            )
        ),
    )

    output = adapter.run(relay_payload)

    assert output["error"] == {
        "code": "claude_relay_start_failed",
        "message": "NeMo Relay gateway failed to start",
        "retryable": False,
        "metadata": {"gateway_log_path": str(relay.gateway.log_path)},
    }
    assert "secret" not in json.dumps(output)
    assert not relay.plugin_path.exists()


@pytest.mark.parametrize(
    "auth_environment",
    [
        {
            "ANTHROPIC_CONFIG_DIR": "/run/anthropic",
            "ANTHROPIC_PROFILE": "production",
        },
        {
            "ANTHROPIC_FEDERATION_RULE_ID": "fdrl_test",
            "ANTHROPIC_ORGANIZATION_ID": "organization-test",
            "ANTHROPIC_SERVICE_ACCOUNT_ID": "svac_test",
            "ANTHROPIC_WORKSPACE_ID": "wrkspc_test",
            "ANTHROPIC_IDENTITY_TOKEN_FILE": "/run/secrets/anthropic/token",
        },
        {
            "ANTHROPIC_FEDERATION_RULE_ID": "fdrl_test",
            "ANTHROPIC_ORGANIZATION_ID": "organization-test",
            "ANTHROPIC_SERVICE_ACCOUNT_ID": "svac_test",
            "ANTHROPIC_IDENTITY_TOKEN": "identity-token",
        },
        {"ANTHROPIC_API_KEY": "default-secret"},
        {"ANTHROPIC_AUTH_TOKEN": "bearer-token"},
        {
            "ANTHROPIC_API_KEY": "",
            "ANTHROPIC_PROFILE": "fallback-profile",
        },
        {
            "ANTHROPIC_AUTH_TOKEN": "",
            "ANTHROPIC_API_KEY": "fallback-api-key",
            "ANTHROPIC_PROFILE": "fallback-profile",
        },
    ],
)
def test_build_options_forwards_anthropic_auth_environment(
    claude_payload, auth_environment
):
    model = claude_payload["config"]["models"]["default"]
    model.pop("api_key_env")
    settings = claude_payload["config"]["harness"]["settings"]
    settings.pop("env")
    for name in ANTHROPIC_AUTH_ENV_NAMES:
        os.environ.pop(name, None)
    os.environ["FABRIC_UNRELATED_SECRET"] = "do-not-forward"
    os.environ.update(auth_environment)

    options = adapter.build_options(claude_payload, resume=None)

    forwarded_auth_environment = {
        name: options.env[name]
        for name in ANTHROPIC_AUTH_ENV_NAMES
        if name in options.env
    }
    assert forwarded_auth_environment == auth_environment
    assert options.env["FABRIC_UNRELATED_SECRET"] == ""


def test_build_options_preserves_unix_user_for_cached_login(
    claude_payload,
):
    os.environ["USER"] = "fabric-user"

    options = adapter.build_options(claude_payload, resume=None)

    assert options.env["USER"] == "fabric-user"


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (CLINotFoundError("raw path", "/secret/claude"), "claude_cli_not_found"),
        (CLIConnectionError("raw connection"), "claude_connection_failed"),
        (
            ProcessError("raw process", exit_code=9, stderr="secret"),
            "claude_process_failed",
        ),
        (CLIJSONDecodeError("secret-json", ValueError("bad")), "claude_invalid_json"),
        (
            MessageParseError("raw parse", data={"secret": "value"}),
            "claude_message_parse_failed",
        ),
        (ClaudeSDKError("raw sdk"), "claude_failed"),
    ],
)
def test_sdk_errors_are_structured_without_raw_provider_data(error, code):
    output = adapter.sdk_failure(error)
    serialized = json.dumps(output)

    assert output["error"]["code"] == code
    assert "secret" not in serialized
    assert "raw " not in serialized


def test_error_result_is_normalized_as_failure(claude_payload):
    result = ResultMessage(
        subtype="error_max_turns",
        duration_ms=100,
        duration_api_ms=80,
        is_error=True,
        num_turns=4,
        session_id="claude-session",
        errors=["provider-specific failure"],
    )

    output = adapter.normalize_result(claude_payload, [], result)

    assert output["failed"] is True
    assert output["error"] == {
        "code": "claude_result_failed",
        "message": "Claude returned an error result",
        "retryable": False,
        "metadata": {"subtype": "error_max_turns"},
    }


def test_error_subtype_is_failure_when_sdk_flag_is_false(claude_payload):
    result = ResultMessage(
        subtype="error_max_budget_usd",
        duration_ms=100,
        duration_api_ms=80,
        is_error=False,
        num_turns=4,
        session_id="claude-session",
    )

    output = adapter.normalize_result(claude_payload, [], result)

    assert output["completed"] is False
    assert output["failed"] is True
    assert output["error"]["metadata"] == {"subtype": "error_max_budget_usd"}


def test_run_normalizes_unexpected_exception(claude_payload, monkeypatch):
    monkeypatch.setattr(
        adapter,
        "run_claude",
        MagicMock(side_effect=RuntimeError("secret")),
    )

    output = adapter.run(claude_payload)

    assert output["error"]["code"] == "claude_adapter_internal_error"
    assert "secret" not in json.dumps(output)


def test_main_normalizes_payload_load_failure(monkeypatch, capsys):
    monkeypatch.setattr(
        adapter.common_utils,
        "load_payload",
        MagicMock(side_effect=ValueError("secret payload")),
    )

    with pytest.raises(SystemExit, match="2"):
        adapter.main()

    output = json.loads(capsys.readouterr().out)
    assert output["error"]["code"] == "claude_adapter_internal_error"
    assert "secret" not in json.dumps(output)
