# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import os
import tomllib
from collections.abc import AsyncIterator
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from claude_agent_sdk import AssistantMessage
from claude_agent_sdk import ClaudeAgentOptions
from claude_agent_sdk import ClaudeSDKError
from claude_agent_sdk import CLIConnectionError
from claude_agent_sdk import CLIJSONDecodeError
from claude_agent_sdk import CLINotFoundError
from claude_agent_sdk import Message
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


def lifecycle_invocation(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "runtime_context": payload["runtime_context"],
        "request": payload["request"],
    }


def install_fake_client(
    monkeypatch: pytest.MonkeyPatch,
    response_factory: Callable[[MagicMock], AsyncIterator[Message]],
) -> list[MagicMock]:
    clients: list[MagicMock] = []
    client_type = adapter.ClaudeSDKClient

    def build_client(options: ClaudeAgentOptions) -> MagicMock:
        client = MagicMock(spec=client_type)
        client.options = options
        client.prompts = []
        client.connect = AsyncMock()
        client.query = AsyncMock(side_effect=client.prompts.append)
        client.receive_response.side_effect = lambda: response_factory(client)
        client.disconnect = AsyncMock()
        client.interrupt = AsyncMock()
        clients.append(client)
        return client

    monkeypatch.setattr(adapter, "ClaudeSDKClient", build_client)
    return clients


def test_claude_descriptor_is_narrow_and_versioned():
    descriptor_path = ROOT / "adapters" / "claude" / "fabric-adapter.json"
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))

    assert descriptor == {
        "contract_version": "fabric.adapter/v1alpha2",
        "adapter_id": "nvidia.fabric.claude",
        "harness": "claude",
        "adapter_kind": "python",
        "runner": {
            "module": "nemo_fabric_adapters.claude.adapter",
        },
        "settings_schema": {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {
                "setting_sources": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["user", "project", "local"],
                    },
                    "default": [],
                    "description": "Claude settings scopes to load.",
                },
                "max_budget_usd": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "description": (
                        "Maximum amount in US dollars that Claude may spend during "
                        "one invocation."
                    ),
                },
                "permission_mode": {
                    "type": "string",
                    "enum": [
                        "default",
                        "acceptEdits",
                        "bypassPermissions",
                        "plan",
                        "dontAsk",
                        "auto",
                    ],
                    "description": "Claude permission handling mode.",
                },
            },
            "required": [],
            "additionalProperties": False,
        },
        "config": {
            "accepts": [
                "models",
                "models.base_url",
                "instructions.system",
                "runtime.max_turns",
                "tools.enabled",
                "tools.blocked",
                "mcp",
                "skills",
            ],
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
                    "permission_mode": "dontAsk",
                    "max_budget_usd": 1.5,
                    "setting_sources": [],
                },
            },
            "instructions": {
                "system": {"content": "Review carefully.", "mode": "replace"}
            },
            "runtime": {"timeout_seconds": 30, "max_turns": 4},
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
            "environment": {
                "workspace": str(workspace),
                "env": {"ANTHROPIC_API_KEY": "configured-secret"},
            },
            "artifacts": {"root": str(tmp_path / "artifacts"), "artifacts": []},
        },
        "request": {"request_id": "request-1", "input": "Inspect the patch"},
        "capability_plan": {
            "native": {
                "tools_configured": True,
                "mcp_servers": {
                    "repo": {
                        "transport": "stdio",
                        "url": "repo-mcp",
                        "args": ["--root", ".", "--config", "repo config.json"],
                        "env": {"REPO_MCP_MODE": "mcp-secret-value"},
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
    options = adapter.build_options(claude_payload)
    assert options.cwd == Path(
        claude_payload["runtime_context"]["environment"]["workspace"]
    )
    assert options.model == "claude-test-model"
    assert options.system_prompt == "Review carefully."
    assert options.tools is None
    assert options.allowed_tools == []
    assert options.disallowed_tools == ["Bash"]
    assert options.hooks is not None
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
    assert isinstance(options.mcp_servers, Path)
    if os.name != "nt":
        assert options.mcp_servers.stat().st_mode & 0o777 == 0o600
        assert options.mcp_servers.parent.stat().st_mode & 0o777 == 0o700
    serialized_mcp = options.mcp_servers.read_text(encoding="utf-8")
    assert "mcp-secret-value" not in serialized_mcp
    mcp_config = json.loads(serialized_mcp)
    projected_value = mcp_config["mcpServers"]["repo"]["env"]["REPO_MCP_MODE"]
    assert projected_value.startswith("${NEMO_FABRIC_CLAUDE_MCP_")
    assert projected_value.endswith("}")
    projected_name = projected_value[2:-1]
    assert options.env[projected_name] == "mcp-secret-value"
    assert mcp_config == {
        "mcpServers": {
            "docs": {"type": "http", "url": "https://mcp.example.test"},
            "repo": {
                "type": "stdio",
                "command": "repo-mcp",
                "args": ["--root", ".", "--config", "repo config.json"],
                "env": {"REPO_MCP_MODE": projected_value},
            },
        }
    }
    assert "NEMO_RELAY_GATEWAY_URL" not in options.env
    assert "ANTHROPIC_BASE_URL" not in options.env


async def test_tool_policy_hooks_gate_built_in_and_mcp_tools(claude_payload):
    claude_payload["config"]["tools"] = {
        "enabled": ["Read", "Edit"],
        "blocked": ["Bash"],
    }

    options = adapter.build_options(claude_payload)

    assert options.tools == ["Read", "Edit"]
    assert options.allowed_tools == ["Read", "Edit"]
    assert options.hooks is not None
    hook = options.hooks["PreToolUse"][0].hooks[0]
    for tool_name in ("Read", "Edit"):
        assert await hook({"tool_name": tool_name}, None, {"signal": None}) == {}
    for tool_name in ("Bash", "mcp__repo__search"):
        output = await hook(
            {"tool_name": tool_name},
            None,
            {"signal": None},
        )
        assert output["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_enabled_tools_do_not_populate_allowed_tools_in_default_mode(claude_payload):
    claude_payload["config"]["harness"]["settings"]["permission_mode"] = "default"
    claude_payload["config"]["tools"] = {"enabled": ["Read"], "blocked": []}

    options = adapter.build_options(claude_payload)

    assert options.permission_mode == "default"
    assert options.tools == ["Read"]
    assert options.allowed_tools == []


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
    relay_payload["config"]["models"]["default"].update(
        {
            "provider": "acme",
            "base_url": "https://acme.example/v1/",
        }
    )
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
    assert relay.gateway.anthropic_base_url == "https://acme.example"
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
    assert (
        hooks["SessionStart"]
        == adapter.relay_hooks.render_relay_hooks("claude", executable)["hooks"][
            "SessionStart"
        ]
    )
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

    options = adapter.build_options(relay_payload, relay=relay)

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

    options = adapter.build_options(relay_payload, relay=relay)

    assert options.tools is None
    assert options.skills is None
    assert options.plugins == [{"type": "local", "path": str(relay.plugin_path)}]


def test_build_options_maps_blocked_tools_to_disallowed_tools(claude_payload):
    claude_payload["config"]["tools"] = {"blocked": ["Bash", "WebFetch"]}

    options = adapter.build_options(claude_payload)

    assert options.tools is None
    assert options.disallowed_tools == ["Bash", "WebFetch"]


def test_build_options_rejects_skill_path_without_skill_manifest(claude_payload):
    skill_path = Path(claude_payload["capability_plan"]["native"]["skill_paths"][0])
    (skill_path / "SKILL.md").unlink()

    with pytest.raises(adapter.AdapterConfigError, match="SKILL.md"):
        adapter.build_options(claude_payload)


def test_build_options_maps_custom_provider_to_claude_gateway_environment(
    claude_payload,
):
    model = claude_payload["config"]["models"]["default"]
    model.update(
        {
            "provider": "acme",
            "model": "aws/anthropic/claude-opus-4-5",
            "api_key_env": "ACME_API_KEY",
            "base_url": "https://acme.example/v1/",
        }
    )
    claude_payload["runtime_context"]["environment"]["env"].pop("ANTHROPIC_API_KEY")
    os.environ["ACME_API_KEY"] = "acme-secret"

    options = adapter.build_options(claude_payload)

    assert options.model == "aws/anthropic/claude-opus-4-5"
    assert options.env["ANTHROPIC_BASE_URL"] == "https://acme.example"
    assert options.env["ANTHROPIC_API_KEY"] == "acme-secret"
    assert options.env["ANTHROPIC_AUTH_TOKEN"] == ""


def test_build_options_requires_custom_provider_api_key_env(
    claude_payload,
):
    model = claude_payload["config"]["models"]["default"]
    model.update(
        {
            "provider": "acme",
            "model": "aws/anthropic/claude-opus-4-5",
            "base_url": "https://acme.example/v1",
        }
    )
    model.pop("api_key_env")
    claude_payload["runtime_context"]["environment"]["env"].pop("ANTHROPIC_API_KEY")

    with pytest.raises(adapter.AdapterConfigError, match="api_key_env is required"):
        adapter.build_options(claude_payload)


def test_build_options_requires_custom_provider_credential(claude_payload):
    model = claude_payload["config"]["models"]["default"]
    model.update(
        {
            "provider": "acme",
            "model": "aws/anthropic/claude-opus-4-5",
            "api_key_env": "ACME_API_KEY",
            "base_url": "https://acme.example/v1",
        }
    )
    os.environ.pop("ACME_API_KEY", None)

    with pytest.raises(adapter.AdapterConfigError, match="ACME_API_KEY is required"):
        adapter.build_options(claude_payload)


def test_build_options_requires_custom_provider_endpoint(claude_payload):
    model = claude_payload["config"]["models"]["default"]
    model.update(
        {
            "provider": "acme",
            "model": "aws/anthropic/claude-opus-4-5",
            "api_key_env": "ACME_API_KEY",
        }
    )
    claude_payload["runtime_context"]["environment"]["env"] = {
        "ACME_API_KEY": "acme-secret"
    }

    with pytest.raises(adapter.AdapterConfigError, match="base_url is required"):
        adapter.build_options(claude_payload)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("ANTHROPIC_API_KEY", "conflicting-secret"),
        ("ANTHROPIC_AUTH_TOKEN", "conflicting-token"),
        ("ANTHROPIC_BASE_URL", "https://other.example"),
    ],
)
def test_build_options_rejects_model_environment_conflicts(
    claude_payload,
    name,
    value,
):
    claude_payload["config"]["models"]["default"] = {
        "provider": "acme",
        "model": "aws/anthropic/claude-opus-4-5",
        "api_key_env": "ACME_API_KEY",
        "base_url": "https://acme.example/v1",
    }
    claude_payload["runtime_context"]["environment"]["env"] = {
        "ACME_API_KEY": "acme-secret",
        name: value,
    }

    with pytest.raises(
        adapter.AdapterConfigError,
        match=rf"environment\.env\.{name} conflicts",
    ):
        adapter.build_options(claude_payload)


def test_selected_model_rejects_empty_provider(claude_payload):
    model = claude_payload["config"]["models"]["default"]
    model["provider"] = ""

    with pytest.raises(adapter.AdapterConfigError, match="non-empty string"):
        adapter.selected_model(claude_payload)


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


async def test_claude_runtime_reuses_one_connected_sdk_client(
    claude_payload, monkeypatch
):
    clients = []

    class FakeClient:
        def __init__(self, options):
            self.options = options
            self.connect_count = 0
            self.disconnect_count = 0
            self.prompts = []
            clients.append(self)

        async def connect(self):
            self.connect_count += 1

        async def query(self, prompt):
            self.prompts.append(prompt)

        async def receive_response(self):
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
                result=f"done-{len(self.prompts)}",
            )

        async def disconnect(self):
            self.disconnect_count += 1

        async def interrupt(self):
            raise AssertionError("successful invocation must not be interrupted")

    monkeypatch.setattr(adapter, "ClaudeSDKClient", FakeClient)

    start_payload = dict(claude_payload)
    start_payload.pop("request")
    runtime = adapter.ClaudeRuntime()
    await runtime.start(start_payload)
    mcp_config_path = clients[0].options.mcp_servers
    assert isinstance(mcp_config_path, Path)
    assert mcp_config_path.exists()
    first = await runtime.invoke(lifecycle_invocation(claude_payload))
    claude_payload["runtime_context"]["invocation_id"] = "invocation-2"
    claude_payload["request"]["input"] = {"not": "text"}
    invalid = await runtime.invoke(lifecycle_invocation(claude_payload))
    claude_payload["runtime_context"]["invocation_id"] = "invocation-3"
    claude_payload["request"]["input"] = "Inspect the tests"
    second = await runtime.invoke(lifecycle_invocation(claude_payload))
    await runtime.stop()

    assert not mcp_config_path.exists()
    assert len(clients) == 1
    assert clients[0].connect_count == 1
    assert clients[0].disconnect_count == 1
    assert clients[0].prompts == ["Inspect the patch", "Inspect the tests"]
    assert first["response"] == "done-1"
    assert second["response"] == "done-2"
    assert invalid["error"]["code"] == "claude_invalid_request"
    assert first["session_id"] == second["session_id"] == "claude-session"


async def test_claude_runtime_removes_mcp_config_after_failed_sdk_connect(
    claude_payload, monkeypatch
):
    staged_paths: list[Path] = []

    class FailingClient:
        def __init__(self, options):
            assert isinstance(options.mcp_servers, Path)
            assert options.mcp_servers.exists()
            staged_paths.append(options.mcp_servers)

        async def connect(self):
            raise CLIConnectionError("connection failed")

    monkeypatch.setattr(adapter, "ClaudeSDKClient", FailingClient)
    start_payload = dict(claude_payload)
    start_payload.pop("request")

    with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
        await adapter.ClaudeRuntime().start(start_payload)

    assert caught.value.code == "claude_connection_failed"
    assert len(staged_paths) == 1
    assert not staged_paths[0].exists()


async def test_claude_runtime_owns_one_relay_gateway_until_stop(
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
    mock_start = MagicMock(return_value=process)
    mock_stop = MagicMock()

    class FakeClient:
        def __init__(self, options):
            self.options = options

        async def connect(self):
            pass

        async def query(self, _prompt):
            pass

        async def receive_response(self):
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

        async def disconnect(self):
            pass

        async def interrupt(self):
            raise AssertionError("successful invocation must not be interrupted")

    monkeypatch.setattr(adapter, "ClaudeSDKClient", FakeClient)
    monkeypatch.setattr(adapter, "prepare_claude_relay", MagicMock(return_value=relay))
    monkeypatch.setattr(adapter.relay_gateway, "start_relay_gateway", mock_start)
    monkeypatch.setattr(adapter.relay_gateway, "stop_relay_gateway", mock_stop)

    start_payload = dict(relay_payload)
    start_payload.pop("request")
    runtime = adapter.ClaudeRuntime()
    await runtime.start(start_payload)
    first = await runtime.invoke(lifecycle_invocation(relay_payload))
    relay_payload["runtime_context"]["invocation_id"] = "invocation-2"
    second = await runtime.invoke(lifecycle_invocation(relay_payload))

    mock_start.assert_called_once_with(
        launch=relay.gateway,
        cwd=Path(relay_payload["runtime_context"]["environment"]["workspace"]),
    )
    mock_stop.assert_not_called()
    assert first["relay_runtime"]["gateway_url"] == relay.gateway.url
    assert second["relay_runtime"]["gateway_url"] == relay.gateway.url

    await runtime.stop()

    mock_stop.assert_called_once_with(process)
    assert not relay.plugin_path.exists()


def atif_plugin_config(output_directory: Path) -> dict[str, Any]:
    return {
        "version": 1,
        "components": [
            {
                "kind": "observability",
                "enabled": True,
                "config": {
                    "atif": {
                        "enabled": True,
                        "output_directory": str(output_directory),
                        "filename_template": "trajectory-{session_id}.atif.json",
                    }
                },
            }
        ],
    }


def relay_settings(
    tmp_path: Path, plugin_config: dict[str, Any]
) -> adapter.ClaudeRelaySettings:
    executable = tmp_path / "nemo-relay"
    executable.touch()
    plugin_path = tmp_path / "relay-plugin"
    plugin_path.mkdir()
    return adapter.ClaudeRelaySettings(
        gateway=adapter.relay_gateway.RelayGatewayLaunch(
            executable=executable,
            config_path=tmp_path / "relay-config" / "config.toml",
            bind="127.0.0.1:43210",
            url="http://127.0.0.1:43210",
            log_path=tmp_path / "relay-config" / "gateway.log",
        ),
        plugin_config=plugin_config,
        plugin_path=plugin_path,
    )


def install_mock_relay(
    monkeypatch: pytest.MonkeyPatch, relay: adapter.ClaudeRelaySettings
) -> tuple[MagicMock, MagicMock, MagicMock]:
    relay.gateway.log_path.parent.mkdir()
    relay.gateway.log_path.write_text("gateway started\n", encoding="utf-8")
    process = MagicMock()
    mock_start = MagicMock(return_value=process)
    mock_stop = MagicMock()
    monkeypatch.setattr(adapter, "prepare_claude_relay", MagicMock(return_value=relay))
    monkeypatch.setattr(adapter.relay_gateway, "start_relay_gateway", mock_start)
    monkeypatch.setattr(adapter.relay_gateway, "stop_relay_gateway", mock_stop)
    return process, mock_start, mock_stop


async def test_runtime_waits_for_delayed_relay_artifact(
    relay_payload, monkeypatch, tmp_path
):
    atif_dir = tmp_path / "atif"
    atif_dir.mkdir()
    atif_path = atif_dir / "trajectory-session.atif.json"
    relay = relay_settings(tmp_path, atif_plugin_config(atif_dir))
    process, mock_start, mock_stop = install_mock_relay(monkeypatch, relay)
    write_task = None

    async def responses(client) -> AsyncIterator[ResultMessage]:
        nonlocal write_task
        assert client.options.env["ANTHROPIC_BASE_URL"] == relay.gateway.url
        assert Path(client.options.plugins[-1]["path"]) == relay.plugin_path

        async def write_atif():
            await asyncio.sleep(0.05)
            atif_path.write_text(
                json.dumps({"schema_version": "ATIF-v1.7", "steps": []}),
                encoding="utf-8",
            )

        write_task = asyncio.create_task(write_atif())
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

    install_fake_client(monkeypatch, responses)
    runtime = adapter.ClaudeRuntime()
    start_payload = {
        key: value for key, value in relay_payload.items() if key != "request"
    }
    await runtime.start(start_payload)
    try:
        output = await runtime.invoke(lifecycle_invocation(relay_payload))
        assert write_task is not None
        await write_task
    finally:
        await runtime.stop()

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


async def test_relay_atif_timeout_fails_successful_turn_explicitly(
    relay_payload, monkeypatch, tmp_path
):
    atif_dir = tmp_path / "atif"
    atif_dir.mkdir()
    stale_atif = atif_dir / "trajectory-existing.atif.json"
    stale_atif.write_text("{}", encoding="utf-8")
    relay = relay_settings(tmp_path, atif_plugin_config(atif_dir))
    install_mock_relay(monkeypatch, relay)
    wait_for_atif = AsyncMock(return_value=None)
    monkeypatch.setattr(
        adapter.relay_artifacts, "wait_for_finalized_atif", wait_for_atif
    )

    async def responses(_client) -> AsyncIterator[ResultMessage]:
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

    install_fake_client(monkeypatch, responses)
    runtime = adapter.ClaudeRuntime()
    start_payload = {
        key: value for key, value in relay_payload.items() if key != "request"
    }
    await runtime.start(start_payload)
    try:
        output = await runtime.invoke(lifecycle_invocation(relay_payload))
        unavailable = await runtime.invoke(lifecycle_invocation(relay_payload))
    finally:
        await runtime.stop()

    assert output["failed"] is True
    assert output["error"] == {
        "code": "claude_relay_atif_timeout",
        "message": "NeMo Relay did not finalize an ATIF artifact before the deadline",
        "retryable": False,
        "metadata": {
            "timeout_seconds": adapter.relay_artifacts.ATIF_FINALIZATION_TIMEOUT_SECONDS
        },
    }
    wait_for_atif.assert_awaited_once()
    assert output["relay_runtime"]["enabled"] is True
    assert output["relay_artifacts"] == []
    assert unavailable["error"]["code"] == "claude_runtime_unavailable"
    assert "relay_runtime" not in unavailable
    assert "relay_artifacts" not in unavailable


async def test_runtime_stop_reports_relay_gateway_failure(
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

    async def responses(_client) -> AsyncIterator[ResultMessage]:
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

    install_fake_client(monkeypatch, responses)
    runtime = adapter.ClaudeRuntime()
    start_payload = {
        key: value for key, value in relay_payload.items() if key != "request"
    }
    await runtime.start(start_payload)
    output = await runtime.invoke(lifecycle_invocation(relay_payload))
    with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
        await runtime.stop()

    assert output["response"] == "done"
    assert output["completed"] is True
    assert caught.value.code == "claude_relay_stop_failed"
    assert caught.value.metadata == {"gateway_log_path": str(relay.gateway.log_path)}
    assert "raw stop failure" not in str(caught.value)
    assert not relay.plugin_path.exists()


async def test_runtime_stop_reports_relay_plugin_cleanup_failure(
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

    async def responses(_client) -> AsyncIterator[ResultMessage]:
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

    install_fake_client(monkeypatch, responses)
    runtime = adapter.ClaudeRuntime()
    start_payload = {
        key: value for key, value in relay_payload.items() if key != "request"
    }
    await runtime.start(start_payload)
    output = await runtime.invoke(lifecycle_invocation(relay_payload))
    with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
        await runtime.stop()

    assert output["response"] == "done"
    assert output["completed"] is True
    assert caught.value.code == "claude_relay_cleanup_failed"
    assert "raw plugin cleanup failure" not in str(caught.value)
    mock_stop.assert_called_once_with(process)
    mock_rmtree.assert_called_once_with(relay.plugin_path)
    assert relay.plugin_path.exists()


@pytest.mark.parametrize(
    "failure", [ClaudeSDKError("sdk failed"), asyncio.CancelledError()]
)
async def test_runtime_stops_relay_after_sdk_failure_or_cancellation(
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

    async def responses(_client) -> AsyncIterator[ResultMessage]:
        raise failure
        yield

    install_fake_client(monkeypatch, responses)
    runtime = adapter.ClaudeRuntime()
    start_payload = {
        key: value for key, value in relay_payload.items() if key != "request"
    }
    await runtime.start(start_payload)
    try:
        if isinstance(failure, asyncio.CancelledError):
            with pytest.raises(asyncio.CancelledError):
                await runtime.invoke(lifecycle_invocation(relay_payload))
        else:
            output = await runtime.invoke(lifecycle_invocation(relay_payload))
            assert output["error"]["code"] == "claude_failed"
            assert output["relay_runtime"]["enabled"] is True
    finally:
        await runtime.stop()

    mock_stop.assert_called_once_with(process)
    assert not relay.plugin_path.exists()


@pytest.mark.parametrize(
    ("subtype", "is_error"),
    [("success", True), ("error_max_budget_usd", False)],
)
async def test_runtime_preserves_failed_result_when_sdk_stream_raises(
    claude_payload,
    monkeypatch,
    caplog,
    subtype,
    is_error,
):
    async def responses(_client) -> AsyncIterator[ResultMessage]:
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

    install_fake_client(monkeypatch, responses)
    runtime = adapter.ClaudeRuntime()
    start_payload = {
        key: value for key, value in claude_payload.items() if key != "request"
    }
    await runtime.start(start_payload)
    output = await runtime.invoke(lifecycle_invocation(claude_payload))
    await runtime.stop()

    assert output["response"] == "Not logged in"
    assert output["error"] == {
        "code": "claude_result_failed",
        "message": "Claude returned an error result",
        "retryable": False,
        "metadata": {"subtype": subtype},
    }
    assert "raw SDK stream error" not in json.dumps(output)
    assert "raw SDK stream error" in caplog.text


async def test_runtime_start_reports_relay_failure_without_raw_diagnostic(
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

    runtime = adapter.ClaudeRuntime()
    start_payload = {
        key: value for key, value in relay_payload.items() if key != "request"
    }
    with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
        await runtime.start(start_payload)

    assert caught.value.code == "claude_relay_start_failed"
    assert caught.value.message == "NeMo Relay gateway failed to start"
    assert caught.value.metadata == {"gateway_log_path": str(relay.gateway.log_path)}
    assert "secret" not in str(caught.value)
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
    claude_payload["runtime_context"]["environment"].pop("env")
    for name in ANTHROPIC_AUTH_ENV_NAMES:
        os.environ.pop(name, None)
    os.environ["FABRIC_UNRELATED_SECRET"] = "do-not-forward"
    os.environ.update(auth_environment)

    options = adapter.build_options(claude_payload)

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

    options = adapter.build_options(claude_payload)

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


def test_main_serves_persistent_runtime(monkeypatch):
    serve = MagicMock()
    monkeypatch.setattr(adapter.lifecycle, "serve", serve)

    adapter.main()

    serve.assert_called_once_with(adapter.ClaudeRuntime)
