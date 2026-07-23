# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from nemo_fabric import Fabric
from nemo_fabric_adapters.codex import adapter
from openai_codex import AsyncCodex, AsyncThread, AsyncTurnHandle
from openai_codex.types import TurnStatus


def lifecycle_start_payload(payload):
    return {key: value for key, value in payload.items() if key != "request"}


def lifecycle_invocation(payload):
    return {
        "runtime_context": payload["runtime_context"],
        "request": payload["request"],
    }


async def invoke_once_async(payload):
    runtime = adapter.CodexRuntime()
    await runtime.start(lifecycle_start_payload(payload))
    try:
        return await runtime.invoke(lifecycle_invocation(payload))
    finally:
        await runtime.stop()


def invoke_once(payload):
    return asyncio.run(invoke_once_async(payload))


def runtime_start_error(payload):
    async def scenario() -> adapter.lifecycle.LifecycleError:
        runtime = adapter.CodexRuntime()
        with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
            await runtime.start(lifecycle_start_payload(payload))
        return caught.value

    return asyncio.run(scenario())


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
    mock_codex.skill_request = AsyncMock()
    mock_codex.close_error = None

    def build_client(*, config):
        mock_client = MagicMock(spec=AsyncCodex)
        mock_client.config = config
        mock_client.closed = False
        mock_client.thread = None
        mock_client._client = SimpleNamespace(request=mock_codex.skill_request)

        async def close():
            if mock_codex.close_error is not None:
                raise mock_codex.close_error
            mock_client.closed = True

        async def thread_start(**_kwargs):
            mock_client.thread = (
                mock_codex.next_thread
                if mock_codex.next_thread is not None
                else mock_thread(mock_codex.next_thread_id, mock_codex.next_result)
            )
            return mock_client.thread

        mock_client.close.side_effect = close
        mock_client.thread_start.side_effect = thread_start
        mock_codex.instances.append(mock_client)
        return mock_client

    mock_codex.side_effect = build_client
    monkeypatch.setattr(adapter, "AsyncCodex", mock_codex)
    return mock_codex


def test_single_invocation_uses_native_thread_and_turn_contract(
    codex_payload, mock_codex, tmp_path
):
    os.environ["CODEX_HOME"] = str(tmp_path / "codex-home")
    os.environ["CODEX_INTERNAL_ORIGINATOR_OVERRIDE"] = "parent-codex"
    os.environ["FABRIC_UNRELATED_SECRET"] = "do-not-forward"
    codex_payload["config"]["harness"]["settings"]["env"] = {
        "CODEX_EXPLICIT": "forward-me"
    }

    output = invoke_once(codex_payload)

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
    assert client.config.env["CODEX_INTERNAL_ORIGINATOR_OVERRIDE"] == "codex_python_sdk"
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
    client.models.assert_not_awaited()
    client._client.request.assert_not_awaited()


def test_runtime_stop_reports_close_failure_after_completed_turn(
    codex_payload, mock_codex, caplog
):
    mock_codex.close_error = RuntimeError("close failed")

    async def scenario() -> tuple[dict[str, Any], adapter.lifecycle.LifecycleError]:
        runtime = adapter.CodexRuntime()
        await runtime.start(lifecycle_start_payload(codex_payload))
        output = await runtime.invoke(lifecycle_invocation(codex_payload))
        with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
            await runtime.stop()
        return output, caught.value

    output, error = asyncio.run(scenario())

    assert output["completed"] is True
    assert output["failed"] is False
    assert output["thread_id"] == "thread-123"
    assert output["response"] == "done"
    assert output["error"] is None
    assert error.code == "codex_sdk_stop_failed"
    assert "Codex SDK client failed to close" in caplog.text
    mock_codex.instances[0].close.assert_awaited_once_with()


def test_start_failure_is_not_masked_by_sdk_close_failure(
    codex_payload, mock_codex, monkeypatch, caplog
):
    mock_codex.close_error = RuntimeError("close failed")
    register_skills = AsyncMock(
        side_effect=adapter.AdapterConfigError(
            "codex_skill_registration_failed",
            "Codex skill registration failed",
        )
    )
    monkeypatch.setattr(adapter, "_register_skill_roots", register_skills)

    async def scenario() -> adapter.lifecycle.LifecycleError:
        runtime = adapter.CodexRuntime()
        with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
            await runtime.start(lifecycle_start_payload(codex_payload))
        return caught.value

    error = asyncio.run(scenario())

    assert error.code == "codex_skill_registration_failed"
    assert error.message == "Codex skill registration failed"
    assert "Codex SDK cleanup after start failure also failed" in caplog.text
    register_skills.assert_awaited_once()
    mock_codex.instances[0].close.assert_awaited_once_with()


def test_sdk_maps_native_mcp_servers_into_thread_config(codex_payload, mock_codex):
    os.environ["FABRIC_TEST_MCP_URL"] = "https://mcp.example.test/mcp"
    codex_payload["capability_plan"] = {
        "native": {
            "mcp_servers": {
                "repo": {
                    "transport": "stdio",
                    "url": "python -m repo_mcp --root .",
                },
                "remote": {
                    "transport": "streamable-http",
                    "url": "${FABRIC_TEST_MCP_URL}",
                },
            }
        }
    }
    codex_payload["config"]["harness"]["settings"]["config_overrides"][
        "mcp_servers.remote.required"
    ] = True

    output = invoke_once(codex_payload)

    assert output["completed"] is True
    config = mock_codex.instances[0].thread_start.await_args.kwargs["config"]
    assert config["mcp_servers"] == {
        "remote": {
            "url": "https://mcp.example.test/mcp",
            "required": True,
        },
        "repo": {
            "command": "python",
            "args": ["-m", "repo_mcp", "--root", "."],
        },
    }


def test_blocked_toolsets_override_codex_config_overrides(codex_payload, mock_codex):
    codex_payload["config"]["tools"] = {
        "blocked": [
            "apps",
            "browser",
            "image_generation",
            "multi_agent",
            "plugins",
            "request_user_input",
            "shell",
            "tool_suggest",
            "web_search",
        ]
    }
    overrides = codex_payload["config"]["harness"]["settings"]["config_overrides"]
    overrides.update(
        {
            "features.apps": True,
            "features.browser_use": True,
            "features.image_generation": True,
            "features.multi_agent": True,
            "features.plugins": True,
            "features.shell_tool": True,
            "features.tool_suggest": True,
            "tools.experimental_request_user_input.enabled": True,
            "web_search": "live",
        }
    )

    output = invoke_once(codex_payload)

    assert output["completed"] is True
    config = mock_codex.instances[0].thread_start.await_args.kwargs["config"]
    assert all(
        adapter._config_has_layer(config, layer)
        for layer in adapter.BLOCKED_TOOL_CONFIG.values()
    )


def test_blocked_toolset_config_does_not_leak_between_runtimes(
    codex_payload, mock_codex
):
    codex_payload["config"]["tools"] = {"blocked": ["shell"]}

    output = invoke_once(codex_payload)

    assert output["completed"] is True
    config = mock_codex.instances[0].thread_start.await_args.kwargs["config"]
    assert config["features"] == {"shell_tool": False, "web_search": False}


def test_blocked_toolsets_preserve_unrelated_codex_tools(codex_payload, mock_codex):
    codex_payload["config"]["tools"] = {"blocked": ["apps", "browser", "plugins"]}
    codex_payload["config"]["harness"]["settings"]["config_overrides"].update(
        {
            "features.standalone_web_search": True,
            "features.tool_suggest": True,
            "web_search": "live",
        }
    )

    output = invoke_once(codex_payload)

    assert output["completed"] is True
    config = mock_codex.instances[0].thread_start.await_args.kwargs["config"]
    assert config["features"]["standalone_web_search"] is True
    assert config["features"]["tool_suggest"] is True
    assert config["web_search"] == "live"


def test_blocked_mcp_tool_uses_raw_server_and_tool_names(codex_payload, mock_codex):
    codex_payload["capability_plan"] = {
        "native": {
            "mcp_servers": {
                "repo": {
                    "transport": "streamable-http",
                    "url": "https://mcp.example.test/repo",
                }
            }
        }
    }
    codex_payload["config"]["tools"] = {
        "blocked": ["mcp:repo:write_file", "mcp:repo:delete:file"]
    }
    codex_payload["config"]["harness"]["settings"]["config_overrides"].update(
        {
            "mcp_servers.repo.disabled_tools": ["existing"],
        }
    )

    output = invoke_once(codex_payload)

    assert output["completed"] is True
    server = mock_codex.instances[0].thread_start.await_args.kwargs["config"][
        "mcp_servers"
    ]["repo"]
    assert server["disabled_tools"] == ["existing", "write_file", "delete:file"]


def test_blocked_mcp_toolset_disables_every_configured_server(
    codex_payload, mock_codex
):
    codex_payload["capability_plan"] = {
        "native": {
            "mcp_servers": {
                "repo": {
                    "transport": "streamable-http",
                    "url": "https://mcp.example.test/repo",
                }
            }
        }
    }
    codex_payload["config"]["tools"] = {"blocked": ["mcp"]}
    codex_payload["config"]["harness"]["settings"]["config_overrides"].update(
        {
            "mcp_servers.repo.enabled": True,
            "mcp_servers.extra.command": "extra-mcp",
        }
    )

    output = invoke_once(codex_payload)

    assert output["completed"] is True
    servers = mock_codex.instances[0].thread_start.await_args.kwargs["config"][
        "mcp_servers"
    ]
    assert servers["repo"]["enabled"] is False
    assert servers["extra"]["enabled"] is False


def test_blocked_app_tool_uses_raw_connector_and_tool_names(codex_payload, mock_codex):
    codex_payload["config"]["tools"] = {"blocked": ["app:google_drive:files/delete"]}
    codex_payload["config"]["harness"]["settings"]["config_overrides"].update(
        {"apps.google_drive.tools.files/delete.enabled": True}
    )

    output = invoke_once(codex_payload)

    assert output["completed"] is True
    config = mock_codex.instances[0].thread_start.await_args.kwargs["config"]
    assert config["apps"]["google_drive"]["tools"]["files/delete"] == {"enabled": False}


@pytest.mark.parametrize(
    "blocked",
    [
        "apply_patch",
        "dynamic:arbitrary",
        "mcp:repo",
        "mcp::tool",
        "app:connector:",
    ],
)
def test_unenforceable_blocked_tool_fails_before_codex_starts(
    codex_payload, mock_codex, blocked
):
    codex_payload["config"]["tools"] = {"blocked": [blocked]}

    error = runtime_start_error(codex_payload)

    assert error.code == "codex_invalid_configuration"
    assert blocked in error.message
    mock_codex.assert_not_called()


def test_blocked_tool_rejects_unknown_mcp_server(codex_payload, mock_codex):
    codex_payload["config"]["tools"] = {"blocked": ["mcp:unknown:delete"]}

    error = runtime_start_error(codex_payload)

    assert error.code == "codex_invalid_configuration"
    assert "unknown Fabric MCP server 'unknown'" in error.message
    mock_codex.assert_not_called()


def test_blocked_tool_rejects_custom_codex_runtime(codex_payload, mock_codex):
    codex_payload["config"]["tools"] = {"blocked": ["shell"]}
    codex_payload["config"]["harness"]["settings"]["codex_bin"] = "/tmp/codex"

    error = runtime_start_error(codex_payload)

    assert error.code == "codex_invalid_configuration"
    assert "runtime pinned by the adapter" in error.message
    mock_codex.assert_not_called()


def test_blocked_tool_validation_is_defense_in_depth(
    codex_payload, mock_codex, monkeypatch
):
    codex_payload["config"]["tools"] = {"blocked": ["shell"]}
    monkeypatch.setattr(adapter, "_apply_blocked_tools_config", lambda *_args: None)

    error = runtime_start_error(codex_payload)

    assert error.code == "codex_invalid_configuration"
    assert "was not enforced for 'shell'" in error.message
    mock_codex.assert_not_called()


def test_sdk_registers_native_skill_roots(codex_payload, mock_codex, tmp_path):
    review = tmp_path / "skills" / "review"
    test = tmp_path / "skills" / "test"
    for skill in (review, test):
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text(
            f"---\nname: {skill.name}\ndescription: Test skill.\n---\n",
            encoding="utf-8",
        )
    codex_payload["capability_plan"] = {
        "native": {"skill_paths": ["skills/review", "skills/test"]}
    }

    output = invoke_once(codex_payload)

    assert output["completed"] is True
    mock_codex.instances[0].thread.turn.assert_awaited_once_with(
        "Inspect the change.",
        effort=None,
        output_schema=None,
    )
    mock_codex.instances[0].models.assert_awaited_once_with()
    mock_codex.instances[0]._client.request.assert_awaited_once_with(
        "skills/extraRoots/set",
        {"extraRoots": [str(review), str(test)]},
        response_model=adapter.SkillsExtraRootsSetResponse,
    )


def test_sdk_closes_when_skill_registration_is_unavailable(
    codex_payload, mock_codex, tmp_path
):
    skill = tmp_path / "skills" / "review"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: review\ndescription: Test skill.\n---\n",
        encoding="utf-8",
    )
    codex_payload["capability_plan"] = {"native": {"skill_paths": ["skills/review"]}}
    mock_codex.skill_request = None

    error = runtime_start_error(codex_payload)

    assert error.code == "codex_invalid_configuration"
    client = mock_codex.instances[0]
    client.thread_start.assert_not_awaited()
    assert client.closed is True


@pytest.mark.parametrize("transport", ["sse", "carrier-pigeon"])
def test_sdk_rejects_unsupported_mcp_transport(codex_payload, mock_codex, transport):
    codex_payload["capability_plan"] = {
        "native": {
            "mcp_servers": {
                "bad": {"transport": transport, "url": "https://mcp.example.test"}
            }
        }
    }

    error = runtime_start_error(codex_payload)

    assert error.code == "codex_invalid_configuration"
    assert f"unsupported Codex MCP transport: {transport}" in error.message
    mock_codex.assert_not_called()


def test_sdk_rejects_invalid_native_skill_path(codex_payload, mock_codex, tmp_path):
    missing = tmp_path / "skills" / "missing"
    codex_payload["capability_plan"] = {"native": {"skill_paths": [str(missing)]}}

    error = runtime_start_error(codex_payload)

    assert error.code == "codex_invalid_configuration"
    assert "directory containing SKILL.md" in error.message
    mock_codex.assert_not_called()


@pytest.mark.parametrize("skill_paths", [None, "", {}, False])
def test_sdk_rejects_falsy_non_list_skill_paths(codex_payload, mock_codex, skill_paths):
    codex_payload["capability_plan"] = {"native": {"skill_paths": skill_paths}}

    error = runtime_start_error(codex_payload)

    assert error.code == "codex_invalid_configuration"
    assert error.message == "native skill_paths must be a list of paths"
    mock_codex.assert_not_called()


def test_sdk_can_use_an_explicit_codex_runtime(codex_payload, mock_codex, tmp_path):
    codex_bin = tmp_path / "bin" / "codex"
    codex_bin.parent.mkdir()
    codex_bin.touch()
    codex_payload["config"]["harness"]["settings"]["codex_bin"] = str(codex_bin)

    output = invoke_once(codex_payload)

    assert output["completed"] is True
    assert mock_codex.instances[0].config.codex_bin == str(codex_bin)


@pytest.mark.parametrize("codex_bin", ["bin/codex", "~/bin/codex"])
def test_sdk_resolves_relative_codex_runtime_from_base_dir(codex_payload, codex_bin):
    codex_payload["config"]["harness"]["settings"]["codex_bin"] = codex_bin

    config = adapter.sdk_config(codex_payload, relay=None)

    base_dir = Path(codex_payload["base_dir"])
    assert config.codex_bin == str((base_dir / codex_bin).resolve())


def test_sdk_keeps_absolute_codex_runtime_path(codex_payload, tmp_path):
    codex_bin = tmp_path / "bin" / ".." / "codex"
    codex_payload["config"]["harness"]["settings"]["codex_bin"] = str(codex_bin)

    config = adapter.sdk_config(codex_payload, relay=None)

    assert config.codex_bin == str(codex_bin)


async def test_persistent_runtime_reuses_one_client_and_thread(
    codex_payload, mock_codex
):
    start_payload = dict(codex_payload)
    start_payload.pop("request")
    runtime = adapter.CodexRuntime()

    await runtime.start(start_payload)
    first = await runtime.invoke(lifecycle_invocation(codex_payload))
    codex_payload["runtime_context"]["invocation_id"] = "invocation-2"
    codex_payload["request"]["input"] = "Continue."
    second = await runtime.invoke(lifecycle_invocation(codex_payload))
    await runtime.stop()

    assert first["thread_id"] == second["thread_id"] == "thread-123"
    assert len(mock_codex.instances) == 1
    client = mock_codex.instances[0]
    client.thread_start.assert_awaited_once()
    assert client.thread.turn.await_count == 2
    assert client.thread.turn.await_args_list[1].args[0] == "Continue."
    client.close.assert_awaited_once()


async def test_persistent_runtime_registers_skills_once_and_maps_mcp(
    codex_payload, mock_codex, tmp_path
):
    skill = tmp_path / "skills" / "review"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: review\ndescription: Test skill.\n---\n",
        encoding="utf-8",
    )
    codex_payload["capability_plan"] = {
        "native": {
            "skill_paths": ["skills/review"],
            "mcp_servers": {
                "review": {
                    "transport": "streamable-http",
                    "url": "https://mcp.example.test/review",
                }
            },
        }
    }
    start_payload = dict(codex_payload)
    start_payload.pop("request")
    runtime = adapter.CodexRuntime()

    await runtime.start(start_payload)
    await runtime.invoke(lifecycle_invocation(codex_payload))
    codex_payload["runtime_context"]["invocation_id"] = "invocation-2"
    await runtime.invoke(lifecycle_invocation(codex_payload))
    await runtime.stop()

    client = mock_codex.instances[0]
    client.models.assert_awaited_once_with()
    client._client.request.assert_awaited_once_with(
        "skills/extraRoots/set",
        {"extraRoots": [str(skill)]},
        response_model=adapter.SkillsExtraRootsSetResponse,
    )
    assert client.thread_start.await_args.kwargs["config"]["mcp_servers"] == {
        "review": {"url": "https://mcp.example.test/review"}
    }
    assert client.thread.turn.await_count == 2


async def test_persistent_runtime_owns_one_relay_gateway(
    codex_payload, mock_codex, monkeypatch, tmp_path
):
    codex_payload["telemetry_plan"] = {
        "providers": ["relay"],
        "relay_enabled": True,
    }
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
    process = MagicMock()
    start_gateway = MagicMock(return_value=process)
    stop_gateway = MagicMock()
    monkeypatch.setattr(adapter, "prepare_codex_relay", MagicMock(return_value=relay))
    monkeypatch.setattr(adapter.relay_gateway, "start_relay_gateway", start_gateway)
    monkeypatch.setattr(adapter.relay_gateway, "stop_relay_gateway", stop_gateway)
    start_payload = dict(codex_payload)
    start_payload.pop("request")
    runtime = adapter.CodexRuntime()

    await runtime.start(start_payload)
    await runtime.invoke(lifecycle_invocation(codex_payload))
    codex_payload["runtime_context"]["invocation_id"] = "invocation-2"
    await runtime.invoke(lifecycle_invocation(codex_payload))
    stop_gateway.assert_not_called()
    await runtime.stop()

    assert len(mock_codex.instances) == 1
    start_gateway.assert_called_once_with(
        launch=gateway,
        cwd=Path(codex_payload["runtime_context"]["environment"]["workspace"]),
    )
    stop_gateway.assert_called_once_with(process)


def test_failed_sdk_turn_is_normalized_and_transport_is_closed(
    codex_payload, mock_codex
):
    mock_codex.next_result = RuntimeError("model request failed")

    output = invoke_once(codex_payload)

    assert output["error"] == {
        "code": "codex_turn_failed",
        "message": "model request failed",
        "retryable": False,
    }
    assert mock_codex.instances[0].closed is True


def test_incomplete_sdk_turn_is_failed(codex_payload, mock_codex):
    result = successful_result(response=None)
    mock_codex.next_result = result

    output = invoke_once(codex_payload)

    assert output["error"]["code"] == "codex_turn_incomplete"
    assert output["turn_status"] == "completed"


def test_selected_model_rejects_unsupported_provider(codex_payload, mock_codex):
    model = codex_payload["config"]["models"]["default"]
    model["provider"] = "anthropic"

    error = runtime_start_error(codex_payload)

    assert error.code == "codex_invalid_configuration"
    assert "provider must be openai or nvidia" in error.message
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

    output = invoke_once(codex_payload)

    assert output["completed"] is True
    client = mock_codex.instances[0]
    assert client.config.env["NVIDIA_API_KEY"] == "nvidia-secret"
    start = client.thread_start.await_args.kwargs
    assert start["model"] == "openai/gpt-oss-120b"
    assert start["model_provider"] == "nvidia"
    assert Path(client.config.env["CODEX_HOME"]).parts[-3:] == (
        ".fabric",
        "codex",
        "nvidia-home",
    )
    assert start["config"]["features"] == {"web_search": False}
    assert start["config"]["model_providers"] == {
        "nvidia": {
            "name": "NVIDIA",
            "base_url": "https://nvidia.example/v1",
            "env_key": "NVIDIA_API_KEY",
            "wire_api": "responses",
        }
    }


def test_nvidia_provider_normalizes_codex_home_creation_failure(
    codex_payload, mock_codex, monkeypatch
):
    model = codex_payload["config"]["models"]["default"]
    model.update(
        {
            "provider": "nvidia",
            "model": "openai/gpt-oss-120b",
            "api_key_env": "NVIDIA_API_KEY",
            "settings": {"base_url": "https://nvidia.example/v1"},
        }
    )
    os.environ["NVIDIA_API_KEY"] = "nvidia-secret"

    async def fail_to_create_home(*_args, **_kwargs):
        raise OSError("read-only filesystem")

    monkeypatch.setattr(adapter.asyncio, "to_thread", fail_to_create_home)

    error = runtime_start_error(codex_payload)

    assert error.code == "codex_runtime_unavailable"
    mock_codex.assert_not_called()


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

    error = runtime_start_error(codex_payload)

    assert error.code == "codex_invalid_configuration"
    assert "NVIDIA_API_KEY is required" in error.message
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

    error = runtime_start_error(codex_payload)

    assert error.code == "codex_invalid_configuration"
    assert "NVIDIA_FRONTIER_BASE_URL" in error.message
    assert not (adapter.state_dir(codex_payload) / "nvidia-home").exists()
    mock_codex.assert_not_called()


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
    monkeypatch.setattr(adapter.relay_gateway, "start_relay_gateway", start_gateway)
    monkeypatch.setattr(adapter.relay_gateway, "stop_relay_gateway", stop_gateway)
    os.environ["FABRIC_RELAY_CONFIG_PATH"] = str(tmp_path / "relay.json")

    output = invoke_once(codex_payload)

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


def test_relay_rejects_nvidia_provider(codex_payload, mock_codex):
    model = codex_payload["config"]["models"]["default"]
    model.update(
        {
            "provider": "nvidia",
            "model": "openai/gpt-oss-120b",
            "api_key_env": "NVIDIA_API_KEY",
            "settings": {"base_url": "https://nvidia.example/v1"},
        }
    )
    codex_payload["telemetry_plan"] = {
        "providers": ["relay"],
        "relay_enabled": True,
    }
    os.environ["NVIDIA_API_KEY"] = "nvidia-secret"

    error = runtime_start_error(codex_payload)

    assert error.code == "codex_invalid_configuration"
    assert error.message == ("NeMo Relay requires the built-in openai model provider")
    mock_codex.assert_not_called()


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
    monkeypatch.setattr(adapter.relay_gateway, "relay_cli_contract", contract)
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
def test_relay_stop_failure_is_reported_by_runtime_stop(
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

    async def scenario() -> tuple[dict[str, Any], adapter.lifecycle.LifecycleError]:
        runtime = adapter.CodexRuntime()
        await runtime.start(lifecycle_start_payload(codex_payload))
        output = await runtime.invoke(lifecycle_invocation(codex_payload))
        with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
            await runtime.stop()
        return output, caught.value

    output, error = asyncio.run(scenario())

    assert output["completed"] is True
    assert error.code == "codex_relay_stop_failed"


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
                            "resource_attributes": {"deployment.environment": "test"},
                        }
                    },
                }
            ]
        },
    }

    output = invoke_once(codex_payload)

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


def test_timeout_interrupts_native_turn_and_closes_sdk(codex_payload, mock_codex):
    mock_blocking_thread = mock_thread("thread-timeout")

    async def block():
        await asyncio.sleep(60)

    mock_blocking_thread.handle.run.side_effect = block
    mock_codex.next_thread = mock_blocking_thread
    codex_payload["config"]["harness"]["settings"]["timeout_seconds"] = 0.01

    output = invoke_once(codex_payload)

    client = mock_codex.instances[0]
    assert output["error"]["code"] == "codex_timed_out"
    assert client.thread.handle.interrupted is True
    assert client.closed is True


@pytest.mark.parametrize(
    "setting", ["codex_command", "codex_args", "codex_profile", "skip_git_repo_check"]
)
def test_cli_only_settings_are_rejected(codex_payload, setting):
    codex_payload["config"]["harness"]["settings"][setting] = "legacy"

    error = runtime_start_error(codex_payload)

    assert error.code == "codex_invalid_configuration"
    assert setting in error.message


@pytest.mark.parametrize(
    ("setting", "normalized_field"),
    [
        ("mcp_servers", "FabricConfig.mcp"),
        ("skills", "FabricConfig.skills"),
        ("tools", "FabricConfig.tools"),
    ],
)
def test_normalized_capabilities_reject_harness_settings(
    codex_payload, mock_codex, setting, normalized_field
):
    codex_payload["config"]["harness"]["settings"][setting] = {}

    error = runtime_start_error(codex_payload)

    assert error.code == "codex_invalid_configuration"
    assert normalized_field in error.message
    mock_codex.assert_not_called()


def test_adapter_rejects_structured_input(codex_payload):
    codex_payload["request"]["input"] = {
        "messages": [{"role": "user", "content": "Inspect the change."}]
    }

    output = invoke_once(codex_payload)

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
    }
    assert descriptor["config"]["accepts"] == [
        "models",
        "mcp",
        "skills",
        "telemetry",
        "tools",
        "tools.blocked",
    ]
    assert "requirements" not in descriptor


def test_codex_config_resolves_sdk_adapter():
    from examples.code_review_agent import BASE_DIR, codex_config

    config = codex_config()
    skill_path = Path(__file__).parents[2] / "skills" / "nemo-fabric-integrate"
    config.add_skill_path(skill_path)
    config.add_mcp_server(
        "github",
        transport="streamable-http",
        url="https://mcp.example.test/mcp",
        exposure="harness_native",
    )
    config.block_tools("shell")
    plan = Fabric().plan(config, base_dir=BASE_DIR)

    assert plan.adapter.adapter_id == "nvidia.fabric.codex"
    assert plan.adapter.harness == "codex"
    assert plan.config.runtime.input_schema == "text"
    assert plan.config.harness.settings["reasoning_effort"] == "high"
    assert plan.config.tools.blocked == ["shell"]
    native = plan["capability_plan"]["native"]
    assert native["tools_configured"] is True
    assert native["skill_paths"] == [str(skill_path)]
    assert native["mcp_servers"]["github"] == {
        "transport": "streamable-http",
        "url": "https://mcp.example.test/mcp",
        "exposure": "harness_native",
    }
    unsupported = plan["capability_plan"]["unsupported"]
    assert unsupported["tools_configured"] is False
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
def test_environment_rejects_non_mapping_runtime_telemetry(codex_payload, telemetry):
    codex_payload["runtime_context"]["telemetry"] = telemetry

    with pytest.raises(
        adapter.AdapterInputError,
        match=r"runtime_context\.telemetry must be a mapping",
    ):
        adapter.child_environment(codex_payload)


def test_main_serves_persistent_runtime(monkeypatch):
    serve = MagicMock()
    monkeypatch.setattr(adapter.lifecycle, "serve", serve)

    adapter.main()

    serve.assert_called_once_with(adapter.CodexRuntime)
