# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import os
import subprocess
import sys
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
            "instructions": {
                "system": {"content": "Review carefully.", "mode": "replace"}
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


def configure_mcp(payload, servers):
    capability_plan = payload.setdefault("capability_plan", {})
    native = capability_plan.setdefault("native", {})
    native["mcp_servers"] = servers


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


def atif_plugin_config(output_directory: Path) -> dict[str, Any]:
    return {
        "version": 1,
        "components": [
            {
                "kind": "observability",
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


def relay_settings(tmp_path: Path, plugin_config: dict[str, Any]):
    return adapter.CodexRelaySettings(
        gateway=adapter.relay_gateway.RelayGatewayLaunch(
            executable=tmp_path / "nemo-relay",
            config_path=tmp_path / "relay" / "config.toml",
            bind="127.0.0.1:43210",
            url="http://127.0.0.1:43210",
            log_path=tmp_path / "relay" / "gateway.log",
        ),
        plugin_config=plugin_config,
    )


def install_mock_relay(monkeypatch, relay: adapter.CodexRelaySettings):
    monkeypatch.setattr(adapter, "prepare_codex_relay", MagicMock(return_value=relay))
    monkeypatch.setattr(
        adapter.relay_gateway,
        "start_relay_gateway",
        MagicMock(return_value=MagicMock()),
    )
    monkeypatch.setattr(adapter.relay_gateway, "stop_relay_gateway", MagicMock())


@pytest.fixture(name="mock_codex")
def mock_codex_fixture(monkeypatch):
    mock_codex = MagicMock(spec=AsyncCodex)
    mock_codex.instances = []
    mock_codex.next_thread_id = "thread-123"
    mock_codex.next_result = None
    mock_codex.next_thread = None
    mock_codex.mcp_auth_statuses = {}
    mock_codex.report_mcp_auth_statuses = True
    mock_codex.mcp_login_success = True
    mock_codex.mcp_login_error = None
    mock_codex.oauth_authorization_url = "https://auth.example.test/authorize"
    mock_codex.login_params = None

    async def protocol_request(method, params, *, response_model):
        if method == "skills/extraRoots/set":
            return None
        if method == "mcpServerStatus/list":
            return response_model(
                data=[
                    {
                        "name": name,
                        "authStatus": status,
                        "resourceTemplates": [],
                        "resources": [],
                        "tools": {},
                    }
                    for name, status in mock_codex.mcp_auth_statuses.items()
                ]
            )
        if method == "mcpServer/oauth/login":
            mock_codex.login_params = params
            return response_model(authorizationUrl=mock_codex.oauth_authorization_url)
        raise AssertionError(f"unexpected Codex protocol request: {method}")

    async def next_notification():
        params = mock_codex.login_params
        assert params is not None
        return SimpleNamespace(
            method="mcpServer/oauthLogin/completed",
            payload=adapter.McpServerOauthLoginCompletedNotification(
                error=mock_codex.mcp_login_error,
                name=params["name"],
                success=mock_codex.mcp_login_success,
                threadId=params["threadId"],
            ),
        )

    mock_codex.skill_request = AsyncMock(side_effect=protocol_request)
    mock_codex.next_notification = AsyncMock(side_effect=next_notification)
    mock_codex.close_error = None

    def build_client(*, config):
        mock_client = MagicMock(spec=AsyncCodex)
        mock_client.config = config
        mock_client.closed = False
        mock_client.thread = None
        mock_client._client = SimpleNamespace(
            request=mock_codex.skill_request,
            next_notification=mock_codex.next_notification,
        )

        async def close():
            if mock_codex.close_error is not None:
                raise mock_codex.close_error
            mock_client.closed = True

        async def thread_start(**kwargs):
            config = kwargs.get("config") or {}
            for name, server in config.get("mcp_servers", {}).items():
                if (
                    server.get("auth") == "oauth"
                    and mock_codex.report_mcp_auth_statuses
                ):
                    mock_codex.mcp_auth_statuses.setdefault(
                        name, adapter.McpAuthStatus.o_auth
                    )
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
    os.environ["DBUS_SESSION_BUS_ADDRESS"] = "unix:path=/run/user/1000/bus"
    os.environ["FABRIC_UNRELATED_SECRET"] = "do-not-forward"
    os.environ["XDG_RUNTIME_DIR"] = "/run/user/1000"
    codex_payload["runtime_context"]["environment"]["env"] = {
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
    assert (
        client.config.env["DBUS_SESSION_BUS_ADDRESS"] == "unix:path=/run/user/1000/bus"
    )
    assert client.config.env["FABRIC_UNRELATED_SECRET"] == ""
    assert client.config.env["XDG_RUNTIME_DIR"] == "/run/user/1000"
    start = client.thread_start.await_args.kwargs
    assert start["model"] == "gpt-5.4"
    assert start["model_provider"] == "openai"
    assert start["base_instructions"] == "Review carefully."
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
    os.environ["FABRIC_TEST_MCP_HEADER"] = "fabric"
    os.environ["FABRIC_TEST_WINDOWS_HEADER"] = "windows"
    os.environ["FABRIC_TEST_UNBRACED_HEADER"] = "unbraced"
    configure_mcp(
        codex_payload,
        {
            "repo": {
                "transport": "stdio",
                "url": "python",
                "args": [
                    "-m",
                    "repo_mcp",
                    "--root",
                    ".",
                    "--config",
                    "repo config.json",
                ],
                "env": {"REPO_MCP_MODE": "test"},
            },
            "remote": {
                "transport": "streamable-http",
                "url": "${FABRIC_TEST_MCP_URL}",
                "custom_headers": {
                    "X-Tenant": "${FABRIC_TEST_MCP_HEADER}",
                    "X-Windows": "%FABRIC_TEST_WINDOWS_HEADER%",
                    "X-Unbraced": "$FABRIC_TEST_UNBRACED_HEADER",
                    "X-Static": "static",
                },
                "authentication": {
                    "type": "oauth2",
                    "scopes": ["read", "write"],
                    "redirect_uri": "http://127.0.0.1:8765/callback",
                },
            },
        },
    )
    codex_payload["config"]["harness"]["settings"]["config_overrides"][
        "mcp_servers.remote.required"
    ] = True

    output = invoke_once(codex_payload)

    assert output["completed"] is True
    config = mock_codex.instances[0].thread_start.await_args.kwargs["config"]
    assert config["mcp_servers"] == {
        "remote": {
            "url": "https://mcp.example.test/mcp",
            "http_headers": {
                "X-Tenant": "fabric",
                "X-Unbraced": "unbraced",
                "X-Windows": (
                    "windows"
                    if os.name == "nt"
                    else "%FABRIC_TEST_WINDOWS_HEADER%"
                ),
                "X-Static": "static",
            },
            "auth": "oauth",
            "scopes": ["read", "write"],
            "required": True,
        },
        "repo": {
            "command": "python",
            "args": [
                "-m",
                "repo_mcp",
                "--root",
                ".",
                "--config",
                "repo config.json",
            ],
            "env": {"REPO_MCP_MODE": "test"},
        },
    }
    assert mock_codex.instances[0].config.env["FABRIC_TEST_MCP_HEADER"] == ""
    assert mock_codex.instances[0].config.env["FABRIC_TEST_WINDOWS_HEADER"] == ""
    assert mock_codex.instances[0].config.env["FABRIC_TEST_UNBRACED_HEADER"] == ""
    assert config["mcp_oauth_callback_url"] == "http://127.0.0.1:8765/callback"


def test_codex_preserves_prefixed_environment_reference_as_static_header(
    codex_payload,
):
    configure_mcp(
        codex_payload,
        {
            "remote": {
                "transport": "streamable-http",
                "url": "https://mcp.example.test/mcp",
                "custom_headers": {"Authorization": "Bearer ${MCP_TOKEN}"},
            }
        },
    )

    assert adapter._native_mcp_servers(codex_payload)["remote"] == {
        "url": "https://mcp.example.test/mcp",
        "http_headers": {"Authorization": "Bearer ${MCP_TOKEN}"},
    }


def test_codex_rejects_mcp_oauth_client_secret(codex_payload):
    configure_mcp(
        codex_payload,
        {
            "remote": {
                "transport": "streamable-http",
                "url": "https://mcp.example.test/mcp",
                "authentication": {
                    "type": "oauth2",
                    "client_id": "fabric-client",
                    "client_secret_env": "FABRIC_MCP_CLIENT_SECRET",
                },
            },
        },
    )

    with pytest.raises(adapter.AdapterConfigError, match="client_secret_env"):
        adapter.thread_config(codex_payload, None)


def test_codex_rejects_mcp_oauth_client_id(codex_payload):
    configure_mcp(
        codex_payload,
        {
            "remote": {
                "transport": "streamable-http",
                "url": "https://mcp.example.test/mcp",
                "authentication": {
                    "type": "oauth2",
                    "client_id": "fabric-client",
                },
            },
        },
    )

    with pytest.raises(adapter.AdapterConfigError, match="client_id"):
        adapter.thread_config(codex_payload, None)


async def test_mcp_auth_statuses_paginates_until_cursor_is_none():
    client = MagicMock()
    client.request = AsyncMock(
        side_effect=[
            SimpleNamespace(
                data=[
                    SimpleNamespace(
                        name="first", auth_status=adapter.McpAuthStatus.o_auth
                    )
                ],
                next_cursor="page-2",
            ),
            SimpleNamespace(
                data=[
                    SimpleNamespace(
                        name="second",
                        auth_status=adapter.McpAuthStatus.not_logged_in,
                    )
                ],
                next_cursor=None,
            ),
        ]
    )

    statuses = await adapter._mcp_auth_statuses(
        client, thread_id="thread-123", timeout=1
    )

    assert statuses == {
        "first": adapter.McpAuthStatus.o_auth,
        "second": adapter.McpAuthStatus.not_logged_in,
    }
    assert client.request.await_args_list[0].args[1] == {
        "detail": "toolsAndAuthOnly",
        "threadId": "thread-123",
    }
    assert client.request.await_args_list[1].args[1] == {
        "detail": "toolsAndAuthOnly",
        "threadId": "thread-123",
        "cursor": "page-2",
    }


async def test_mcp_auth_statuses_rejects_repeated_cursor():
    client = MagicMock()
    client.request = AsyncMock(
        side_effect=[
            SimpleNamespace(data=[], next_cursor="repeated"),
            SimpleNamespace(data=[], next_cursor="repeated"),
        ]
    )

    with pytest.raises(
        adapter.AdapterConfigError, match="returned a repeated cursor"
    ) as caught:
        await adapter._mcp_auth_statuses(client, thread_id="thread-123", timeout=1)

    assert caught.value.code == "codex_mcp_authentication_failed"
    assert client.request.await_count == 2


async def test_mcp_auth_statuses_respects_invocation_timeout():
    async def block_request(*args, **kwargs):
        await asyncio.Event().wait()

    client = MagicMock()
    client.request = AsyncMock(side_effect=block_request)

    with pytest.raises(
        adapter.AdapterConfigError, match="status listing timed out"
    ) as caught:
        await adapter._mcp_auth_statuses(client, thread_id="thread-123", timeout=0.01)

    assert caught.value.code == "codex_mcp_authentication_failed"


async def test_mcp_auth_statuses_distinguishes_request_timeout():
    request_timeout = TimeoutError("request transport timed out")
    client = MagicMock()
    client.request = AsyncMock(side_effect=request_timeout)

    with pytest.raises(
        adapter.AdapterConfigError, match="status listing request timed out"
    ) as caught:
        await adapter._mcp_auth_statuses(client, thread_id="thread-123", timeout=1)

    assert caught.value.code == "codex_mcp_authentication_failed"
    assert caught.value.__cause__ is request_timeout


@pytest.mark.parametrize(
    ("invocation_timeout", "oauth_timeout", "expected_timeout"),
    [(30, 12, 12), (5, 12, 5), (5.1, 12, 6)],
)
def test_codex_logs_into_mcp_server_before_first_turn(
    codex_payload,
    mock_codex,
    monkeypatch,
    invocation_timeout,
    oauth_timeout,
    expected_timeout,
):
    codex_payload["config"]["runtime"]["timeout_seconds"] = invocation_timeout
    configure_mcp(
        codex_payload,
        {
            "remote": {
                "transport": "streamable-http",
                "url": "https://mcp.example.test/mcp",
                "authentication": {
                    "type": "oauth2",
                    "scopes": ["read", "write"],
                    "authorization_timeout_seconds": oauth_timeout,
                },
            },
        },
    )
    mock_codex.mcp_auth_statuses["remote"] = adapter.McpAuthStatus.not_logged_in
    open_browser = AsyncMock(return_value=True)
    monkeypatch.setattr(adapter, "_open_authorization_url", open_browser)

    output = invoke_once(codex_payload)

    assert output["completed"] is True
    requests = mock_codex.skill_request.await_args_list
    assert [call.args[0] for call in requests] == [
        "mcpServerStatus/list",
        "mcpServer/oauth/login",
    ]
    assert requests[1].args[1] == {
        "name": "remote",
        "scopes": ["read", "write"],
        "threadId": "thread-123",
        "timeoutSecs": expected_timeout,
    }
    open_browser.assert_awaited_once_with(mock_codex.oauth_authorization_url)
    mock_codex.next_notification.assert_awaited_once_with()
    mock_codex.instances[0].thread.turn.assert_awaited_once()


def test_codex_reports_failed_mcp_oauth_login_before_turn(
    codex_payload, mock_codex, monkeypatch
):
    configure_mcp(
        codex_payload,
        {
            "remote": {
                "transport": "streamable-http",
                "url": "https://mcp.example.test/mcp",
                "authentication": {"type": "oauth2"},
            },
        },
    )
    mock_codex.mcp_auth_statuses["remote"] = adapter.McpAuthStatus.not_logged_in
    mock_codex.mcp_login_success = False
    mock_codex.mcp_login_error = "authorization denied"
    monkeypatch.setattr(
        adapter,
        "_open_authorization_url",
        AsyncMock(return_value=True),
    )

    output = invoke_once(codex_payload)

    assert output["failed"] is True
    assert output["error"]["code"] == "codex_mcp_authentication_failed"
    assert output["error"]["message"] == (
        "Codex MCP OAuth login failed for server 'remote'"
    )
    mock_codex.instances[0].thread.turn.assert_not_awaited()


def test_codex_reports_missing_mcp_auth_status_before_turn(codex_payload, mock_codex):
    configure_mcp(
        codex_payload,
        {
            "remote": {
                "transport": "streamable-http",
                "url": "https://mcp.example.test/mcp",
                "authentication": {"type": "oauth2"},
            },
        },
    )
    mock_codex.report_mcp_auth_statuses = False

    output = invoke_once(codex_payload)

    assert output["failed"] is True
    assert output["error"] == {
        "code": "codex_mcp_authentication_failed",
        "message": "Codex did not report a status for MCP server 'remote'",
        "retryable": False,
    }
    assert [call.args[0] for call in mock_codex.skill_request.await_args_list] == [
        "mcpServerStatus/list"
    ]
    mock_codex.next_notification.assert_not_awaited()
    mock_codex.instances[0].thread.turn.assert_not_awaited()


@pytest.mark.parametrize("opened", [True, False])
async def test_codex_opens_mcp_authorization_url_without_blocking(monkeypatch, opened):
    open_browser = MagicMock(return_value=opened)
    monkeypatch.setattr(adapter.webbrowser, "open", open_browser)
    to_thread = AsyncMock(return_value=opened)
    monkeypatch.setattr(adapter.asyncio, "to_thread", to_thread)

    assert await adapter._open_authorization_url("https://auth.example.test") is opened
    to_thread.assert_awaited_once_with(open_browser, "https://auth.example.test")


def test_codex_rejects_mcp_service_account_authentication(codex_payload):
    configure_mcp(
        codex_payload,
        {
            "remote": {
                "transport": "streamable-http",
                "url": "https://mcp.example.test/mcp",
                "authentication": {
                    "type": "service_account",
                    "client_id": "fabric-client",
                    "client_secret_env": "FABRIC_MCP_CLIENT_SECRET",
                    "token_url": "https://auth.example.test/token",
                },
            },
        },
    )

    with pytest.raises(adapter.AdapterConfigError, match="service_account"):
        adapter.thread_config(codex_payload, None)


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
    configure_mcp(
        codex_payload,
        {"bad": {"transport": transport, "url": "https://mcp.example.test"}},
    )

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


def test_sdk_test_override_can_use_an_explicit_codex_runtime(
    codex_payload, mock_codex, tmp_path, monkeypatch
):
    codex_bin = tmp_path / "bin" / "codex"
    codex_bin.parent.mkdir()
    codex_bin.touch()
    monkeypatch.setenv("FABRIC_TEST_CODEX_BIN", str(codex_bin))

    output = invoke_once(codex_payload)

    assert output["completed"] is True
    assert mock_codex.instances[0].config.codex_bin == str(codex_bin)


@pytest.mark.parametrize("codex_bin", ["bin/codex", "~/bin/codex"])
def test_sdk_test_override_resolves_relative_runtime_from_base_dir(
    codex_payload, codex_bin, monkeypatch
):
    monkeypatch.setenv("FABRIC_TEST_CODEX_BIN", codex_bin)

    config = adapter.sdk_config(codex_payload, relay=None)

    base_dir = Path(codex_payload["base_dir"])
    assert config.codex_bin == str((base_dir / codex_bin).resolve())


def test_sdk_test_override_keeps_absolute_runtime_path(
    codex_payload, tmp_path, monkeypatch
):
    codex_bin = tmp_path / "bin" / ".." / "codex"
    monkeypatch.setenv("FABRIC_TEST_CODEX_BIN", str(codex_bin))

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
        }
    }
    configure_mcp(
        codex_payload,
        {
            "review": {
                "transport": "streamable-http",
                "url": "https://mcp.example.test/review",
            }
        },
    )
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


async def test_relay_waits_for_delayed_atif_before_collecting_artifacts(
    codex_payload, mock_codex, monkeypatch, tmp_path
):
    codex_payload["telemetry_plan"] = {
        "providers": ["relay"],
        "relay_enabled": True,
    }
    atif_dir = tmp_path / "relay" / "atif"
    atif_dir.mkdir(parents=True)
    atif_file = atif_dir / "trajectory-session.atif.json"
    relay = relay_settings(tmp_path, atif_plugin_config(atif_dir))
    install_mock_relay(monkeypatch, relay)
    mock_sdk_thread = mock_thread("thread-123")
    write_task = None

    async def finish_turn():
        nonlocal write_task

        async def write_atif():
            await asyncio.sleep(0.05)
            atif_file.write_text(
                json.dumps({"schema_version": "ATIF-v1.7", "steps": []}),
                encoding="utf-8",
            )

        write_task = asyncio.create_task(write_atif())
        return successful_result()

    mock_sdk_thread.handle.run.side_effect = finish_turn
    mock_codex.next_thread = mock_sdk_thread
    runtime = adapter.CodexRuntime()

    await runtime.start(lifecycle_start_payload(codex_payload))
    try:
        output = await runtime.invoke(lifecycle_invocation(codex_payload))
        assert write_task is not None
        await write_task
    finally:
        await runtime.stop()

    assert output["completed"] is True
    assert output["relay_artifacts"] == [{"kind": "atif", "path": str(atif_file)}]


async def test_relay_atif_timeout_fails_successful_turn_explicitly(
    codex_payload, mock_codex, monkeypatch, tmp_path
):
    codex_payload["telemetry_plan"] = {
        "providers": ["relay"],
        "relay_enabled": True,
    }
    atif_dir = tmp_path / "relay" / "atif"
    atif_dir.mkdir(parents=True)
    stale_atif = atif_dir / "trajectory-existing.atif.json"
    stale_atif.write_text('{"schema_version":"ATIF-v1.7","steps":[]}', encoding="utf-8")
    late_atif = atif_dir / "trajectory-late.atif.json"
    relay = relay_settings(tmp_path, atif_plugin_config(atif_dir))
    install_mock_relay(monkeypatch, relay)
    wait_for_atif = AsyncMock(return_value=None)
    monkeypatch.setattr(
        adapter.relay_artifacts, "wait_for_finalized_atif", wait_for_atif
    )
    runtime = adapter.CodexRuntime()

    await runtime.start(lifecycle_start_payload(codex_payload))
    try:
        output = await runtime.invoke(lifecycle_invocation(codex_payload))
        late_atif.write_text(
            '{"schema_version":"ATIF-v1.7","steps":[]}', encoding="utf-8"
        )
        unavailable = await runtime.invoke(lifecycle_invocation(codex_payload))
    finally:
        await runtime.stop()

    assert output["failed"] is True
    assert output["error"] == {
        "code": "codex_relay_atif_timeout",
        "message": "NeMo Relay did not finalize an ATIF artifact before the deadline",
        "retryable": False,
        "metadata": {
            "timeout_seconds": adapter.relay_artifacts.ATIF_FINALIZATION_TIMEOUT_SECONDS
        },
    }
    wait_for_atif.assert_awaited_once()
    assert output["relay_runtime"]["enabled"] is True
    assert output["relay_artifacts"] == []
    assert unavailable["error"]["code"] == "codex_runtime_unavailable"
    assert "relay_runtime" not in unavailable
    assert "relay_artifacts" not in unavailable


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


def test_custom_provider_requires_explicit_api_key_env(codex_payload, mock_codex):
    model = codex_payload["config"]["models"]["default"]
    model.update(
        {
            "provider": "acme",
            "model": "acme/code-model",
            "base_url": "https://acme.example/v1",
        }
    )

    error = runtime_start_error(codex_payload)

    assert error.code == "codex_invalid_configuration"
    assert "api_key_env is required" in error.message
    mock_codex.assert_not_called()


def test_custom_provider_uses_responses_api_and_configured_credential(
    codex_payload, mock_codex
):
    model = codex_payload["config"]["models"]["default"]
    model.update(
        {
            "provider": "acme",
            "model": "acme/code-model",
            "api_key_env": "ACME_API_KEY",
            "base_url": "https://acme.example/v1/",
        }
    )
    os.environ["ACME_API_KEY"] = "acme-secret"

    output = invoke_once(codex_payload)

    assert output["completed"] is True
    client = mock_codex.instances[0]
    assert client.config.env["ACME_API_KEY"] == "acme-secret"
    start = client.thread_start.await_args.kwargs
    assert start["model"] == "acme/code-model"
    assert start["model_provider"] == "acme"
    assert Path(client.config.env["CODEX_HOME"]).parts[-3:] == (
        ".fabric",
        "codex",
        "custom-provider-home",
    )
    assert start["config"]["features"] == {"web_search": False}
    assert start["config"]["model_providers"] == {
        "acme": {
            "name": "acme",
            "base_url": "https://acme.example/v1",
            "env_key": "ACME_API_KEY",
            "wire_api": "responses",
        }
    }


def test_custom_provider_normalizes_codex_home_creation_failure(
    codex_payload, mock_codex, monkeypatch
):
    model = codex_payload["config"]["models"]["default"]
    model.update(
        {
            "provider": "acme",
            "model": "acme/code-model",
            "api_key_env": "ACME_API_KEY",
            "base_url": "https://acme.example/v1",
        }
    )
    os.environ["ACME_API_KEY"] = "acme-secret"

    async def fail_to_create_home(*_args, **_kwargs):
        raise OSError("read-only filesystem")

    monkeypatch.setattr(adapter.asyncio, "to_thread", fail_to_create_home)

    error = runtime_start_error(codex_payload)

    assert error.code == "codex_runtime_unavailable"
    mock_codex.assert_not_called()


def test_custom_provider_requires_credential(codex_payload, mock_codex):
    model = codex_payload["config"]["models"]["default"]
    model.update(
        {
            "provider": "acme",
            "model": "acme/code-model",
            "api_key_env": "ACME_API_KEY",
            "base_url": "https://acme.example/v1",
        }
    )
    os.environ.pop("ACME_API_KEY", None)

    error = runtime_start_error(codex_payload)

    assert error.code == "codex_invalid_configuration"
    assert "ACME_API_KEY is required" in error.message
    assert not (adapter.state_dir(codex_payload) / "custom-provider-home").exists()
    mock_codex.assert_not_called()


def test_custom_provider_requires_explicit_endpoint(codex_payload, mock_codex):
    model = codex_payload["config"]["models"]["default"]
    model.update(
        {
            "provider": "acme",
            "model": "acme/code-model",
            "api_key_env": "ACME_API_KEY",
        }
    )
    os.environ["ACME_API_KEY"] = "acme-secret"

    error = runtime_start_error(codex_payload)

    assert error.code == "codex_invalid_configuration"
    assert "base_url is required" in error.message
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
    executable_arg = str(executable)
    if sys.platform == "win32":
        executable_arg = subprocess.list2cmdline([executable_arg.replace("\\", "/")])
    assert config["hooks"]["SessionStart"][0]["hooks"][0] == {
        "type": "command",
        "command": f"{executable_arg} hook-forward codex",
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


def test_relay_routes_custom_provider_through_gateway(codex_payload, tmp_path):
    model = codex_payload["config"]["models"]["default"]
    model.update(
        {
            "provider": "acme",
            "model": "acme/code-model",
            "api_key_env": "ACME_API_KEY",
            "base_url": "https://acme.example/v1",
        }
    )
    codex_payload["telemetry_plan"] = {
        "providers": ["relay"],
        "relay_enabled": True,
    }
    os.environ["ACME_API_KEY"] = "acme-secret"
    gateway = adapter.relay_gateway.RelayGatewayLaunch(
        executable=tmp_path / "nemo-relay",
        config_path=tmp_path / "relay" / "config.toml",
        bind="127.0.0.1:43210",
        url="http://127.0.0.1:43210",
        log_path=tmp_path / "relay" / "gateway.log",
        openai_base_url="https://acme.example/v1",
    )
    relay = adapter.CodexRelaySettings(
        gateway=gateway,
        plugin_config={"version": 1, "components": []},
    )

    adapter.validate_runtime_payload(codex_payload)
    config = adapter.thread_config(codex_payload, relay)

    assert config["model_providers"]["acme"] == {
        "name": "acme",
        "base_url": gateway.url,
        "env_key": "ACME_API_KEY",
        "wire_api": "responses",
    }


def test_prepare_relay_reuses_one_resolved_executable(
    codex_payload, monkeypatch, tmp_path
):
    codex_payload["config"]["models"]["default"].update(
        {
            "provider": "acme",
            "model": "acme/code-model",
            "api_key_env": "ACME_API_KEY",
            "base_url": "https://acme.example/v1/",
        }
    )
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
    assert relay.gateway.openai_base_url == "https://acme.example/v1"
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
    assert "service_name" not in start
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
    codex_payload["config"]["runtime"]["timeout_seconds"] = 0.01

    output = invoke_once(codex_payload)

    client = mock_codex.instances[0]
    assert output["error"]["code"] == "codex_timed_out"
    assert client.thread.handle.interrupted is True
    assert client.closed is True


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
        "models.base_url",
        "instructions.system",
        "mcp",
        "mcp.auth.oauth2",
        "skills",
    ]
    assert descriptor["model_schema"]["if"]["properties"]["provider"] == {
        "const": "openai"
    }
    assert descriptor["model_schema"]["else"]["required"] == [
        "base_url",
        "api_key_env",
    ]
    assert descriptor["model_schema"]["properties"]["settings"] == {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }
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
    plan = Fabric().plan(config, base_dir=BASE_DIR)

    assert plan.adapter.adapter_id == "nvidia.fabric.codex"
    assert plan.adapter.harness == "codex"
    assert plan.config.runtime.input_schema == "text"
    assert plan.config.harness.settings["reasoning_effort"] == "high"
    native = plan["capability_plan"]["native"]
    assert native["skill_paths"] == [str(skill_path)]
    assert native["mcp_servers"]["github"] == {
        "transport": "streamable-http",
        "url": "https://mcp.example.test/mcp",
        "exposure": "harness_native",
    }
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
    codex_payload["runtime_context"]["environment"]["env"] = {
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
