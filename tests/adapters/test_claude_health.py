# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Claude health observations of the owned CLI and its control interface."""

import asyncio
import time
from unittest.mock import MagicMock

import pytest
from anyio.abc import Process
from claude_agent_sdk import ClaudeSDKClient
from claude_agent_sdk._internal.transport.subprocess_cli import SubprocessCLITransport
from nemo_fabric_adapter_contract.models import (
    AgentConfig,
    AdapterCheckResult,
    AgentRunRequest,
    RuntimeContext,
)
from nemo_fabric_adapters.claude.adapter import ClaudeRuntime


@pytest.fixture(name="claude_health")
def claude_health_fixture():
    runtime = ClaudeRuntime()
    client = MagicMock(spec=ClaudeSDKClient)
    transport = MagicMock(spec=SubprocessCLITransport)
    process = MagicMock(spec=Process)
    process.returncode = None
    transport._process = process
    transport.is_ready.return_value = True
    client._transport = transport
    client.get_mcp_status.return_value = {
        "mcpServers": [{"name": "repo", "status": "connected"}]
    }
    runtime._client = client
    runtime._fabric_runtime_id = "runtime-health-test"
    runtime._agent_config = AgentConfig.from_mapping(
        {
            "models": {"default": {"provider": "anthropic", "model": "claude-test"}},
            "mcp": {"servers": {"repo": {"transport": "stdio", "url": "repo-mcp"}}},
        }
    )
    return runtime, client, transport, process


@pytest.mark.parametrize("level", ["live", "active", "ready"])
async def test_claude_health_uses_current_native_evidence(claude_health, level):
    runtime, client, _, _ = claude_health
    result = await runtime.check(level)
    AdapterCheckResult.from_mapping(result)
    assert result["outcome"] == "passed"
    assert result["activity"] == "idle"
    assert result["reason_code"] is None
    assert result["observed_at"] is not None
    assert (
        result["source"]
        == {
            "live": "fabric.adapter.liveness",
            "active": "fabric.adapter.interface",
            "ready": "fabric.adapter.readiness",
        }[level]
    )
    assert client.get_mcp_status.await_count == (level != "live")
    client.query.assert_not_called()
    client.get_server_info.assert_not_called()
    client.get_context_usage.assert_not_called()
    client.interrupt.assert_not_called()
    assert runtime._claude_session_id is None
    assert runtime._cost_total is None


@pytest.mark.parametrize("level", ["live", "active", "ready"])
async def test_claude_health_detects_owned_process_exit(claude_health, level):
    runtime, client, _, process = claude_health
    process.returncode = 7
    result = await runtime.check(level)
    assert (result["outcome"], result["reason_code"]) == ("failed", "agent_exited")
    assert result["activity"] == "stopped"
    client.get_mcp_status.assert_not_called()


@pytest.mark.parametrize("missing", ["client", "transport", "process"])
async def test_claude_health_does_not_treat_retained_handles_as_liveness(
    claude_health, missing
):
    runtime, client, transport, _ = claude_health
    if missing == "client":
        runtime._client = None
    elif missing == "transport":
        client._transport = None
    else:
        transport._process = None
    result = await runtime.check("live")
    assert (result["outcome"], result["reason_code"]) == (
        "unknown",
        "runtime_unavailable",
    )


@pytest.mark.parametrize(
    "response",
    [
        None,
        {},
        {"mcpServers": None},
        {"mcpServers": [{}]},
        {"mcpServers": [{"name": "repo", "status": []}]},
        {"mcpServers": [{"name": "repo", "status": "new-status"}]},
        {"mcpServers": [{"name": "repo", "status": "connected"}] * 2},
    ],
)
async def test_claude_health_rejects_malformed_native_responses(
    claude_health, response
):
    runtime, client, _, _ = claude_health
    client.get_mcp_status.return_value = response
    result = await runtime.check("active")
    assert (result["outcome"], result["reason_code"]) == ("unknown", "probe_error")


@pytest.mark.parametrize("status", ["failed", "needs-auth", "disabled", "pending"])
async def test_claude_readiness_uses_native_response_independently_of_mcp_connections(
    claude_health, status
):
    runtime, client, _, _ = claude_health
    client.get_mcp_status.return_value = {
        "mcpServers": [
            {
                "name": "repo",
                "status": status,
                "error": "credential-bearing native error",
                "config": {"env": {"SECRET": "do-not-report"}},
            }
        ]
    }
    assert (await runtime.check("active"))["outcome"] == "passed"
    result = await runtime.check("ready")
    assert (result["outcome"], result["reason_code"]) == ("passed", None)
    assert "credential-bearing" not in str(result)
    assert "do-not-report" not in str(result)


async def test_claude_readiness_accepts_native_status_with_or_without_mcp_configuration(
    claude_health,
):
    runtime, client, _, _ = claude_health
    client.get_mcp_status.return_value = {"mcpServers": []}
    result = await runtime.check("ready")
    assert (result["outcome"], result["reason_code"]) == ("passed", None)
    runtime._agent_config = AgentConfig.from_mapping(
        {"models": {"default": {"provider": "anthropic", "model": "claude-test"}}}
    )
    assert (await runtime.check("ready"))["outcome"] == "passed"


@pytest.mark.parametrize(
    "change", ["stop", "replace-client", "replace-process", "exit", "close-interface"]
)
async def test_claude_health_rechecks_ownership_after_native_response(
    claude_health, change
):
    runtime, client, transport, process = claude_health

    async def respond():
        if change == "stop":
            await runtime.stop()
        elif change == "replace-client":
            runtime._client = MagicMock(spec=ClaudeSDKClient)
        elif change == "replace-process":
            transport._process = MagicMock(spec=Process)
        elif change == "close-interface":
            transport.is_ready.return_value = False
        else:
            process.returncode = 9
        return {"mcpServers": [{"name": "repo", "status": "connected"}]}

    client.get_mcp_status.side_effect = respond
    result = await runtime.check("ready")
    expected = (
        ("failed", "agent_exited")
        if change == "exit"
        else ("failed", "endpoint_unavailable")
        if change == "close-interface"
        else ("unknown", "observation_changed")
    )
    assert (result["outcome"], result["reason_code"]) == expected


async def test_claude_health_reports_probe_errors_without_claiming_process_death(
    claude_health,
):
    runtime, client, _, _ = claude_health
    client.get_mcp_status.side_effect = RuntimeError("secret-native-detail")
    result = await runtime.check("active")
    assert (result["outcome"], result["reason_code"]) == ("unknown", "probe_error")
    assert "secret-native-detail" not in str(result)
    assert (await runtime.check("live"))["outcome"] == "passed"


async def test_claude_health_cancellation_bounds_native_requests_and_recovery(
    claude_health,
):
    runtime, client, _, _ = claude_health
    entered, release = asyncio.Event(), asyncio.Event()

    async def respond():
        entered.set()
        await release.wait()
        return {"mcpServers": [{"name": "repo", "status": "connected"}]}

    client.get_mcp_status.side_effect = respond
    probe = asyncio.create_task(runtime.check("ready"))
    await asyncio.wait_for(entered.wait(), 1)
    probe.cancel()
    with pytest.raises(asyncio.CancelledError):
        await probe
    try:
        for _ in range(3):
            with pytest.raises(TimeoutError):
                async with asyncio.timeout(0.01):
                    await runtime.check("ready")
        assert client.get_mcp_status.await_count == 1
        client.interrupt.assert_not_called()
    finally:
        release.set()
    assert (await runtime.check("ready"))["outcome"] == "passed"
    assert client.get_mcp_status.await_count == 2


async def test_claude_health_unusable_runtime_and_closed_interface(claude_health):
    runtime, client, transport, _ = claude_health
    transport.is_ready.return_value = False
    assert (await runtime.check("live"))["outcome"] == "passed"
    result = await runtime.check("active")
    assert (result["outcome"], result["reason_code"]) == (
        "failed",
        "endpoint_unavailable",
    )
    transport.is_ready.return_value = True
    runtime._unusable = True
    assert (await runtime.check("ready"))["outcome"] == "failed"
    client.get_mcp_status.assert_not_called()


async def test_claude_health_does_not_wait_for_invocation(claude_health):
    runtime, client, _, _ = claude_health
    entered, release = asyncio.Event(), asyncio.Event()

    async def receive():
        entered.set()
        await release.wait()
        yield

    client.receive_response.side_effect = receive
    context = RuntimeContext.from_mapping(
        {
            "runtime_id": "runtime-health-test",
            "request_id": "health-test",
            "invocation_id": "health-invocation",
            "deadline_millis": int((time.time() + 10) * 1000),
            "environment": {
                "environment_id": "health-test",
                "provider": "local",
                "control_location": "external_control",
                "ownership": "caller_owned",
                "env": {},
            },
            "artifacts": {},
        }
    )
    invocation = asyncio.create_task(
        runtime.invoke(AgentRunRequest(input="keep running"), context)
    )
    try:
        await asyncio.wait_for(entered.wait(), 1)
        result = await asyncio.wait_for(runtime.check("ready"), 1)
        assert result["outcome"] == "passed"
        assert result["activity"] == "busy"
        assert not invocation.done()
        client.query.assert_awaited_once_with("keep running")
        client.interrupt.assert_not_called()
        assert runtime._claude_session_id is None
    finally:
        release.set()
        await invocation
    assert (await runtime.check("live"))["activity"] == "idle"
