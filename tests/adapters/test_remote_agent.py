# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end protocol tests for the remote-agent adapter."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import httpx
import pytest
from nemo_fabric_adapter_contract.models import AgentConfig
from nemo_fabric_adapter_contract.models import AgentRunRequest
from nemo_fabric_adapter_contract.models import RuntimeContext
from nemo_fabric_adapters.remote_agent import adapter


def _context() -> RuntimeContext:
    return RuntimeContext.from_mapping(
        {
            "runtime_id": "remote-agent-runtime",
            "invocation_id": "remote-agent-invocation",
            "request_id": "remote-agent-request",
            "environment": {
                "environment_id": "remote-agent-environment",
                "provider": "local",
                "control_location": "in_env_control",
                "workspace": ".",
                "env": {},
                "ownership": "caller_owned",
            },
            "artifacts": {},
        }
    )


async def test_sse_events_flushes_unterminated_final_event():
    async def lines():
        yield "event: response.completed"
        yield 'data: {"response": {"id": "response-id"}}'

    mock_response = MagicMock()
    mock_response.aiter_lines.return_value = lines()

    events = [event async for event in adapter._sse_events(mock_response)]

    assert events == [("response.completed", {"response": {"id": "response-id"}})]


@pytest.mark.parametrize(
    ("api_type", "path"),
    [
        pytest.param("openai-responses", "/responses", id="openai-responses"),
        pytest.param(
            "openai-completions", "/chat/completions", id="openai-completions"
        ),
        pytest.param("anthropic-messages", "/messages", id="anthropic-messages"),
    ],
)
async def test_remote_agent_invokes_supported_protocol(
    api_server: str, api_type: str, path: str, repo_root: Path
):
    config = AgentConfig.from_mapping(
        {
            "harness": {
                "settings": {"base_url": f"{api_server}/v1", "api_type": api_type}
            },
            "instructions": {"system": {"content": "Be concise."}},
            "models": {
                "default": {
                    "provider": "test",
                    "model": "fabric-echo",
                    "temperature": 0.2,
                    "settings": {"max_tokens": 64},
                }
            },
        }
    )
    context = _context()
    runtime = adapter.RemoteAgentRuntime()
    await runtime.start(
        {
            "config": config,
            "runtime_context": context.to_mapping(),
            "base_dir": str(repo_root),
        }
    )

    try:
        assert runtime._endpoint == f"{api_server}/v1{path}"

        result = await runtime.invoke(AgentRunRequest(input="Hello."), context)
    finally:
        await runtime.stop()

    async with httpx.AsyncClient() as control_client:
        captured = (await control_client.get(f"{api_server}/_requests")).json()
    assert result.status == "succeeded"
    expected_text = (
        "echo user_count=1 latest=Hello."
        if api_type == "openai-completions"
        else "echo response"
    )
    assert result.output == {"response": expected_text}
    assert result.usage.input_tokens == 0
    assert captured[-1]["model"] == "fabric-echo"
    assert captured[-1]["temperature"] == 0.2
    assert captured[-1].get("stream", False) is (api_type != "openai-completions")
    assert captured[-1]["messages" if api_type != "openai-responses" else "input"]


async def test_remote_agent_retains_transcript_and_reports_http_failure(
    api_server: str,
    repo_root: Path,
):
    config = AgentConfig.from_mapping(
        {
            "harness": {"settings": {"base_url": f"{api_server}/v1"}},
            "models": {"default": {"provider": "test", "model": "fabric-echo"}},
        }
    )
    context = _context()
    runtime = adapter.RemoteAgentRuntime()
    await runtime.start(
        {
            "config": config,
            "runtime_context": context.to_mapping(),
            "base_dir": str(repo_root),
        }
    )

    try:
        await runtime.invoke(AgentRunRequest(input="First."), context)
        await runtime.invoke(AgentRunRequest(input="Second."), context)

        async with httpx.AsyncClient() as control_client:
            captured = (await control_client.get(f"{api_server}/_requests")).json()
        assert [message["role"] for message in captured[-1]["input"]] == [
            "user",
            "assistant",
            "user",
        ]

        async with httpx.AsyncClient() as control_client:
            await control_client.post(
                f"{api_server}/_scenario", json={"status_code": 503}
            )
        result = await runtime.invoke(AgentRunRequest(input="Retry."), context)
    finally:
        await runtime.stop()

    assert result.status == "failed"
    assert result.error.code == "remote_agent_http_error"
    assert result.error.retryable is True


async def test_remote_agent_configures_http_client(
    monkeypatch: pytest.MonkeyPatch, repo_root: Path
):
    mock_client = MagicMock()
    mock_client.aclose = AsyncMock()
    mock_async_client = MagicMock(return_value=mock_client)
    monkeypatch.setattr(adapter.httpx, "AsyncClient", mock_async_client)

    config = AgentConfig.from_mapping(
        {
            "harness": {
                "settings": {
                    "base_url": "https://agents.example.test/v1",
                    "connect_timeout_seconds": 2.5,
                    "read_timeout_seconds": 30,
                }
            },
            "models": {"default": {"provider": "test", "model": "fabric-echo"}},
        }
    )
    context = _context()
    runtime = adapter.RemoteAgentRuntime()
    await runtime.start(
        {
            "config": config,
            "runtime_context": context.to_mapping(),
            "base_dir": str(repo_root),
        }
    )
    await runtime.stop()
    mock_client.aclose.assert_awaited_once()

    assert mock_async_client.call_args.kwargs["http2"] is True
    timeout = mock_async_client.call_args.kwargs["timeout"]
    assert timeout.connect == 2.5
    assert timeout.read == 30
    assert timeout.write == 30
    assert timeout.pool == 2.5


async def test_remote_agent_maps_timeout_and_closes_client(
    monkeypatch: pytest.MonkeyPatch, repo_root: Path
):
    request = httpx.Request("POST", "https://agents.example.test/v1/chat/completions")
    mock_client = MagicMock()
    mock_client.post = AsyncMock(
        side_effect=httpx.ReadTimeout("remote agent timed out", request=request)
    )
    mock_client.aclose = AsyncMock()
    monkeypatch.setattr(
        adapter.httpx, "AsyncClient", MagicMock(return_value=mock_client)
    )

    config = AgentConfig.from_mapping(
        {
            "harness": {
                "settings": {
                    "base_url": "https://agents.example.test/v1",
                    "api_type": "openai-completions",
                }
            },
            "models": {"default": {"provider": "test", "model": "fabric-echo"}},
        }
    )
    context = _context()
    runtime = adapter.RemoteAgentRuntime()
    await runtime.start(
        {
            "config": config,
            "runtime_context": context.to_mapping(),
            "base_dir": str(repo_root),
        }
    )

    try:
        with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
            await runtime.invoke(AgentRunRequest(input="Hello."), context)
    finally:
        await runtime.stop()

    assert caught.value.code == "remote_agent_transport_failed"
    assert caught.value.retryable is True
    mock_client.aclose.assert_awaited_once()


async def test_remote_agent_rejects_append_system_instruction(repo_root: Path):
    config = AgentConfig.from_mapping(
        {
            "harness": {"settings": {"base_url": "https://agents.example.test/v1"}},
            "instructions": {
                "system": {"content": "Additional guidance.", "mode": "append"}
            },
            "models": {"default": {"provider": "test", "model": "fabric-echo"}},
        }
    )
    context = _context()
    runtime = adapter.RemoteAgentRuntime()

    with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
        await runtime.start(
            {
                "config": config,
                "runtime_context": context.to_mapping(),
                "base_dir": str(repo_root),
            }
        )

    assert caught.value.code == "unsupported_system_instruction_mode"
    assert caught.value.metadata["field"] == "instructions.system.mode"


def test_remote_agent_descriptor_and_module_entrypoint(repo_root: Path):
    descriptor = json.loads(
        (
            repo_root / "adapters/remote-agent/remote-agent.fabric-adapter.json"
        ).read_text(encoding="utf-8")
    )
    result = subprocess.run(
        [sys.executable, "-m", "nemo_fabric_adapters.remote_agent.adapter"],
        input="",
        text=True,
        capture_output=True,
        check=False,
    )

    assert descriptor["adapter_id"] == "nvidia.fabric.remote-agent"
    assert descriptor["settings_schema"]["properties"]["api_type"]["default"] == (
        "openai-responses"
    )
    assert descriptor["config"]["system_instruction_modes"] == ["replace"]
    assert descriptor["capabilities"]["streaming"] is False
    assert "telemetry" not in descriptor
    assert result.returncode == 0, result.stderr
