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

import pytest
import requests
from nemo_fabric_adapter_contract.models import AgentConfig
from nemo_fabric_adapter_contract.models import AgentRunRequest
from nemo_fabric_adapter_contract.models import RuntimeContext
from nemo_fabric_adapters.remote_agent import adapter


ROOT = Path(__file__).resolve().parents[2]


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
    api_server: str, api_type: str, path: str
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
            "base_dir": str(ROOT),
        }
    )
    assert runtime._endpoint == f"{api_server}/v1{path}"

    result = await runtime.invoke(AgentRunRequest(input="Hello."), context)
    await runtime.stop()

    captured = requests.get(f"{api_server}/_requests", timeout=5).json()
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
            "base_dir": str(ROOT),
        }
    )
    await runtime.invoke(AgentRunRequest(input="First."), context)
    await runtime.invoke(AgentRunRequest(input="Second."), context)

    captured = requests.get(f"{api_server}/_requests", timeout=5).json()
    assert [message["role"] for message in captured[-1]["input"]] == [
        "user",
        "assistant",
        "user",
    ]

    requests.post(f"{api_server}/_scenario", json={"status_code": 503}, timeout=5)
    result = await runtime.invoke(AgentRunRequest(input="Retry."), context)
    await runtime.stop()

    assert result.status == "failed"
    assert result.error.code == "remote_agent_http_error"
    assert result.error.retryable is True


async def test_remote_agent_enables_http2(monkeypatch: pytest.MonkeyPatch):
    mock_client = MagicMock()
    mock_client.aclose = AsyncMock()
    mock_async_client = MagicMock(return_value=mock_client)
    monkeypatch.setattr(adapter.httpx, "AsyncClient", mock_async_client)

    config = AgentConfig.from_mapping(
        {
            "harness": {"settings": {"base_url": "https://agents.example.test/v1"}},
            "models": {"default": {"provider": "test", "model": "fabric-echo"}},
        }
    )
    context = _context()
    runtime = adapter.RemoteAgentRuntime()
    await runtime.start(
        {
            "config": config,
            "runtime_context": context.to_mapping(),
            "base_dir": str(ROOT),
        }
    )
    await runtime.stop()

    assert mock_async_client.call_args.kwargs["http2"] is True


def test_remote_agent_descriptor_and_module_entrypoint():
    descriptor = json.loads(
        (ROOT / "adapters/remote-agent/remote-agent.fabric-adapter.json").read_text(
            encoding="utf-8"
        )
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
    assert result.returncode == 0, result.stderr
