# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused contract tests for the mini-SWE-agent adapter."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from nemo_fabric_adapter_contract.models import AgentConfig
from nemo_fabric_adapters.mini_swe_agent import adapter
from minisweagent.exceptions import Submitted

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(name="mock_mini")
def mock_mini_fixture(monkeypatch):
    model = MagicMock()
    model.format_message.side_effect = lambda **message: message
    environment = MagicMock()
    queries = []

    def query(messages):
        queries.append(messages.copy())
        return {
            "role": "assistant",
            "content": "Working.",
            "extra": {"actions": [{"command": "submit", "tool_call_id": "call-1"}]},
        }

    model.query.side_effect = query
    model.format_observation_messages.side_effect = lambda _message, outputs, *_: [
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": outputs[0]["output"],
        }
    ]
    environment.execute.side_effect = Submitted(
        {
            "role": "exit",
            "content": "done",
            "extra": {"exit_status": "Submitted", "submission": "done"},
        }
    )
    model_factory = MagicMock(return_value=model)
    environment_factory = MagicMock(return_value=environment)
    monkeypatch.setattr(
        "minisweagent.environments.local.LocalEnvironment", environment_factory
    )
    monkeypatch.setattr("minisweagent.models.litellm_model.LitellmModel", model_factory)
    return {
        "environment_factory": environment_factory,
        "model": model,
        "model_factory": model_factory,
        "queries": queries,
    }


@pytest.fixture(name="mini_payload")
def mini_payload_fixture(tmp_path: Path) -> dict:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return {
        "config": {
            "harness": {"settings": {"timeout": 45}},
            "instructions": {
                "system": {"content": "Work with the literal {{template}}."}
            },
            "models": {
                "default": {
                    "provider": "nvidia",
                    "model": "nvidia/test-model",
                    "api_key_env": "TEST_MINI_API_KEY",
                    "base_url": "https://example.test/v1",
                    "temperature": 0.2,
                }
            },
            "runtime": {"max_turns": 3},
        },
        "runtime_context": {
            "runtime_id": "mini-runtime",
            "invocation_id": "mini-invocation",
            "request_id": "mini-request",
            "environment": {
                "environment_id": "mini-environment",
                "provider": "local",
                "control_location": "in_env_control",
                "workspace": str(workspace),
                "env": {"TEST_MINI_API_KEY": "test-key"},
                "ownership": "caller_owned",
            },
            "artifacts": {},
        },
        "request": {"request_id": "mini-request", "input": "Fix the test."},
    }


def test_mini_swe_agent_descriptor_is_narrow_and_versioned():
    descriptor = json.loads(
        (ROOT / "adapters/mini-swe-agent/fabric-adapter.json").read_text(
            encoding="utf-8"
        )
    )

    assert descriptor["contract_version"] == "fabric.adapter/v1alpha2"
    assert descriptor["adapter_id"] == "nvidia.fabric.mini-swe-agent"
    assert descriptor["harness"] == "mini-swe-agent"
    assert descriptor["runner"] == {
        "module": "nemo_fabric_adapters.mini_swe_agent.adapter"
    }
    assert descriptor["config"] == {
        "input": "agent_config",
        "accepts": [
            "models",
            "models.base_url",
            "models.temperature",
            "instructions.system",
            "runtime.max_turns",
        ],
    }
    assert descriptor["capabilities"] == {
        "service": False,
        "streaming": False,
        "updates": False,
        "cancellation": False,
    }


async def test_mini_swe_agent_maps_config_and_returns_normalized_output(
    mock_mini, mini_payload, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("TEST_MINI_API_KEY", "test-key")
    runtime = adapter.MiniSweAgentRuntime()
    start = {**mini_payload, "config": AgentConfig.from_mapping(mini_payload["config"])}
    await runtime.start(start)

    mock_mini["model_factory"].assert_called_once_with(
        model_name="nvidia/test-model",
        model_kwargs={
            "api_key": "test-key",
            "api_base": "https://example.test/v1",
            "temperature": 0.2,
        },
    )
    mock_mini["environment_factory"].assert_called_once_with(timeout=45)

    result = await runtime.invoke(mini_payload)

    assert runtime._agent.config.system_template == "{{system_instruction}}"
    assert runtime._agent.config.instance_template == adapter.INSTANCE_TEMPLATE
    assert runtime._agent.config.step_limit == 3
    assert mock_mini["queries"][0][0]["content"] == (
        "Work with the literal {{template}}."
    )
    assert result == {
        "failed": False,
        "output": "done",
        "usage": {"api_calls": 1},
    }
    await runtime.stop()


async def test_mini_swe_agent_reports_an_incomplete_loop_as_failed(
    mock_mini, mini_payload
):
    mock_mini["model"].query.side_effect = lambda _: {
        "role": "assistant",
        "extra": {"actions": []},
    }
    mock_mini["model"].format_observation_messages.side_effect = lambda *_: [
        {
            "role": "exit",
            "extra": {"exit_status": "LimitsExceeded", "submission": ""},
        }
    ]
    runtime = adapter.MiniSweAgentRuntime()
    start = {**mini_payload, "config": AgentConfig.from_mapping(mini_payload["config"])}
    await runtime.start(start)

    result = await runtime.invoke(mini_payload)

    assert result["failed"] is True
    assert result["error"]["code"] == "mini_swe_agent_incomplete"


async def test_mini_swe_agent_retains_history_between_invocations(
    mock_mini, mini_payload
):
    runtime = adapter.MiniSweAgentRuntime()
    start = {**mini_payload, "config": AgentConfig.from_mapping(mini_payload["config"])}
    await runtime.start(start)

    await runtime.invoke(mini_payload)
    mini_payload["request"]["input"] = "Continue the task."
    await runtime.invoke(mini_payload)

    agent = runtime._agent
    assert mock_mini["model"].query.call_count == 2
    assert [message["role"] for message in mock_mini["queries"][1]] == [
        "system",
        "user",
        "assistant",
        "tool",
        "user",
    ]
    assert [message["role"] for message in agent.messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "user",
        "assistant",
        "tool",
        "exit",
    ]
    assert agent.messages[4]["content"] == "Continue the task."


def test_mini_swe_agent_module_entrypoint_exits_cleanly():
    result = subprocess.run(
        [sys.executable, "-m", "nemo_fabric_adapters.mini_swe_agent.adapter"],
        input="",
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
