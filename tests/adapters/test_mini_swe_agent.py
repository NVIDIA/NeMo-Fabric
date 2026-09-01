# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused contract tests for the mini-SWE-agent adapter."""

from __future__ import annotations

import json
import subprocess
import sys
from contextlib import asynccontextmanager
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from nemo_fabric_adapter_contract.models import AgentConfig
from nemo_fabric_adapter_contract.models import AgentRunRequest
from nemo_fabric_adapter_contract.models import RuntimeContext
from nemo_fabric_adapters.mini_swe_agent import adapter
from nemo_fabric_adapters.mini_swe_agent.agents import RelayRetainingDefaultAgent
from nemo_fabric_adapters.mini_swe_agent.agents import RetainingDefaultAgent
from nemo_fabric_adapters.mini_swe_agent.agents import _submitted_output
from minisweagent.exceptions import Submitted

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def run_agent_inline_fixture(monkeypatch: pytest.MonkeyPatch):
    async def run_inline(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(adapter.asyncio, "to_thread", run_inline)


def invocation(payload: dict) -> tuple[AgentRunRequest, RuntimeContext]:
    request = {
        key: value for key, value in payload["request"].items() if key != "request_id"
    }
    return (
        AgentRunRequest.from_mapping(request),
        RuntimeContext.from_mapping(payload["runtime_context"]),
    )


@pytest.fixture(name="mock_mini")
def mock_mini_fixture(monkeypatch):
    model = MagicMock()
    model.config.model_name = "nvidia/test-model"
    model.format_message.side_effect = lambda **message: message
    environment = MagicMock()
    queries = []

    def query(messages):
        queries.append(messages.copy())
        return {
            "role": "assistant",
            "content": "Working.",
            "extra": {
                "actions": [{"command": "submit", "tool_call_id": "call-1"}],
                "cost": 0.25,
                "response": {
                    "id": "chatcmpl-mini-1",
                    "model": "nvidia/test-model",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "Working."},
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 8,
                        "completion_tokens": 4,
                        "total_tokens": 12,
                    },
                },
            },
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


@pytest.fixture(name="mock_relay")
def mock_relay_fixture(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict:
    import nemo_relay

    calls: dict[str, list] = {
        "plugin_configs": [],
        "request_scopes": [],
        "step_starts": [],
        "step_ends": [],
        "llm_starts": [],
        "llm_ends": [],
        "tool_starts": [],
        "tool_ends": [],
    }
    request_handle = SimpleNamespace(uuid="request-scope")
    step_handle = SimpleNamespace(uuid="step-scope")
    llm_handle = SimpleNamespace(uuid="llm-scope")
    tool_handle = SimpleNamespace(uuid="tool-scope")

    @asynccontextmanager
    async def plugin_context(config):
        calls["plugin_configs"].append(config)
        yield {"diagnostics": []}

    @contextmanager
    def request_scope(name, scope_type, **kwargs):
        calls["request_scopes"].append((name, scope_type, kwargs))
        yield request_handle

    def push_step(name, scope_type, **kwargs):
        calls["step_starts"].append((name, scope_type, kwargs))
        return step_handle

    def pop_step(handle, **kwargs):
        calls["step_ends"].append((handle, kwargs))

    def llm_start(name, request, **kwargs):
        calls["llm_starts"].append((name, request, kwargs))
        return llm_handle

    def llm_end(handle, response, **kwargs):
        calls["llm_ends"].append((handle, response, kwargs))

    def tool_start(name, arguments, **kwargs):
        calls["tool_starts"].append((name, arguments, kwargs))
        return tool_handle

    def tool_end(handle, output, **kwargs):
        calls["tool_ends"].append((handle, output, kwargs))

    monkeypatch.setattr(nemo_relay.plugin, "plugin", plugin_context)
    monkeypatch.setattr(nemo_relay.scope, "scope", request_scope)
    monkeypatch.setattr(nemo_relay.scope, "push", push_step)
    monkeypatch.setattr(nemo_relay.scope, "pop", pop_step)
    monkeypatch.setattr(nemo_relay.scope, "get_handle", lambda: None)
    monkeypatch.setattr(nemo_relay.llm, "call", llm_start)
    monkeypatch.setattr(nemo_relay.llm, "call_end", llm_end)
    monkeypatch.setattr(nemo_relay.tools, "call", tool_start)
    monkeypatch.setattr(nemo_relay.tools, "call_end", tool_end)

    plugin_config = {"version": 1, "components": []}
    artifacts = [{"kind": "atof", "path": str(tmp_path / "events.atof.jsonl")}]
    monkeypatch.setattr(
        adapter.common_utils, "load_relay_plugin_config", lambda _payload: plugin_config
    )
    monkeypatch.setattr(
        adapter.common_utils, "collect_relay_artifacts", lambda _config: artifacts
    )
    calls["plugin_config"] = plugin_config
    calls["artifacts"] = artifacts
    calls["request_handle"] = request_handle
    calls["step_handle"] = step_handle
    return calls


def test_mini_swe_agent_descriptor_is_narrow_and_versioned():
    descriptor = json.loads(
        (ROOT / "adapters/mini-swe-agent/mini-swe-agent.fabric-adapter.json").read_text(
            encoding="utf-8"
        )
    )

    assert descriptor["contract_version"] == "fabric.adapter/v1alpha2"
    assert descriptor["adapter_id"] == "nvidia.fabric.mini-swe-agent"
    assert descriptor["runner"] == {
        "module": "nemo_fabric_adapters.mini_swe_agent.adapter"
    }
    assert descriptor["config"] == {
        "accepts": [
            "models",
            "models.base_url",
            "models.temperature",
            "instructions.system",
            "runtime.max_turns",
        ],
        "system_instruction_modes": ["replace"],
    }
    assert descriptor["capabilities"] == {
        "service": False,
        "streaming": False,
        "updates": False,
        "cancellation": False,
    }
    assert descriptor["telemetry"] == {
        "providers": {
            "relay": {"outputs": ["atif", "otel", "openinference"]},
        }
    }


async def test_runtime_start_rejects_append_system_instruction(mini_payload):
    mini_payload["config"]["instructions"]["system"]["mode"] = "append"
    start = {
        **mini_payload,
        "config": AgentConfig.from_mapping(mini_payload["config"]),
    }

    with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
        await adapter.MiniSweAgentRuntime().start(start)

    assert caught.value.code == "unsupported_system_instruction_mode"
    assert caught.value.metadata["field"] == "instructions.system.mode"


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

    result = await runtime.invoke(*invocation(mini_payload))

    assert runtime._agent.config.system_template == "{{system_instruction}}"
    assert runtime._agent.config.instance_template == adapter.INSTANCE_TEMPLATE
    assert runtime._agent.config.step_limit == 3
    assert mock_mini["queries"][0][0]["content"] == (
        "Work with the literal {{template}}."
    )
    assert "include it in the same command" in mock_mini["queries"][0][1]["content"]
    assert type(runtime._agent) is RetainingDefaultAgent
    assert result.output == {
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

    result = await runtime.invoke(*invocation(mini_payload))

    assert result.status == "failed"
    assert result.error.code == "mini_swe_agent_incomplete"


async def test_mini_swe_agent_retains_history_between_invocations(
    mock_mini, mini_payload
):
    runtime = adapter.MiniSweAgentRuntime()
    start = {**mini_payload, "config": AgentConfig.from_mapping(mini_payload["config"])}
    await runtime.start(start)

    await runtime.invoke(*invocation(mini_payload))
    mini_payload["request"]["input"] = "Continue the task."
    await runtime.invoke(*invocation(mini_payload))

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


async def test_relay_enabled_uses_instrumented_subclass_and_reports_artifacts(
    mock_mini,
    mini_payload,
    mock_relay,
):
    mini_payload["runtime_context"]["telemetry"] = {
        "relay_enabled": True,
        "metadata": {"telemetry_providers": ["relay"]},
    }
    runtime = adapter.MiniSweAgentRuntime()
    start = {**mini_payload, "config": AgentConfig.from_mapping(mini_payload["config"])}
    await runtime.start(start)

    result = await runtime.invoke(*invocation(mini_payload))

    assert type(runtime._agent) is RelayRetainingDefaultAgent
    assert result.status == "succeeded"
    assert result.output["telemetry"] == {
        "enabled": True,
        "provider": "relay",
        "emitter": "mini-swe-agent.subclass/nemo_relay",
    }
    assert result.output["relay_artifacts"] == mock_relay["artifacts"]
    assert mock_relay["plugin_configs"] == [mock_relay["plugin_config"]]

    request_scope = mock_relay["request_scopes"][0]
    assert request_scope[0] == "mini-swe-agent.request"
    assert request_scope[2]["metadata"] == {
        "nemo_fabric_request_id": "mini-request",
        "nemo_fabric_invocation_id": "mini-invocation",
    }
    assert mock_relay["step_starts"][0][2]["handle"] is mock_relay["request_handle"]
    assert mock_relay["llm_starts"][0][2]["handle"] is mock_relay["step_handle"]
    assert mock_relay["llm_starts"][0][1].content["model"] == "nvidia/test-model"
    assert mock_relay["llm_ends"][0][1]["usage"]["cost"] == 0.25
    assert mock_relay["tool_starts"][0][0:2] == (
        "bash",
        {"command": "submit"},
    )
    assert mock_relay["tool_starts"][0][2]["tool_call_id"] == "call-1"
    assert mock_relay["tool_ends"][0][2]["metadata"]["status"] == "submitted"
    assert mock_relay["step_ends"][0][1]["output"] == {
        "status": "interrupted",
        "interrupt_type": "Submitted",
    }


async def test_relay_event_failure_degrades_telemetry_without_failing_agent(
    mock_mini,
    mini_payload,
    mock_relay,
    monkeypatch: pytest.MonkeyPatch,
):
    import nemo_relay

    mini_payload["runtime_context"]["telemetry"] = {"relay_enabled": True}
    monkeypatch.setattr(
        nemo_relay.llm,
        "call_end",
        MagicMock(side_effect=RuntimeError("relay llm end failed")),
    )
    runtime = adapter.MiniSweAgentRuntime()
    start = {**mini_payload, "config": AgentConfig.from_mapping(mini_payload["config"])}
    await runtime.start(start)

    result = await runtime.invoke(*invocation(mini_payload))

    assert result.status == "succeeded"
    assert result.output["output"] == "done"
    assert result.output["telemetry"]["degraded"] is True
    assert "RuntimeError: relay llm end failed" in result.output["telemetry"]["error"]
    assert "telemetry unreliable" not in result.output["telemetry"]["error"]
    assert result.output["relay_artifacts"] == mock_relay["artifacts"]

    second = await runtime.invoke(*invocation(mini_payload))

    assert second.status == "succeeded"
    assert second.output["telemetry"]["degraded"] is True
    assert second.output["relay_artifacts"] == mock_relay["artifacts"]
    assert len(mock_relay["request_scopes"]) == 2


@pytest.mark.parametrize(
    "error",
    [Submitted(), Submitted({"role": "exit"})],
)
def test_submitted_output_handles_missing_content(error):
    assert _submitted_output(error) == {
        "output": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n",
        "returncode": 0,
        "exception_info": "",
    }


async def test_stop_resets_relay_quarantine():
    runtime = adapter.MiniSweAgentRuntime()
    runtime._telemetry_quarantine = "dirty scope"

    await runtime.stop()

    assert runtime._telemetry_quarantine is None


async def test_relay_enabled_requires_optional_dependency(
    mock_mini,
    mini_payload,
    monkeypatch: pytest.MonkeyPatch,
):
    mini_payload["runtime_context"]["telemetry"] = {"relay_enabled": True}
    monkeypatch.setattr(adapter.importlib.util, "find_spec", lambda _name: None)
    runtime = adapter.MiniSweAgentRuntime()
    start = {**mini_payload, "config": AgentConfig.from_mapping(mini_payload["config"])}

    with pytest.raises(
        adapter.lifecycle.LifecycleError,
        match=r"mini-swe-agent\[relay\]",
    ) as exc_info:
        await runtime.start(start)
    assert exc_info.value.code == "mini_swe_agent_relay_missing"


def test_mini_swe_agent_module_entrypoint_exits_cleanly():
    result = subprocess.run(
        [sys.executable, "-m", "nemo_fabric_adapters.mini_swe_agent.adapter"],
        input="",
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
