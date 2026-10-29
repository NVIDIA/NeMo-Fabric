# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused contract tests for the OpenHands adapter."""

from __future__ import annotations

import asyncio
import json
import os
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import PropertyMock

import pytest
from nemo_fabric_adapter_contract.models import AgentConfig
from nemo_fabric_adapter_contract.models import AgentRunRequest
from nemo_fabric_adapter_contract.models import AgentRunStatus
from nemo_fabric_adapter_contract.models import RuntimeContext
from nemo_fabric_adapters.openhands import adapter

ROOT = Path(__file__).resolve().parents[2]


class MockStatus(str, Enum):
    FINISHED = "finished"
    ERROR = "error"


@pytest.fixture(name="openhands_payload")
def openhands_payload_fixture(tmp_path: Path) -> dict:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    skill = tmp_path / "skills" / "review"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: review\ndescription: Review code.\n---\nReview carefully.\n",
        encoding="utf-8",
    )
    return {
        "base_dir": str(tmp_path),
        "config": {
            "models": {
                "default": {
                    "provider": "nvidia",
                    "model": "nvidia/test-model",
                    "api_key_env": "TEST_OPENHANDS_API_KEY",
                    "base_url": "https://example.test/v1",
                    "temperature": 0.2,
                    "top_p": 0.8,
                    "max_tokens": 256,
                }
            },
            "instructions": {
                "system": {"content": "Review carefully.", "mode": "append"}
            },
            "runtime": {"max_turns": 7},
            "tools": {"enabled": ["bash", "edit"], "blocked": []},
            "skills": {"paths": ["skills/review"]},
            "mcp": {
                "servers": {
                    "local": {
                        "transport": "stdio",
                        "url": "python",
                        "args": ["server.py"],
                        "env": {"MODE": "test"},
                    },
                    "remote": {
                        "transport": "streamable-http",
                        "url": "https://mcp.example.test",
                        "custom_headers": {},
                    },
                }
            },
        },
        "runtime_context": {
            "runtime_id": "openhands-runtime",
            "invocation_id": "openhands-invocation",
            "request_id": "openhands-request",
            "environment": {
                "environment_id": "openhands-environment",
                "provider": "local",
                "control_location": "in_env_control",
                "workspace": "workspace",
                "ownership": "caller_owned",
            },
            "artifacts": {},
        },
    }


@pytest.fixture(name="mock_openhands")
def mock_openhands_fixture(monkeypatch: pytest.MonkeyPatch) -> dict:
    calls: dict[str, object] = {}
    usage = SimpleNamespace(prompt_tokens=11, completion_tokens=5)
    metrics_delta = SimpleNamespace(
        accumulated_token_usage=usage,
        accumulated_cost=0.25,
        token_usages=[usage],
        costs=[SimpleNamespace(cost=0.25)],
        response_latencies=[SimpleNamespace(response_id="response-test")],
    )
    metrics = MagicMock()
    metrics.deep_copy.return_value = object()
    metrics.diff.return_value = metrics_delta
    llm = SimpleNamespace(metrics=metrics)

    llm_factory = MagicMock(return_value=llm)
    tool_factory = MagicMock(side_effect=lambda **kwargs: kwargs)
    agent_context_factory = MagicMock(side_effect=lambda **kwargs: kwargs)
    agent_factory = MagicMock(side_effect=lambda **kwargs: kwargs)
    mcp_server_factory = MagicMock(side_effect=lambda **kwargs: kwargs)
    skill = SimpleNamespace(name="review")
    skill_type = MagicMock()
    skill_type.load.return_value = skill

    state = SimpleNamespace(events=[], execution_status=MockStatus.FINISHED)
    conversation = SimpleNamespace(state=state)
    replies = iter(["first response", "second response"])

    async def arun():
        state.events.append(next(replies))
        state.execution_status = MockStatus.FINISHED

    conversation.send_message = MagicMock()
    conversation.arun = AsyncMock(side_effect=arun)
    conversation.close = MagicMock()
    conversation_factory = MagicMock(return_value=conversation)
    default_condenser = MagicMock()

    api = adapter.OpenHandsApi(
        Agent=agent_factory,
        AgentContext=agent_context_factory,
        Conversation=conversation_factory,
        ConversationExecutionStatus=MockStatus,
        LLM=llm_factory,
        MCPServer=mcp_server_factory,
        SecretStr=MagicMock(side_effect=lambda value: f"secret:{value}"),
        Skill=skill_type,
        Tool=tool_factory,
        TerminalTool=SimpleNamespace(name="TerminalTool"),
        FileEditorTool=SimpleNamespace(name="FileEditorTool"),
        get_agent_final_response=lambda events: events[-1] if events else "",
        default_condenser=default_condenser,
    )
    monkeypatch.setattr(adapter, "_load_openhands_api", lambda: api)
    calls.update(
        {
            "agent_context_factory": agent_context_factory,
            "agent_factory": agent_factory,
            "conversation": conversation,
            "conversation_factory": conversation_factory,
            "default_condenser": default_condenser,
            "llm_factory": llm_factory,
            "mcp_server_factory": mcp_server_factory,
            "skill_type": skill_type,
            "tool_factory": tool_factory,
        }
    )
    return calls


def _start_payload(payload: dict) -> dict:
    return {**payload, "config": AgentConfig.from_mapping(payload["config"])}


def _invocation(payload: dict, text: str) -> tuple[AgentRunRequest, RuntimeContext]:
    return (
        AgentRunRequest(input=text),
        RuntimeContext.from_mapping(payload["runtime_context"]),
    )


def test_descriptor_declares_only_initial_openhands_surface():
    descriptor = json.loads(
        (ROOT / "adapters/python/openhands/openhands.fabric-adapter.json").read_text(
            encoding="utf-8"
        )
    )

    assert descriptor["contract_version"] == "fabric.adapter/v1alpha2"
    assert descriptor["adapter_id"] == "nvidia.fabric.openhands"
    assert descriptor["adapter_kind"] == "python"
    assert descriptor["runner"] == {"module": "nemo_fabric_adapters.openhands.adapter"}
    assert descriptor["config"] == {
        "accepts": [
            "models",
            "models.base_url",
            "models.temperature",
            "models.top_p",
            "models.max_tokens",
            "instructions.system",
            "runtime.max_turns",
            "tools.enabled",
            "tools.blocked",
            "mcp",
            "skills",
        ],
        "system_instruction_modes": ["replace", "append"],
    }
    assert descriptor["capabilities"] == {
        "service": False,
        "streaming": False,
        "updates": False,
        "cancellation": False,
        "metadata": {"health_checks": ["live", "active"]},
    }
    assert descriptor["settings_schema"]["properties"] == {
        "condenser": {
            "type": "string",
            "enum": ["default", "none"],
            "default": "default",
            "description": (
                "OpenHands conversation history condenser. Use none to keep "
                "the full history."
            ),
        }
    }
    assert descriptor["settings_schema"]["additionalProperties"] is False
    assert descriptor["extension_schemas"]["run_error"]["properties"] == {
        "sdk_error": {"type": "string"}
    }
    assert "telemetry" not in descriptor


def test_missing_openhands_parent_package_is_reported(
    monkeypatch: pytest.MonkeyPatch,
):
    def missing(_name: str):
        raise ModuleNotFoundError("No module named 'openhands'")

    monkeypatch.setattr(adapter.importlib.util, "find_spec", missing)

    with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
        adapter._load_openhands_api()

    assert caught.value.code == "openhands_harness_unavailable"


async def test_start_maps_normalized_configuration(
    openhands_payload: dict,
    mock_openhands: dict,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("TEST_OPENHANDS_API_KEY", "test-key")
    runtime = adapter.OpenHandsRuntime()
    await runtime.start(_start_payload(openhands_payload))

    mock_openhands["llm_factory"].assert_called_once_with(
        model="nvidia_nim/nvidia/test-model",
        usage_id="agent",
        api_key="secret:test-key",
        base_url="https://example.test/v1",
        temperature=0.2,
        top_p=0.8,
        max_output_tokens=256,
    )
    context_kwargs = mock_openhands["agent_context_factory"].call_args.kwargs
    assert context_kwargs["skills"] == [SimpleNamespace(name="review")]
    assert context_kwargs["system_message_suffix"] == "Review carefully."
    assert context_kwargs["load_user_skills"] is False
    assert context_kwargs["load_public_skills"] is False
    assert context_kwargs["load_project_skills"] is False
    assert context_kwargs["load_memory"] is False
    mock_openhands["skill_type"].load.assert_called_once_with(
        Path(openhands_payload["base_dir"]) / "skills/review/SKILL.md",
        strict=True,
        skip_mcp=True,
    )

    agent_kwargs = mock_openhands["agent_factory"].call_args.kwargs
    assert "system_prompt" not in agent_kwargs
    assert agent_kwargs["tools"] == [
        {"name": "TerminalTool"},
        {"name": "FileEditorTool"},
    ]
    assert agent_kwargs["mcp_config"] == {
        "local": {
            "transport": "stdio",
            "command": "python",
            "args": ["server.py"],
            "env": {"MODE": "test"},
        },
        "remote": {
            "transport": "streamable-http",
            "url": "https://mcp.example.test",
            "headers": None,
        },
    }
    mock_openhands["conversation_factory"].assert_called_once_with(
        agent=agent_kwargs,
        workspace=str(Path(openhands_payload["base_dir"]) / "workspace"),
        max_iteration_per_run=7,
        persistence_dir=None,
        profile_store_dir=ANY,
        delete_on_close=True,
        visualizer=None,
    )
    profile_store = Path(
        mock_openhands["conversation_factory"].call_args.kwargs["profile_store_dir"]
    )
    assert profile_store.is_dir()


async def test_start_condenses_history_with_the_agent_llm(
    openhands_payload: dict,
    mock_openhands: dict,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("TEST_OPENHANDS_API_KEY", "test-key")
    await adapter.OpenHandsRuntime().start(_start_payload(openhands_payload))

    llm = mock_openhands["llm_factory"].return_value
    default_condenser = mock_openhands["default_condenser"]
    default_condenser.assert_called_once_with(llm)
    agent_kwargs = mock_openhands["agent_factory"].call_args.kwargs
    assert agent_kwargs["condenser"] is default_condenser.return_value


async def test_start_without_condenser_when_disabled(
    openhands_payload: dict,
    mock_openhands: dict,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("TEST_OPENHANDS_API_KEY", "test-key")
    openhands_payload["config"]["harness"] = {"settings": {"condenser": "none"}}
    await adapter.OpenHandsRuntime().start(_start_payload(openhands_payload))

    mock_openhands["default_condenser"].assert_not_called()
    agent_kwargs = mock_openhands["agent_factory"].call_args.kwargs
    assert agent_kwargs.get("condenser") is None


@pytest.mark.parametrize(
    "condenser", ["off", "None", None], ids=["off", "string-None", "null"]
)
async def test_start_rejects_unknown_condenser_setting(
    openhands_payload: dict,
    mock_openhands: dict,
    monkeypatch: pytest.MonkeyPatch,
    condenser: str | None,
):
    monkeypatch.setenv("TEST_OPENHANDS_API_KEY", "test-key")
    openhands_payload["config"]["harness"] = {"settings": {"condenser": condenser}}

    with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
        await adapter.OpenHandsRuntime().start(_start_payload(openhands_payload))

    assert caught.value.code == "openhands_condenser_invalid"
    mock_openhands["default_condenser"].assert_not_called()
    mock_openhands["agent_factory"].assert_not_called()


# Event limit: the system prompt, the user message, and 40 think action and
# observation pairs exceed the 80-event limit, so the condenser runs before the
# 41st agent call. Context window error: the rejected 11th call triggers the
# condensation, and the 12th call finishes.
@pytest.mark.parametrize(
    ("context_error_at_call", "expected_agent_calls"),
    [(None, 41), (11, 12)],
    ids=["event-limit", "context-window-error"],
)
async def test_invoke_condenses_history_and_counts_summary_usage(
    openhands_payload: dict,
    monkeypatch: pytest.MonkeyPatch,
    context_error_at_call: int | None,
    expected_agent_calls: int,
):
    pytest.importorskip("openhands.sdk")
    pytest.importorskip("openhands.tools")
    from litellm.exceptions import ContextWindowExceededError
    from litellm.types.utils import ModelResponse
    from openhands.sdk.event import Condensation
    from openhands.sdk.llm import llm as openhands_llm

    calls = {"agent": 0, "summary": 0}

    def respond(**kwargs) -> ModelResponse:
        # Agent steps send tools; the condenser's summary request does not.
        if kwargs.get("tools"):
            calls["agent"] += 1
            if calls["agent"] == context_error_at_call:
                raise ContextWindowExceededError(
                    message="context length exceeded",
                    model="test-model",
                    llm_provider="openai",
                )
            tool, arguments = (
                # Distinct thoughts keep the SDK stuck detector quiet.
                ("think", {"thought": f"Step {calls['agent']}."})
                if calls["summary"] == 0
                else ("finish", {"message": "done"})
            )
            message = {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": f"call-{calls['agent']}",
                        "type": "function",
                        "function": {"name": tool, "arguments": json.dumps(arguments)},
                    }
                ],
            }
            prompt_tokens, completion_tokens = 10, 1
        else:
            calls["summary"] += 1
            message = {"role": "assistant", "content": "Summary of earlier steps."}
            prompt_tokens, completion_tokens = 1000, 100
        return ModelResponse(
            model="test-model",
            choices=[{"index": 0, "finish_reason": "stop", "message": message}],
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        )

    monkeypatch.setattr(
        openhands_llm, "litellm_completion", MagicMock(side_effect=respond)
    )
    monkeypatch.setattr(
        openhands_llm, "litellm_acompletion", AsyncMock(side_effect=respond)
    )
    monkeypatch.setenv("TEST_OPENHANDS_API_KEY", "test-key")
    config = openhands_payload["config"]
    config["tools"] = {"enabled": [], "blocked": []}
    config["runtime"] = {"max_turns": 100}
    del config["skills"], config["mcp"]
    runtime = adapter.OpenHandsRuntime()
    await runtime.start(_start_payload(openhands_payload))
    try:
        result = await runtime.invoke(
            *_invocation(openhands_payload, "Think until the history condenses.")
        )
        events = list(runtime._conversation.state.events)
    finally:
        await runtime.stop()

    assert result.status is AgentRunStatus.SUCCEEDED
    assert result.output == {"response": "done"}
    assert calls["summary"] == 1
    assert calls["agent"] == expected_agent_calls
    assert sum(isinstance(event, Condensation) for event in events) == 1
    # A rejected request reports no usage; every other call counts once.
    answered_calls = calls["agent"] - (context_error_at_call is not None)
    assert result.usage.input_tokens == 10 * answered_calls + 1000
    assert result.usage.output_tokens == answered_calls + 100


async def test_replace_instruction_uses_exact_system_prompt(
    openhands_payload: dict,
    mock_openhands: dict,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("TEST_OPENHANDS_API_KEY", "test-key")
    openhands_payload["config"]["instructions"]["system"]["mode"] = "replace"
    runtime = adapter.OpenHandsRuntime()
    await runtime.start(_start_payload(openhands_payload))

    assert mock_openhands["agent_factory"].call_args.kwargs["system_prompt"] == (
        "Review carefully."
    )
    assert (
        mock_openhands["agent_context_factory"].call_args.kwargs[
            "system_message_suffix"
        ]
        is None
    )


async def test_start_prefixes_an_unqualified_model_with_its_provider(
    openhands_payload: dict,
    mock_openhands: dict,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("TEST_OPENHANDS_API_KEY", "test-key")
    openhands_payload["config"]["models"]["default"]["model"] = "test-model"

    await adapter.OpenHandsRuntime().start(_start_payload(openhands_payload))

    assert mock_openhands["llm_factory"].call_args.kwargs["model"] == (
        "nvidia_nim/test-model"
    )


@pytest.mark.parametrize(
    "model_name",
    ["nvidia/test-model", "nvidia_nim/nvidia/test-model"],
)
@pytest.mark.parametrize("with_tools", [False, True])
def test_nvidia_custom_endpoint_preserves_served_model_id(
    model_name: str,
    with_tools: bool,
    monkeypatch: pytest.MonkeyPatch,
):
    pytest.importorskip("openhands.sdk")
    pytest.importorskip("openhands.tools")
    import httpx
    import litellm
    from litellm.caching.caching import InMemoryCache
    from openhands.sdk.llm import Message
    from openhands.sdk.llm import TextContent
    from openhands.sdk.tool import FinishTool

    requests = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-nim",
                "object": "chat.completion",
                "created": 0,
                "model": "nvidia/test-model",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "done"},
                    }
                ],
                "usage": {
                    "prompt_tokens": 11,
                    "prompt_tokens_details": {"cached_tokens": 3},
                    "completion_tokens": 5,
                    "total_tokens": 16,
                },
            },
        )

    model = AgentConfig.from_mapping(
        {
            "models": {
                "default": {
                    "provider": "nvidia",
                    "model": model_name,
                    "base_url": "http://nim.example.test/v1",
                    "api_key_env": "TEST_OPENHANDS_API_KEY",
                }
            }
        }
    ).models["default"]
    monkeypatch.setenv("TEST_OPENHANDS_API_KEY", "unused")
    # LiteLLM caches OpenAI clients across calls, including their HTTP transport.
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", InMemoryCache())
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr(litellm, "client_session", client)
        llm = adapter._llm(adapter._load_openhands_api(), model)
        llm.num_retries = 0
        baseline = llm.metrics.deep_copy()
        llm.completion(
            [Message(role="user", content=[TextContent(text="finish")])],
            tools=FinishTool.create() if with_tools else None,
        )

    assert len(requests) == 1
    assert str(requests[0].url) == "http://nim.example.test/v1/chat/completions"
    payload = json.loads(requests[0].content)
    assert payload["model"] == "nvidia/test-model"
    if with_tools:
        assert payload["tools"][0]["function"]["name"] == "finish"
    assert llm.metrics.accumulated_token_usage.prompt_tokens == 11
    assert llm.metrics.accumulated_token_usage.completion_tokens == 5
    assert llm.metrics.accumulated_token_usage.cache_read_tokens == 3
    usage = adapter._usage(llm, baseline)
    assert usage.input_tokens == 11
    assert usage.input_tokens_include_cache is True
    assert usage.output_tokens == 5
    assert usage.total_tokens == 16
    assert usage.cached_input_tokens is None
    assert usage.cost_usd is None


async def test_repeated_invocation_returns_only_current_turn_and_usage(
    openhands_payload: dict,
    mock_openhands: dict,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("TEST_OPENHANDS_API_KEY", "test-key")
    runtime = adapter.OpenHandsRuntime()
    await runtime.start(_start_payload(openhands_payload))

    first = await runtime.invoke(*_invocation(openhands_payload, "first"))
    second = await runtime.invoke(*_invocation(openhands_payload, "second"))

    assert first.status is AgentRunStatus.SUCCEEDED
    assert first.output == {"response": "first response"}
    assert second.output == {"response": "second response"}
    assert second.usage is not None
    assert second.usage.to_mapping() == {
        "input_tokens": 11,
        "input_tokens_include_cache": True,
        "output_tokens": 5,
        "total_tokens": 16,
        "cost_usd": 0.25,
    }
    assert mock_openhands["conversation"].send_message.call_args_list[0].args == (
        "first",
    )
    assert mock_openhands["conversation"].send_message.call_args_list[1].args == (
        "second",
    )


async def test_nonfinished_conversation_returns_failed_result(
    openhands_payload: dict,
    mock_openhands: dict,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("TEST_OPENHANDS_API_KEY", "test-key")
    runtime = adapter.OpenHandsRuntime()
    await runtime.start(_start_payload(openhands_payload))
    conversation = mock_openhands["conversation"]

    async def fail():
        conversation.state.execution_status = MockStatus.ERROR

    conversation.arun.side_effect = fail
    result = await runtime.invoke(*_invocation(openhands_payload, "fail"))

    assert result.status is AgentRunStatus.FAILED
    assert result.error is not None
    assert result.error.code == "openhands_error"


async def test_send_message_failure_returns_failed_result(
    openhands_payload: dict,
    mock_openhands: dict,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("TEST_OPENHANDS_API_KEY", "test-key")
    runtime = adapter.OpenHandsRuntime()
    await runtime.start(_start_payload(openhands_payload))
    conversation = mock_openhands["conversation"]
    conversation.send_message.side_effect = RuntimeError("failed")

    result = await runtime.invoke(*_invocation(openhands_payload, "fail"))

    assert result.status is AgentRunStatus.FAILED
    assert result.error is not None
    assert result.error.code == "openhands_invocation_failed"
    assert result.error.message == "OpenHands could not complete the invocation."
    assert result.error.retryable is False
    assert result.error.extensions == {"sdk_error": "RuntimeError"}
    conversation.arun.assert_not_awaited()


@pytest.mark.parametrize(
    ("cause", "sdk_error"),
    [
        (ConnectionError("model endpoint unavailable"), "ConnectionError"),
        (None, "RuntimeError"),
    ],
)
async def test_invocation_failure_reports_the_underlying_sdk_error(
    openhands_payload: dict,
    mock_openhands: dict,
    monkeypatch: pytest.MonkeyPatch,
    cause: BaseException | None,
    sdk_error: str,
):
    monkeypatch.setenv("TEST_OPENHANDS_API_KEY", "test-key")
    runtime = adapter.OpenHandsRuntime()
    await runtime.start(_start_payload(openhands_payload))
    try:
        error = RuntimeError("Conversation run failed")
        error.__cause__ = cause
        mock_openhands["conversation"].arun.side_effect = error

        result = await runtime.invoke(*_invocation(openhands_payload, "fail"))

        assert result.status is AgentRunStatus.FAILED
        assert result.error is not None
        assert result.error.code == "openhands_invocation_failed"
        assert result.error.message == "OpenHands could not complete the invocation."
        assert result.error.retryable is False
        assert result.error.extensions == {"sdk_error": sdk_error}
        assert result.usage.input_tokens == 11
        assert result.usage.output_tokens == 5
        assert result.usage.cost_usd == 0.25
    finally:
        await runtime.stop()


async def test_invoke_rejects_a_different_runtime(
    openhands_payload: dict,
    mock_openhands: dict,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("TEST_OPENHANDS_API_KEY", "test-key")
    runtime = adapter.OpenHandsRuntime()
    await runtime.start(_start_payload(openhands_payload))
    request, context = _invocation(openhands_payload, "review")
    context = RuntimeContext.from_mapping(
        {**context.to_mapping(), "runtime_id": "another-runtime"}
    )

    with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
        await runtime.invoke(request, context)

    assert caught.value.code == "openhands_runtime_mismatch"
    mock_openhands["conversation"].send_message.assert_not_called()


async def test_stop_closes_conversation_idempotently(
    openhands_payload: dict,
    mock_openhands: dict,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("TEST_OPENHANDS_API_KEY", "test-key")
    runtime = adapter.OpenHandsRuntime()
    await runtime.start(_start_payload(openhands_payload))
    profile_store = Path(
        mock_openhands["conversation_factory"].call_args.kwargs["profile_store_dir"]
    )

    await runtime.stop()
    await runtime.stop()

    mock_openhands["conversation"].close.assert_called_once_with()
    assert not profile_store.exists()


async def test_start_cleans_profile_store_after_conversation_failure(
    openhands_payload: dict,
    mock_openhands: dict,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    monkeypatch.setenv("TEST_OPENHANDS_API_KEY", "test-key")
    profile_store = tmp_path / "profiles"
    profile_store.mkdir()
    temporary_directory = SimpleNamespace(name=str(profile_store), cleanup=MagicMock())
    monkeypatch.setattr(
        adapter.tempfile,
        "TemporaryDirectory",
        MagicMock(return_value=temporary_directory),
    )
    mock_openhands["conversation_factory"].side_effect = RuntimeError("failed")

    with pytest.raises(RuntimeError, match="failed"):
        await adapter.OpenHandsRuntime().start(_start_payload(openhands_payload))

    temporary_directory.cleanup.assert_called_once_with()


@pytest.mark.parametrize("field", ["allowed_tools", "blocked_tools"])
async def test_start_rejects_unenforced_mcp_tool_policy(
    openhands_payload: dict,
    mock_openhands: dict,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
):
    monkeypatch.setenv("TEST_OPENHANDS_API_KEY", "test-key")
    server = openhands_payload["config"]["mcp"]["servers"]["local"]
    server[field] = ["read"]

    with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
        await adapter.OpenHandsRuntime().start(_start_payload(openhands_payload))

    assert caught.value.code == "openhands_mcp_tool_policy_unsupported"


@pytest.mark.parametrize(
    ("server_name", "field", "value"),
    [
        ("local", "custom_headers", {"X-Test": "value"}),
        ("remote", "args", ["--verbose"]),
        ("remote", "extensions", {"example.test/option": True}),
    ],
)
async def test_start_rejects_unapplied_mcp_server_fields(
    openhands_payload: dict,
    mock_openhands: dict,
    monkeypatch: pytest.MonkeyPatch,
    server_name: str,
    field: str,
    value: object,
):
    monkeypatch.setenv("TEST_OPENHANDS_API_KEY", "test-key")
    server = openhands_payload["config"]["mcp"]["servers"][server_name]
    server[field] = value

    with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
        await adapter.OpenHandsRuntime().start(_start_payload(openhands_payload))

    assert caught.value.code == "openhands_mcp_configuration_unsupported"


async def test_start_rejects_unapplied_mcp_extensions(
    openhands_payload: dict,
    mock_openhands: dict,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("TEST_OPENHANDS_API_KEY", "test-key")
    openhands_payload["config"]["mcp"]["extensions"] = {"example.test/option": True}

    with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
        await adapter.OpenHandsRuntime().start(_start_payload(openhands_payload))

    assert caught.value.code == "openhands_mcp_configuration_unsupported"


async def test_start_rejects_missing_named_credential(
    openhands_payload: dict,
    mock_openhands: dict,
):
    with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
        await adapter.OpenHandsRuntime().start(_start_payload(openhands_payload))

    assert caught.value.code == "openhands_credential_missing"
    assert "TEST_OPENHANDS_API_KEY" in str(caught.value)


async def test_start_rejects_unknown_tool(
    openhands_payload: dict,
    mock_openhands: dict,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("TEST_OPENHANDS_API_KEY", "test-key")
    openhands_payload["config"]["tools"]["enabled"] = ["read"]

    with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
        await adapter.OpenHandsRuntime().start(_start_payload(openhands_payload))

    assert caught.value.code == "openhands_tool_unsupported"


async def test_blocked_tool_alias_blocks_canonical_default(
    openhands_payload: dict,
    mock_openhands: dict,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("TEST_OPENHANDS_API_KEY", "test-key")
    openhands_payload["config"]["tools"] = {"blocked": ["bash"]}

    await adapter.OpenHandsRuntime().start(_start_payload(openhands_payload))

    assert mock_openhands["agent_factory"].call_args.kwargs["tools"] == [
        {"name": "FileEditorTool"}
    ]


async def test_start_rejects_empty_mcp_target(
    openhands_payload: dict,
    mock_openhands: dict,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("TEST_OPENHANDS_API_KEY", "test-key")
    openhands_payload["config"]["mcp"]["servers"]["local"]["url"] = (
        "$EMPTY_OPENHANDS_MCP_TARGET"
    )
    monkeypatch.setenv("EMPTY_OPENHANDS_MCP_TARGET", "")

    with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
        await adapter.OpenHandsRuntime().start(_start_payload(openhands_payload))

    assert caught.value.code == "openhands_mcp_target_invalid"


@pytest.mark.parametrize(
    ("level", "initialized", "outcome", "reason"),
    [
        ("live", False, "passed", None),
        ("active", True, "passed", None),
        ("active", False, "unknown", "capabilities_unverified"),
        ("ready", True, "unsupported", "unsupported_level"),
        ("operational", True, "unsupported", "unsupported_level"),
    ],
)
async def test_health_uses_native_conversation_lifecycle(
    level, initialized, outcome, reason
):
    runtime = adapter.OpenHandsRuntime()
    conversation = MagicMock()
    conversation._cleanup_initiated = False
    conversation._cleanup_complete = False
    conversation._agent_ready = initialized
    conversation._arun_task = None
    runtime._conversation = conversation
    runtime._runtime_id = "runtime-health"
    result = await runtime.check(level)
    assert (result["outcome"], result["reason_code"]) == (outcome, reason)
    if outcome == "passed":
        assert result["observed_at"] is not None
        assert result["activity"] == "idle"
    conversation.send_message.assert_not_called()
    conversation.arun.assert_not_called()
    conversation.close.assert_not_called()


@pytest.mark.parametrize("flag", ["_cleanup_initiated", "_cleanup_complete"])
@pytest.mark.parametrize("level", ["live", "active"])
async def test_health_detects_native_closure(flag, level):
    runtime = adapter.OpenHandsRuntime()
    conversation = MagicMock()
    conversation._cleanup_initiated = False
    conversation._cleanup_complete = False
    conversation._agent_ready = True
    conversation._arun_task = None
    setattr(conversation, flag, True)
    runtime._conversation = conversation
    runtime._runtime_id = "runtime-health"
    result = await runtime.check(level)
    assert (result["outcome"], result["reason_code"]) == ("failed", "session_closed")
    assert result["activity"] == "stopped"


async def test_health_rejects_unqualified_native_shape_and_missing_runtime():
    runtime = adapter.OpenHandsRuntime()
    assert (await runtime.check("live"))["reason_code"] == "runtime_unavailable"
    runtime._runtime_id = "runtime-health"
    runtime._conversation = SimpleNamespace()
    result = await runtime.check("live")
    assert (result["outcome"], result["reason_code"], result["observed_at"]) == (
        "unknown",
        "capabilities_unverified",
        None,
    )


async def test_health_remains_available_during_native_arun(
    openhands_payload, mock_openhands
):
    os.environ["TEST_OPENHANDS_API_KEY"] = "test-key"
    runtime = adapter.OpenHandsRuntime()
    await runtime.start(_start_payload(openhands_payload))
    conversation = mock_openhands["conversation"]
    conversation._cleanup_initiated = False
    conversation._cleanup_complete = False
    conversation._agent_ready = True
    entered, release = asyncio.Event(), asyncio.Event()
    original_arun = conversation.arun.side_effect

    async def arun():
        conversation._arun_task = asyncio.current_task()
        entered.set()
        await release.wait()
        await original_arun()

    conversation.arun.side_effect = arun
    task = asyncio.create_task(
        runtime.invoke(*_invocation(openhands_payload, "private prompt"))
    )
    try:
        await asyncio.wait_for(entered.wait(), 5)
        for level in ("live", "active"):
            result = await asyncio.wait_for(runtime.check(level), 0.1)
            assert (result["outcome"], result["activity"]) == ("passed", "busy")
        assert not task.done()
        conversation.close.assert_not_called()
        release.set()
        assert (await task).status is AgentRunStatus.SUCCEEDED
    finally:
        release.set()
        await task
        await runtime.stop()


@pytest.mark.parametrize("raises", [False, True])
async def test_health_discards_observation_when_native_read_changes_ownership(raises):
    runtime = adapter.OpenHandsRuntime()
    conversation = MagicMock()
    runtime._conversation = conversation
    runtime._runtime_id = "runtime-health"

    def changed():
        runtime._generation += 1
        if raises:
            raise RuntimeError("private native error")
        return False

    type(conversation)._cleanup_initiated = PropertyMock(side_effect=changed)
    conversation._cleanup_complete = False
    conversation._agent_ready = True
    conversation._arun_task = None
    result = await runtime.check("live")
    assert (result["outcome"], result["reason_code"], result["observed_at"]) == (
        "unknown",
        "observation_changed",
        None,
    )
    assert "private" not in json.dumps(result)


@pytest.mark.parametrize("response_without_usage", [False, True])
def test_native_metrics_defaults_do_not_establish_zero_usage(response_without_usage):
    pytest.importorskip("openhands.sdk")
    from openhands.sdk.llm.utils.metrics import Metrics

    metrics = Metrics(model_name="test-model")
    baseline = metrics.deep_copy()
    if response_without_usage:
        metrics.add_response_latency(0.1, "response-test")
    assert adapter._usage(SimpleNamespace(metrics=metrics), baseline) is None


@pytest.mark.parametrize("tokens", [(11, 5), (0, 0)])
@pytest.mark.parametrize("cost", [None, 0.0, 0.25])
def test_native_metrics_preserve_recorded_usage_and_only_recorded_cost(tokens, cost):
    pytest.importorskip("openhands.sdk")
    from openhands.sdk.llm.utils.metrics import Metrics

    metrics = Metrics(model_name="test-model")
    baseline = metrics.deep_copy()
    metrics.add_response_latency(0.1, "response-test")
    metrics.add_token_usage(
        prompt_tokens=tokens[0],
        completion_tokens=tokens[1],
        cache_read_tokens=0,
        cache_write_tokens=0,
        context_window=0,
        response_id="response-test",
    )
    if cost is not None:
        metrics.add_cost(cost)
    usage = adapter._usage(SimpleNamespace(metrics=metrics), baseline)
    assert usage.input_tokens == tokens[0]
    assert usage.output_tokens == tokens[1]
    assert usage.total_tokens == sum(tokens)
    assert usage.cost_usd == cost
    # An unchanged warm conversation supplies no new per-call evidence.
    assert adapter._usage(SimpleNamespace(metrics=metrics), metrics.deep_copy()) is None


def test_native_metrics_do_not_report_partially_priced_cost():
    pytest.importorskip("openhands.sdk")
    from openhands.sdk.llm.utils.metrics import Metrics

    metrics = Metrics(model_name="test-model")
    baseline = metrics.deep_copy()
    for response_id in ["first", "second"]:
        metrics.add_response_latency(0.1, response_id)
        metrics.add_token_usage(
            prompt_tokens=11,
            completion_tokens=5,
            cache_read_tokens=0,
            cache_write_tokens=0,
            context_window=0,
            response_id=response_id,
        )
    metrics.add_cost(0.25)
    usage = adapter._usage(SimpleNamespace(metrics=metrics), baseline)
    assert usage.input_tokens == 22
    assert usage.output_tokens == 10
    assert usage.cost_usd is None


@pytest.mark.parametrize("response_without_tokens", [False, True])
def test_usage_requires_tokens_for_every_observed_response(response_without_tokens):
    usage = SimpleNamespace(prompt_tokens=11, completion_tokens=5)
    metrics = MagicMock()
    metrics.diff.return_value = SimpleNamespace(
        accumulated_token_usage=usage,
        token_usages=[usage],
        response_latencies=[object()] * (2 if response_without_tokens else 1),
        accumulated_cost=0,
        costs=[],
    )
    result = adapter._usage(SimpleNamespace(metrics=metrics), object())
    if response_without_tokens:
        assert result is None
    else:
        # A later failed request without a response adds no response record.
        assert result.input_tokens == 11
        assert result.output_tokens == 5


@pytest.mark.parametrize("token_records", [0, 1])
@pytest.mark.parametrize("cost_records", [1, 2])
def test_usage_retains_only_complete_cost_when_token_coverage_is_partial(
    token_records, cost_records
):
    metrics = MagicMock()
    metrics.diff.return_value = SimpleNamespace(
        accumulated_token_usage=SimpleNamespace(prompt_tokens=11, completion_tokens=5),
        token_usages=[object()] * token_records,
        response_latencies=[object(), object()],
        accumulated_cost=0.25,
        costs=[object()] * cost_records,
    )
    result = adapter._usage(SimpleNamespace(metrics=metrics), object())
    if cost_records == 2:
        assert result.cost_usd == 0.25
        assert result.input_tokens is None
        assert result.output_tokens is None
        assert result.total_tokens is None
    else:
        assert result is None
