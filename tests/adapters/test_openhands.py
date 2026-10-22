# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused contract tests for the OpenHands adapter."""

from __future__ import annotations

import json
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from nemo_fabric_adapter_contract.models import AgentConfig
from nemo_fabric_adapter_contract.models import AgentRunRequest
from nemo_fabric_adapter_contract.models import AgentRunStatus
from nemo_fabric_adapter_contract.models import RuntimeContext
from nemo_fabric_adapters.openhands import adapter

ROOT = Path(__file__).resolve().parents[2]


def _base_usage(usage):
    return {
        key: value for key, value in usage.to_mapping().items() if key != "extensions"
    }


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
    # Shaped like OpenHands Metrics.diff(): summed counters plus one record per
    # model response and per response that reported usage.
    usage = SimpleNamespace(
        prompt_tokens=11,
        completion_tokens=5,
        cache_read_tokens=3,
        cache_write_tokens=0,
        reasoning_tokens=2,
        response_id="resp-1",
    )
    metrics_delta = SimpleNamespace(
        accumulated_token_usage=usage,
        accumulated_cost=0.25,
        token_usages=[usage],
        response_latencies=[SimpleNamespace(latency=0.5, response_id="resp-1")],
    )
    metrics = MagicMock()
    metrics.deep_copy.return_value = object()
    metrics.diff.return_value = metrics_delta
    telemetry = SimpleNamespace(log_enabled=False, completion_log=None)
    telemetry.set_log_completions_callback = lambda callback: setattr(
        telemetry, "completion_log", callback
    )
    llm = SimpleNamespace(metrics=metrics, telemetry=telemetry)

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
    usage_log = {"cached_tokens": 3}

    async def arun():
        # OpenHands logs each model response, here with cache details.
        telemetry.completion_log(
            "test-model.json",
            json.dumps(
                {
                    "response": {
                        "id": "resp-1",
                        "usage": {"prompt_tokens_details": usage_log},
                    }
                }
            ),
        )
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
            "metrics": metrics,
            "telemetry": telemetry,
            "usage_log": usage_log,
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
        model="nvidia/test-model",
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


def _litellm_usage(
    prompt: int | None, completion: int | None, cached: int | None = None
) -> dict:
    """Build a LiteLLM usage mapping that omits the counters given as None."""
    usage: dict = {"total_tokens": (prompt or 0) + (completion or 0)}
    if prompt is not None:
        usage["prompt_tokens"] = prompt
    if completion is not None:
        usage["completion_tokens"] = completion
    if cached is not None:
        usage["prompt_tokens_details"] = {"cached_tokens": cached}
    return usage


def _cache_hits_after_first_call(call: int) -> dict:
    # A conversation's first request finds nothing cached.
    return _litellm_usage(10, 1, cached=0 if call == 1 else 8)


ALL_COUNTS = ("input", "cached", "output")


# Event limit: the system prompt, the user message, and 40 think action and
# observation pairs exceed the 80-event limit, so the condenser runs before the
# 41st agent call. Context window error: the rejected 11th call triggers the
# condensation, and the 12th call finishes. The other cases change only the
# usage that the agent and summary responses report. ``known`` lists the
# counters the adapter must report, as the sum of what the responses sent;
# None means usage is unknown.
@pytest.mark.parametrize(
    (
        "context_error_at_call",
        "expected_agent_calls",
        "agent_usage",
        "summary_usage",
        "known",
    ),
    [
        (
            None,
            41,
            lambda call: _litellm_usage(10, 1, cached=8),
            _litellm_usage(1000, 100, cached=600),
            ALL_COUNTS,
        ),
        (
            11,
            12,
            lambda call: _litellm_usage(10, 1, cached=8),
            _litellm_usage(1000, 100, cached=600),
            None,
        ),
        (
            None,
            41,
            _cache_hits_after_first_call,
            _litellm_usage(1000, 100, cached=0),
            ALL_COUNTS,
        ),
        (
            None,
            41,
            lambda call: _litellm_usage(10, 1, cached=8),
            _litellm_usage(1000, 100),
            ("input", "output"),
        ),
        (
            None,
            41,
            lambda call: _litellm_usage(10, 1),
            _litellm_usage(1000, 100),
            ("input", "output"),
        ),
        (
            None,
            41,
            lambda call: _litellm_usage(10, None, cached=8),
            _litellm_usage(1000, None, cached=600),
            ("input", "cached"),
        ),
        (
            None,
            41,
            lambda call: _litellm_usage(None, 1),
            _litellm_usage(None, 100),
            ("output",),
        ),
        (
            None,
            41,
            lambda call: _litellm_usage(10, 1, cached=8),
            _litellm_usage(1000, None, cached=600),
            ("input", "cached"),
        ),
        (None, 41, lambda call: _litellm_usage(10, 1, cached=8), None, None),
        (
            None,
            41,
            lambda call: _litellm_usage(10, 1, cached=8),
            _litellm_usage(0, 0),
            None,
        ),
    ],
    ids=[
        "event-limit",
        "context-window-error",
        "cache-hits-after-first-call",
        "summary-without-cache-details",
        "no-cache-details",
        "prompt-only",
        "completion-only",
        "complete-then-prompt-only",
        "summary-without-usage",
        "summary-with-zero-usage",
    ],
)
async def test_invoke_condenses_history_and_counts_summary_usage(
    openhands_payload: dict,
    monkeypatch: pytest.MonkeyPatch,
    context_error_at_call: int | None,
    expected_agent_calls: int,
    agent_usage: Callable[[int], dict],
    summary_usage: dict | None,
    known: tuple[str, ...] | None,
    tmp_path: Path,
):
    pytest.importorskip("openhands.sdk")
    pytest.importorskip("openhands.tools")
    from litellm.exceptions import ContextWindowExceededError
    from litellm.types.utils import ModelResponse
    from openhands.sdk.event import Condensation
    from openhands.sdk.llm import llm as openhands_llm
    from openhands.sdk.llm import llm_profile_store

    # Isolate automatic vision-profile discovery from the user's saved models.
    monkeypatch.setattr(
        llm_profile_store, "_DEFAULT_PROFILE_DIR", tmp_path / "profiles"
    )

    calls = {"agent": 0, "summary": 0}
    sent = {"input": 0, "cached": 0, "output": 0}

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
            usage = agent_usage(calls["agent"])
        else:
            calls["summary"] += 1
            message = {"role": "assistant", "content": "Summary of earlier steps."}
            usage = summary_usage
        if usage is not None:
            sent["input"] += usage.get("prompt_tokens", 0)
            sent["output"] += usage.get("completion_tokens", 0)
            sent["cached"] += usage.get("prompt_tokens_details", {}).get(
                "cached_tokens", 0
            )
        return ModelResponse(
            model="test-model",
            choices=[{"index": 0, "finish_reason": "stop", "message": message}],
            **({"usage": usage} if usage is not None else {}),
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
    if known is None:
        assert result.usage.input_tokens is None
        assert result.usage.output_tokens is None
        return
    expected = {name: sent[name] if name in known else None for name in sent}
    assert result.usage.input_tokens == expected["input"]
    assert result.usage.cached_input_tokens == expected["cached"]
    assert result.usage.input_tokens_include_cache is (
        True if "input" in known else None
    )
    assert result.usage.output_tokens == expected["output"]
    # Core derives the normalized total; LiteLLM's synthesized total is not
    # promoted as an independent provider report.
    assert result.usage.total_tokens is None
    # LiteLLM cannot price the test model, and OpenHands costs are estimates.
    assert result.usage.cost_usd is None


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
        "nvidia/test-model"
    )


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
    assert _base_usage(second.usage) == {
        "input_tokens": 11,
        "cached_input_tokens": 3,
        "input_tokens_include_cache": True,
        "output_tokens": 5,
    }
    assert mock_openhands["conversation"].send_message.call_args_list[0].args == (
        "first",
    )
    assert mock_openhands["conversation"].send_message.call_args_list[1].args == (
        "second",
    )


# Gemini's API leaves out a cache-read count of 0, so for Gemini an absent
# count is a reported 0. Other providers send it when they report cache reads.
@pytest.mark.parametrize(
    ("provider", "model", "expected_cached"),
    [
        ("nvidia", "nvidia/test-model", None),
        ("gemini", "gemini-2.5-flash", 0),
        ("vertex_ai", "gemini-2.5-pro", 0),
    ],
    ids=["openai-compatible", "gemini", "vertex-ai-gemini"],
)
async def test_invoke_without_cache_details_depends_on_provider(
    openhands_payload: dict,
    mock_openhands: dict,
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    model: str,
    expected_cached: int | None,
):
    monkeypatch.setenv("TEST_OPENHANDS_API_KEY", "test-key")
    openhands_payload["config"]["models"]["default"].update(
        provider=provider, model=model
    )
    if provider in {"gemini", "vertex_ai"}:
        openhands_payload["config"]["models"]["default"].pop("base_url")
    mock_openhands["usage_log"].clear()
    runtime = adapter.OpenHandsRuntime()
    await runtime.start(_start_payload(openhands_payload))

    result = await runtime.invoke(*_invocation(openhands_payload, "review"))

    assert result.status is AgentRunStatus.SUCCEEDED
    assert result.usage.input_tokens == 11
    assert result.usage.cached_input_tokens == expected_cached


# Each case replaces the one usage record the mocked OpenHands metrics hold.
@pytest.mark.parametrize(
    ("record", "expected"),
    [
        (
            {"prompt_tokens": 1 << 64},
            {"output_tokens": 5, "input_tokens": None, "total_tokens": None},
        ),
        (
            {"prompt_tokens": (1 << 64) - 1},
            {"input_tokens": (1 << 64) - 1, "total_tokens": None},
        ),
        ({"cache_read_tokens": 12}, {"input_tokens": 11, "cached_input_tokens": None}),
        ({"response_id": "other"}, {"input_tokens": None, "cached_input_tokens": None}),
    ],
    ids=[
        "input-past-uint64",
        "total-past-uint64",
        "cache-reads-above-input",
        "cache-report-for-another-response",
    ],
)
async def test_invoke_leaves_inconsistent_counts_unknown(
    openhands_payload: dict,
    mock_openhands: dict,
    monkeypatch: pytest.MonkeyPatch,
    record: dict,
    expected: dict,
):
    monkeypatch.setenv("TEST_OPENHANDS_API_KEY", "test-key")
    delta = mock_openhands["metrics"].diff.return_value
    usage = SimpleNamespace(**{**vars(delta.token_usages[0]), **record})
    delta.token_usages = [usage]
    if "cache_read_tokens" in record:
        mock_openhands["usage_log"]["cached_tokens"] = record["cache_read_tokens"]
    runtime = adapter.OpenHandsRuntime()
    await runtime.start(_start_payload(openhands_payload))

    result = await runtime.invoke(*_invocation(openhands_payload, "review"))

    assert result.status is AgentRunStatus.SUCCEEDED
    usage_mapping = _base_usage(result.usage)
    for name, value in expected.items():
        assert usage_mapping.get(name) == value


# Each response is (prompt, completion, cache reads), and every response
# reports its cache-read count. Different responses leave different counters
# incomplete, and a single response can report more cache reads than input.
@pytest.mark.parametrize(
    ("responses", "expected"),
    [
        (
            [(10, 0, 2), (0, 4, 0)],
            {"cached_input_tokens": 2},
        ),
        (
            [(10, 0, 20), (100, 3, 0)],
            {"input_tokens": 110, "input_tokens_include_cache": True},
        ),
    ],
    ids=["only-cache-reads-complete", "one-response-caches-more-than-input"],
)
async def test_invoke_checks_each_response_separately(
    openhands_payload: dict,
    mock_openhands: dict,
    monkeypatch: pytest.MonkeyPatch,
    responses: list[tuple[int, int, int]],
    expected: dict,
):
    monkeypatch.setenv("TEST_OPENHANDS_API_KEY", "test-key")
    records = [
        SimpleNamespace(
            prompt_tokens=prompt,
            completion_tokens=completion,
            cache_read_tokens=cached,
            response_id=f"resp-{index}",
        )
        for index, (prompt, completion, cached) in enumerate(responses)
    ]
    delta = mock_openhands["metrics"].diff.return_value
    delta.token_usages = records
    delta.response_latencies = [
        SimpleNamespace(latency=0.5, response_id=r.response_id) for r in records
    ]
    telemetry = mock_openhands["telemetry"]
    state = mock_openhands["conversation"].state

    async def arun():
        for record in records:
            telemetry.completion_log(
                "test-model.json",
                json.dumps(
                    {
                        "response": {
                            "id": record.response_id,
                            "usage": {
                                "prompt_tokens_details": {
                                    "cached_tokens": record.cache_read_tokens
                                }
                            },
                        }
                    }
                ),
            )
        state.events.append("done")

    mock_openhands["conversation"].arun = AsyncMock(side_effect=arun)
    runtime = adapter.OpenHandsRuntime()
    await runtime.start(_start_payload(openhands_payload))

    result = await runtime.invoke(*_invocation(openhands_payload, "review"))

    assert result.status is AgentRunStatus.SUCCEEDED
    assert _base_usage(result.usage) == expected


def _completion_log(response_id: str, usage: dict) -> str:
    return json.dumps({"response": {"id": response_id, "usage": usage}})


# LiteLLM declares cached_tokens as optional, so a null count is unreported.
# The Responses API carries cache reads in input_tokens_details instead.


async def test_failed_invocation_reports_usage_from_its_own_responses(
    openhands_payload: dict,
    mock_openhands: dict,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("TEST_OPENHANDS_API_KEY", "test-key")
    runtime = adapter.OpenHandsRuntime()
    await runtime.start(_start_payload(openhands_payload))
    # The first invocation logs a response without cache details.
    mock_openhands["usage_log"].clear()
    await runtime.invoke(*_invocation(openhands_payload, "first"))
    mock_openhands["usage_log"]["cached_tokens"] = 3
    telemetry = mock_openhands["telemetry"]

    async def arun():
        telemetry.completion_log(
            "test-model.json",
            _completion_log("resp-1", {"prompt_tokens_details": {"cached_tokens": 3}}),
        )
        raise RuntimeError("model call failed after a response")

    mock_openhands["conversation"].arun = AsyncMock(side_effect=arun)

    result = await runtime.invoke(*_invocation(openhands_payload, "second"))

    assert result.status is AgentRunStatus.FAILED
    assert result.error.code == "openhands_invocation_failed"
    # Only the second invocation's report counts, so cache reads are known.
    assert _base_usage(result.usage) == {
        "input_tokens": 11,
        "cached_input_tokens": 3,
        "input_tokens_include_cache": True,
        "output_tokens": 5,
    }


async def test_invoke_without_model_response_reports_unknown_usage(
    openhands_payload: dict,
    mock_openhands: dict,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("TEST_OPENHANDS_API_KEY", "test-key")
    # OpenHands Metrics.diff() starts every counter at 0.
    usage = SimpleNamespace(
        prompt_tokens=0,
        completion_tokens=0,
        cache_read_tokens=0,
        cache_write_tokens=0,
        reasoning_tokens=0,
    )
    mock_openhands["metrics"].diff.return_value = SimpleNamespace(
        accumulated_token_usage=usage,
        accumulated_cost=0.0,
        token_usages=[],
        response_latencies=[],
    )
    runtime = adapter.OpenHandsRuntime()
    await runtime.start(_start_payload(openhands_payload))

    result = await runtime.invoke(*_invocation(openhands_payload, "review"))

    assert result.status is AgentRunStatus.SUCCEEDED
    assert result.usage.input_tokens is None
    assert result.usage.output_tokens is None


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
    conversation.arun.assert_not_awaited()


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


@pytest.mark.parametrize("override", ["global", "environment"])
async def test_custom_gemini_route_does_not_infer_omitted_cache_zero(
    openhands_payload, mock_openhands, monkeypatch, override
):
    litellm = pytest.importorskip("litellm")
    monkeypatch.setenv("TEST_OPENHANDS_API_KEY", "test-key")
    model = openhands_payload["config"]["models"]["default"]
    model.update(provider="gemini", model="gemini-2.5-flash")
    model.pop("base_url")
    if override == "global":
        monkeypatch.setattr(litellm, "api_base", "https://custom.test/v1")
    else:
        monkeypatch.setenv("GEMINI_API_BASE", "https://custom.test/v1")
    mock_openhands["usage_log"].clear()
    runtime = adapter.OpenHandsRuntime()
    await runtime.start(_start_payload(openhands_payload))
    try:
        result = await runtime.invoke(*_invocation(openhands_payload, "review"))
    finally:
        await runtime.stop()
    assert result.usage.input_tokens == 11
    assert result.usage.cached_input_tokens is None
