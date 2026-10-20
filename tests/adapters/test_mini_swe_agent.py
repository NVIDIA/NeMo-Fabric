# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused contract tests for the mini-SWE-agent adapter."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from contextlib import asynccontextmanager
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import litellm
import openai
import pytest
from nemo_fabric_adapter_contract.models import AgentConfig
from nemo_fabric_adapter_contract.models import AgentRunRequest
from nemo_fabric_adapter_contract.models import AgentUsage
from nemo_fabric_adapter_contract.models import RuntimeContext
from nemo_fabric_adapters.mini_swe_agent import adapter
from nemo_fabric_adapters.mini_swe_agent.agents import RelayRetainingDefaultAgent
from nemo_fabric_adapters.mini_swe_agent.agents import RetainingDefaultAgent
from nemo_fabric_adapters.mini_swe_agent.agents import _submitted_output
from litellm.llms.custom_httpx.http_handler import HTTPHandler
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
    monkeypatch.setattr(
        "nemo_fabric_adapters.mini_swe_agent.agents.UsageRecordingLitellmModel",
        model_factory,
    )
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
                    "top_p": 0.8,
                    "max_tokens": 256,
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
        "propagation_contexts": [],
        "propagation_stacks": [],
        "used_propagation_stacks": [],
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
        yield SimpleNamespace(report={"config": {"diagnostics": []}})

    @contextmanager
    def request_scope(name, scope_type, **kwargs):
        calls["request_scopes"].append((name, scope_type, kwargs))
        yield request_handle

    def propagation_context(parent_uuid, root_uuid=None):
        calls["propagation_contexts"].append((parent_uuid, root_uuid))
        return SimpleNamespace(parent_uuid=parent_uuid, root_uuid=root_uuid)

    def create_propagation_stack(context):
        calls["propagation_stacks"].append(context)
        return context

    @contextmanager
    def use_propagation_stack(stack):
        calls["used_propagation_stacks"].append(stack)
        yield stack

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

    monkeypatch.setattr(nemo_relay.plugin, "activate", plugin_context)
    monkeypatch.setattr(nemo_relay, "PropagationContext", propagation_context)
    monkeypatch.setattr(
        nemo_relay,
        "create_scope_stack_from_propagation",
        create_propagation_stack,
    )
    monkeypatch.setattr(nemo_relay, "use_scope_stack", use_propagation_stack)
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
        (
            ROOT / "adapters/python/mini-swe-agent/mini-swe-agent.fabric-adapter.json"
        ).read_text(encoding="utf-8")
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
            "models.top_p",
            "models.max_tokens",
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
            "top_p": 0.8,
            "max_tokens": 256,
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


def litellm_response(
    usage: dict | None, command: str | None = "submit"
) -> tuple[str, dict]:
    """Build a provider response body for one scripted model call.

    The usage keys pick the provider: ``input_tokens`` or ``output_tokens`` for
    the Anthropic Messages API, ``inputTokens`` or ``outputTokens`` for Bedrock
    Converse, and Chat Completions otherwise. ``command=None`` returns text
    without a tool call, which fails mini-SWE-agent's tool-call parsing.
    """

    usage_keys = set(usage or {})
    if usage_keys & {"input_tokens", "output_tokens"}:
        content = (
            [{"type": "tool_use", "id": "toolu_1", "name": "bash", "input": {}}]
            if command
            else [{"type": "text", "text": "Thinking."}]
        )
        if command:
            content[0]["input"] = {"command": command}
        return "anthropic", {
            "id": "msg_mini",
            "type": "message",
            "role": "assistant",
            "model": "claude-test",
            "content": content,
            "stop_reason": "tool_use" if command else "end_turn",
            "stop_sequence": None,
            "usage": usage,
        }
    if usage_keys & {"inputTokens", "outputTokens"}:
        content = (
            [{"toolUse": {"toolUseId": "t1", "name": "bash", "input": {}}}]
            if command
            else [{"text": "Thinking."}]
        )
        if command:
            content[0]["toolUse"]["input"] = {"command": command}
        return "bedrock", {
            "output": {"message": {"role": "assistant", "content": content}},
            "stopReason": "tool_use" if command else "end_turn",
            "usage": usage,
            "metrics": {"latencyMs": 1},
        }
    tool_calls = [
        {
            "id": "call-1",
            "type": "function",
            "function": {"name": "bash", "arguments": json.dumps({"command": command})},
        }
    ]
    body = {
        "object": "chat.completion",
        "created": 1760000000,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None if command else "Thinking.",
                    "tool_calls": tool_calls if command else None,
                },
                "finish_reason": "tool_calls" if command else "stop",
            }
        ],
    }
    if usage is not None:
        body["usage"] = usage
    return "openai", body


@pytest.fixture(name="litellm_mini")
def litellm_mini_fixture(monkeypatch: pytest.MonkeyPatch) -> list:
    """Run the real LitellmModel and litellm.completion on scripted HTTP bodies.

    Only the HTTP transport is faked, so LiteLLM's own provider conversion and
    its ``logger_fn`` call run as they do against a real endpoint.
    """

    responses: list = []
    environment = MagicMock()
    environment.get_template_vars.return_value = {}

    def execute(action):
        if action["command"] == "submit":
            raise Submitted(
                {
                    "role": "exit",
                    "content": "done",
                    "extra": {"exit_status": "Submitted", "submission": "done"},
                }
            )
        return {"output": "ok", "returncode": 0, "exception_info": ""}

    environment.execute.side_effect = execute
    monkeypatch.setattr(
        "minisweagent.environments.local.LocalEnvironment",
        MagicMock(return_value=environment),
    )
    completion = litellm.completion
    calls = iter(range(1, 1000))

    def scripted_completion(**kwargs):
        provider, body = responses.pop(0)
        kwargs["api_key"] = "test-key"
        transport = httpx.MockTransport(lambda _request: httpx.Response(200, json=body))
        if provider == "openai":
            body.setdefault("id", f"chatcmpl-mini-{next(calls)}")
            kwargs["model"] = "openai/test-model"
            kwargs["client"] = openai.OpenAI(
                api_key="test-key",
                base_url="https://example.test/v1",
                http_client=httpx.Client(transport=transport),
            )
        else:
            kwargs["model"] = (
                "anthropic/claude-sonnet-4-5"
                if provider == "anthropic"
                else "bedrock/converse/anthropic.claude-sonnet-4-5"
            )
            kwargs["client"] = HTTPHandler(client=httpx.Client(transport=transport))
            kwargs.update(
                aws_access_key_id="test",
                aws_secret_access_key="test",
                aws_region_name="us-east-1",
            )
        return completion(**kwargs)

    monkeypatch.setattr(litellm, "completion", scripted_completion)
    monkeypatch.setattr(
        litellm.cost_calculator, "completion_cost", lambda *_, **__: 0.01
    )
    return responses


# LiteLLM signs Bedrock requests with botocore. The lockfile installs boto3
# only on Python 3.12 and later, so the Bedrock cases run there.
_REQUIRES_BOTO3 = pytest.mark.skipif(
    importlib.util.find_spec("boto3") is None,
    reason="LiteLLM's Bedrock provider needs boto3",
)

_OMIT = object()


def _present(**counts: object) -> dict:
    return {key: value for key, value in counts.items() if value is not _OMIT}


def openai_usage(
    prompt=_OMIT, completion=_OMIT, total=_OMIT, cached=_OMIT, **extra
) -> dict:
    """Chat Completions usage; an omitted argument leaves its field out."""

    usage = _present(
        prompt_tokens=prompt, completion_tokens=completion, total_tokens=total
    )
    if cached is not _OMIT:
        usage["prompt_tokens_details"] = {"cached_tokens": cached}
    return usage | extra


def anthropic_usage(
    uncached=_OMIT, output=_OMIT, read=_OMIT, write=_OMIT, **extra
) -> dict:
    """Anthropic Messages usage; an omitted argument leaves its field out."""

    return (
        _present(
            input_tokens=uncached,
            output_tokens=output,
            cache_read_input_tokens=read,
            cache_creation_input_tokens=write,
        )
        | extra
    )


def bedrock_usage(
    uncached=_OMIT, output=_OMIT, total=_OMIT, read=_OMIT, write=_OMIT
) -> dict:
    """Bedrock Converse usage; an omitted argument leaves its field out."""

    return _present(
        inputTokens=uncached,
        outputTokens=output,
        totalTokens=total,
        cacheReadInputTokens=read,
        cacheWriteInputTokens=write,
    )


def expected_usage(input=None, cached=None, output=None, total=None) -> AgentUsage:
    return AgentUsage(
        input_tokens=input,
        cached_input_tokens=cached,
        input_tokens_include_cache=(
            True if input is not None or cached is not None else None
        ),
        output_tokens=output,
        total_tokens=total,
    )


OPENAI_CACHED_USAGE = {
    **openai_usage(1200, 80, 1280),
    "prompt_tokens_details": {"cached_tokens": 1024, "audio_tokens": 0},
    "completion_tokens_details": {"reasoning_tokens": 64, "audio_tokens": 0},
}
ANTHROPIC_NO_CACHE = anthropic_usage(100, 20, read=0, write=0)


@pytest.mark.parametrize(
    ("responses", "expected"),
    [
        # OpenAI-compatible usage, read from LiteLLM's normalized counts.
        pytest.param(
            [
                litellm_response(OPENAI_CACHED_USAGE, command="ls"),
                litellm_response(openai_usage(1500, 40, 1540, cached=1280)),
            ],
            expected_usage(2700, 2304, 120, 2820),
            id="openai-cached-two-calls",
        ),
        pytest.param(
            [
                litellm_response(openai_usage(1200, 80, 1280, cached=0), "ls"),
                litellm_response(openai_usage(1400, 60, 1460, cached=1152), "ls"),
                litellm_response(openai_usage(1600, 40, 1640, cached=1408)),
            ],
            expected_usage(4200, 2560, 180, 4380),
            id="openai-first-call-without-cache-hit-keeps-counts",
        ),
        pytest.param(
            [
                litellm_response(openai_usage(100, 10, 110), command=None),
                litellm_response(openai_usage(200, 20, 220)),
            ],
            expected_usage(300, output=30, total=330),
            id="format-error-call-counted",
        ),
        pytest.param(
            [litellm_response(openai_usage(1200))],
            expected_usage(1200),
            id="omitted-output-and-total-stay-unknown",
        ),
        pytest.param(
            [litellm_response(openai_usage(100, 20))],
            expected_usage(100, output=20, total=120),
            id="zero-filled-total-derived-from-known-parts",
        ),
        pytest.param(
            [litellm_response(openai_usage(1 << 64, 20))],
            expected_usage(output=20),
            id="input-above-contract-bound-is-unknown",
        ),
        pytest.param(
            [litellm_response(openai_usage(100, 20, 1 << 64))],
            expected_usage(100, output=20),
            id="total-above-contract-bound-is-unknown",
        ),
        pytest.param(
            [litellm_response(openai_usage(100, 20, -1))],
            expected_usage(100, output=20),
            id="negative-total-is-unknown",
        ),
        pytest.param(
            [litellm_response(openai_usage(100, total=120))],
            expected_usage(100, total=120),
            id="reported-total-kept-without-output",
        ),
        pytest.param(
            [
                litellm_response(openai_usage(100, total=120), command="ls"),
                litellm_response(openai_usage(200, 30, 230)),
            ],
            expected_usage(300, total=350),
            id="reported-totals-summed-across-partial-calls",
        ),
        pytest.param(
            [
                litellm_response(openai_usage(100, 20, 120), command="ls"),
                litellm_response(openai_usage(200)),
            ],
            expected_usage(300),
            id="call-with-unknown-total-makes-total-unknown",
        ),
        pytest.param(
            [
                litellm_response(
                    openai_usage(100, total=120, cache_read_input_tokens=None)
                )
            ],
            expected_usage(100, total=120),
            id="null-cache-key-keeps-reported-total",
        ),
        pytest.param(
            [
                litellm_response(OPENAI_CACHED_USAGE, command="ls"),
                litellm_response(None),
            ],
            None,
            id="call-without-usage-makes-counts-unknown",
        ),
        pytest.param(
            [litellm_response(openai_usage(0, 0, 0))],
            None,
            id="zero-filled-counts-are-unknown",
        ),
        # Anthropic usage, read from the provider body LiteLLM passes to
        # logger_fn before it zero-fills omitted counts.
        pytest.param(
            [litellm_response(anthropic_usage(50, 120, read=2000, write=300))],
            expected_usage(2350, 2000, 120, 2470),
            id="anthropic-cache-folded-into-input",
        ),
        pytest.param(
            [
                litellm_response(anthropic_usage(12, 90, read=0, write=1500), "ls"),
                litellm_response(anthropic_usage(0, 70, read=1500, write=200), "ls"),
                litellm_response(anthropic_usage(8, 50, read=1700, write=0)),
            ],
            expected_usage(4920, 3200, 210, 5130),
            id="anthropic-first-call-without-cache-hit-keeps-counts",
        ),
        pytest.param(
            [litellm_response(anthropic_usage(output=120, read=2000))],
            expected_usage(cached=2000, output=120),
            id="anthropic-omitted-input-with-cache-hit-stays-unknown",
        ),
        pytest.param(
            [litellm_response(anthropic_usage(100, 20))],
            expected_usage(output=20),
            id="anthropic-omitted-cache-counts-stay-unknown",
        ),
        pytest.param(
            [litellm_response(anthropic_usage(100, 20, read=None, write=None))],
            expected_usage(output=20),
            id="anthropic-null-cache-counts-stay-unknown",
        ),
        pytest.param(
            [litellm_response(anthropic_usage(100, 20, read=-1))],
            expected_usage(output=20),
            id="anthropic-invalid-cache-part-leaves-input-unknown",
        ),
        pytest.param(
            [litellm_response(anthropic_usage(100, read=0, write=0))],
            expected_usage(100, 0),
            id="anthropic-omitted-output-leaves-total-unknown",
        ),
        pytest.param(
            [
                litellm_response(
                    {
                        **ANTHROPIC_NO_CACHE,
                        "iterations": [
                            anthropic_usage(100, 20, read=0, write=0),
                            anthropic_usage(30, 5, read=100, write=0),
                        ],
                    }
                )
            ],
            expected_usage(230, 100, 25, 255),
            id="anthropic-iterations-summed-with-presence-rules",
        ),
        pytest.param(
            [
                litellm_response(
                    {
                        **ANTHROPIC_NO_CACHE,
                        "iterations": [
                            anthropic_usage(100, 20, read=0, write=0),
                            anthropic_usage(30, 5),
                        ],
                    }
                )
            ],
            expected_usage(output=25),
            id="anthropic-iteration-without-cache-counts-leaves-input-unknown",
        ),
        # Bedrock Converse usage, read from the provider body. Inclusive input is
        # inputTokens plus cache reads and writes; totalTokens includes them.
        pytest.param(
            [litellm_response(bedrock_usage(100, 0, 100))],
            expected_usage(100, output=0, total=100),
            id="bedrock-reported-zero-output-and-total-kept",
            marks=_REQUIRES_BOTO3,
        ),
        pytest.param(
            [litellm_response(bedrock_usage(100, 20, -1))],
            expected_usage(100, output=20),
            id="bedrock-negative-total-is-unknown",
            marks=_REQUIRES_BOTO3,
        ),
        pytest.param(
            [litellm_response(bedrock_usage(100, 20, 1 << 64))],
            expected_usage(100, output=20),
            id="bedrock-overflowing-total-is-unknown",
            marks=_REQUIRES_BOTO3,
        ),
        pytest.param(
            [
                litellm_response(bedrock_usage(1500, 90, 3090, write=1500), "ls"),
                litellm_response(bedrock_usage(12, 50, 1762, read=1700, write=0)),
            ],
            expected_usage(4712, output=140, total=4852),
            id="bedrock-cache-write-then-read-keeps-input-output-and-total",
            marks=_REQUIRES_BOTO3,
        ),
    ],
)
async def test_mini_swe_agent_usage_sums_litellm_calls_with_unknown_counts_left_unset(
    litellm_mini, mini_payload, responses, expected
):
    litellm_mini.extend(responses)
    runtime = adapter.MiniSweAgentRuntime()
    start = {**mini_payload, "config": AgentConfig.from_mapping(mini_payload["config"])}
    await runtime.start(start)

    result = await runtime.invoke(*invocation(mini_payload))

    assert result.status == "succeeded"
    assert result.usage == expected
    assert result.output["usage"] == {"api_calls": len(responses)}


@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        pytest.param(
            anthropic_usage(0, 20, read=1500, write=0),
            expected_usage(cached=1500, output=20),
            id="cache-hit",
        ),
        pytest.param(
            ANTHROPIC_NO_CACHE, expected_usage(output=20), id="explicit-zero-cache"
        ),
        pytest.param(
            anthropic_usage(100, 20), expected_usage(output=20), id="missing-cache"
        ),
        pytest.param(
            anthropic_usage(100, 20, read=None, write=None),
            expected_usage(output=20),
            id="null-cache",
        ),
    ],
)
async def test_mini_swe_agent_usage_without_provider_usage_keeps_unproven_counts_unknown(
    litellm_mini, mini_payload, usage, expected
):
    litellm_mini.append(litellm_response(usage))
    runtime = adapter.MiniSweAgentRuntime()
    start = {**mini_payload, "config": AgentConfig.from_mapping(mini_payload["config"])}
    await runtime.start(start)
    await runtime.invoke(*invocation(mini_payload))

    # LiteLLM's normalized usage alone cannot show whether a cache count was
    # omitted, so inclusive input, its total, and a cached 0 stay unknown.
    assert adapter._invocation_usage(runtime._agent.invocation_responses, 1, {}) == (
        expected
    )


async def test_mini_swe_agent_failed_result_keeps_usage(litellm_mini, mini_payload):
    litellm_mini.append(litellm_response(OPENAI_CACHED_USAGE, command="ls"))
    mini_payload["config"]["runtime"]["max_turns"] = 1
    runtime = adapter.MiniSweAgentRuntime()
    start = {**mini_payload, "config": AgentConfig.from_mapping(mini_payload["config"])}
    await runtime.start(start)

    result = await runtime.invoke(*invocation(mini_payload))

    assert result.status == "failed"
    assert result.error.code == "mini_swe_agent_incomplete"
    assert result.usage == expected_usage(1200, 1024, 80, 1280)


async def test_mini_swe_agent_usage_counts_only_the_current_invocation(
    litellm_mini, mini_payload
):
    litellm_mini.extend(
        [
            litellm_response(OPENAI_CACHED_USAGE),
            litellm_response(openai_usage(1500, 40, 1540)),
        ]
    )
    runtime = adapter.MiniSweAgentRuntime()
    start = {**mini_payload, "config": AgentConfig.from_mapping(mini_payload["config"])}
    await runtime.start(start)

    await runtime.invoke(*invocation(mini_payload))
    mini_payload["request"]["input"] = "Continue the task."
    result = await runtime.invoke(*invocation(mini_payload))

    assert result.usage == expected_usage(1500, output=40, total=1540)


async def test_mini_swe_agent_usage_is_unknown_when_a_call_left_no_response(
    mock_mini, mini_payload
):
    # The first call stores no LiteLLM response, as mini-SWE-agent 2.0.0 does
    # for a call that fails tool-call parsing.
    query_with_response = mock_mini["model"].query.side_effect
    mock_mini["model"].query.side_effect = [
        {"role": "assistant", "extra": {"actions": [{"command": "ls"}]}},
        query_with_response([]),
    ]
    mock_mini["environment_factory"].return_value.execute.side_effect = [
        {"output": "ok", "returncode": 0, "exception_info": ""},
        Submitted(
            {
                "role": "exit",
                "content": "done",
                "extra": {"exit_status": "Submitted", "submission": "done"},
            }
        ),
    ]
    runtime = adapter.MiniSweAgentRuntime()
    start = {**mini_payload, "config": AgentConfig.from_mapping(mini_payload["config"])}
    await runtime.start(start)

    result = await runtime.invoke(*invocation(mini_payload))

    assert result.status == "succeeded"
    assert result.output["usage"] == {"api_calls": 2}
    assert result.usage is None


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
        "nemo_fabric_runtime_id": "mini-runtime",
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


async def test_uuid_request_id_seeds_relay_propagation(
    mock_mini,
    mini_payload,
    mock_relay,
):
    request_id = "018f47a4-3af7-7d94-8e61-9f0f89b5d312"
    mini_payload["runtime_context"].update(
        {
            "request_id": request_id,
            "telemetry": {"relay_enabled": True},
        }
    )
    runtime = adapter.MiniSweAgentRuntime()
    start = {**mini_payload, "config": AgentConfig.from_mapping(mini_payload["config"])}
    await runtime.start(start)

    result = await runtime.invoke(*invocation(mini_payload))

    assert result.status == "succeeded"
    assert mock_relay["propagation_contexts"] == [(request_id, request_id)]
    assert mock_relay["used_propagation_stacks"] == mock_relay["propagation_stacks"]
    assert mock_relay["request_scopes"][0][2]["metadata"] == {
        "nemo_fabric_request_id": request_id,
        "nemo_fabric_invocation_id": "mini-invocation",
        "nemo_fabric_runtime_id": "mini-runtime",
    }


SESSION_ROOT = "018f47a4-0000-7d94-8e61-9f0f89b5d312"
REQUEST_UUID = "018f47a4-3af7-7d94-8e61-9f0f89b5d312"


@pytest.mark.parametrize(
    ("request_id", "expected_parent"),
    [(REQUEST_UUID, REQUEST_UUID), ("request-1", SESSION_ROOT)],
)
async def test_typed_session_root_roots_relay_propagation(
    mock_mini,
    mini_payload,
    mock_relay,
    request_id,
    expected_parent,
):
    mini_payload["runtime_context"].update(
        {
            "request_id": request_id,
            "telemetry": {"relay_enabled": True},
        }
    )
    mini_payload["request"]["relay_session_root"] = SESSION_ROOT
    mini_payload["request"]["context"] = {"relay_session_root": REQUEST_UUID}
    runtime = adapter.MiniSweAgentRuntime()
    start = {**mini_payload, "config": AgentConfig.from_mapping(mini_payload["config"])}
    await runtime.start(start)

    result = await runtime.invoke(*invocation(mini_payload))

    assert result.status == "succeeded"
    assert mock_relay["propagation_contexts"] == [(expected_parent, SESSION_ROOT)]
    assert mock_relay["used_propagation_stacks"] == mock_relay["propagation_stacks"]
    assert mock_relay["request_scopes"][0][2]["metadata"] == {
        "nemo_fabric_request_id": request_id,
        "nemo_fabric_session_root": SESSION_ROOT,
        "nemo_fabric_invocation_id": "mini-invocation",
        "nemo_fabric_runtime_id": "mini-runtime",
    }


@pytest.mark.parametrize(
    "session_root",
    ["", "not-a-uuid", None],
    ids=["empty", "non-uuid", "context-keys-ignored"],
)
async def test_unusable_session_root_falls_back_to_the_request_root(
    mock_mini,
    mini_payload,
    mock_relay,
    session_root,
):
    mini_payload["runtime_context"].update(
        {
            "request_id": REQUEST_UUID,
            "telemetry": {"relay_enabled": True},
        }
    )
    mini_payload["request"]["relay_session_root"] = session_root
    mini_payload["request"]["context"] = {
        "relay_session_root": SESSION_ROOT,
        "session_id": SESSION_ROOT,
    }
    runtime = adapter.MiniSweAgentRuntime()
    start = {**mini_payload, "config": AgentConfig.from_mapping(mini_payload["config"])}
    await runtime.start(start)

    result = await runtime.invoke(*invocation(mini_payload))

    assert result.status == "succeeded"
    assert mock_relay["propagation_contexts"] == [(REQUEST_UUID, REQUEST_UUID)]
    assert (
        "nemo_fabric_session_root" not in mock_relay["request_scopes"][0][2]["metadata"]
    )


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
