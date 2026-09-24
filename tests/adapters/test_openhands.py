# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused contract tests for the OpenHands adapter."""

from __future__ import annotations

import json
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
    )
    monkeypatch.setattr(adapter, "_load_openhands_api", lambda: api)
    calls.update(
        {
            "agent_context_factory": agent_context_factory,
            "agent_factory": agent_factory,
            "conversation": conversation,
            "conversation_factory": conversation_factory,
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
    assert second.usage.to_mapping() == {
        "input_tokens": 11,
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
