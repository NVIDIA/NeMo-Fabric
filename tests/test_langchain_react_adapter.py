# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the langchain-react Fabric adapter."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
LANGCHAIN_REACT_SRC = ROOT / "adapters" / "langchain-react" / "src"
COMMON_SRC = ROOT / "adapters" / "common" / "src"
for path in (LANGCHAIN_REACT_SRC, COMMON_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from nemo_fabric_adapters.langchain_react import adapter  # noqa: E402
from nemo_fabric_adapters.langchain_react.llm import resolve_model_config  # noqa: E402
from nemo_fabric_adapters.langchain_react.react.output_parser import ReActOutputParser  # noqa: E402
from nemo_fabric_adapters.langchain_react.tools import ToolResolutionContext  # noqa: E402
from nemo_fabric_adapters.langchain_react.tools import build_calculator_tools  # noqa: E402
from nemo_fabric_adapters.langchain_react.tools import resolve_tools  # noqa: E402


def test_build_chat_model_allows_keyless_gateway(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from nemo_fabric_adapters.langchain_react.llm import build_chat_model

    model = build_chat_model(
        {
            "model": "default/qwen3-8b",
            "base_url": "http://10.0.0.51:8080/apis/inference-gateway/v2/workspaces/default/openai/-/v1",
            "allow_empty_api_key": True,
            "api_key": "not-used",
            "temperature": 0.0,
            "top_p": 1.0,
        }
    )
    assert model.model_name == "default/qwen3-8b"


def test_output_parser_parses_final_answer() -> None:
    text = "Thought: done\nFinal Answer: Alexander Graham Bell"
    result = ReActOutputParser().parse(text)
    assert result.return_values["output"] == "Alexander Graham Bell"


def test_output_parser_parses_action() -> None:
    text = "Thought: search\nAction: wiki\nAction Input: telephone inventor"
    result = ReActOutputParser().parse(text)
    assert result.tool == "wiki"
    assert result.tool_input == "telephone inventor"


def test_resolve_model_config_applies_direct_overrides() -> None:
    models = {"default": {"model": "test-model", "temperature": 0.0, "top_p": 1.0}}
    resolved = resolve_model_config(models, "default", {}, {"temperature": 0.6, "top_p": 0.8})
    assert resolved["temperature"] == 0.6
    assert resolved["top_p"] == 0.8


def test_resolve_tools_expands_calculator_group() -> None:
    tools = resolve_tools(
        ["calculator", "clock"],
        {
            "calculator": {"kind": "function_group", "include": ["add", "multiply"]},
            "clock": {"kind": "current_datetime"},
        },
    )
    names = {tool.name for tool in tools}
    assert names == {"add", "multiply", "clock"}


def test_resolve_tools_supports_exa_and_code_generation() -> None:
    class FakeLLM:
        async def ainvoke(self, inputs):
            return SimpleNamespace(content="print('hello')")

    tools = resolve_tools(
        ["web", "coder"],
        {
            "web": {"kind": "exa_internet_search", "max_results": 2},
            "coder": {"kind": "code_generation", "programming_language": "Python"},
        },
        context=ToolResolutionContext(build_llm=lambda _name: FakeLLM()),
    )
    assert {tool.name for tool in tools} == {"web", "coder"}
    assert tools[0].description is not None
    assert "code generation" in tools[1].description.lower()


def test_resolve_tools_rejects_removed_tavily_kind() -> None:
    with pytest.raises(RuntimeError, match="tavily_internet_search"):
        resolve_tools(["search"], {"search": {"kind": "tavily_internet_search"}})


def test_request_messages_trims_history() -> None:
    from nemo_fabric_adapters.langchain_react.config import request_messages

    payload = {
        "request": {
            "input": [
                {"role": "user", "content": "one"},
                {"role": "assistant", "content": "two"},
                {"role": "user", "content": "three"},
            ]
        }
    }
    messages = request_messages(payload, max_history=10)
    assert len(messages) == 3
    trimmed = request_messages(payload, max_history=2)
    assert len(trimmed) == 1
    assert str(trimmed[-1].content) == "three"


@pytest.mark.asyncio
async def test_code_generation_tool_uses_bound_llm() -> None:
    from nemo_fabric_adapters.langchain_react.tools.code_generation import build_code_generation_tool

    class FakeLLM:
        async def ainvoke(self, inputs):
            del inputs
            return SimpleNamespace(content="code:fibonacci")

    tool = build_code_generation_tool(llm=FakeLLM(), name="code_generation")
    assert await tool.ainvoke({"question": "fibonacci"}) == "code:fibonacci"


@pytest.mark.asyncio
async def test_calculator_add_tool() -> None:
    add_tool = next(tool for tool in build_calculator_tools(["add"]) if tool.name == "add")
    assert await add_tool.ainvoke({"numbers": [2, 3]}) == 5.0


def _react_payload(*, overrides: dict[str, Any] | None = None, user_message: str = "2+2") -> dict[str, Any]:
    request: dict[str, Any] = {
        "request_id": "req-1",
        "input": user_message,
    }
    if overrides is not None:
        request["overrides"] = overrides
    return {
        "effective_config": {
            "agent_name": "demo",
            "config": {
                "harness": {
                    "settings": {
                        "workflow": {
                            "tool_names": ["calculator"],
                            "llm_name": "default",
                            "use_native_tool_calling": True,
                            "parse_agent_response_max_retries": 1,
                            "max_tool_calls": 3,
                        },
                        "tools": {
                            "calculator": {
                                "kind": "function_group",
                                "include": ["add"],
                            }
                        },
                    }
                },
                "models": {
                    "default": {
                        "provider": "openai",
                        "model": "test-model",
                        "api_key_env": "TEST_API_KEY",
                    }
                },
            },
        },
        "request": request,
    }


@pytest.mark.asyncio
async def test_adapter_run_with_mocked_graph(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    class FakeGraph:
        async def ainvoke(self, state, config=None):
            captured["recursion_limit"] = (config or {}).get("recursion_limit")
            captured["question"] = state.messages[-1].content
            state.messages.append(SimpleNamespace(type="ai", content="4"))
            state.final_answer = "4"
            return state

    class FakeBuilder:
        def __init__(self, *args, **kwargs) -> None:
            captured["workflow"] = kwargs["config"]

        async def build_graph(self):
            return FakeGraph()

    monkeypatch.setenv("TEST_API_KEY", "secret")
    monkeypatch.setattr(adapter, "ReActAgentGraph", FakeBuilder)
    monkeypatch.setattr(
        adapter,
        "build_chat_model",
        lambda model_config: SimpleNamespace(model=model_config["model"]),
    )
    monkeypatch.setattr(
        adapter,
        "resolve_tools",
        lambda tool_names, tools_cfg, context=None: [SimpleNamespace(name="add")],
    )

    output = await adapter.run_langchain_react(
        _react_payload(overrides={"temperature": 0.4, "top_p": 0.7})
    )

    assert output["failed"] is False
    assert output["response"] == "4"
    assert output["temperature"] == 0.4
    assert output["top_p"] == 0.7
    assert captured["recursion_limit"] == 8
    assert captured["workflow"].use_native_tool_calling is True


@pytest.mark.asyncio
async def test_adapter_registry_descriptor_exists() -> None:
    descriptor_path = ROOT / "adapters" / "langchain-react" / "fabric-adapter.json"
    assert descriptor_path.is_file()
    import json

    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    assert descriptor["adapter_id"] == "nvidia.fabric.langchain.react"
    assert descriptor["runner"]["callable"] == "run"


@pytest.mark.asyncio
async def test_react_graph_returns_final_answer_with_fake_llm() -> None:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from langchain_core.messages import HumanMessage

    from nemo_fabric_adapters.langchain_react.react.graph import ReActAgentGraph
    from nemo_fabric_adapters.langchain_react.react.graph import ReActGraphState
    from nemo_fabric_adapters.langchain_react.react.graph import WorkflowSettings
    from nemo_fabric_adapters.langchain_react.react.graph import create_react_agent_prompt
    from nemo_fabric_adapters.langchain_react.tools import build_calculator_tools

    class FakeChatModel(BaseChatModel):
        @property
        def _llm_type(self) -> str:
            return "fake"

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content="Thought: done\nFinal Answer: 42"))]
            )

        async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
            return self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    workflow = WorkflowSettings(
        tool_names=["add"],
        parse_agent_response_max_retries=1,
        max_tool_calls=2,
    )
    graph_builder = ReActAgentGraph(
        llm=FakeChatModel(),
        prompt=create_react_agent_prompt(workflow),
        tools=build_calculator_tools(["add"]),
        config=workflow,
    )
    graph = await graph_builder.build_graph()
    state = ReActGraphState(messages=[HumanMessage(content="What is 6+6?")])
    result = await graph.ainvoke(state, config={"recursion_limit": 6})
    final_state = ReActGraphState.model_validate(result)
    assert final_state.final_answer == "42"


@pytest.mark.asyncio
async def test_react_graph_native_tool_call_invokes_calculator() -> None:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from langchain_core.messages import HumanMessage

    from nemo_fabric_adapters.langchain_react.react.graph import ReActAgentGraph
    from nemo_fabric_adapters.langchain_react.react.graph import ReActGraphState
    from nemo_fabric_adapters.langchain_react.react.graph import WorkflowSettings
    from nemo_fabric_adapters.langchain_react.react.graph import create_react_agent_prompt
    from nemo_fabric_adapters.langchain_react.tools import build_calculator_tools

    responses = [
        AIMessage(
            content="",
            tool_calls=[{"name": "add", "args": {"numbers": [12.0, 8.0]}, "id": "call-1", "type": "tool_call"}],
        ),
        AIMessage(content="Thought: done\nFinal Answer: 20"),
    ]

    class FakeToolChatModel(BaseChatModel):
        def __init__(self) -> None:
            super().__init__()
            self._calls = 0

        @property
        def _llm_type(self) -> str:
            return "fake-tool"

        def bind_tools(self, tools, **kwargs):
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            message = responses[min(self._calls, len(responses) - 1)]
            self._calls += 1
            return ChatResult(generations=[ChatGeneration(message=message)])

        async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
            return self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    workflow = WorkflowSettings(
        tool_names=["add"],
        use_native_tool_calling=True,
        parse_agent_response_max_retries=1,
        max_tool_calls=3,
    )
    graph_builder = ReActAgentGraph(
        llm=FakeToolChatModel(),
        prompt=create_react_agent_prompt(workflow),
        tools=build_calculator_tools(["add"]),
        config=workflow,
    )
    graph = await graph_builder.build_graph()
    state = ReActGraphState(messages=[HumanMessage(content="What is 12 + 8?")])
    result = await graph.ainvoke(state, config={"recursion_limit": 8})
    final_state = ReActGraphState.model_validate(result)
    assert final_state.final_answer == "20"


def test_fabric_cli_discovers_langchain_react_adapter() -> None:
    import json
    import subprocess

    completed = subprocess.run(
        ["cargo", "run", "-q", "-p", "fabric-cli", "--", "plan", "examples/react-optimize-agent"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        pytest.skip(f"fabric-cli unavailable: {completed.stderr.strip()}")
    plan = json.loads(completed.stdout)
    assert plan["adapter_descriptor"]["descriptor"]["adapter_id"] == "nvidia.fabric.langchain.react"
    assert plan["agent_name"] == "react-optimize-agent"

