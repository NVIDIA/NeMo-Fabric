# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for the source-only LangGraph reference adapter."""

from __future__ import annotations

import json
import os
import runpy
import shutil
import sys
from pathlib import Path
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from nemo_fabric import Fabric
from nemo_fabric import FabricConfig
from nemo_fabric import FabricConfigError
from nemo_fabric import HarnessConfig
from nemo_fabric import InstructionConfig
from nemo_fabric import InstructionsConfig
from nemo_fabric import McpConfig
from nemo_fabric import McpServerConfig
from nemo_fabric import MetadataConfig
from nemo_fabric import ModelConfig
from nemo_fabric import SkillConfig
from nemo_fabric import ToolsConfig
from nemo_fabric import WorkflowConfig
from nemo_fabric import WorkflowEntrypointConfig


ROOT = Path(__file__).parents[2]
LANGGRAPH_SOURCE = ROOT / "external" / "langgraph" / "src"
DESCRIPTOR = ROOT / "external" / "langgraph" / "fabric-adapter.json"
sys.path.insert(0, str(LANGGRAPH_SOURCE))

from nemo_fabric_adapters.langgraph import adapter  # noqa: E402


def _stage_adapter(base_dir: Path) -> None:
    destination = base_dir / "adapters" / "langgraph" / "fabric-adapter.json"
    destination.parent.mkdir(parents=True)
    shutil.copyfile(DESCRIPTOR, destination)


def _config(
    *, ref: str = "agent_graph:build_graph", settings: dict[str, object] | None = None
) -> FabricConfig:
    return FabricConfig(
        metadata=MetadataConfig(name="langgraph-reference-test"),
        harness=HarnessConfig(
            adapter_id="example.fabric.langgraph",
            resolution="preinstalled",
        ),
        workflow=WorkflowConfig(
            entrypoint=WorkflowEntrypointConfig(kind="langgraph_factory", ref=ref),
            settings={"prefix": "Fabric: "} if settings is None else settings,
        ),
    )


def _normalized_config(*, ref: str = "capturing_graph:build_graph") -> FabricConfig:
    return FabricConfig(
        metadata=MetadataConfig(name="langgraph-normalized-config-test"),
        harness=HarnessConfig(
            adapter_id="example.fabric.langgraph",
            resolution="preinstalled",
        ),
        workflow=WorkflowConfig(
            entrypoint=WorkflowEntrypointConfig(kind="langgraph_factory", ref=ref),
            settings={
                "llm_name": "nim_llm",
                "tool_names": ["current_timezone", "mcp_math"],
            },
        ),
        models={
            "nim_llm": ModelConfig(
                provider="nim",
                model="meta/llama-3.1-70b-instruct",
                api_key_env="NVIDIA_API_KEY",
                temperature=0.25,
            )
        },
        instructions=InstructionsConfig(
            system=InstructionConfig(content="Use the configured tools when useful.")
        ),
        mcp=McpConfig(
            servers={
                "mcp_math": McpServerConfig(
                    transport="streamable-http",
                    url="http://127.0.0.1:9901/mcp",
                    allowed_tools=["multiply"],
                    blocked_tools=["divide"],
                )
            }
        ),
        tools=ToolsConfig(
            enabled=["mcp_math"],
            blocked=["current_timezone"],
        ),
        skills=SkillConfig(paths=["skills"]),
    )


def _runtime_payload(config: FabricConfig, base_dir: Path) -> dict[str, object]:
    return {
        "base_dir": str(base_dir),
        "config": config.to_mapping(),
        "capability_plan": {
            "native": {"mcp_servers": {}, "skill_paths": []},
            "tools_configured": False,
        },
    }


def test_descriptor_declares_the_source_reference_contract():
    descriptor = json.loads(DESCRIPTOR.read_text(encoding="utf-8"))

    assert descriptor["adapter_id"] == "example.fabric.langgraph"
    assert descriptor["harness"] == "langgraph"
    assert descriptor["adapter_kind"] == "python"
    assert descriptor["runner"] == {"module": "nemo_fabric_adapters.langgraph.adapter"}
    assert descriptor["requirements"] == {}
    assert descriptor["config"]["accepts"] == [
        "models",
        "models.base_url",
        "models.temperature",
        "instructions.system",
        "tools.enabled",
        "tools.blocked",
        "mcp",
        "mcp.tool_filters",
        "skills",
    ]
    assert descriptor["settings_schema"]["additionalProperties"] is False
    workflow_schema = descriptor["workflow_schema"]
    assert workflow_schema["required"] == ["entrypoint"]
    assert workflow_schema["additionalProperties"] is False
    entrypoint_schema = workflow_schema["properties"]["entrypoint"]
    assert entrypoint_schema["properties"]["kind"]["const"] == "langgraph_factory"
    assert entrypoint_schema["required"] == ["kind", "ref"]
    assert entrypoint_schema["additionalProperties"] is False
    assert descriptor["capabilities"] == {
        "cancellation": False,
        "service": False,
        "streaming": False,
        "updates": False,
    }


def test_plan_validates_factory_shape_without_importing_application_code(tmp_path: Path):
    _stage_adapter(tmp_path)

    plan = Fabric().plan(_config(), base_dir=tmp_path)

    assert plan.config.workflow == _config().to_mapping()["workflow"]
    assert (
        plan["adapter_descriptor"]["descriptor"]["workflow_schema"]["properties"][
            "entrypoint"
        ]["properties"]["kind"]["const"]
        == "langgraph_factory"
    )

    with pytest.raises(FabricConfigError, match=r"workflow\.entrypoint\.ref"):
        Fabric().plan(_config(ref="agent graph:build_graph"), base_dir=tmp_path)

    invalid_settings = _config()
    invalid_settings.harness.settings["unknown"] = True
    with pytest.raises(FabricConfigError, match=r"harness\.settings"):
        Fabric().plan(invalid_settings, base_dir=tmp_path)


def test_plan_routes_supported_normalized_config(tmp_path: Path):
    _stage_adapter(tmp_path)

    plan = Fabric().plan(_normalized_config(), base_dir=tmp_path)

    native = plan["capability_plan"]["native"]
    assert native["mcp_servers"]["mcp_math"]["allowed_tools"] == ["multiply"]
    assert native["mcp_servers"]["mcp_math"]["blocked_tools"] == ["divide"]
    assert plan["capability_plan"]["tools"] == {
        "enabled": ["mcp_math"],
        "blocked": ["current_timezone"],
    }
    assert plan["capability_plan"]["native"]["skill_paths"] == [
        str(tmp_path / "skills")
    ]


@pytest.mark.parametrize("example", ["calculator.py", "email_phishing.py"])
def test_examples_build_plan_valid_workflow_configurations(
    tmp_path: Path,
    example: str,
):
    _stage_adapter(tmp_path)
    namespace = runpy.run_path(
        str(ROOT / "external" / "langgraph" / "examples" / example)
    )
    build_config = namespace["build_config"]
    config = (
        build_config("http://127.0.0.1:9901/mcp")
        if example == "calculator.py"
        else build_config()
    )

    plan = Fabric().plan(config, base_dir=tmp_path)

    assert plan.config.workflow == config.to_mapping()["workflow"]


async def test_email_graph_adapts_raw_text_to_application_state():
    namespace = runpy.run_path(
        str(ROOT / "external" / "langgraph" / "examples" / "email_phishing.py")
    )
    mock_compiled_graph = MagicMock(name="compiled_graph")
    mock_compiled_graph.ainvoke = AsyncMock(return_value={"assessment": "phishing"})
    mock_graph = MagicMock(name="graph")
    mock_graph.compile.return_value = mock_compiled_graph

    compiled_graph = namespace["TextInputGraph"](mock_graph).compile()
    result = await compiled_graph.ainvoke("Suspicious email")

    assert result == {"assessment": "phishing"}
    mock_compiled_graph.ainvoke.assert_awaited_once_with({"input": "Suspicious email"})


def test_email_graph_loads_skill_instructions(tmp_path: Path):
    namespace = runpy.run_path(
        str(ROOT / "external" / "langgraph" / "examples" / "email_phishing.py")
    )
    skill_directory = tmp_path / "phishing-triage"
    skill_directory.mkdir()
    (skill_directory / "SKILL.md").write_text(
        "Treat credential requests as suspicious.", encoding="utf-8"
    )

    instructions = namespace["load_skill_instructions"]([str(skill_directory)])

    assert instructions == "Treat credential requests as suspicious."


async def test_runtime_compiles_factory_graph_once_and_invokes_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _stage_adapter(tmp_path)
    source_path = str(LANGGRAPH_SOURCE)
    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    monkeypatch.setenv(
        "PYTHONPATH",
        (
            source_path
            if not existing_pythonpath
            else f"{source_path}{os.pathsep}{existing_pythonpath}"
        ),
    )
    compiled_marker = tmp_path / "compiled"
    (tmp_path / "agent_graph.py").write_text(
        """
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class GraphState(TypedDict):
    user_id: str
    message: str
    answer: str


class CompileMarker:
    def __init__(self, graph, marker: str):
        self.graph = graph
        self.marker = marker

    def compile(self):
        marker = Path(self.marker)
        count = int(marker.read_text(encoding="utf-8")) if marker.exists() else 0
        marker.write_text(str(count + 1), encoding="utf-8")
        return self.graph.compile()


def build_graph(prefix: str):
    graph = StateGraph(GraphState)

    def answer(state: GraphState) -> dict[str, str]:
        return {"answer": f"{prefix}{state['message']}"}

    graph.add_node("answer", answer)
    graph.add_edge(START, "answer")
    graph.add_edge("answer", END)
    return CompileMarker(graph, __COMPILED_MARKER__)
""".replace("__COMPILED_MARKER__", repr(str(compiled_marker))),
        encoding="utf-8",
    )

    async with await Fabric().start_runtime(_config(), base_dir=tmp_path) as runtime:
        assert compiled_marker.read_text(encoding="utf-8") == "1"
        result = await runtime.invoke(input={"user_id": "user-42", "message": "hello"})
        second_result = await runtime.invoke(
            input={"user_id": "user-42", "message": "again"}
        )

    assert result["status"] == "succeeded"
    assert result.output == {
        "user_id": "user-42",
        "message": "hello",
        "answer": "Fabric: hello",
    }
    assert second_result.output == {
        "user_id": "user-42",
        "message": "again",
        "answer": "Fabric: again",
    }
    assert compiled_marker.read_text(encoding="utf-8") == "1"


async def test_runtime_translates_normalized_config_for_the_factory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _stage_adapter(tmp_path)
    source_path = str(LANGGRAPH_SOURCE)
    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    monkeypatch.setenv(
        "PYTHONPATH",
        (
            source_path
            if not existing_pythonpath
            else f"{source_path}{os.pathsep}{existing_pythonpath}"
        ),
    )
    captured_arguments = tmp_path / "captured-arguments.json"
    (tmp_path / "capturing_graph.py").write_text(
        """
import json
from pathlib import Path


class Graph:
    def compile(self):
        return self

    async def ainvoke(self, graph_input):
        return {"answer": graph_input}


def build_graph(
    *, model_config, system_instruction, mcp_servers, skill_paths, tool_names, marker: str
):
    Path(marker).write_text(
        json.dumps(
            {
                "model_config": model_config,
                "system_instruction": system_instruction,
                "mcp_servers": mcp_servers,
                "skill_paths": skill_paths,
                "tool_names": tool_names,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return Graph()
""",
        encoding="utf-8",
    )

    config = _normalized_config()
    config.workflow.settings["marker"] = str(captured_arguments)
    async with await Fabric().start_runtime(config, base_dir=tmp_path) as runtime:
        result = await runtime.invoke(input="hello")

    assert result.output == {"answer": "hello"}
    assert json.loads(captured_arguments.read_text(encoding="utf-8")) == {
        "mcp_servers": {
            "mcp_math": {
                "allowed_tools": ["multiply"],
                "blocked_tools": ["divide"],
                "exposure": "harness_native",
                "transport": "streamable-http",
                "url": "http://127.0.0.1:9901/mcp",
            }
        },
        "model_config": {
            "api_key_env": "NVIDIA_API_KEY",
            "model": "meta/llama-3.1-70b-instruct",
            "provider": "nim",
            "temperature": 0.25,
        },
        "skill_paths": [str(tmp_path / "skills")],
        "system_instruction": "Use the configured tools when useful.",
        "tool_names": ["mcp_math"],
    }


async def test_runtime_awaits_an_async_factory_and_clears_state_on_stop(tmp_path: Path):
    (tmp_path / "async_factory.py").write_text(
        """
class Graph:
    def compile(self):
        return self

    async def ainvoke(self, graph_input):
        return {"answer": graph_input}


async def build_graph():
    return Graph()
""",
        encoding="utf-8",
    )
    payload = _runtime_payload(
        _config(ref="async_factory:build_graph", settings={}), tmp_path
    )
    runtime = adapter.LangGraphRuntime()

    await runtime.start(payload)
    assert await runtime.invoke({"request": {"input": "first"}}) == {
        "answer": "first"
    }
    await runtime.stop()
    with pytest.raises(adapter.lifecycle.LifecycleError, match="not started"):
        await runtime.invoke({"request": {"input": "second"}})

    await runtime.start(payload)
    assert await runtime.invoke({"request": {"input": "third"}}) == {
        "answer": "third"
    }
    await runtime.stop()


async def test_runtime_isolates_factory_modules_by_base_dir(tmp_path: Path):
    first_base_dir = tmp_path / "first"
    second_base_dir = tmp_path / "second"
    for base_dir, source in ((first_base_dir, "first"), (second_base_dir, "second")):
        base_dir.mkdir()
        (base_dir / "shared_factory.py").write_text(
            f"""
class Graph:
    def compile(self):
        return self

    async def ainvoke(self, graph_input):
        return {{"source": "{source}"}}


def build_graph():
    return Graph()
""",
            encoding="utf-8",
        )

    config = _config(ref="shared_factory:build_graph", settings={})
    try:
        first_runtime = adapter.LangGraphRuntime()
        await first_runtime.start(_runtime_payload(config, first_base_dir))
        first_result = await first_runtime.invoke({"request": {"input": "ignored"}})
        await first_runtime.stop()

        second_runtime = adapter.LangGraphRuntime()
        await second_runtime.start(_runtime_payload(config, second_base_dir))
        second_result = await second_runtime.invoke({"request": {"input": "ignored"}})
        await second_runtime.stop()
    finally:
        sys.modules.pop("shared_factory", None)

    assert first_result == {"source": "first"}
    assert second_result == {"source": "second"}


async def test_runtime_normalizes_a_graph_invocation_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _stage_adapter(tmp_path)
    source_path = str(LANGGRAPH_SOURCE)
    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    monkeypatch.setenv(
        "PYTHONPATH",
        (
            source_path
            if not existing_pythonpath
            else f"{source_path}{os.pathsep}{existing_pythonpath}"
        ),
    )
    (tmp_path / "failing_graph.py").write_text(
        """
class Graph:
    def compile(self):
        return self

    async def ainvoke(self, graph_input):
        raise RuntimeError("sensitive graph detail")


def build_graph():
    return Graph()
""",
        encoding="utf-8",
    )

    result = await Fabric().run(
        _config(ref="failing_graph:build_graph", settings={}),
        base_dir=tmp_path,
        input="hello",
    )

    assert result["status"] == "failed"
    assert result.output == {
        "failed": True,
        "response": None,
        "error": {
            "code": "langgraph_graph_invoke_failed",
            "message": "Compiled graph failed during invocation",
            "retryable": False,
        },
    }
    assert result.error.to_mapping() == {
        "stage": "invoke",
        "code": "langgraph_graph_invoke_failed",
        "message": "Compiled graph failed during invocation",
        "retryable": False,
        "metadata": {},
    }
    assert "sensitive graph detail" not in json.dumps(result.to_mapping())


async def test_runtime_invokes_a_synchronous_compiled_graph(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _stage_adapter(tmp_path)
    source_path = str(LANGGRAPH_SOURCE)
    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    monkeypatch.setenv(
        "PYTHONPATH",
        (
            source_path
            if not existing_pythonpath
            else f"{source_path}{os.pathsep}{existing_pythonpath}"
        ),
    )
    (tmp_path / "sync_graph.py").write_text(
        """
class Graph:
    def compile(self):
        return CompiledGraph()


class CompiledGraph:
    def invoke(self, graph_input):
        return {"answer": graph_input["message"].upper()}


def build_graph(prefix: str):
    return Graph()
""",
        encoding="utf-8",
    )

    config = _config(ref="sync_graph:build_graph")
    async with await Fabric().start_runtime(config, base_dir=tmp_path) as runtime:
        result = await runtime.invoke(input={"message": "hello"})

    assert result["status"] == "succeeded"
    assert result.output == {"answer": "HELLO"}
