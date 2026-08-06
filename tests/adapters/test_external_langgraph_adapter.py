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
from nemo_fabric import MetadataConfig
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


def _config(*, ref: str = "agent_graph:build_graph") -> FabricConfig:
    return FabricConfig(
        metadata=MetadataConfig(name="langgraph-reference-test"),
        harness=HarnessConfig(
            adapter_id="example.fabric.langgraph",
            resolution="preinstalled",
        ),
        workflow=WorkflowConfig(
            entrypoint=WorkflowEntrypointConfig(kind="langgraph_factory", ref=ref),
            settings={"prefix": "Fabric: "},
        ),
    )


def test_descriptor_declares_the_source_reference_contract():
    descriptor = json.loads(DESCRIPTOR.read_text(encoding="utf-8"))

    assert descriptor["adapter_id"] == "example.fabric.langgraph"
    assert descriptor["harness"] == "langgraph"
    assert descriptor["adapter_kind"] == "python"
    assert descriptor["runner"] == {"module": "nemo_fabric_adapters.langgraph.adapter"}
    assert descriptor["requirements"] == {}
    assert descriptor["config"]["accepts"] == []
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
