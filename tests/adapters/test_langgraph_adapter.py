# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end coverage for the LangGraph graph-factory adapter."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from nemo_fabric import Fabric
from nemo_fabric import FabricConfig
from nemo_fabric import FabricConfigError
from nemo_fabric import HarnessConfig
from nemo_fabric import MetadataConfig
from nemo_fabric import WorkflowConfig
from nemo_fabric import WorkflowEntrypointConfig


ROOT = Path(__file__).resolve().parents[2]
DESCRIPTOR = ROOT / "adapters" / "langgraph" / "fabric-adapter.json"


def _stage_adapter(base_dir: Path) -> None:
    destination = base_dir / "adapters" / "langgraph" / "fabric-adapter.json"
    destination.parent.mkdir(parents=True)
    shutil.copyfile(DESCRIPTOR, destination)


def _config(*, ref: str = "agent_graph:build_graph") -> FabricConfig:
    return FabricConfig(
        metadata=MetadataConfig(name="langgraph-test"),
        harness=HarnessConfig(
            adapter_id="nvidia.fabric.langgraph",
            resolution="preinstalled",
        ),
        workflow=WorkflowConfig(
            entrypoint=WorkflowEntrypointConfig(kind="langgraph_factory", ref=ref),
            settings={"prefix": "Fabric: "},
        ),
    )


def test_plan_requires_a_langgraph_factory_reference(tmp_path: Path):
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


async def test_runtime_compiles_factory_graph_once_and_invokes_it(tmp_path: Path):
    _stage_adapter(tmp_path)
    compiled_marker = tmp_path / "compiled"
    (tmp_path / "agent_graph.py").write_text(
        """
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class GraphState(TypedDict):
    input: str
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
        return {"answer": f"{prefix}{state['input']}"}

    graph.add_node("answer", answer)
    graph.add_edge(START, "answer")
    graph.add_edge("answer", END)
    return CompileMarker(graph, __COMPILED_MARKER__)
""".replace("__COMPILED_MARKER__", repr(str(compiled_marker))),
        encoding="utf-8",
    )

    async with await Fabric().start_runtime(_config(), base_dir=tmp_path) as runtime:
        assert compiled_marker.read_text(encoding="utf-8") == "1"
        result = await runtime.invoke(input={"input": "hello"})
        second_result = await runtime.invoke(input={"input": "again"})

    assert result["status"] == "succeeded"
    assert result.output == {"input": "hello", "answer": "Fabric: hello"}
    assert second_result.output == {"input": "again", "answer": "Fabric: again"}
    assert compiled_marker.read_text(encoding="utf-8") == "1"
