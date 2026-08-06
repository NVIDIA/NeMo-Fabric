<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Fabric LangGraph Adapter

The LangGraph adapter starts a user-owned graph from a Python factory. It
validates the static factory reference while planning, then imports the factory
and calls `graph.compile()` during runtime start. The compiled graph remains
owned by that runtime and receives the raw NeMo Fabric request input on every
invocation.

## Install

| Installation | Runtime | Adapter | LangGraph |
| --- | --- | --- | --- |
| `pip install "nemo-fabric[langgraph]"` | Yes | Yes | Yes |
| `pip install "nemo-fabric-adapters-langgraph[harness]"` | No | Yes | Yes |
| `pip install nemo-fabric-adapters-langgraph` | No | Yes | No |

The adapter supports `langgraph>=1.2,<2.0`. For a split runtime and adapter
environment, set `ADAPTER_PYTHON` to the adapter environment's interpreter.

## Configure a Graph Factory

`workflow.entrypoint` is required. Use `kind: langgraph_factory` and set `ref`
to an importable Python factory in `module:attribute` form. The adapter imports
the module from the agent's `base_dir`; `workflow.settings` are passed to the
factory as keyword arguments.

```python
from nemo_fabric import FabricConfig
from nemo_fabric import HarnessConfig
from nemo_fabric import MetadataConfig
from nemo_fabric import WorkflowConfig
from nemo_fabric import WorkflowEntrypointConfig

config = FabricConfig(
    metadata=MetadataConfig(name="review-agent"),
    harness=HarnessConfig(
        adapter_id="nvidia.fabric.langgraph",
        resolution="preinstalled",
    ),
    workflow=WorkflowConfig(
        entrypoint=WorkflowEntrypointConfig(
            kind="langgraph_factory",
            ref="my_agent.graph:build_graph",
        ),
        settings={"prefix": "Review: "},
    ),
)
```

The selected `build_graph` factory must accept the configured keyword arguments
and return an uncompiled graph with a callable `compile()` method. Its
`compile()` result must provide `ainvoke(input)` or `invoke(input)`. For
example:

```python
from typing import TypedDict

from langgraph.graph import END
from langgraph.graph import START
from langgraph.graph import StateGraph


class GraphState(TypedDict):
    input: str
    answer: str


def build_graph(prefix: str):
    graph = StateGraph(GraphState)

    def answer(state: GraphState) -> dict[str, str]:
        return {"answer": f"{prefix}{state['input']}"}

    graph.add_node("answer", answer)
    graph.add_edge(START, "answer")
    graph.add_edge("answer", END)
    return graph
```

Planning checks the entry-point kind and `module:attribute` syntax without
importing the application module. Runtime start imports the factory, forwards
the settings, compiles the graph once, and fails before accepting an invocation
if construction or compilation fails.

## Current Capability Boundary

This adapter establishes the graph factory lifecycle only. It does not yet map
NeMo Fabric models, instructions, tools, MCP servers, skills, or telemetry into
the graph factory. The descriptor therefore rejects those normalized mappings
during planning instead of silently ignoring them. Graph topology, state,
checkpointers, and application resources remain owned by the graph factory.
