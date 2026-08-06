# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Application-owned graph factory for the calculator example."""

from __future__ import annotations

from typing import Any
from typing import TypedDict

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import END
from langgraph.graph import START
from langgraph.graph import StateGraph


class CalculatorState(TypedDict):
    """Per-invocation calculator state."""

    user_id: str
    operation: str
    left: float
    right: float
    answer: Any


async def build_graph(mcp_url: str) -> StateGraph[CalculatorState]:
    """Return an uncompiled graph that calls the configured calculator server."""

    client = MultiServerMCPClient(
        {
            "mcp_math": {
                "transport": "streamable-http",
                "url": mcp_url,
            }
        }
    )

    async def calculate(state: CalculatorState) -> dict[str, Any]:
        tools = await client.get_tools(server_name="mcp_math")
        tool_by_name = {tool.name: tool for tool in tools}
        tool = tool_by_name.get(state["operation"])
        if tool is None:
            raise ValueError(f"Unsupported calculator operation {state['operation']!r}")
        answer = await tool.ainvoke({"left": state["left"], "right": state["right"]})
        return {"answer": answer}

    graph = StateGraph(CalculatorState)
    graph.add_node("calculate", calculate)
    graph.add_edge(START, "calculate")
    graph.add_edge("calculate", END)
    return graph
