# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a per-user calculator graph backed by a Streamable HTTP MCP server."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any
from typing import TypedDict

from nemo_fabric import Fabric
from nemo_fabric import FabricConfig
from nemo_fabric import HarnessConfig
from nemo_fabric import MetadataConfig
from nemo_fabric import RuntimeConfig
from nemo_fabric import WorkflowConfig
from nemo_fabric import WorkflowEntrypointConfig


class CalculatorState(TypedDict):
    """Per-invocation calculator state."""

    user_id: str
    operation: str
    left: float
    right: float
    answer: Any


async def build_graph(mcp_url: str) -> Any:
    """Return an uncompiled graph that calls the configured calculator server."""

    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langgraph.graph import END
    from langgraph.graph import START
    from langgraph.graph import StateGraph

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


def build_config(mcp_url: str) -> FabricConfig:
    """Build the calculator graph configuration."""

    return FabricConfig(
        metadata=MetadataConfig(
            name="langgraph-calculator",
            description="Runs calculator operations for an application user.",
        ),
        harness=HarnessConfig(
            adapter_id="example.fabric.langgraph",
            resolution="preinstalled",
        ),
        workflow=WorkflowConfig(
            entrypoint=WorkflowEntrypointConfig(
                kind="langgraph_factory",
                ref="calculator:build_graph",
            ),
            settings={"mcp_url": mcp_url},
        ),
        runtime=RuntimeConfig(input_schema="json", output_schema="json"),
    )


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--mcp-url", default="http://127.0.0.1:9901/mcp")
    parser.add_argument(
        "--input",
        default='{"user_id":"user-42","operation":"multiply","left":21,"right":2}',
    )
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()

    fabric = Fabric()
    config = build_config(args.mcp_url)
    output = (
        fabric.plan(config, base_dir=args.base_dir)
        if args.plan
        else await fabric.run(
            config,
            base_dir=args.base_dir,
            input=json.loads(args.input),
        )
    )
    print(json.dumps(output.to_mapping(), indent=2))


if __name__ == "__main__":
    asyncio.run(main())
