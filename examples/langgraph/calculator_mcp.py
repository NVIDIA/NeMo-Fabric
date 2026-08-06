# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A per-user LangGraph ReAct agent with local and MCP calculator tools."""

from __future__ import annotations

import argparse
import asyncio
import os
from collections.abc import Callable
from typing import Any

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver

from examples.langgraph.config import LangGraphExampleConfig
from examples.langgraph.config import McpServerConfig
from examples.langgraph.config import build_nim_chat_model
from examples.langgraph.config import load_config


@tool
def current_timezone() -> str:
    """Return the server's configured IANA time zone, or ``UTC`` when unset."""

    return os.environ.get("TZ", "UTC")


def mcp_connection(server: McpServerConfig) -> dict[str, str]:
    """Translate the example's hyphenated transport into LangChain MCP syntax."""

    return {"transport": "streamable_http", "url": str(server.url)}


def _selected_mcp_tools(
    tools: list[Any], server: McpServerConfig
) -> list[Any]:
    if not server.include:
        return tools
    by_name = {item.name: item for item in tools}
    missing = set(server.include) - set(by_name)
    if missing:
        raise RuntimeError(
            "MCP server did not expose configured tool(s): " + ", ".join(sorted(missing))
        )
    return [by_name[name] for name in server.include]


class PerUserReactAgent:
    """Create an isolated LangGraph, MCP client, and checkpoint store per user."""

    def __init__(
        self,
        config: LangGraphExampleConfig,
        *,
        model_factory: Callable[[Any], Any] = build_nim_chat_model,
        mcp_client_factory: Callable[[dict[str, Any]], Any] = MultiServerMCPClient,
        graph_factory: Callable[..., Any] = create_agent,
    ) -> None:
        if config.workflow.entrypoint != "langgraph:per_user_react_agent":
            raise ValueError("calculator example requires langgraph:per_user_react_agent")
        if config.mcp is None or "mcp_math" not in config.mcp.servers:
            raise ValueError("calculator example requires mcp.servers.mcp_math")
        if "current_timezone" not in config.tools:
            raise ValueError("calculator example requires tools.current_timezone")

        self._config = config
        self._model_factory = model_factory
        self._mcp_client_factory = mcp_client_factory
        self._graph_factory = graph_factory
        self._sessions: dict[str, Any] = {}

    async def graph_for(self, user_id: str) -> Any:
        """Return the user-owned graph, creating it and its MCP client on first use."""

        if not user_id.strip():
            raise ValueError("user_id must be a non-empty string")
        graph = self._sessions.get(user_id)
        if graph is not None:
            return graph

        server = self._config.mcp.servers["mcp_math"]  # validated in __init__
        client = self._mcp_client_factory(
            {"mcp_math": mcp_connection(server)}, tool_name_prefix=False
        )
        mcp_tools = _selected_mcp_tools(list(await client.get_tools()), server)
        model = self._model_factory(self._config.selected_model())
        graph = self._graph_factory(
            model,
            [current_timezone, *mcp_tools],
            checkpointer=InMemorySaver(),
            debug=bool(self._config.workflow.settings.get("verbose", False)),
            name="per_user_calculator",
        )
        self._sessions[user_id] = graph
        return graph

    async def ainvoke(self, user_id: str, message: str) -> dict[str, Any]:
        """Run a message in the graph and persisted conversation for ``user_id``."""

        graph = await self.graph_for(user_id)
        return await graph.ainvoke(
            {"messages": [{"role": "user", "content": message}]},
            {"configurable": {"thread_id": user_id}},
        )


def build_per_user_react_agent(config: LangGraphExampleConfig) -> PerUserReactAgent:
    """Build the calculator example's declared LangGraph factory entry point."""

    return PerUserReactAgent(config)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="examples/langgraph/configs/calculator_mcp.yaml")
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--input", required=True)
    return parser.parse_args()


async def _run() -> None:
    args = _parse_args()
    graph = build_per_user_react_agent(load_config(args.config))
    result = await graph.ainvoke(args.user_id, args.input)
    print(result["messages"][-1].content)


def main() -> None:
    """Run the calculator example from the command line."""

    asyncio.run(_run())


if __name__ == "__main__":
    main()
