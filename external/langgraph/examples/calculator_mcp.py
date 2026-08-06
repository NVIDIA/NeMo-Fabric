# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Calculator MCP server for the LangGraph reference example."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP


server = FastMCP("mcp_math", port=9901)


@server.tool()
def add(left: float, right: float) -> float:
    """Add two numbers."""

    return left + right


@server.tool()
def subtract(left: float, right: float) -> float:
    """Subtract the right number from the left number."""

    return left - right


@server.tool()
def multiply(left: float, right: float) -> float:
    """Multiply two numbers."""

    return left * right


@server.tool()
def divide(left: float, right: float) -> float:
    """Divide the left number by the right number."""

    if right == 0:
        raise ValueError("cannot divide by zero")
    return left / right


if __name__ == "__main__":
    server.run(transport="streamable-http")
