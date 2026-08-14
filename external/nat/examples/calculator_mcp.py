# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Self-contained calculator MCP server for the NeMo Agent Toolkit reference example."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

server = FastMCP("calculator")


@server.tool()
def add(left: float, right: float) -> float:
    """Add two numbers."""

    return left + right


@server.tool()
def subtract(left: float, right: float) -> float:
    """Subtract the right value from the left value."""

    return left - right


@server.tool()
def multiply(left: float, right: float) -> float:
    """Multiply two numbers."""

    return left * right


@server.tool()
def divide(left: float, right: float) -> float:
    """Divide the left value by the right value."""

    if right == 0:
        raise ValueError("cannot divide by zero")
    return left / right


if __name__ == "__main__":
    server.run(transport="stdio")
