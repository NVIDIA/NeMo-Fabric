# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal stdio MCP server for Hermes adapter integration tests."""

from mcp.server import MCPServer


server = MCPServer("fabric-time-test")


@server.tool()
def get_current_time(timezone: str) -> str:
    return f"timezone={timezone}"


if __name__ == "__main__":
    server.run(transport="stdio")
