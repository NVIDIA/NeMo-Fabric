# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A local streamable-HTTP MCP server for the calculator example."""

from __future__ import annotations

import argparse

from mcp.server.fastmcp import FastMCP


def create_server(*, host: str = "127.0.0.1", port: int = 9901) -> FastMCP:
    """Create an MCP server with the four calculator tools used by the example."""

    server = FastMCP("NeMo Fabric calculator example", host=host, port=port)

    @server.tool()
    def calculator__add(left: float, right: float) -> float:
        """Add two numbers."""

        return left + right

    @server.tool()
    def calculator__subtract(left: float, right: float) -> float:
        """Subtract ``right`` from ``left``."""

        return left - right

    @server.tool()
    def calculator__multiply(left: float, right: float) -> float:
        """Multiply two numbers."""

        return left * right

    @server.tool()
    def calculator__divide(left: float, right: float) -> float:
        """Divide ``left`` by ``right``."""

        if right == 0:
            raise ValueError("right must not be zero")
        return left / right

    return server


def main() -> None:
    """Run the server with streamable HTTP transport."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=9901, type=int)
    args = parser.parse_args()
    create_server(host=args.host, port=args.port).run(transport="streamable-http")


if __name__ == "__main__":
    main()
