# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stdio MCP fixture with observable tool execution and a private response token."""

import argparse
from pathlib import Path

from mcp.server.fastmcp import FastMCP


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--marker", type=Path, required=True)
    parser.add_argument("--token", required=True)
    args = parser.parse_args()
    server = FastMCP("fabric-probe")

    @server.tool()
    def probe() -> str:
        """Record a probe call and return its verification token."""
        args.marker.write_text(args.token, encoding="utf-8")
        return args.token

    server.run(transport="stdio")


if __name__ == "__main__":
    main()
