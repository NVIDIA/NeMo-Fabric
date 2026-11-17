# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Controlled source reader for real code-review deadline qualification."""

import argparse
import asyncio
from pathlib import Path

try:
    from mcp.server import MCPServer
except ImportError:
    from mcp.server.fastmcp import FastMCP as MCPServer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--delay", type=float, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--token", required=True)
    args = parser.parse_args()
    server = MCPServer("deadline-review")

    @server.tool()
    async def read_review_source(part: int) -> str:
        """Read calculator.py in two sequential parts. Read part 1, then part 2."""
        if part not in (1, 2):
            raise ValueError("part must be 1 or 2")
        if part == 2 and not args.receipt.exists():
            raise ValueError("Read part 1 and wait for its response before part 2")
        await asyncio.sleep(args.delay)
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        with args.receipt.open("a") as receipt:
            receipt.write(f"{part}\n")
        if part == 1:
            return "Source header received. Now call read_review_source with part=2."
        return f"{args.source.read_text()}\nReview receipt: {args.token}"

    server.run(transport="stdio")


if __name__ == "__main__":
    main()
