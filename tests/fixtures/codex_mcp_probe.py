# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP


marker = Path(sys.argv[1])
server = FastMCP("fabric-codex-probe")


@server.tool()
def mark() -> str:
    """Create the configured marker file."""

    marker.touch()
    return "marker created"


if __name__ == "__main__":
    server.run()
