# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A dependency-free, read-only MCP server for the Harbor capability example."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def reply(request: dict[str, Any]) -> dict[str, Any] | None:
    request_id = request.get("id")
    method = request.get("method")
    if request_id is None:
        return None
    if method == "initialize":
        result = {
            "protocolVersion": "2025-03-26",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "fabric-repo-inspector", "version": "0.1.0"},
        }
    elif method == "tools/list":
        result = {
            "tools": [
                {
                    "name": "repo_summary",
                    "description": "Summarize the current repository without modifying it.",
                    "inputSchema": {"type": "object", "properties": {}},
                }
            ]
        }
    elif method == "tools/call":
        root = Path("/app")
        names = sorted(path.name for path in root.iterdir())[:50]
        result = {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps({"root": str(root), "entries": names}),
                }
            ]
        }
    else:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32601, "message": f"unknown method: {method}"},
        }
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def main() -> None:
    for line in sys.stdin:
        try:
            response = reply(json.loads(line))
            if response is not None:
                print(json.dumps(response), flush=True)
        except Exception as error:  # noqa: BLE001 - JSON-RPC process boundary
            print(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {"code": -32603, "message": str(error)},
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
