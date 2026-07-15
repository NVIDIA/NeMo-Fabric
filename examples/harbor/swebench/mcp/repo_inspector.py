# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A dependency-free, read-only MCP server for the Harbor capability example."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

SUPPORTED_PROTOCOL_VERSIONS = ("2025-03-26",)
UNSUPPORTED_PROTOCOL_VERSION = -32004
JsonRpcId = str | int | float | None


def request_id_from(request: object) -> JsonRpcId:
    if not isinstance(request, dict) or "id" not in request:
        return None
    request_id = request["id"]
    if request_id is None or isinstance(request_id, str):
        return request_id
    if isinstance(request_id, (int, float)) and not isinstance(request_id, bool):
        return request_id
    return None


def error_response(
    request_id: JsonRpcId,
    code: int,
    message: str,
    *,
    data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    error: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return {"jsonrpc": "2.0", "id": request_id, "error": error}


def reply(request: object) -> dict[str, Any] | None:
    request_id = request_id_from(request)
    if not isinstance(request, dict):
        return error_response(None, -32600, "invalid request")
    method = request.get("method")
    if (
        request.get("jsonrpc") != "2.0"
        or not isinstance(method, str)
        or ("id" in request and request_id is None and request["id"] is not None)
    ):
        return error_response(request_id, -32600, "invalid request")
    if "id" not in request:
        return None
    if method == "initialize":
        params = request.get("params")
        requested_version = params.get("protocolVersion") if isinstance(params, dict) else None
        if requested_version not in SUPPORTED_PROTOCOL_VERSIONS:
            return error_response(
                request_id,
                UNSUPPORTED_PROTOCOL_VERSION,
                "UNSUPPORTED_PROTOCOL_VERSION",
                data={
                    "requested": requested_version,
                    "supported": list(SUPPORTED_PROTOCOL_VERSIONS),
                },
            )
        result = {
            "protocolVersion": requested_version,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "fabric-repo-inspector", "version": "0.1.0"},
        }
    elif method == "tools/list":
        result = {
            "tools": [
                {
                    "name": "repo_summary",
                    "description": "Summarize the current repository without modifying it.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                }
            ]
        }
    elif method == "tools/call":
        params = request.get("params")
        if not isinstance(params, dict):
            return error_response(request_id, -32602, "tools/call params must be an object")
        name = params.get("name")
        arguments = params.get("arguments", {})
        if name != "repo_summary":
            return error_response(
                request_id,
                -32602,
                "unknown tool",
                data={"requested": name, "supported": ["repo_summary"]},
            )
        if not isinstance(arguments, dict) or arguments:
            return error_response(
                request_id,
                -32602,
                "repo_summary arguments must be an empty object",
            )
        root = Path("/testbed")
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
        return error_response(request_id, -32601, f"unknown method: {method}")
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def main() -> None:
    for line in sys.stdin:
        request_id = None
        try:
            request = json.loads(line)
            request_id = request_id_from(request)
            response = reply(request)
            if response is not None:
                print(json.dumps(response), flush=True)
        except json.JSONDecodeError as error:
            print(json.dumps(error_response(None, -32700, str(error))), flush=True)
        except Exception as error:  # noqa: BLE001 - JSON-RPC process boundary
            print(json.dumps(error_response(request_id, -32603, str(error))), flush=True)


if __name__ == "__main__":
    main()
