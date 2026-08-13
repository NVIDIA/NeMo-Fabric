# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a streamable-HTTP MCP server that logs every request header."""

from __future__ import annotations

import argparse
from collections.abc import Awaitable, Callable
import json
from pathlib import Path

import uvicorn
from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import Receive, Scope, Send


AsgiApp = Callable[[Scope, Receive, Send], Awaitable[None]]


class PrintHeadersMiddleware:
    """Print and optionally log HTTP request headers before forwarding."""

    def __init__(self, app: AsgiApp, log_requests: Path | None = None) -> None:
        self.app = app
        self.log_requests = log_requests

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            print(f"> {scope['method']} {scope['path']}", flush=True)
            headers = {
                name.decode("latin-1"): value.decode("latin-1")
                for name, value in scope["headers"]
            }
            for name, value in headers.items():
                print(
                    f"{name}: {value}",
                    flush=True,
                )
            print(flush=True)
            if self.log_requests is not None:
                request = {
                    "method": scope["method"],
                    "path": scope["path"],
                    "headers": headers,
                }
                with self.log_requests.open("a", encoding="utf-8") as stream:
                    json.dump(request, stream)
                    stream.write("\n")
        await self.app(scope, receive, send)


server = FastMCP("header-printer", stateless_http=True, json_response=True)


@server.custom_route("/health", methods=["GET"])
async def health(_request: Request) -> JSONResponse:
    """Report that the MCP server is ready to receive requests."""

    return JSONResponse({"status": "ok"})


@server.tool()
def ping() -> str:
    """Return a simple response to verify the MCP connection."""

    return "pong"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an MCP server that prints all HTTP request headers."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--log-requests",
        type=Path,
        help="Append HTTP requests to this JSONL file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = PrintHeadersMiddleware(
        server.streamable_http_app(), log_requests=args.log_requests
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
