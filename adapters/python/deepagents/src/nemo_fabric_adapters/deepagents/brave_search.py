# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MCP server that exposes Brave web search to Deep Agents.

Declare it as a stdio MCP server that runs
``python -m nemo_fabric_adapters.deepagents.brave_search`` with
``BRAVE_API_KEY`` in its environment.
"""

import json
import os
import time

SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"
RESPONSE_LIMIT_BYTES = 1024 * 1024
TIMEOUT_SECONDS = 20
FAILURE = "Brave search failed; check the integration credential and service"


class _ResponseRejected(Exception):
    """The response exceeded its size or time limit."""


def web_search(query: str, count: int = 5) -> dict:
    """Search the web for current information and return Brave's results."""
    if (
        not query.strip()
        or len(query) > 2048
        or type(count) is not int
        or not 1 <= count <= 10
    ):
        raise ValueError("search requires a nonempty query and count between 1 and 10")
    key = os.environ.get("BRAVE_API_KEY")
    if not key:
        raise RuntimeError("Brave search is not configured; set BRAVE_API_KEY")
    import httpx

    # httpx timeouts bound inactivity; the deadline also bounds a response
    # that keeps trickling in.
    deadline = time.monotonic() + TIMEOUT_SECONDS
    try:
        with httpx.stream(
            "GET",
            SEARCH_URL,
            params={"q": query, "count": count},
            headers={"X-Subscription-Token": key, "Accept": "application/json"},
            timeout=TIMEOUT_SECONDS,
            follow_redirects=False,
        ) as response:
            response.raise_for_status()
            body = bytearray()
            for chunk in response.iter_bytes():
                if (
                    len(body) + len(chunk) > RESPONSE_LIMIT_BYTES
                    or time.monotonic() > deadline
                ):
                    raise _ResponseRejected
                body.extend(chunk)
        result = json.loads(body)
    except (httpx.HTTPError, ValueError, _ResponseRejected):
        # Error details can include the request URL or response text.
        raise RuntimeError(FAILURE) from None
    if not isinstance(result, dict):
        raise RuntimeError(FAILURE)
    return result


if __name__ == "__main__":
    from mcp.server.fastmcp import FastMCP

    server = FastMCP("fabric-brave")
    server.tool()(web_search)
    server.run()
