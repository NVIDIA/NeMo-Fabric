# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MCP server that exposes Brave web search to Deep Agents.

Declare it as a stdio MCP server that runs
``python -m nemo_fabric_adapters.deepagents.brave_search`` with
``BRAVE_API_KEY`` in its environment.
"""

import asyncio
import json
import os
import zlib

SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"
RESPONSE_LIMIT_BYTES = 1024 * 1024
TIMEOUT_SECONDS = 20
FAILURE = "Brave search failed; check the integration credential and service"


class _ResponseRejected(Exception):
    """The response is too large or uses an unsupported encoding."""


class _BoundedBody:
    """Collect a response body, decompressing gzip, up to the size limit.

    The limit applies to decompressed bytes, so a small compressed response
    cannot expand past it.
    """

    def __init__(self, encoding: str) -> None:
        if encoding not in ("", "identity", "gzip"):
            raise _ResponseRejected
        self._decoder = (
            zlib.decompressobj(16 + zlib.MAX_WBITS) if encoding == "gzip" else None
        )
        self.data = bytearray()
        self._received = 0

    def add(self, chunk: bytes) -> None:
        self._received += len(chunk)
        room = RESPONSE_LIMIT_BYTES - len(self.data)
        if self._received > RESPONSE_LIMIT_BYTES:
            raise _ResponseRejected
        if self._decoder is None:
            self.data.extend(chunk)
            return
        if self._decoder.eof:
            raise _ResponseRejected  # Data after the end of the gzip stream.
        data = self._decoder.decompress(chunk, room + 1)
        if len(data) > room or self._decoder.unconsumed_tail or self._decoder.unused_data:
            raise _ResponseRejected
        self.data.extend(data)

    def finish(self) -> bytes:
        if self._decoder is not None and not self._decoder.eof:
            raise _ResponseRejected  # Truncated gzip stream.
        return bytes(self.data)


async def web_search(query: str, count: int = 5) -> dict:
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

    try:
        # The total deadline also cancels a read that is blocked waiting for data.
        async with asyncio.timeout(TIMEOUT_SECONDS):
            async with httpx.AsyncClient(
                timeout=TIMEOUT_SECONDS, follow_redirects=False
            ) as client:
                async with client.stream(
                    "GET",
                    SEARCH_URL,
                    params={"q": query, "count": count},
                    headers={
                        "X-Subscription-Token": key,
                        "Accept": "application/json",
                        "Accept-Encoding": "gzip",
                    },
                ) as response:
                    response.raise_for_status()
                    body = _BoundedBody(
                        response.headers.get("content-encoding", "").strip().lower()
                    )
                    async for chunk in response.aiter_raw():
                        body.add(chunk)
        result = json.loads(body.finish())
    except (httpx.HTTPError, ValueError, TimeoutError, zlib.error, _ResponseRejected):
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
