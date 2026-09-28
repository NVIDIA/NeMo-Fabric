# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The packaged Brave search MCP server bounds every request and hides failure details."""

import asyncio
import gzip
import importlib.util
import sys
from functools import partial

import httpx
import pytest
from nemo_fabric_adapters.deepagents import brave_search
from nemo_fabric_adapters.deepagents.brave_search import FAILURE, web_search

RESULTS = b'{"web": {"results": []}}'


def _response(status, content, headers=None):
    async def body():  # Streamed like a network body, so raw reads are possible.
        yield content

    return httpx.Response(status, headers=headers, content=body())


@pytest.fixture(name="brave")
def brave_fixture(monkeypatch):
    """Serve Brave responses from a handler and record each request."""

    server = {"requests": [], "respond": lambda request: _response(200, RESULTS)}

    def handle(request):
        server["requests"].append(request)
        return server["respond"](request)

    monkeypatch.setenv("BRAVE_API_KEY", "placeholder")
    transport = httpx.MockTransport(handle)
    monkeypatch.setattr(
        httpx, "AsyncClient", partial(httpx.AsyncClient, transport=transport)
    )
    return server


async def test_a_search_sends_the_credential_and_returns_brave_results(brave):
    assert await web_search("question", 3) == {"web": {"results": []}}
    request = brave["requests"][0]
    assert request.headers["X-Subscription-Token"] == "placeholder"
    assert request.headers["Accept-Encoding"] == "identity"
    assert request.url.params["count"] == "3"


@pytest.mark.parametrize(
    ("status", "content", "headers"),
    [
        (401, b"secret response", {}),
        (302, b"", {"location": "https://elsewhere.example"}),
        (200, b"x" * (brave_search.RESPONSE_LIMIT_BYTES + 1), {}),
        (200, gzip.compress(RESULTS), {"content-encoding": "gzip"}),
        (200, b"[]", {}),
        (200, b"not json", {}),
    ],
    ids=["http-error", "redirect", "too-large", "encoded", "not-an-object", "not-json"],
)
async def test_rejected_responses_report_only_a_generic_failure(
    brave, status, content, headers
):
    brave["respond"] = lambda request: _response(status, content, headers)
    with pytest.raises(RuntimeError) as raised:
        await web_search("question")
    assert str(raised.value) == FAILURE
    assert len(brave["requests"]) == 1, "redirects are not followed"


async def test_a_blocked_response_stops_at_the_total_deadline(brave, monkeypatch):
    async def trickle():
        yield b'{"web":'
        await asyncio.sleep(1)
        yield b"{}}"

    brave["respond"] = lambda request: httpx.Response(200, content=trickle())
    monkeypatch.setattr(brave_search, "TIMEOUT_SECONDS", 0.1)
    with pytest.raises(RuntimeError, match="Brave search failed"):
        await asyncio.wait_for(web_search("question"), 1)


async def test_invalid_input_and_a_missing_credential_fail_without_a_request(
    brave, monkeypatch
):
    for query, count in [("", 5), ("x", 0), ("x", True), ("x" * 2049, 1)]:
        with pytest.raises(ValueError):
            await web_search(query, count)
    monkeypatch.setenv("BRAVE_API_KEY", "")
    with pytest.raises(RuntimeError, match="BRAVE_API_KEY"):
        await web_search("question")
    assert brave["requests"] == []


@pytest.mark.skipif(
    not importlib.util.find_spec("langchain_mcp_adapters"),
    reason="requires Deep Agents",
)
async def test_deep_agents_runs_the_search_tool_over_stdio_without_network():
    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient(
        {
            "brave": {
                "transport": "stdio",
                "command": sys.executable,
                "args": ["-m", "nemo_fabric_adapters.deepagents.brave_search"],
                "env": {"BRAVE_API_KEY": ""},
            }
        }
    )
    [tool] = await client.get_tools()
    assert tool.name == "web_search" and "query" in tool.args
    # Without a credential the tool fails before any network request.
    with pytest.raises(Exception, match="BRAVE_API_KEY"):
        await tool.ainvoke({"query": "current events"})
