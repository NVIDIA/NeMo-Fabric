# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import gzip
import importlib.util
import sys
import unittest
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import patch

from nemo_fabric_adapters.deepagents import brave_search
from nemo_fabric_adapters.deepagents.brave_search import web_search


class FakeHTTPError(Exception):
    pass


class FakeResponse:
    def __init__(self, chunks, encoding=None, delay=0.0):
        self.chunks = chunks
        self.headers = {} if encoding is None else {"content-encoding": encoding}
        self.delay = delay
        self.error = None

    def raise_for_status(self):
        if self.error is not None:
            raise self.error

    async def aiter_raw(self):
        for chunk in self.chunks:
            if self.delay:
                await asyncio.sleep(self.delay)
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk


class BraveSearch(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.response = FakeResponse([b'{"web":{"results":[]}}'])
        self.requests = []
        test = self

        class AsyncClient:
            def __init__(self, **options):
                test.requests.append({"client": options})

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                return False

            @asynccontextmanager
            async def stream(self, method, url, **options):
                test.requests[-1].update(options, method=method, url=url)
                yield test.response

        httpx = SimpleNamespace(AsyncClient=AsyncClient, HTTPError=FakeHTTPError)
        for patcher in (
            patch.dict("os.environ", {"BRAVE_API_KEY": "placeholder"}),
            patch.dict("sys.modules", {"httpx": httpx}),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    async def test_request_is_bounded_and_credentials_are_not_error_details(self):
        self.assertEqual(await web_search("question"), {"web": {"results": []}})
        request = self.requests[-1]
        self.assertFalse(request["client"]["follow_redirects"])
        self.assertEqual(request["headers"]["X-Subscription-Token"], "placeholder")
        self.assertEqual(request["headers"]["Accept-Encoding"], "gzip")
        for chunks in ([b"x" * (1024 * 1024 + 1)], [b"[]"], [b"not json"]):
            self.response = FakeResponse(chunks)
            with self.assertRaisesRegex(RuntimeError, "Brave search failed"):
                await web_search("question")
        self.response = FakeResponse([b"{}"])
        self.response.error = FakeHTTPError("secret response")
        with self.assertRaisesRegex(
            RuntimeError,
            "^Brave search failed; check the integration credential and service$",
        ):
            await web_search("question")
        for query, count in [("", 5), ("x", 0), ("x", True), ("x" * 2049, 1)]:
            with self.assertRaises(ValueError):
                await web_search(query, count)

    async def test_compressed_responses_are_limited_after_decompression(self):
        body = b'{"web":{"results":[{"title":"a"}]}}'
        self.response = FakeResponse([gzip.compress(body)], encoding="gzip")
        self.assertEqual(await web_search("question"), {"web": {"results": [{"title": "a"}]}})
        bomb = gzip.compress(b" " * (8 * 1024 * 1024))
        self.assertLess(len(bomb), brave_search.RESPONSE_LIMIT_BYTES)
        self.response = FakeResponse([bomb], encoding="gzip")
        with self.assertRaisesRegex(RuntimeError, "Brave search failed"):
            await web_search("question")
        self.response = FakeResponse([body], encoding="br")
        with self.assertRaisesRegex(RuntimeError, "Brave search failed"):
            await web_search("question")

    async def test_a_blocked_response_stops_at_the_total_deadline(self):
        self.response = FakeResponse([b'{"web":', b'{"results":[]}}'], delay=0.2)
        with patch.object(brave_search, "TIMEOUT_SECONDS", 0.1):
            with self.assertRaisesRegex(RuntimeError, "Brave search failed"):
                await asyncio.wait_for(web_search("question"), 1)

    async def test_unexpected_errors_are_not_reported_as_search_failures(self):
        self.response = FakeResponse([TypeError("adapter bug")])
        with self.assertRaisesRegex(TypeError, "adapter bug"):
            await web_search("question")

    async def test_missing_credential_is_reported_without_a_request(self):
        with patch.dict("os.environ", {"BRAVE_API_KEY": ""}):
            with self.assertRaisesRegex(RuntimeError, "BRAVE_API_KEY"):
                await web_search("question")
        self.assertEqual(self.requests, [])


@unittest.skipUnless(
    importlib.util.find_spec("langchain_mcp_adapters"), "requires Deep Agents image"
)
class NativeSearchDiscovery(unittest.IsolatedAsyncioTestCase):
    async def test_native_mcp_client_runs_search_without_inference_or_network(self):
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
        tools = await client.get_tools()
        self.assertEqual([tool.name for tool in tools], ["web_search"])
        self.assertIn("query", tools[0].args)
        # Without a credential the tool fails before any network request.
        with self.assertRaisesRegex(Exception, "BRAVE_API_KEY"):
            await tools[0].ainvoke({"query": "current events"})
