# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from nemo_fabric_adapters.deepagents.brave_search import web_search


class FakeHTTPError(Exception):
    pass


class BraveSearch(unittest.TestCase):
    def setUp(self):
        self.response = MagicMock()
        self.response.iter_bytes.return_value = [b'{"web":{"results":[]}}']
        self.stream = MagicMock()
        self.stream.return_value.__enter__.return_value = self.response
        httpx = SimpleNamespace(stream=self.stream, HTTPError=FakeHTTPError)
        for patcher in (
            patch.dict("os.environ", {"BRAVE_API_KEY": "placeholder"}),
            patch.dict("sys.modules", {"httpx": httpx}),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_request_is_bounded_and_credentials_are_not_error_details(self):
        self.assertEqual(web_search("question"), {"web": {"results": []}})
        self.assertFalse(self.stream.call_args.kwargs["follow_redirects"])
        self.assertEqual(
            self.stream.call_args.kwargs["headers"]["X-Subscription-Token"],
            "placeholder",
        )
        for chunks in ([b"x" * (1024 * 1024 + 1)], [b"[]"], [b"not json"]):
            self.response.iter_bytes.return_value = chunks
            with self.assertRaisesRegex(RuntimeError, "Brave search failed"):
                web_search("question")
        self.response.raise_for_status.side_effect = FakeHTTPError("secret response")
        with self.assertRaisesRegex(
            RuntimeError,
            "^Brave search failed; check the integration credential and service$",
        ):
            web_search("question")
        for query, count in [("", 5), ("x", 0), ("x", True), ("x" * 2049, 1)]:
            with self.assertRaises(ValueError):
                web_search(query, count)

    def test_a_slow_response_stops_at_the_total_deadline(self):
        self.response.iter_bytes.return_value = [b'{"web":', b'{"results":[]}}']
        clock = iter([0, 1, 1000])
        with patch(
            "nemo_fabric_adapters.deepagents.brave_search.time.monotonic",
            lambda: next(clock),
        ):
            with self.assertRaisesRegex(RuntimeError, "Brave search failed"):
                web_search("question")

    def test_unexpected_errors_are_not_reported_as_search_failures(self):
        self.response.iter_bytes.side_effect = TypeError("adapter bug")
        with self.assertRaisesRegex(TypeError, "adapter bug"):
            web_search("question")

    def test_missing_credential_is_reported_without_a_request(self):
        with patch.dict("os.environ", {"BRAVE_API_KEY": ""}):
            with self.assertRaisesRegex(RuntimeError, "BRAVE_API_KEY"):
                web_search("question")
        self.stream.assert_not_called()


@unittest.skipUnless(
    importlib.util.find_spec("langchain_mcp_adapters"), "requires Deep Agents image"
)
class NativeSearchDiscovery(unittest.IsolatedAsyncioTestCase):
    async def test_native_mcp_client_discovers_search_without_inference_or_network(
        self,
    ):
        from langchain_mcp_adapters.client import MultiServerMCPClient

        client = MultiServerMCPClient(
            {
                "brave": {
                    "transport": "stdio",
                    "command": sys.executable,
                    "args": ["-m", "nemo_fabric_adapters.deepagents.brave_search"],
                }
            }
        )
        tools = await client.get_tools()
        self.assertEqual([tool.name for tool in tools], ["web_search"])
        self.assertIn("query", tools[0].args)
