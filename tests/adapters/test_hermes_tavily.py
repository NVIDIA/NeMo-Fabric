# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import io
import json
import os
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

import sys
import pytest

if sys.version_info >= (3, 14):
    pytest.skip("Hermes adapter supports Python 3.11–3.13", allow_module_level=True)

from nemo_fabric_adapters.hermes.plugins import tavily


class TavilyRequests(unittest.TestCase):
    def test_search_uses_fixed_endpoint_and_bearer_placeholder(self):
        response = {
            "results": [
                {"url": "https://example.com", "title": "Example", "content": "Text"}
            ]
        }
        with (
            patch.dict(os.environ, {"TAVILY_API_KEY": "fixture-placeholder"}),
            patch.object(tavily.urllib.request, "build_opener") as opener,
        ):
            opener.return_value.open.return_value = io.BytesIO(
                json.dumps(response).encode()
            )
            provider = tavily.Tavily()
            self.assertTrue(provider.is_available())
            opener.assert_not_called()
            result = provider.search("owned query", limit=2)
            request = opener.return_value.open.call_args.args[0]
            self.assertEqual(request.full_url, "https://api.tavily.com/search")
            self.assertEqual(request.get_method(), "POST")
            self.assertEqual(
                request.get_header("Authorization"), "Bearer fixture-placeholder"
            )
            self.assertEqual(
                json.loads(request.data), {"query": "owned query", "max_results": 2}
            )
            self.assertEqual(
                result,
                {
                    "success": True,
                    "data": {
                        "web": [
                            {
                                "title": "Example",
                                "url": "https://example.com",
                                "description": "Text",
                                "position": 1,
                            }
                        ]
                    },
                },
            )

    def test_extract_preserves_successes_and_reports_failed_urls_without_echoing_errors(
        self,
    ):
        response = {
            "results": [
                {"url": "https://example.com", "raw_content": "Extracted text"}
            ],
            "failed_results": [
                {"url": "https://example.org", "error": "untrusted-secret\u001b[2J"}
            ],
        }
        with (
            patch.dict(os.environ, {"TAVILY_API_KEY": "fixture-placeholder"}),
            patch.object(tavily.urllib.request, "build_opener") as opener,
        ):
            opener.return_value.open.return_value = io.BytesIO(
                json.dumps(response).encode()
            )
            urls = ["https://example.com", "https://example.org"]
            result = tavily.Tavily().extract(urls)
            request = opener.return_value.open.call_args.args[0]
            self.assertEqual(request.full_url, "https://api.tavily.com/extract")
            self.assertEqual(
                request.get_header("Authorization"), "Bearer fixture-placeholder"
            )
            self.assertEqual(json.loads(request.data), {"urls": urls})
            self.assertEqual(result[0]["content"], "Extracted text")
            self.assertEqual(result[0]["raw_content"], "Extracted text")
            self.assertEqual(
                result[1],
                {"url": urls[1], "error": "Tavily could not extract this URL"},
            )
            self.assertNotIn("untrusted-secret", json.dumps(result))

    def test_missing_key_invalid_responses_and_redirects_do_not_fall_back(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(tavily.urllib.request, "build_opener") as opener,
        ):
            provider = tavily.Tavily()
            self.assertFalse(provider.is_available())
            self.assertFalse(provider.search("query")["success"])
            self.assertIn("error", provider.extract(["https://example.com"])[0])
            opener.assert_not_called()
        for raw in (
            b"not-json",
            b"{}",
            b'{"results":null}',
            b"x" * (4 * 1024 * 1024 + 1),
        ):
            with (
                self.subTest(response_length=len(raw)),
                patch.dict(os.environ, {"TAVILY_API_KEY": "fixture-placeholder"}),
                patch.object(tavily.urllib.request, "build_opener") as opener,
            ):
                opener.return_value.open.return_value = io.BytesIO(raw)
                self.assertFalse(tavily.Tavily().search("query")["success"])
                self.assertEqual(opener.return_value.open.call_count, 1)
        self.assertIsNone(
            tavily.NoRedirect().redirect_request(
                None, None, 302, "", {}, "https://other.example"
            )
        )
        with (
            patch.dict(os.environ, {"TAVILY_API_KEY": "fixture-placeholder"}),
            patch.object(tavily.urllib.request, "build_opener") as opener,
        ):
            opener.return_value.open.side_effect = urllib.error.HTTPError(
                "https://api.tavily.com/search",
                401,
                "untrusted-secret\u001b[2J",
                {},
                None,
            )
            result = tavily.Tavily().search("query")
            self.assertEqual(
                result, {"success": False, "error": "Tavily search failed"}
            )
            self.assertEqual(opener.return_value.open.call_count, 1)

    def _call(self, method, response, *args, **kwargs):
        with (
            patch.dict(os.environ, {"TAVILY_API_KEY": "fixture-placeholder"}),
            patch.object(tavily.urllib.request, "build_opener") as opener,
        ):
            opener.return_value.open.return_value = io.BytesIO(
                json.dumps(response).encode()
            )
            result = getattr(tavily.Tavily(), method)(*args, **kwargs)
            return result, json.loads(opener.return_value.open.call_args.args[0].data)

    def test_search_limits_are_clamped_to_the_supported_range(self):
        for limit, expected in [(0, 1), (-3, 1), (5, 5), (50, 20)]:
            with self.subTest(limit=limit):
                result, body = self._call("search", {"results": []}, "q", limit=limit)
                self.assertTrue(result["success"])
                self.assertEqual(body["max_results"], expected)

    def test_extraction_results_follow_the_requested_order(self):
        urls = ["https://a.example", "https://b.example", "https://c.example"]
        response = {
            "results": [
                {"url": "https://c.example", "raw_content": "C"},
                {"url": "https://a.example", "raw_content": "A"},
            ],
            "failed_results": [],
        }
        result, _ = self._call("extract", response, urls)
        self.assertEqual([entry["url"] for entry in result], urls)
        self.assertEqual(result[0]["content"], "A")
        self.assertEqual(result[2]["content"], "C")
        self.assertEqual(result[1]["error"], "Tavily could not extract this URL")

