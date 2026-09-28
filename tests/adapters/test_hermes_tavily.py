# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The packaged Tavily plugin calls a fixed endpoint and hides failure details."""

import io
import json
import sys
import urllib.error
from unittest.mock import MagicMock

import pytest

if sys.version_info >= (3, 14):
    pytest.skip("Hermes adapter supports Python 3.11–3.13", allow_module_level=True)

from nemo_fabric_adapters.hermes.plugins import tavily


@pytest.fixture(name="api")
def api_fixture(monkeypatch):
    """Answer Tavily requests with api.response and record them."""

    api = MagicMock(requests=[], response={})

    def open_(request, timeout):
        api.requests.append(request)
        if isinstance(api.response, Exception):
            raise api.response
        raw = api.response
        return io.BytesIO(raw if isinstance(raw, bytes) else json.dumps(raw).encode())

    monkeypatch.setenv("TAVILY_API_KEY", "placeholder")
    monkeypatch.setattr(
        tavily.urllib.request, "build_opener", lambda *handlers: MagicMock(open=open_)
    )
    return api


@pytest.mark.parametrize(("limit", "sent"), [(2, 2), (0, 1), (50, 20)])
def test_search_posts_to_tavily_and_maps_results(api, limit, sent):
    api.response = {
        "results": [{"url": "https://a.example", "title": "A", "content": "Text"}]
    }
    result = tavily.Tavily().search("query", limit=limit)
    assert result == {
        "success": True,
        "data": {
            "web": [
                {
                    "title": "A",
                    "url": "https://a.example",
                    "description": "Text",
                    "position": 1,
                }
            ]
        },
    }
    [request] = api.requests
    assert request.full_url == "https://api.tavily.com/search"
    assert request.get_header("Authorization") == "Bearer placeholder"
    assert json.loads(request.data) == {"query": "query", "max_results": sent}


def test_extraction_follows_request_order_without_echoing_errors(api):
    urls = ["https://a.example", "https://b.example", "https://c.example"]
    api.response = {
        "results": [
            {"url": "https://c.example", "raw_content": "C"},
            {"url": "https://a.example", "raw_content": "A"},
        ],
        "failed_results": [{"url": "https://b.example", "error": "secret\u001b[2J"}],
    }
    result = tavily.Tavily().extract(urls)
    assert [entry["url"] for entry in result] == urls
    assert [entry.get("content") for entry in result] == ["A", None, "C"]
    assert result[1]["error"] == "Tavily could not extract this URL"
    assert "secret" not in json.dumps(result)


@pytest.mark.parametrize(
    "response",
    [
        b"not-json",
        {},
        {"results": None},
        b"x" * (4 * 1024 * 1024 + 1),
        urllib.error.HTTPError("https://api.tavily.com", 401, "secret", {}, None),
    ],
    ids=["not-json", "no-results", "null-results", "too-large", "http-error"],
)
def test_failed_searches_report_only_a_generic_error(api, response):
    api.response = response
    assert tavily.Tavily().search("query") == {
        "success": False,
        "error": "Tavily search failed",
    }


def test_without_a_credential_nothing_is_sent_and_redirects_are_refused(
    api, monkeypatch
):
    monkeypatch.delenv("TAVILY_API_KEY")
    provider = tavily.Tavily()
    assert not provider.is_available()
    assert not provider.search("query")["success"]
    assert "error" in provider.extract(["https://a.example"])[0]
    assert api.requests == []
    assert (
        tavily.NoRedirect().redirect_request(None, None, 302, "", {}, "https://x")
        is None
    )
