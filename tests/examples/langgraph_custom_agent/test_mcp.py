# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for normalized MCP-to-LangChain translation."""

from __future__ import annotations

import asyncio
import json

import pytest
from langchain_core.tools import tool
from nemo_fabric_adapter_contract.models import AgentConfig
from nemo_fabric_adapters.common import lifecycle

from examples.langgraph_custom_agent.adapter import mcp as mcp_module
from examples.langgraph_custom_agent.mcp.url_inspector import inspect_url


def _config(
    *,
    transport: str = "stdio",
    allowed_tools: list[str] | None = None,
    blocked_tools: list[str] | None = None,
) -> AgentConfig:
    server = {
        "transport": transport,
        "url": "python",
        "args": ["url_inspector.py"],
    }
    if allowed_tools is not None:
        server["allowed_tools"] = allowed_tools
    if blocked_tools:
        server["blocked_tools"] = blocked_tools
    return AgentConfig.from_mapping({"mcp": {"servers": {"links": server}}})


def test_stdio_connection_maps_the_normalized_command_and_arguments():
    server = _config().mcp.servers["links"]

    assert mcp_module._stdio_connection("links", server) == {
        "transport": "stdio",
        "command": "python",
        "args": ["url_inspector.py"],
    }


def test_resolver_applies_per_server_tool_filters(monkeypatch):
    @tool
    def inspect_url(url: str) -> str:
        """Inspect one URL."""

        return json.dumps({"hostname": "example.invalid", "indicators": []})

    @tool
    def unused_tool(value: str) -> str:
        """An unrelated server tool."""

        return value

    async def discover(_connections):
        return {"links": [inspect_url, unused_tool]}

    monkeypatch.setattr(mcp_module, "_discover_tools", discover)

    selected = asyncio.run(
        mcp_module.resolve_url_inspector(
            _config(allowed_tools=["inspect_url"], blocked_tools=[])
        )
    )

    assert selected is inspect_url


def test_resolver_rejects_an_unknown_allowed_tool(monkeypatch):
    async def discover(_connections):
        return {"links": []}

    monkeypatch.setattr(mcp_module, "_discover_tools", discover)

    with pytest.raises(lifecycle.LifecycleError) as error:
        asyncio.run(
            mcp_module.resolve_url_inspector(_config(allowed_tools=["missing"]))
        )

    assert error.value.code == "email_phishing_invalid_mcp"
    assert error.value.metadata == {"field": "mcp.servers.links.allowed_tools"}


def test_resolver_rejects_a_policy_that_blocks_the_required_tool(monkeypatch):
    @tool
    def inspect_url(url: str) -> str:
        """Inspect one URL."""

        return url

    async def discover(_connections):
        return {"links": [inspect_url]}

    monkeypatch.setattr(mcp_module, "_discover_tools", discover)

    with pytest.raises(lifecycle.LifecycleError) as error:
        asyncio.run(
            mcp_module.resolve_url_inspector(
                _config(blocked_tools=["inspect_url"])
            )
        )

    assert error.value.metadata == {"field": "mcp.servers"}


def test_resolver_rejects_non_stdio_transport():
    with pytest.raises(lifecycle.LifecycleError) as error:
        asyncio.run(mcp_module.resolve_url_inspector(_config(transport="http")))

    assert error.value.metadata == {"field": "mcp.servers.links.transport"}


def test_resolver_reports_discovery_context_without_exposing_raw_error_details(
    monkeypatch,
):
    async def fail_discovery(_connections):
        raise FileNotFoundError("secret command argument")

    monkeypatch.setattr(mcp_module, "_discover_tools", fail_discovery)

    with pytest.raises(lifecycle.LifecycleError) as error:
        asyncio.run(mcp_module.resolve_url_inspector(_config()))

    assert error.value.metadata == {
        "field": "mcp.servers",
        "cause_type": "FileNotFoundError",
        "servers": ["links"],
    }
    assert "secret command argument" not in error.value.message
    assert "verify each stdio command" in error.value.message


def test_example_url_inspector_is_deterministic():
    assert inspect_url("https://example.invalid/login") == {
        "hostname": "example.invalid",
        "indicators": ["reserved_test_domain"],
    }
