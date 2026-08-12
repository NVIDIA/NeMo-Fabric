# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Translate normalized MCP servers into native LangChain tools."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool
from nemo_fabric_adapter_contract.models import AgentConfig
from nemo_fabric_adapter_contract.models import AgentMcpServerConfig
from nemo_fabric_adapters.common import lifecycle

URL_INSPECTOR_TOOL = "inspect_url"


def _config_error(
    field: str,
    message: str,
    *,
    metadata: dict[str, Any] | None = None,
) -> lifecycle.LifecycleError:
    details: dict[str, Any] = {"field": field}
    details.update(metadata or {})
    return lifecycle.LifecycleError(
        "email_phishing_invalid_mcp",
        message,
        metadata=details,
    )


def _stdio_connection(
    name: str,
    server: AgentMcpServerConfig,
) -> dict[str, Any]:
    field = f"mcp.servers.{name}"
    if server.transport != "stdio":
        raise _config_error(
            f"{field}.transport",
            "The email-phishing example accepts only stdio MCP servers",
        )
    if server.extensions:
        raise _config_error(
            f"{field}.extensions",
            "The email-phishing example does not accept MCP server extensions",
        )

    connection: dict[str, Any] = {
        "transport": "stdio",
        "command": server.url,
        "args": list(server.args),
    }
    if server.env:
        connection["env"] = dict(server.env)
    return connection


async def _discover_tools(
    connections: dict[str, dict[str, Any]],
) -> dict[str, list[BaseTool]]:
    # The optional dependency is imported only when MCP is configured.
    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient(connections)
    return {
        name: list(await client.get_tools(server_name=name)) for name in connections
    }


async def resolve_url_inspector(agent_config: AgentConfig) -> BaseTool | None:
    """Resolve one optional MCP tool used by the custom graph."""

    if agent_config.mcp is None or not agent_config.mcp.servers:
        return None
    if agent_config.mcp.extensions:
        raise _config_error(
            "mcp.extensions",
            "The email-phishing example does not accept MCP extensions",
        )

    connections = {
        name: _stdio_connection(name, server)
        for name, server in agent_config.mcp.servers.items()
    }
    try:
        discovered = await _discover_tools(connections)
    except Exception as error:
        raise _config_error(
            "mcp.servers",
            "The email-phishing adapter could not discover the configured MCP tools; "
            "verify each stdio command, its arguments, and server startup",
            metadata={
                "cause_type": type(error).__name__,
                "servers": sorted(connections),
            },
        ) from error

    selected: list[BaseTool] = []
    for name, server in agent_config.mcp.servers.items():
        tools = discovered[name]
        discovered_names = {tool.name for tool in tools}
        if server.allowed_tools is not None:
            missing = set(server.allowed_tools).difference(discovered_names)
            if missing:
                tool_name = sorted(missing)[0]
                raise _config_error(
                    f"mcp.servers.{name}.allowed_tools",
                    f"MCP server {name!r} does not provide tool {tool_name!r}",
                )
        selected.extend(
            tool
            for tool in tools
            if (server.allowed_tools is None or tool.name in server.allowed_tools)
            and tool.name not in server.blocked_tools
        )

    matches = [tool for tool in selected if tool.name == URL_INSPECTOR_TOOL]
    if len(matches) > 1:
        raise _config_error(
            "mcp.servers",
            f"More than one MCP server provides {URL_INSPECTOR_TOOL!r}",
        )
    if not matches:
        raise _config_error(
            "mcp.servers",
            f"The configured MCP tool policy does not expose {URL_INSPECTOR_TOOL!r}",
        )
    return matches[0]
