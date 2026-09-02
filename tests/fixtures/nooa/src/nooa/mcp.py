# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic OO Agents MCP surface for subprocess adapter tests."""

from __future__ import annotations


class _MCPTool:
    def __init__(self, server_name: str, target: str) -> None:
        self._server_name = server_name
        self._target = target

    async def echo(self, *, message: str) -> str:
        return f"{self._server_name} via {self._target}: {message}"


class MCPManager:
    """Create fixture tools with the public NOOA MCP factory signatures."""

    @staticmethod
    async def create_stdio_server(
        server_name: str,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> _MCPTool:
        del args, env
        return _MCPTool(server_name, command)

    @staticmethod
    async def create_url_server(
        server_name: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        transport: str = "streamable-http",
    ) -> _MCPTool:
        del headers, transport
        return _MCPTool(server_name, url)
