# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic OO Agents MCP surface for subprocess adapter tests.

Like NOOA, the factories connect eagerly and wrap a connection failure in a
``RuntimeError`` whose cause is the transport error. A stdio command that is not
an existing file and a URL whose host starts with ``unreachable`` fail that way.
"""

from __future__ import annotations

import os
from urllib.parse import urlsplit


class _MCPTool:
    def __init__(self, server_name: str, target: str) -> None:
        self._server_name = server_name
        self._target = target

    async def echo(self, *, message: str) -> str:
        return f"{self._server_name} via {self._target}: {message}"


def _connection_failure(server_name: str, cause: Exception) -> RuntimeError:
    """Wrap a transport error the way NOOA reports a failed connection."""
    error = RuntimeError(f"Could not connect to MCP server {server_name!r}: {cause}")
    error.__cause__ = cause
    return error


class MCPManager:
    """Create fixture tools with the public NOOA MCP factory signatures."""

    @staticmethod
    async def create_stdio_server(
        server_name: str,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> _MCPTool:
        """Return a fixture tool, or fail like NOOA when the command does not exist."""
        del args, env
        if not os.path.isfile(command):
            raise _connection_failure(
                server_name, FileNotFoundError(f"no such command: {command}")
            )
        return _MCPTool(server_name, command)

    @staticmethod
    async def create_url_server(
        server_name: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        transport: str = "streamable-http",
    ) -> _MCPTool:
        """Return a fixture tool, or refuse like NOOA for an unreachable host."""
        del headers, transport
        if (urlsplit(url).hostname or "").startswith("unreachable"):
            raise _connection_failure(
                server_name, ConnectionRefusedError(f"refused: {url}")
            )
        return _MCPTool(server_name, url)
