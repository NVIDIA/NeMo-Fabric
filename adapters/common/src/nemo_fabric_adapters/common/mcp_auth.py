# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared MCP OAuth configuration and protocol helpers for Fabric adapters."""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import webbrowser
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qs
from urllib.parse import urlparse

LOGGER = logging.getLogger(__name__)


class McpAuthConfigError(ValueError):
    """Invalid normalized MCP authentication configuration."""


@dataclass(frozen=True)
class McpOAuth2Config:
    """Harness-neutral OAuth2 fields from an MCP server configuration."""

    client_id: str | None
    client_secret_env: str | None
    scopes: tuple[str, ...]
    redirect_uri: str | None

    @property
    def scope(self) -> str | None:
        value = " ".join(self.scopes)
        return value or None


def parse_oauth2_config(server_name: str, value: Any) -> McpOAuth2Config:
    """Parse Fabric's normalized OAuth2 mapping without applying harness policy."""

    if not isinstance(value, Mapping) or value.get("type") != "oauth2":
        raise McpAuthConfigError(
            f"MCP server {server_name!r} has unsupported authentication type"
        )
    scopes = value.get("scopes") or []
    return McpOAuth2Config(
        client_id=(str(client_id) if (client_id := value.get("client_id")) else None),
        client_secret_env=(
            str(secret_env) if (secret_env := value.get("client_secret_env")) else None
        ),
        scopes=tuple(str(scope) for scope in scopes),
        redirect_uri=(
            str(redirect_uri) if (redirect_uri := value.get("redirect_uri")) else None
        ),
    )


def validate_stdio_options(server_name: str, server: Mapping[str, Any]) -> None:
    """Reject HTTP-only MCP options on a stdio server."""

    if server.get("authentication"):
        raise McpAuthConfigError(
            f"MCP server {server_name!r} authentication is not supported for stdio transport"
        )
    if server.get("custom_headers"):
        raise McpAuthConfigError(
            f"MCP server {server_name!r} custom_headers are not supported for stdio transport"
        )


def normalize_custom_headers(server_name: str, value: Any) -> dict[str, str]:
    """Validate and stringify an MCP custom-header mapping."""

    if not isinstance(value, Mapping):
        raise McpAuthConfigError(
            f"MCP server {server_name!r} custom_headers must be a mapping"
        )
    return {str(name): str(item) for name, item in value.items()}


def resolve_client_secret(
    server_name: str,
    config: McpOAuth2Config,
    environment: Mapping[str, str] | None = None,
    *,
    require_client_id: bool = False,
) -> str | None:
    """Resolve a configured OAuth client secret without retaining or logging it."""

    if config.client_secret_env is None:
        return None
    if require_client_id and config.client_id is None:
        raise McpAuthConfigError(
            f"MCP server {server_name!r} authentication.client_secret_env requires client_id"
        )
    source = os.environ if environment is None else environment
    secret = source.get(config.client_secret_env)
    if not secret:
        raise McpAuthConfigError(
            f"MCP server {server_name!r} authentication.client_secret_env references an unset environment variable"
        )
    return secret


def loopback_callback_port(redirect_uri: str) -> int:
    """Return the explicit port from an HTTP loopback OAuth redirect URI."""

    parsed = urlparse(redirect_uri)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost"}
        or parsed.port is None
    ):
        raise McpAuthConfigError(
            "authentication.redirect_uri must be an HTTP loopback URI with an explicit port"
        )
    return parsed.port


async def open_authorization_url(authorization_url: str) -> bool:
    """Open an OAuth authorization URL without blocking the event loop."""

    return await asyncio.to_thread(webbrowser.open, authorization_url)


class _McpOAuthMemoryStorage:
    def __init__(self, client_info: Any = None):
        self.tokens: Any = None
        self.client_info = client_info

    async def get_tokens(self) -> Any:
        return self.tokens

    async def set_tokens(self, tokens: Any) -> None:
        self.tokens = tokens

    async def get_client_info(self) -> Any:
        return self.client_info

    async def set_client_info(self, client_info: Any) -> None:
        self.client_info = client_info


def _available_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def create_mcp_oauth_provider(
    server_name: str,
    server_url: str,
    config: McpOAuth2Config,
    *,
    client_name: str,
) -> Any:
    """Build an MCP SDK OAuth provider for a harness without native OAuth support."""

    from mcp.client.auth import OAuthClientProvider
    from mcp.shared.auth import OAuthClientInformationFull
    from mcp.shared.auth import OAuthClientMetadata

    if config.redirect_uri is not None:
        redirect_uri = config.redirect_uri
        callback_port = loopback_callback_port(redirect_uri)
    else:
        callback_port = _available_loopback_port()
        redirect_uri = f"http://127.0.0.1:{callback_port}/callback"

    metadata = OAuthClientMetadata(
        client_name=client_name,
        redirect_uris=[redirect_uri],
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
        token_endpoint_auth_method=(
            "client_secret_post" if config.client_secret_env else "none"
        ),
        scope=config.scope,
    )

    client_info = None
    if config.client_id is not None:
        client_info = OAuthClientInformationFull(
            client_id=config.client_id,
            client_secret=resolve_client_secret(
                server_name,
                config,
                require_client_id=True,
            ),
            redirect_uris=[redirect_uri],
            grant_types=metadata.grant_types,
            response_types=metadata.response_types,
            token_endpoint_auth_method=metadata.token_endpoint_auth_method,
            scope=metadata.scope,
        )
    elif config.client_secret_env is not None:
        raise McpAuthConfigError(
            f"MCP server {server_name!r} authentication.client_secret_env requires client_id"
        )

    async def redirect_handler(authorization_url: str) -> None:
        LOGGER.warning(
            "MCP server '%s' requires OAuth authorization: %s",
            server_name,
            authorization_url,
        )
        await open_authorization_url(authorization_url)

    async def callback_handler() -> tuple[str, str | None]:
        loop = asyncio.get_running_loop()
        result: asyncio.Future[tuple[str, str | None]] = loop.create_future()

        async def handle_callback(
            reader: asyncio.StreamReader, writer: asyncio.StreamWriter
        ) -> None:
            request = await reader.readuntil(b"\r\n\r\n")
            target_path = request.split(b" ", 2)[1].decode("ascii")
            query = parse_qs(urlparse(target_path).query)
            code = query.get("code", [""])[0]
            state = query.get("state", [None])[0]
            body = b"OAuth authorization complete. You may close this window."
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: "
                + str(len(body)).encode("ascii")
                + b"\r\nConnection: close\r\n\r\n"
                + body
            )
            await writer.drain()
            writer.close()
            if not result.done():
                result.set_result((code, state))

        server = await asyncio.start_server(handle_callback, "127.0.0.1", callback_port)
        async with server:
            return await result

    return OAuthClientProvider(
        server_url=server_url,
        client_metadata=metadata,
        storage=_McpOAuthMemoryStorage(client_info),
        redirect_handler=redirect_handler,
        callback_handler=callback_handler,
    )
