# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared MCP OAuth configuration and protocol helpers for Fabric adapters."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import math
import os
import socket
import time
import webbrowser
from collections.abc import Awaitable
from collections.abc import Callable
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qs
from urllib.parse import quote
from urllib.parse import urlencode
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
    enable_dynamic_registration: bool = True
    client_name: str | None = None
    token_endpoint_auth_method: str | None = None
    authorization_timeout_seconds: float = 300.0

    @property
    def scope(self) -> str | None:
        value = " ".join(self.scopes)
        return value or None


@dataclass(frozen=True)
class McpServiceAccountConfig:
    """Harness-neutral OAuth client-credentials fields."""

    client_id: str
    client_secret_env: str
    token_url: str
    scopes: tuple[str, ...]
    token_endpoint_auth_method: str = "client_secret_basic"
    token_cache_buffer_seconds: float = 300.0

    @property
    def scope(self) -> str | None:
        value = " ".join(self.scopes)
        return value or None


TOKEN_ENDPOINT_AUTH_METHODS = {
    "none",
    "client_secret_post",
    "client_secret_basic",
}


def _string_tuple(server_name: str, field: str, value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise McpAuthConfigError(
            f"MCP server {server_name!r} authentication.{field} must be a list of strings"
        )
    return tuple(value)


def _positive_timeout(server_name: str, field: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise McpAuthConfigError(
            f"MCP server {server_name!r} authentication.{field} must be greater than zero"
        )
    return float(value)


def _token_endpoint_auth_method(
    server_name: str,
    value: Any,
    *,
    default: str | None,
) -> str | None:
    if value is None:
        return default
    if value not in TOKEN_ENDPOINT_AUTH_METHODS:
        supported = ", ".join(sorted(TOKEN_ENDPOINT_AUTH_METHODS))
        raise McpAuthConfigError(
            f"MCP server {server_name!r} authentication.token_endpoint_auth_method "
            f"must be one of: {supported}"
        )
    return str(value)


def parse_oauth2_config(server_name: str, value: Any) -> McpOAuth2Config:
    """Parse Fabric's normalized OAuth2 mapping without applying harness policy."""

    if not isinstance(value, Mapping) or value.get("type") != "oauth2":
        raise McpAuthConfigError(
            f"MCP server {server_name!r} has unsupported authentication type"
        )
    client_id = (
        str(raw_client_id) if (raw_client_id := value.get("client_id")) else None
    )
    secret_env = (
        str(raw_secret_env)
        if (raw_secret_env := value.get("client_secret_env"))
        else None
    )
    dynamic_registration = value.get("enable_dynamic_registration", True)
    if not isinstance(dynamic_registration, bool):
        raise McpAuthConfigError(
            f"MCP server {server_name!r} authentication.enable_dynamic_registration must be a boolean"
        )
    if secret_env and not client_id:
        raise McpAuthConfigError(
            f"MCP server {server_name!r} authentication.client_secret_env requires client_id"
        )
    if not client_id and not dynamic_registration:
        raise McpAuthConfigError(
            f"MCP server {server_name!r} authentication.client_id is required when "
            "dynamic registration is disabled"
        )
    method = _token_endpoint_auth_method(
        server_name,
        value.get("token_endpoint_auth_method"),
        default=None,
    )
    if (
        method in {"client_secret_basic", "client_secret_post"}
        and client_id
        and not secret_env
    ):
        raise McpAuthConfigError(
            f"MCP server {server_name!r} authentication.token_endpoint_auth_method "
            "requires client_secret_env for a pre-registered client"
        )
    if method == "none" and secret_env:
        raise McpAuthConfigError(
            f"MCP server {server_name!r} authentication.token_endpoint_auth_method "
            "'none' cannot use client_secret_env"
        )
    return McpOAuth2Config(
        client_id=client_id,
        client_secret_env=secret_env,
        scopes=_string_tuple(server_name, "scopes", value.get("scopes")),
        redirect_uri=(
            str(redirect_uri) if (redirect_uri := value.get("redirect_uri")) else None
        ),
        enable_dynamic_registration=dynamic_registration,
        client_name=(
            str(client_name) if (client_name := value.get("client_name")) else None
        ),
        token_endpoint_auth_method=method,
        authorization_timeout_seconds=_positive_timeout(
            server_name,
            "authorization_timeout_seconds",
            value.get("authorization_timeout_seconds", 300),
        ),
    )


def parse_service_account_config(
    server_name: str, value: Any
) -> McpServiceAccountConfig:
    """Parse Fabric's normalized OAuth client-credentials mapping."""

    if not isinstance(value, Mapping) or value.get("type") != "service_account":
        raise McpAuthConfigError(
            f"MCP server {server_name!r} has unsupported authentication type"
        )
    required: dict[str, str] = {}
    for field in ("client_id", "client_secret_env", "token_url"):
        item = value.get(field)
        if not isinstance(item, str) or not item.strip():
            raise McpAuthConfigError(
                f"MCP server {server_name!r} authentication.{field} is required"
            )
        required[field] = item
    method = _token_endpoint_auth_method(
        server_name,
        value.get("token_endpoint_auth_method"),
        default="client_secret_basic",
    )
    if method == "none":
        raise McpAuthConfigError(
            f"MCP server {server_name!r} service_account authentication does not support "
            "token_endpoint_auth_method 'none'"
        )
    buffer = value.get("token_cache_buffer_seconds", 300)
    if isinstance(buffer, bool) or not isinstance(buffer, (int, float)) or buffer < 0:
        raise McpAuthConfigError(
            f"MCP server {server_name!r} authentication.token_cache_buffer_seconds "
            "must be zero or greater"
        )
    return McpServiceAccountConfig(
        client_id=required["client_id"],
        client_secret_env=required["client_secret_env"],
        token_url=required["token_url"],
        scopes=_string_tuple(server_name, "scopes", value.get("scopes")),
        token_endpoint_auth_method=method,
        token_cache_buffer_seconds=float(buffer),
    )


def normalize_custom_headers(server_name: str, value: Any) -> dict[str, str]:
    """Validate and stringify an MCP custom-header mapping."""

    if not isinstance(value, Mapping):
        raise McpAuthConfigError(
            f"MCP server {server_name!r} custom_headers must be a mapping"
        )
    for name, item in value.items():
        if any(character in str(name) + str(item) for character in ("\r", "\n")):
            raise McpAuthConfigError(
                f"MCP server {server_name!r} custom_headers contain invalid characters in {name!r}"
            )
    return {str(name): str(item) for name, item in value.items()}


def resolve_client_secret(
    server_name: str,
    config: McpOAuth2Config | McpServiceAccountConfig,
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


AuthorizationUrlHandler = Callable[[str, str], Awaitable[bool | None]]


async def _default_authorization_url_handler(
    _server_name: str, authorization_url: str
) -> bool:
    return await open_authorization_url(authorization_url)


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


class _LoopbackOAuthCallback:
    """Own a single-use loopback callback listener."""

    def __init__(self, redirect_uri: str | None, timeout: float):
        self._timeout = timeout
        self._server: asyncio.AbstractServer | None = None
        self._result: asyncio.Future[tuple[str, str | None]] | None = None
        self._start_lock = asyncio.Lock()
        self._socket: socket.socket | None = None

        if redirect_uri is None:
            self._host = "127.0.0.1"
            self._path = "/callback"
            self._port = 0
            self._reserve_socket()
            self.redirect_uri = f"http://127.0.0.1:{self._port}{self._path}"
        else:
            parsed = urlparse(redirect_uri)
            self._port = loopback_callback_port(redirect_uri)
            self._host = parsed.hostname or "127.0.0.1"
            self._path = parsed.path or "/"
            self.redirect_uri = redirect_uri
            self._reserve_socket()

    def _reserve_socket(self) -> None:
        listener: socket.socket | None = None
        try:
            if self._host == "localhost":
                addresses = socket.getaddrinfo(
                    self._host,
                    self._port,
                    type=socket.SOCK_STREAM,
                    proto=socket.IPPROTO_TCP,
                )
                resolved = next(
                    (
                        address
                        for address in addresses
                        if address[4][0] in {"127.0.0.1", "::1"}
                    ),
                    None,
                )
                if resolved is None:
                    raise McpAuthConfigError(
                        "localhost did not resolve to an IP loopback address"
                    )
                family, socktype, proto, _, bind_address = resolved
            else:
                family = socket.AF_INET
                socktype = socket.SOCK_STREAM
                proto = socket.IPPROTO_TCP
                bind_address = (self._host, self._port)
            listener = socket.socket(family, socktype, proto)
            listener.bind(bind_address)
            listener.setblocking(False)
        except OSError as error:
            if listener is not None:
                listener.close()
            raise McpAuthConfigError(
                f"could not bind MCP OAuth callback listener on {self._host}:{self._port}"
            ) from error
        self._socket = listener
        self._port = int(listener.getsockname()[1])

    async def start(self) -> None:
        async with self._start_lock:
            if self._server is not None:
                return
            if self._socket is None:
                raise McpAuthConfigError(
                    "MCP OAuth callback listener cannot be restarted after it is closed"
                )
            loop = asyncio.get_running_loop()
            self._result = loop.create_future()
            listener = self._socket
            self._socket = None
            if listener is None:
                raise RuntimeError("OAuth callback socket was not reserved")
            try:
                self._server = await asyncio.start_server(
                    self._handle_callback,
                    sock=listener,
                    limit=16 * 1024,
                )
            except BaseException:
                listener.close()
                self._result.cancel()
                self._result = None
                raise

    async def wait(self) -> tuple[str, str | None]:
        if self._result is None:
            raise RuntimeError("OAuth callback listener was not started")
        try:
            async with asyncio.timeout(self._timeout):
                return await self._result
        except TimeoutError as error:
            raise McpAuthConfigError(
                f"MCP OAuth authorization timed out after {self._timeout:g} seconds"
            ) from error
        finally:
            await self.close()

    async def close(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        if self._result is not None and not self._result.done():
            self._result.cancel()
        self._result = None

    def close_reserved_socket(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None

    def __del__(self) -> None:
        self.close_reserved_socket()

    async def _write_response(
        self,
        writer: asyncio.StreamWriter,
        status: bytes,
        body: bytes,
    ) -> None:
        writer.write(
            b"HTTP/1.1 "
            + status
            + b"\r\nContent-Type: text/plain; charset=utf-8\r\nContent-Length: "
            + str(len(body)).encode("ascii")
            + b"\r\nConnection: close\r\n\r\n"
            + body
        )
        try:
            await writer.drain()
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except (AttributeError, OSError):
                pass

    async def _handle_callback(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            request = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=5.0)
            request_line = request.split(b"\r\n", 1)[0].decode("ascii")
            method, target, _ = request_line.split(" ", 2)
            parsed = urlparse(target)
        except (
            TimeoutError,
            ValueError,
            UnicodeDecodeError,
            asyncio.IncompleteReadError,
            asyncio.LimitOverrunError,
        ):
            await self._write_response(writer, b"400 Bad Request", b"Invalid request.")
            return

        if method != "GET" or parsed.path != self._path:
            await self._write_response(writer, b"404 Not Found", b"Not found.")
            return

        query = parse_qs(parsed.query, keep_blank_values=True)
        error_code = query.get("error", [None])[0]
        error_description = query.get("error_description", [None])[0]
        code = query.get("code", [""])[0]
        state = query.get("state", [None])[0]
        if error_code:
            if self._result is not None and not self._result.done():
                message = f"MCP OAuth authorization failed: {error_code!r}"
                if error_description:
                    message += f": {error_description!r}"
                self._result.set_exception(McpAuthConfigError(message))
            await self._write_response(
                writer,
                b"400 Bad Request",
                b"OAuth authorization was not completed.",
            )
            return
        if not code or not state:
            if self._result is not None and not self._result.done():
                self._result.set_exception(
                    McpAuthConfigError(
                        "MCP OAuth callback did not include code and state"
                    )
                )
            await self._write_response(
                writer, b"400 Bad Request", b"Invalid OAuth callback."
            )
            return

        if self._result is not None and not self._result.done():
            self._result.set_result((code, state))
        await self._write_response(
            writer,
            b"200 OK",
            b"OAuth authorization complete. You may close this window.",
        )


def create_mcp_oauth_provider(
    server_name: str,
    server_url: str,
    config: McpOAuth2Config,
    *,
    client_name: str,
    authorization_url_handler: AuthorizationUrlHandler | None = None,
) -> Any:
    """Build an MCP SDK OAuth provider for a harness without native OAuth support."""

    from mcp.client.auth import OAuthClientProvider
    from mcp.shared.auth import OAuthClientInformationFull
    from mcp.shared.auth import OAuthClientMetadata

    callback = _LoopbackOAuthCallback(
        config.redirect_uri,
        config.authorization_timeout_seconds,
    )
    redirect_uri = callback.redirect_uri
    token_endpoint_auth_method = config.token_endpoint_auth_method or (
        "client_secret_post" if config.client_secret_env else "none"
    )

    metadata = OAuthClientMetadata(
        client_name=config.client_name or client_name,
        redirect_uris=[redirect_uri],
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
        token_endpoint_auth_method=token_endpoint_auth_method,
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

    handler = authorization_url_handler or _default_authorization_url_handler

    async def redirect_handler(authorization_url: str) -> None:
        await callback.start()
        LOGGER.warning("MCP server '%s' requires OAuth authorization", server_name)
        try:
            opened = await handler(server_name, authorization_url)
        except BaseException:
            await callback.close()
            raise
        if opened is False:
            await callback.close()
            raise McpAuthConfigError(
                f"MCP server {server_name!r} authorization URL could not be opened"
            )

    provider = OAuthClientProvider(
        server_url=server_url,
        client_metadata=metadata,
        storage=_McpOAuthMemoryStorage(client_info),
        redirect_handler=redirect_handler,
        callback_handler=callback.wait,
        timeout=config.authorization_timeout_seconds,
    )
    setattr(provider, "_fabric_oauth_callback", callback)
    return provider


def create_mcp_service_account_auth(
    server_name: str,
    config: McpServiceAccountConfig,
    environment: Mapping[str, str] | None = None,
) -> Any:
    """Build an expiry-aware HTTPX auth provider for OAuth client credentials."""

    import httpx

    client_secret = resolve_client_secret(server_name, config, environment)
    if client_secret is None:
        raise McpAuthConfigError(
            f"MCP server {server_name!r} authentication.client_secret_env is required"
        )

    class ServiceAccountAuth(httpx.Auth):
        requires_response_body = True

        def __init__(self) -> None:
            self._access_token: str | None = None
            self._expires_at = 0.0
            self._lock = asyncio.Lock()

        def _token_is_valid(self) -> bool:
            return (
                self._access_token is not None and time.monotonic() < self._expires_at
            )

        def _token_request(self) -> httpx.Request:
            data = {"grant_type": "client_credentials"}
            if config.scope:
                data["scope"] = config.scope
            headers = {"Content-Type": "application/x-www-form-urlencoded"}
            if config.token_endpoint_auth_method == "client_secret_basic":
                encoded_id = quote(config.client_id, safe="")
                encoded_secret = quote(client_secret, safe="")
                credentials = base64.b64encode(
                    f"{encoded_id}:{encoded_secret}".encode()
                ).decode()
                headers["Authorization"] = f"Basic {credentials}"
            else:
                data["client_id"] = config.client_id
                data["client_secret"] = client_secret
            return httpx.Request(
                "POST",
                config.token_url,
                headers=headers,
                content=urlencode(data).encode(),
            )

        async def _accept_token_response(self, response: httpx.Response) -> None:
            if response.status_code != 200:
                await response.aread()
                raise McpAuthConfigError(
                    f"MCP server {server_name!r} service-account token request failed "
                    f"with HTTP {response.status_code}"
                )
            try:
                payload = json.loads((await response.aread()).decode("utf-8"))
                access_token = payload["access_token"]
                token_type = payload.get("token_type", "")
                expires_in = float(payload.get("expires_in", 3600))
                if (
                    not isinstance(access_token, str)
                    or not access_token
                    or not math.isfinite(expires_in)
                    or expires_in <= 0
                ):
                    raise ValueError
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise McpAuthConfigError(
                    f"MCP server {server_name!r} returned an invalid service-account token response"
                ) from error
            if not isinstance(token_type, str) or token_type.lower() != "bearer":
                raise McpAuthConfigError(
                    f"MCP server {server_name!r} returned unsupported token_type {token_type!r}"
                )
            self._access_token = access_token
            usable_for = max(0.0, expires_in - config.token_cache_buffer_seconds)
            self._expires_at = time.monotonic() + usable_for

        async def async_auth_flow(self, request: httpx.Request) -> Any:
            if not self._token_is_valid():
                async with self._lock:
                    if not self._token_is_valid():
                        token_response = yield self._token_request()
                        await self._accept_token_response(token_response)
            assert self._access_token is not None
            request_token = self._access_token
            request.headers["Authorization"] = f"Bearer {request_token}"
            response = yield request
            if response.status_code == 401:
                async with self._lock:
                    if self._access_token == request_token:
                        self._access_token = None
                        token_response = yield self._token_request()
                        await self._accept_token_response(token_response)
                assert self._access_token is not None
                request.headers["Authorization"] = f"Bearer {self._access_token}"
                yield request

    return ServiceAccountAuth()
