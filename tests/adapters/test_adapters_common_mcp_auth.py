# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import os
import socket
from urllib.parse import urlparse
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
import httpx
from nemo_fabric_adapters.common import mcp_auth


def test_parse_oauth2_config_normalizes_fields():
    config = mcp_auth.parse_oauth2_config(
        "docs",
        {
            "type": "oauth2",
            "client_id": "fabric-client",
            "client_secret_env": "FABRIC_MCP_CLIENT_SECRET",
            "scopes": ["read", "write"],
            "redirect_uri": "http://127.0.0.1:8765/callback",
        },
    )

    assert config == mcp_auth.McpOAuth2Config(
        client_id="fabric-client",
        client_secret_env="FABRIC_MCP_CLIENT_SECRET",
        scopes=("read", "write"),
        redirect_uri="http://127.0.0.1:8765/callback",
    )
    assert config.scope == "read write"


def test_parse_oauth2_config_allows_dynamic_registration_to_supply_client_secret():
    config = mcp_auth.parse_oauth2_config(
        "docs",
        {
            "type": "oauth2",
            "token_endpoint_auth_method": "client_secret_post",
        },
    )

    assert config.client_id is None
    assert config.client_secret_env is None
    assert config.enable_dynamic_registration is True
    assert config.token_endpoint_auth_method == "client_secret_post"


@pytest.mark.parametrize("value", [None, {}, {"type": "bearer"}])
def test_parse_oauth2_config_rejects_unsupported_authentication(value):
    with pytest.raises(mcp_auth.McpAuthConfigError, match="unsupported"):
        mcp_auth.parse_oauth2_config("docs", value)


def test_normalize_custom_headers_requires_mapping():
    with pytest.raises(mcp_auth.McpAuthConfigError, match="must be a mapping"):
        mcp_auth.normalize_custom_headers("docs", ["X-Tenant", "fabric"])


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("X-Tenant", False),
        ("X-Tenant\r", True),
        ("X-Tenant\nvalue", True),
        ("X-Tenant\r\nvalue", True),
    ],
)
def test_contains_crlf(value, expected):
    assert mcp_auth.contains_crlf(value) is expected


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("X-Foo\r", "bar"),
        ("X-Foo\n", "bar"),
        ("X-Foo", "bar\r"),
        ("X-Foo", "bar\nX-Evil: injected"),
    ],
)
def test_normalize_custom_headers_rejects_newlines(name, value):
    with pytest.raises(mcp_auth.McpAuthConfigError) as error:
        mcp_auth.normalize_custom_headers("docs", {name: value})

    assert str(error.value) == (
        f"MCP server 'docs' custom_headers contain invalid characters in {name!r}"
    )


def test_normalize_custom_headers_expands_environment_variables():
    os.environ["FABRIC_HEADER_VALUE"] = "fabric"

    assert mcp_auth.normalize_custom_headers(
        "docs", {"X-Tenant": "${FABRIC_HEADER_VALUE}"}
    ) == {"X-Tenant": "fabric"}


def test_normalize_custom_headers_rejects_newlines_after_expansion():
    os.environ["FABRIC_HEADER_VALUE"] = "fabric\r\nX-Evil: injected"

    with pytest.raises(mcp_auth.McpAuthConfigError, match="invalid characters"):
        mcp_auth.normalize_custom_headers(
            "docs", {"X-Tenant": "${FABRIC_HEADER_VALUE}"}
        )


def test_resolve_client_secret_uses_named_environment_variable():
    os.environ["FABRIC_MCP_CLIENT_SECRET"] = "oauth-secret"
    config = mcp_auth.McpOAuth2Config(
        client_id="fabric-client",
        client_secret_env="FABRIC_MCP_CLIENT_SECRET",
        scopes=(),
        redirect_uri=None,
    )

    assert (
        mcp_auth.resolve_client_secret("docs", config, require_client_id=True)
        == "oauth-secret"
    )


def test_resolve_client_secret_requires_client_id_when_requested():
    config = mcp_auth.McpOAuth2Config(
        client_id=None,
        client_secret_env="FABRIC_MCP_CLIENT_SECRET",
        scopes=(),
        redirect_uri=None,
    )

    with pytest.raises(mcp_auth.McpAuthConfigError, match="requires client_id"):
        mcp_auth.resolve_client_secret(
            "docs",
            config,
            {"FABRIC_MCP_CLIENT_SECRET": "oauth-secret"},
            require_client_id=True,
        )


def test_resolve_client_secret_rejects_unset_environment_variable():
    config = mcp_auth.McpOAuth2Config(
        client_id="fabric-client",
        client_secret_env="FABRIC_MCP_CLIENT_SECRET",
        scopes=(),
        redirect_uri=None,
    )

    with pytest.raises(mcp_auth.McpAuthConfigError, match="unset"):
        mcp_auth.resolve_client_secret("docs", config, {})


@pytest.mark.parametrize(
    "redirect_uri",
    [
        "https://127.0.0.1:8765/callback",
        "http://example.com:8765/callback",
        "http://127.0.0.1/callback",
    ],
)
def test_loopback_callback_port_rejects_unsupported_redirects(redirect_uri):
    with pytest.raises(mcp_auth.McpAuthConfigError, match="loopback"):
        mcp_auth.loopback_callback_port(redirect_uri)


@pytest.mark.parametrize(
    "redirect_uri",
    ["http://127.0.0.1:8765/callback", "http://localhost:8765/callback"],
)
def test_loopback_callback_port_accepts_ip_and_hostname(redirect_uri):
    assert mcp_auth.loopback_callback_port(redirect_uri) == 8765


@pytest.mark.parametrize("opened", [True, False])
async def test_open_authorization_url_uses_shared_browser_helper(monkeypatch, opened):
    open_browser = MagicMock(return_value=opened)
    monkeypatch.setattr(mcp_auth.webbrowser, "open", open_browser)
    to_thread = AsyncMock(return_value=opened)
    monkeypatch.setattr(mcp_auth.asyncio, "to_thread", to_thread)

    assert await mcp_auth.open_authorization_url("https://auth.example.test") is opened

    to_thread.assert_awaited_once_with(
        open_browser,
        "https://auth.example.test",
    )


async def test_create_mcp_oauth_provider_maps_client_configuration():
    os.environ["FABRIC_MCP_CLIENT_SECRET"] = "oauth-secret"
    config = mcp_auth.McpOAuth2Config(
        client_id="fabric-client",
        client_secret_env="FABRIC_MCP_CLIENT_SECRET",
        scopes=("read", "write"),
        redirect_uri="http://127.0.0.1:8765/callback",
        client_name="Configured Client",
        token_endpoint_auth_method="client_secret_basic",
        authorization_timeout_seconds=42,
    )

    auth = mcp_auth.create_mcp_oauth_provider(
        "jira",
        "https://mcp.example.test/jira",
        config,
        client_name="NeMo Fabric Test",
    )

    assert str(auth.context.server_url) == "https://mcp.example.test/jira"
    assert auth.context.client_metadata.client_name == "Configured Client"
    assert auth.context.client_metadata.scope == "read write"
    assert (
        auth.context.client_metadata.token_endpoint_auth_method == "client_secret_basic"
    )
    assert auth.context.timeout == 42
    client_info = await auth.context.storage.get_client_info()
    assert client_info.client_id == "fabric-client"
    assert client_info.client_secret == "oauth-secret"


async def test_mcp_oauth_provider_starts_listener_before_authorization_handler():
    config = mcp_auth.McpOAuth2Config(
        client_id=None,
        client_secret_env=None,
        scopes=(),
        redirect_uri=None,
    )
    callback_uri = ""

    async def authorization_handler(_name, _url):
        parsed = urlparse(callback_uri)
        reader, writer = await asyncio.open_connection(parsed.hostname, parsed.port)
        writer.write(
            f"GET {parsed.path}?code=oauth-code&state=oauth-state HTTP/1.1\r\n"
            f"Host: {parsed.hostname}\r\n\r\n".encode()
        )
        await writer.drain()
        response = await reader.read()
        writer.close()
        await writer.wait_closed()
        assert b"200 OK" in response
        return True

    auth = mcp_auth.create_mcp_oauth_provider(
        "docs",
        "https://mcp.example.test/docs",
        config,
        client_name="NeMo Fabric Test",
        authorization_url_handler=authorization_handler,
    )
    callback_uri = str(auth.context.client_metadata.redirect_uris[0])

    await auth.context.redirect_handler("https://auth.example.test/authorize")
    assert await auth.context.callback_handler() == (
        "oauth-code",
        "oauth-state",
    )


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        (
            "error=access_denied&state=oauth-state",
            "MCP OAuth authorization failed: 'access_denied'",
        ),
        (
            "error=access_denied&error_description=User+denied+access&state=oauth-state",
            "MCP OAuth authorization failed: 'access_denied': 'User denied access'",
        ),
    ],
)
async def test_loopback_callback_reports_oauth_error(query, expected):
    callback = mcp_auth._LoopbackOAuthCallback(None, timeout=1)
    await callback.start()
    parsed = urlparse(callback.redirect_uri)
    reader, writer = await asyncio.open_connection(parsed.hostname, parsed.port)
    writer.write(
        f"GET {parsed.path}?{query} HTTP/1.1\r\n"
        f"Host: {parsed.hostname}\r\n\r\n".encode()
    )
    await writer.drain()
    response = await reader.read()
    writer.close()
    await writer.wait_closed()

    assert b"400 Bad Request" in response
    with pytest.raises(mcp_auth.McpAuthConfigError) as error:
        await callback.wait()

    assert str(error.value) == expected


async def test_loopback_callback_times_out_closes_listener_and_cannot_restart():
    callback = mcp_auth._LoopbackOAuthCallback(None, timeout=0.01)
    await callback.start()

    with pytest.raises(mcp_auth.McpAuthConfigError, match="timed out"):
        await callback.wait()

    with pytest.raises(mcp_auth.McpAuthConfigError, match="cannot be restarted"):
        await callback.start()


async def test_loopback_callback_listens_on_preferred_localhost_address():
    reservation = mcp_auth._LoopbackOAuthCallback(None, timeout=1)
    port = urlparse(reservation.redirect_uri).port
    reservation.close_reserved_socket()
    assert port is not None

    callback = mcp_auth._LoopbackOAuthCallback(
        f"http://localhost:{port}/callback",
        timeout=1,
    )
    await callback.start()
    family, _, _, _, address = next(
        entry
        for entry in socket.getaddrinfo(
            "localhost",
            port,
            type=socket.SOCK_STREAM,
            proto=socket.IPPROTO_TCP,
        )
        if entry[4][0] in {"127.0.0.1", "::1"}
    )
    reader, writer = await asyncio.open_connection(
        address[0],
        address[1],
        family=family,
    )
    writer.write(
        b"GET /callback?code=oauth-code&state=oauth-state HTTP/1.1\r\n"
        b"Host: localhost\r\n\r\n"
    )
    await writer.drain()
    response = await reader.read()
    writer.close()
    await writer.wait_closed()

    assert b"200 OK" in response
    assert await callback.wait() == ("oauth-code", "oauth-state")


def test_parse_service_account_config_normalizes_fields():
    config = mcp_auth.parse_service_account_config(
        "automation",
        {
            "type": "service_account",
            "client_id": "fabric-client",
            "client_secret_env": "FABRIC_MCP_CLIENT_SECRET",
            "token_url": "https://auth.example.test/token",
            "scopes": ["mcp:invoke"],
            "token_endpoint_auth_method": "client_secret_post",
            "token_cache_buffer_seconds": 60,
        },
    )

    assert config == mcp_auth.McpServiceAccountConfig(
        client_id="fabric-client",
        client_secret_env="FABRIC_MCP_CLIENT_SECRET",
        token_url="https://auth.example.test/token",
        scopes=("mcp:invoke",),
        token_endpoint_auth_method="client_secret_post",
        token_cache_buffer_seconds=60,
    )


async def test_service_account_auth_caches_token_and_refreshes_after_401():
    config = mcp_auth.McpServiceAccountConfig(
        client_id="fabric client",
        client_secret_env="FABRIC_MCP_CLIENT_SECRET",
        token_url="https://auth.example.test/token",
        scopes=("mcp:invoke",),
        token_cache_buffer_seconds=60,
    )
    auth = mcp_auth.create_mcp_service_account_auth(
        "automation",
        config,
        {"FABRIC_MCP_CLIENT_SECRET": "oauth secret"},
    )

    request = httpx.Request("POST", "https://mcp.example.test/mcp")
    flow = auth.async_auth_flow(request)
    token_request = await anext(flow)
    assert str(token_request.url) == "https://auth.example.test/token"
    assert token_request.headers["Authorization"].startswith("Basic ")
    assert token_request.content == b"grant_type=client_credentials&scope=mcp%3Ainvoke"

    authorized = await flow.asend(
        httpx.Response(
            200,
            json={
                "access_token": "first-token",
                "token_type": "Bearer",
                "expires_in": 3600,
            },
            request=token_request,
        )
    )
    assert authorized.headers["Authorization"] == "Bearer first-token"
    with pytest.raises(StopAsyncIteration):
        await flow.asend(httpx.Response(200, request=authorized))

    cached_flow = auth.async_auth_flow(
        httpx.Request("POST", "https://mcp.example.test/mcp")
    )
    cached_request = await anext(cached_flow)
    assert str(cached_request.url) == "https://mcp.example.test/mcp"
    assert cached_request.headers["Authorization"] == "Bearer first-token"

    retry_token_request = await cached_flow.asend(
        httpx.Response(401, request=cached_request)
    )
    retried_request = await cached_flow.asend(
        httpx.Response(
            200,
            json={
                "access_token": "second-token",
                "token_type": "bearer",
                "expires_in": 3600,
            },
            request=retry_token_request,
        )
    )
    assert retried_request.headers["Authorization"] == "Bearer second-token"
    with pytest.raises(StopAsyncIteration):
        await cached_flow.asend(httpx.Response(200, request=retried_request))


@pytest.mark.parametrize(
    ("payload", "token_type"),
    [
        ({"access_token": "token", "expires_in": 3600}, ""),
        (
            {"access_token": "token", "token_type": "mac", "expires_in": 3600},
            "mac",
        ),
    ],
)
async def test_service_account_auth_rejects_unsupported_token_type(payload, token_type):
    config = mcp_auth.McpServiceAccountConfig(
        client_id="fabric-client",
        client_secret_env="FABRIC_MCP_CLIENT_SECRET",
        token_url="https://auth.example.test/token",
        scopes=(),
    )
    auth = mcp_auth.create_mcp_service_account_auth(
        "automation",
        config,
        {"FABRIC_MCP_CLIENT_SECRET": "oauth-secret"},
    )
    flow = auth.async_auth_flow(httpx.Request("POST", "https://mcp.example.test/mcp"))
    token_request = await anext(flow)

    with pytest.raises(mcp_auth.McpAuthConfigError) as error:
        await flow.asend(httpx.Response(200, json=payload, request=token_request))

    assert str(error.value) == (
        f"MCP server 'automation' returned unsupported token_type {token_type!r}"
    )
