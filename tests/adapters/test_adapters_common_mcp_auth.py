# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
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


@pytest.mark.parametrize("value", [None, {}, {"type": "bearer"}])
def test_parse_oauth2_config_rejects_unsupported_authentication(value):
    with pytest.raises(mcp_auth.McpAuthConfigError, match="unsupported"):
        mcp_auth.parse_oauth2_config("docs", value)


@pytest.mark.parametrize("field", ["authentication", "custom_headers"])
def test_validate_stdio_options_rejects_http_only_fields(field):
    with pytest.raises(mcp_auth.McpAuthConfigError, match=field):
        mcp_auth.validate_stdio_options("local", {field: {"type": "oauth2"}})


def test_normalize_custom_headers_requires_mapping():
    with pytest.raises(mcp_auth.McpAuthConfigError, match="must be a mapping"):
        mcp_auth.normalize_custom_headers("docs", ["X-Tenant", "fabric"])


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
    )

    auth = mcp_auth.create_mcp_oauth_provider(
        "jira",
        "https://mcp.example.test/jira",
        config,
        client_name="NeMo Fabric Test",
    )

    assert str(auth.context.server_url) == "https://mcp.example.test/jira"
    assert auth.context.client_metadata.client_name == "NeMo Fabric Test"
    assert auth.context.client_metadata.scope == "read write"
    client_info = await auth.context.storage.get_client_info()
    assert client_info.client_id == "fabric-client"
    assert client_info.client_secret == "oauth-secret"


async def test_mcp_oauth_provider_receives_loopback_callback(monkeypatch):
    port = 8765
    config = mcp_auth.McpOAuth2Config(
        client_id=None,
        client_secret_env=None,
        scopes=(),
        redirect_uri=f"http://127.0.0.1:{port}/callback",
    )
    auth = mcp_auth.create_mcp_oauth_provider(
        "docs",
        "https://mcp.example.test/docs",
        config,
        client_name="NeMo Fabric Test",
    )
    server = MagicMock()
    server.__aenter__ = AsyncMock(return_value=server)
    server.__aexit__ = AsyncMock(return_value=False)
    start_server = AsyncMock(return_value=server)
    monkeypatch.setattr(mcp_auth.asyncio, "start_server", start_server)
    callback = asyncio.create_task(auth.context.callback_handler())
    await asyncio.sleep(0)

    handle_callback = start_server.await_args.args[0]
    reader = asyncio.StreamReader()
    reader.feed_data(
        b"GET /callback?code=oauth-code&state=oauth-state HTTP/1.1\r\n"
        b"Host: 127.0.0.1\r\n\r\n"
    )
    reader.feed_eof()
    writer = MagicMock()
    writer.drain = AsyncMock()

    await handle_callback(reader, writer)

    assert b"200 OK" in writer.write.call_args.args[0]
    writer.close.assert_called_once_with()
    assert await asyncio.wait_for(callback, timeout=1) == (
        "oauth-code",
        "oauth-state",
    )
