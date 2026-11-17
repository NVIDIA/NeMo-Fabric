# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the shared MCP browser sign-in (``oauth2``) auth."""

from __future__ import annotations

import asyncio
import json
import os
import stat
import time
import types
from pathlib import Path
from unittest.mock import AsyncMock
from urllib.parse import parse_qs
from urllib.parse import urlencode
from urllib.parse import urlsplit

import httpx
import pytest
from mcp.shared.auth import OAuthClientInformationFull
from mcp.shared.auth import OAuthToken
from nemo_fabric_adapters.common import mcp_oauth

MCP_URL = "https://mcp.example.test/mcp"
AUTH_URL = "https://auth.example.test"
SECRET = "oauth-secret-do-not-report"


def _assert_owner_only(path: Path, mode: int = 0o600) -> None:
    # Windows has no POSIX permission bits to check.
    if os.name == "posix":
        assert stat.S_IMODE(path.stat().st_mode) == mode


@pytest.mark.parametrize(
    ("name", "expected"),
    [("jira", "jira"), ("my-server_1.v2", "my-server_1.v2")],
)
def test_storage_name_keeps_safe_names(name, expected):
    assert mcp_oauth.storage_name(name) == expected


@pytest.mark.parametrize("name", ["../escape", "a/b", ".hidden", "with space"])
def test_storage_name_cannot_leave_the_directory(name):
    safe = mcp_oauth.storage_name(name)

    assert "/" not in safe
    assert not safe.startswith(".")
    assert safe != mcp_oauth.storage_name(name.replace("/", "_") + "x")


async def test_tokens_are_stored_owner_only_with_absolute_expiry(tmp_path):
    clock = types.SimpleNamespace(now=1000.0)
    directory = tmp_path / "mcp-tokens"
    storage = mcp_oauth.FileTokenStorage(directory, "jira", clock=lambda: clock.now)

    await storage.set_tokens(
        OAuthToken(access_token="at-1", refresh_token="rt-1", expires_in=600)
    )

    path = directory / "jira" / "tokens.json"
    _assert_owner_only(directory, 0o700)
    _assert_owner_only(directory / "jira", 0o700)
    _assert_owner_only(path)
    assert json.loads(path.read_text())["expires_at"] == 1600.0
    assert storage.has_tokens()
    clock.now = 1450.0
    tokens = await storage.get_tokens()
    assert (tokens.access_token, tokens.refresh_token, tokens.expires_in) == (
        "at-1",
        "rt-1",
        150,
    )
    clock.now = 2000.0
    assert (await storage.get_tokens()).expires_in == 0


@pytest.mark.parametrize("content", ["not json", "[]", '{"token_type": "Bearer"}'])
async def test_unreadable_tokens_are_ignored(tmp_path, content):
    (tmp_path / "jira").mkdir()
    (tmp_path / "jira" / "tokens.json").write_text(content)
    storage = mcp_oauth.FileTokenStorage(tmp_path, "jira")

    assert await storage.get_tokens() is None
    assert not storage.has_tokens()


async def test_servers_do_not_share_storage_files(tmp_path):
    jira = mcp_oauth.FileTokenStorage(tmp_path, "jira")
    other = mcp_oauth.FileTokenStorage(tmp_path, "jira.client")

    await other.set_tokens(OAuthToken(access_token="other-token"))
    await jira.set_client_info(
        OAuthClientInformationFull(
            client_id="jira-client", redirect_uris=["http://127.0.0.1:8765/callback"]
        )
    )

    assert (await other.get_tokens()).access_token == "other-token"
    assert (await jira.get_client_info()).client_id == "jira-client"
    assert await other.get_client_info() is None
    assert await jira.get_tokens() is None


@pytest.mark.parametrize(
    ("uri", "valid"),
    [
        ("http://127.0.0.1:8765/callback", True),
        ("http://localhost:8765/oauth", True),
        ("http://[::1]:8765/callback", True),
        ("http://127.0.0.1/callback", False),
        ("https://127.0.0.1:8765/callback", False),
        ("http://example.test:8765/callback", False),
    ],
)
def test_validate_redirect_uri(uri, valid):
    if valid:
        assert mcp_oauth.validate_redirect_uri("jira", uri) == uri
    else:
        with pytest.raises(ValueError, match="'jira' authentication.redirect_uri"):
            mcp_oauth.validate_redirect_uri("jira", uri)


@pytest.fixture(name="redirect_uri")
def redirect_uri_fixture() -> str:
    return f"http://127.0.0.1:{mcp_oauth._free_loopback_port()}/callback"


async def test_loopback_callback_receives_the_code(redirect_uri):
    callback = mcp_oauth.LoopbackCallback(redirect_uri)
    await callback.start()
    try:
        async with httpx.AsyncClient() as client:
            missing = await client.get(redirect_uri.replace("/callback", "/favicon.ico"))
            page = await client.get(f"{redirect_uri}?code=c-1&state=s-1")
        assert missing.status_code == 404
        assert page.status_code == 200
        assert "Sign-in complete" in page.text
        assert await callback.wait() == ("c-1", "s-1")
    finally:
        await callback.close()


@pytest.mark.parametrize(
    ("query", "message"),
    [("error=access_denied", "access_denied"), ("state=s-1", "has no code")],
)
async def test_loopback_callback_reports_a_failed_redirect(redirect_uri, query, message):
    callback = mcp_oauth.LoopbackCallback(redirect_uri)
    await callback.start()
    try:
        async with httpx.AsyncClient() as client:
            await client.get(f"{redirect_uri}?{query}")
        with pytest.raises(mcp_oauth.McpSignInError, match=message):
            await callback.wait()
    finally:
        await callback.close()


async def test_sign_in_is_refused_until_it_is_turned_on(redirect_uri):
    open_browser = AsyncMock(return_value=True)
    sign_in = mcp_oauth.BrowserSignIn("jira", redirect_uri, open_browser=open_browser)

    with pytest.raises(mcp_oauth.McpSignInRequired, match="'jira' needs a browser sign-in"):
        await sign_in.redirect("https://auth.example.test/authorize?state=s")

    open_browser.assert_not_awaited()


async def test_sign_in_that_is_not_completed_times_out(redirect_uri, capsys):
    sign_in = mcp_oauth.BrowserSignIn(
        "jira", redirect_uri, timeout=0.05, open_browser=AsyncMock(return_value=False)
    )
    sign_in.interactive = True

    await sign_in.redirect("https://auth.example.test/authorize?state=s")
    with pytest.raises(mcp_oauth.McpSignInError, match="not completed within 0.05 seconds"):
        await sign_in.callback()

    assert "https://auth.example.test/authorize?state=s" in capsys.readouterr().err
    # The listener is closed, so the port is free again.
    await mcp_oauth.LoopbackCallback(redirect_uri).start()


@pytest.mark.parametrize("failure", [RuntimeError("no browser"), asyncio.CancelledError()])
async def test_sign_in_stops_listening_when_the_browser_cannot_open(redirect_uri, failure):
    sign_in = mcp_oauth.BrowserSignIn(
        "jira", redirect_uri, open_browser=AsyncMock(side_effect=failure)
    )
    sign_in.interactive = True

    with pytest.raises(type(failure)):
        await sign_in.redirect("https://auth.example.test/authorize?state=s")

    listener = mcp_oauth.LoopbackCallback(redirect_uri)
    await listener.start()
    await listener.close()


async def test_secure_client_refuses_plaintext_requests_and_redirects():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/downgrade":
            return httpx.Response(307, headers={"Location": "http://auth.example.test/token"})
        return httpx.Response(200)

    # Keep any environment proxy from replacing the mock transport.
    os.environ["NO_PROXY"] = os.environ["no_proxy"] = "*"
    client = mcp_oauth.secure_http_client_factory("jira")()
    client._transport = httpx.MockTransport(handler)
    async with client:
        assert (await client.post("https://auth.example.test/token")).status_code == 200
        assert (await client.post("http://127.0.0.1:8765/token")).status_code == 200
        for url in ("http://auth.example.test/token", "https://auth.example.test/downgrade"):
            with pytest.raises(
                mcp_oauth.McpInsecureRequestError,
                match=r"'jira' refused a request to http://auth\.example\.test",
            ):
                await client.post(url, data={"refresh_token": "rt-1"})


def test_is_sign_in_error():
    from mcp.client.auth import OAuthFlowError

    assert mcp_oauth.is_sign_in_error(mcp_oauth.McpSignInRequired("x"))
    assert mcp_oauth.is_sign_in_error(OAuthFlowError("x"))
    assert not mcp_oauth.is_sign_in_error(ConnectionError("x"))


@pytest.fixture(name="servers")
def servers_fixture() -> types.SimpleNamespace:
    """An MCP server and its authorization server, behind one mock transport."""

    servers = types.SimpleNamespace(
        valid={"at-1", "at-2"},
        registrations=[],
        token_requests=[],
        mcp_tokens=[],
        expires_in=600,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url).split("?")[0]
        if url == MCP_URL:
            token = request.headers.get("Authorization", "").removeprefix("Bearer ")
            servers.mcp_tokens.append(token)
            if token in servers.valid:
                return httpx.Response(200, json={"ok": True})
            return httpx.Response(
                401,
                headers={
                    "WWW-Authenticate": 'Bearer resource_metadata="https://mcp.example.test'
                    '/.well-known/oauth-protected-resource/mcp"'
                },
            )
        if url == "https://mcp.example.test/.well-known/oauth-protected-resource/mcp":
            return httpx.Response(
                200,
                json={
                    "resource": MCP_URL,
                    "authorization_servers": [AUTH_URL],
                    "scopes_supported": ["everything"],
                },
            )
        if url == f"{AUTH_URL}/.well-known/oauth-authorization-server":
            return httpx.Response(
                200,
                json={
                    "issuer": AUTH_URL,
                    "authorization_endpoint": f"{AUTH_URL}/oauth/authorize",
                    "token_endpoint": f"{AUTH_URL}/oauth/token",
                    "registration_endpoint": f"{AUTH_URL}/oauth/register",
                    "response_types_supported": ["code"],
                    "code_challenge_methods_supported": ["S256"],
                },
            )
        if url == f"{AUTH_URL}/oauth/register":
            body = json.loads(request.content)
            servers.registrations.append(body)
            return httpx.Response(201, json={**body, "client_id": "registered-client"})
        if url == f"{AUTH_URL}/oauth/token":
            form = {key: values[0] for key, values in parse_qs(request.content.decode()).items()}
            servers.token_requests.append((form, request.headers.get("Authorization")))
            if form["grant_type"] == "authorization_code" and form["code"] == "code-1":
                access = "at-1"
            elif form["grant_type"] == "refresh_token" and form["refresh_token"] == "rt-1":
                access = "at-2"
            else:
                return httpx.Response(400, json={"error": "invalid_grant"})
            return httpx.Response(
                200,
                json={
                    "access_token": access,
                    "refresh_token": "rt-1",
                    "token_type": "Bearer",
                    "expires_in": servers.expires_in,
                },
            )
        return httpx.Response(404)

    servers.transport = httpx.MockTransport(handler)
    return servers


@pytest.fixture(name="browser")
def browser_fixture() -> types.SimpleNamespace:
    """A user who approves every authorization request in the browser."""

    browser = types.SimpleNamespace(opened=[])

    async def open_browser(authorization_url: str) -> bool:
        query = {key: values[0] for key, values in parse_qs(urlsplit(authorization_url).query).items()}
        browser.opened.append(query)
        redirect = f"{query['redirect_uri']}?{urlencode({'code': 'code-1', 'state': query['state']})}"

        async def approve() -> None:
            async with httpx.AsyncClient() as client:
                await client.get(redirect)

        asyncio.get_running_loop().create_task(approve())
        return True

    browser.open = open_browser
    return browser


async def _call_mcp(auth, servers) -> httpx.Response:
    async with httpx.AsyncClient(transport=servers.transport, auth=auth) as client:
        return await client.post(MCP_URL, json={"jsonrpc": "2.0", "id": 1})


async def test_browser_sign_in_registers_and_stores_tokens(tmp_path, servers, browser):
    auth = mcp_oauth.browser_sign_in_auth(
        "jira", MCP_URL, tmp_path, scopes=["jira.read"], open_browser=browser.open
    )
    auth.sign_in.interactive = True

    response = await _call_mcp(auth, servers)

    assert response.status_code == 200
    assert servers.mcp_tokens == ["", "at-1"]
    (authorization,) = browser.opened
    # Configured scopes win over the scopes the server advertises.
    assert authorization["scope"] == "jira.read"
    assert authorization["client_id"] == "registered-client"
    assert authorization["code_challenge_method"] == "S256"
    assert authorization["redirect_uri"] == auth.sign_in.redirect_uri
    (registration,) = servers.registrations
    assert registration["client_name"] == "NeMo Fabric"
    assert registration["token_endpoint_auth_method"] == "none"
    for name in ("tokens.json", "client.json", "metadata.json"):
        _assert_owner_only(tmp_path / "jira" / name)
    assert json.loads((tmp_path / "jira" / "metadata.json").read_text())["token_endpoint"] == (
        f"{AUTH_URL}/oauth/token"
    )


async def test_stored_tokens_are_reused_without_a_browser(tmp_path, servers, browser):
    first = mcp_oauth.browser_sign_in_auth("jira", MCP_URL, tmp_path, open_browser=browser.open)
    first.sign_in.interactive = True
    await _call_mcp(first, servers)

    later = mcp_oauth.browser_sign_in_auth("jira", MCP_URL, tmp_path, open_browser=browser.open)
    response = await _call_mcp(later, servers)

    assert response.status_code == 200
    assert servers.mcp_tokens[-1] == "at-1"
    assert len(browser.opened) == 1
    # The registered client's redirect URI, and so its port, is reused.
    assert later.sign_in.redirect_uri == first.sign_in.redirect_uri


async def test_expired_tokens_are_refreshed_at_the_stored_token_endpoint(
    tmp_path, servers, browser
):
    first = mcp_oauth.browser_sign_in_auth("jira", MCP_URL, tmp_path, open_browser=browser.open)
    first.sign_in.interactive = True
    await _call_mcp(first, servers)
    tokens_path = tmp_path / "jira" / "tokens.json"
    stored = json.loads(tokens_path.read_text())
    tokens_path.write_text(json.dumps({**stored, "expires_at": time.time() - 1}))

    later = mcp_oauth.browser_sign_in_auth("jira", MCP_URL, tmp_path, open_browser=browser.open)
    response = await _call_mcp(later, servers)

    assert response.status_code == 200
    assert servers.mcp_tokens[-1] == "at-2"
    form, _ = servers.token_requests[-1]
    assert form["grant_type"] == "refresh_token"
    assert len(browser.opened) == 1


async def test_new_sign_in_is_refused_while_sign_in_is_off(tmp_path, servers, browser):
    auth = mcp_oauth.browser_sign_in_auth("jira", MCP_URL, tmp_path, open_browser=browser.open)

    with pytest.raises(mcp_oauth.McpSignInRequired):
        await _call_mcp(auth, servers)

    assert browser.opened == []
    assert not (tmp_path / "jira.json").exists()


async def test_configured_client_signs_in_with_its_secret(tmp_path, servers, browser):
    auth = mcp_oauth.browser_sign_in_auth(
        "jira",
        MCP_URL,
        tmp_path,
        client_id="fabric-client",
        client_secret=SECRET,
        client_name="Helix agent",
        token_endpoint_auth_method="client_secret_basic",
        open_browser=browser.open,
    )
    auth.sign_in.interactive = True

    response = await _call_mcp(auth, servers)

    assert response.status_code == 200
    assert servers.registrations == []
    assert browser.opened[0]["client_id"] == "fabric-client"
    form, authorization = servers.token_requests[0]
    assert authorization.startswith("Basic ")
    assert "client_secret" not in form
    assert SECRET not in repr(auth)
    _assert_owner_only(tmp_path / "jira" / "client.json")
