# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the shared MCP client-credentials token client and auth."""

from __future__ import annotations

import asyncio
import base64
import types
from urllib.parse import parse_qs

import httpx
import pytest
from nemo_fabric_adapters.common import mcp_service_account

TOKEN_URL = "https://auth.example.test/oauth2/token"
MCP_URL = "https://mcp.example.test/mcp"
SECRET = "s3cret:with/chars"


@pytest.fixture(name="token_endpoint")
def token_endpoint_fixture() -> types.SimpleNamespace:
    """A token endpoint that issues ``token-1``, ``token-2``, ... and records requests."""

    endpoint = types.SimpleNamespace(
        requests=[], status=200, body=None, expires_in=3600, error=None
    )

    def handler(request: httpx.Request) -> httpx.Response:
        endpoint.requests.append(request)
        if endpoint.error is not None:
            raise endpoint.error
        if endpoint.status != 200:
            return httpx.Response(endpoint.status, json={"error": f"detail {SECRET}"})
        if endpoint.body is not None:
            return httpx.Response(200, json=endpoint.body)
        return httpx.Response(
            200,
            json={
                "access_token": f"token-{len(endpoint.requests)}",
                "token_type": "Bearer",
                "expires_in": endpoint.expires_in,
            },
        )

    endpoint.transport = httpx.MockTransport(handler)
    return endpoint


@pytest.fixture(name="clock")
def clock_fixture() -> types.SimpleNamespace:
    clock = types.SimpleNamespace(now=1000.0)
    clock.read = lambda: clock.now
    return clock


@pytest.fixture(name="make_client")
def make_client_fixture(token_endpoint, clock):
    def make(**overrides) -> mcp_service_account.ServiceAccountTokenClient:
        options = {
            "client_id": "fabric agent",
            "client_secret": SECRET,
            "token_url": TOKEN_URL,
            "scopes": ["jira.read", "jira.write"],
            "clock": clock.read,
            "http_client_factory": lambda: httpx.AsyncClient(
                transport=token_endpoint.transport
            ),
        }
        options.update(overrides)
        return mcp_service_account.ServiceAccountTokenClient(**options)

    return make


async def test_basic_auth_token_request(make_client, token_endpoint):
    token = await make_client().get_access_token()

    assert token == "token-1"
    (request,) = token_endpoint.requests
    assert request.method == "POST"
    assert str(request.url) == TOKEN_URL
    assert parse_qs(request.content.decode()) == {
        "grant_type": ["client_credentials"],
        "scope": ["jira.read jira.write"],
    }
    scheme, encoded = request.headers["Authorization"].split(" ")
    assert scheme == "Basic"
    # RFC 6749 section 2.3.1 form-encodes the id and secret before Basic encoding.
    assert base64.b64decode(encoded).decode() == "fabric%20agent:s3cret%3Awith%2Fchars"


async def test_post_auth_token_request(make_client, token_endpoint):
    await make_client(
        token_endpoint_auth_method="client_secret_post", scopes=()
    ).get_access_token()

    (request,) = token_endpoint.requests
    assert "Authorization" not in request.headers
    assert parse_qs(request.content.decode()) == {
        "grant_type": ["client_credentials"],
        "client_id": ["fabric agent"],
        "client_secret": [SECRET],
    }


async def test_token_is_cached_until_the_renewal_buffer(
    make_client, token_endpoint, clock
):
    client = make_client(token_cache_buffer_seconds=300)

    assert await client.get_access_token() == "token-1"
    clock.now += 3299
    assert await client.get_access_token() == "token-1"
    clock.now += 1
    assert await client.get_access_token() == "token-2"
    assert len(token_endpoint.requests) == 2


@pytest.mark.parametrize(
    ("expires_in", "renew_after"),
    [
        (60, 30),  # buffer capped at half of a short lifetime
        (None, 3300),  # missing lifetime defaults to one hour
        ("bad", 3300),
        (0, 3300),
    ],
)
async def test_token_lifetime(make_client, token_endpoint, clock, expires_in, renew_after):
    token_endpoint.expires_in = expires_in
    client = make_client()

    await client.get_access_token()
    clock.now += renew_after - 1
    assert await client.get_access_token() == "token-1"
    clock.now += 1
    assert await client.get_access_token() == "token-2"


async def test_stale_token_forces_one_new_token(make_client, token_endpoint):
    client = make_client()
    first = await client.get_access_token()

    assert await client.get_access_token(stale=first) == "token-2"
    # A caller holding the old token after another caller renewed it reuses the new one.
    assert await client.get_access_token(stale=first) == "token-2"
    assert len(token_endpoint.requests) == 2


async def test_concurrent_callers_share_one_token_request(make_client):
    entered = asyncio.Event()
    release = asyncio.Event()
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        entered.set()
        await release.wait()
        return httpx.Response(200, json={"access_token": "token-1", "expires_in": 3600})

    client = make_client(
        http_client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )

    first = asyncio.ensure_future(client.get_access_token())
    await entered.wait()
    # The first request is still in flight while the other callers arrive.
    others = [asyncio.ensure_future(client.get_access_token()) for _ in range(4)]
    await asyncio.sleep(0)
    release.set()
    tokens = await asyncio.gather(first, *others)

    assert tokens == ["token-1"] * 5
    assert len(requests) == 1


@pytest.mark.parametrize(
    ("status", "rejected"), [(400, True), (401, True), (403, False), (500, False)]
)
async def test_token_endpoint_errors(make_client, token_endpoint, status, rejected):
    token_endpoint.status = status

    with pytest.raises(mcp_service_account.ServiceAccountTokenError) as raised:
        await make_client().get_access_token()

    assert raised.value.rejected is rejected
    assert str(status) in str(raised.value)
    assert SECRET not in str(raised.value)


@pytest.mark.parametrize("body", [{}, {"access_token": ""}, ["token"]])
async def test_token_response_without_access_token(make_client, token_endpoint, body):
    token_endpoint.body = body

    with pytest.raises(
        mcp_service_account.ServiceAccountTokenError, match="no access_token"
    ) as raised:
        await make_client().get_access_token()

    assert raised.value.rejected is False


async def test_unreachable_token_endpoint(make_client, token_endpoint):
    token_endpoint.error = httpx.ConnectError(f"connect failed {TOKEN_URL}")

    with pytest.raises(mcp_service_account.ServiceAccountTokenError) as raised:
        await make_client().get_access_token()

    assert raised.value.rejected is False
    assert str(raised.value) == "token request failed (ConnectError)"
    assert raised.value.__cause__ is None


def test_client_repr_omits_the_secret(make_client):
    assert SECRET not in repr(make_client())
    assert SECRET not in repr(mcp_service_account.service_account_auth(make_client()))


def test_unknown_token_endpoint_auth_method(make_client):
    with pytest.raises(ValueError, match="client_secret_basic or client_secret_post"):
        make_client(token_endpoint_auth_method="none")


@pytest.mark.parametrize(
    ("token_url", "valid"),
    [
        ("https://auth.example.test/token", True),
        (" https://auth.example.test/token ", True),
        ("http://localhost:8080/token", True),
        ("http://127.0.0.1/token", True),
        ("http://[::1]:9000/token", True),
        ("http://auth.example.test/token", False),
        ("ftp://auth.example.test/token", False),
        ("https:///token", False),
        ("auth.example.test/token", False),
    ],
)
def test_validate_token_url(token_url, valid):
    if valid:
        assert mcp_service_account.validate_token_url("jira", token_url) == token_url.strip()
    else:
        with pytest.raises(ValueError, match="'jira' authentication.token_url must use https"):
            mcp_service_account.validate_token_url("jira", token_url)


@pytest.mark.parametrize("value", [None, "", "  "])
def test_missing_client_secret_names_the_variable(value):
    environ = {} if value is None else {"JIRA_SECRET": value}

    with pytest.raises(ValueError, match="'JIRA_SECRET' is not set"):
        mcp_service_account.resolve_client_secret("jira", "JIRA_SECRET", environ)


def test_client_secret_is_read_from_the_environment():
    assert (
        mcp_service_account.resolve_client_secret("jira", "JIRA_SECRET", {"JIRA_SECRET": SECRET})
        == SECRET
    )


def test_secret_env_names():
    authentications = [
        None,
        types.SimpleNamespace(type="oauth2", client_secret_env="OAUTH_SECRET"),
        types.SimpleNamespace(type="service_account", client_secret_env="SA_SECRET"),
    ]

    assert mcp_service_account.secret_env_names(authentications) == {"SA_SECRET"}


@pytest.fixture(name="mcp_server")
def mcp_server_fixture() -> types.SimpleNamespace:
    """An MCP endpoint that answers 401 to the tokens listed in ``rejected``."""

    server = types.SimpleNamespace(seen=[], rejected=set())

    def handler(request: httpx.Request) -> httpx.Response:
        authorization = request.headers.get("Authorization", "")
        server.seen.append((authorization, request.content))
        if authorization.removeprefix("Bearer ") in server.rejected:
            return httpx.Response(401)
        return httpx.Response(200, json={"ok": True})

    server.transport = httpx.MockTransport(handler)
    return server


async def test_auth_sends_the_bearer_token(make_client, mcp_server):
    auth = mcp_service_account.service_account_auth(make_client())

    async with httpx.AsyncClient(transport=mcp_server.transport, auth=auth) as client:
        first = await client.post(MCP_URL, json={"id": 1})
        second = await client.post(MCP_URL, json={"id": 2})

    assert (first.status_code, second.status_code) == (200, 200)
    assert [seen[0] for seen in mcp_server.seen] == ["Bearer token-1", "Bearer token-1"]


async def test_auth_renews_once_after_401(make_client, mcp_server, token_endpoint):
    mcp_server.rejected = {"token-1"}
    auth = mcp_service_account.service_account_auth(make_client())

    async with httpx.AsyncClient(transport=mcp_server.transport, auth=auth) as client:
        response = await client.post(MCP_URL, json={"id": 1})

    assert response.status_code == 200
    assert [seen[0] for seen in mcp_server.seen] == ["Bearer token-1", "Bearer token-2"]
    # The retried request carries the original body.
    assert mcp_server.seen[0][1] == mcp_server.seen[1][1] == b'{"id":1}'
    assert len(token_endpoint.requests) == 2


async def test_auth_returns_a_second_401(make_client, mcp_server, token_endpoint):
    mcp_server.rejected = {"token-1", "token-2"}
    auth = mcp_service_account.service_account_auth(make_client())

    async with httpx.AsyncClient(transport=mcp_server.transport, auth=auth) as client:
        response = await client.post(MCP_URL, json={"id": 1})

    assert response.status_code == 401
    assert len(mcp_server.seen) == 2
    assert len(token_endpoint.requests) == 2


async def test_auth_propagates_token_errors(make_client, mcp_server, token_endpoint):
    token_endpoint.status = 401
    auth = mcp_service_account.service_account_auth(make_client())

    async with httpx.AsyncClient(transport=mcp_server.transport, auth=auth) as client:
        with pytest.raises(mcp_service_account.ServiceAccountTokenError):
            await client.post(MCP_URL, json={"id": 1})

    assert mcp_server.seen == []


@pytest.mark.parametrize(
    ("url", "sent"),
    [("http://mcp.example.test/mcp", False), ("http://127.0.0.1:8000/mcp", True)],
)
async def test_auth_sends_tokens_over_http_only_to_loopback(
    make_client, mcp_server, token_endpoint, url, sent
):
    auth = mcp_service_account.service_account_auth(make_client())

    async with httpx.AsyncClient(transport=mcp_server.transport, auth=auth) as client:
        if sent:
            await client.post(url, json={"id": 1})
        else:
            with pytest.raises(
                mcp_service_account.ServiceAccountTokenError, match="over http"
            ):
                await client.post(url, json={"id": 1})

    assert len(mcp_server.seen) == len(token_endpoint.requests) == int(sent)


def test_auth_rejects_sync_clients(make_client, mcp_server):
    auth = mcp_service_account.service_account_auth(make_client())

    with httpx.Client(transport=mcp_server.transport, auth=auth) as client:
        with pytest.raises(RuntimeError, match="requires an async client"):
            client.post(MCP_URL, json={"id": 1})
