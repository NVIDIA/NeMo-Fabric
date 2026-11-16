# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OAuth 2.0 client-credentials tokens for MCP servers (``service_account``).

An adapter whose MCP client accepts an ``httpx.Auth`` builds a
:class:`ServiceAccountTokenClient` from the server's ``service_account``
authentication and passes :func:`service_account_auth` to the client. Every
request then carries ``Authorization: Bearer <token>``. The token is fetched
from the configured ``token_url``, cached in memory only, and renewed before it
expires. If the MCP server still answers ``401``, the auth fetches a new token
and retries the request once.

The token cache and early renewal follow the NeMo Agent Toolkit service-account
token client.

This module needs ``httpx``, which it imports only when a token is requested or
an auth is built. Adapters that use it must depend on ``httpx`` themselves.
Error messages never include the client secret, the access token or the token
endpoint's response body.
"""

from __future__ import annotations

import asyncio
import base64
import ipaddress
import os
import time
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Mapping
from typing import Any
from urllib.parse import quote
from urllib.parse import urlsplit

CLIENT_SECRET_BASIC = "client_secret_basic"
CLIENT_SECRET_POST = "client_secret_post"
TOKEN_ENDPOINT_AUTH_METHODS = frozenset({CLIENT_SECRET_BASIC, CLIENT_SECRET_POST})
# Lifetime assumed when the token response omits ``expires_in``.
DEFAULT_TOKEN_LIFETIME_SECONDS = 3600.0
DEFAULT_TOKEN_CACHE_BUFFER_SECONDS = 300.0
TOKEN_REQUEST_TIMEOUT_SECONDS = 30.0


class ServiceAccountTokenError(RuntimeError):
    """The token endpoint did not issue an access token.

    ``rejected`` is true when the endpoint refused the client credentials
    (HTTP 400 or 401), so the server needs different credentials rather than a
    retry. Otherwise the endpoint was unreachable or failed.
    """

    def __init__(self, message: str, *, rejected: bool) -> None:
        super().__init__(message)
        self.rejected = rejected


def validate_token_url(server_name: str, token_url: str) -> str:
    """Return ``token_url`` if it is ``https``, or ``http`` on a loopback host.

    Client credentials must not cross the network in clear text.
    """

    url = token_url.strip()
    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    host = parts.hostname or ""
    if scheme == "https" and host:
        return url
    if scheme == "http" and _is_loopback(host):
        return url
    raise ValueError(
        f"MCP server {server_name!r} authentication.token_url must use https "
        "(http is accepted only for a loopback host)"
    )


def _is_loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def resolve_client_secret(
    server_name: str,
    client_secret_env: str,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Read the client secret from the named environment variable.

    Raises ``ValueError`` naming the variable, never its value, when it is
    unset or empty.
    """

    environ = os.environ if environ is None else environ
    secret = environ.get(client_secret_env, "")
    if not secret.strip():
        raise ValueError(
            f"MCP server {server_name!r} authentication.client_secret_env "
            f"{client_secret_env!r} is not set in the environment"
        )
    return secret


def secret_env_names(authentications: Iterable[Any]) -> set[str]:
    """Return the ``client_secret_env`` names of ``service_account`` configs."""

    return {
        auth.client_secret_env
        for auth in authentications
        if getattr(auth, "type", None) == "service_account"
        and getattr(auth, "client_secret_env", None)
    }


class ServiceAccountTokenClient:
    """Fetch and cache one MCP server's client-credentials access token."""

    def __init__(
        self,
        *,
        client_id: str,
        client_secret: str,
        token_url: str,
        scopes: Iterable[str] = (),
        token_endpoint_auth_method: str | None = None,
        token_cache_buffer_seconds: float = DEFAULT_TOKEN_CACHE_BUFFER_SECONDS,
        timeout: float = TOKEN_REQUEST_TIMEOUT_SECONDS,
        clock: Callable[[], float] = time.monotonic,
        http_client_factory: Callable[[], Any] | None = None,
    ) -> None:
        method = token_endpoint_auth_method or CLIENT_SECRET_BASIC
        if method not in TOKEN_ENDPOINT_AUTH_METHODS:
            raise ValueError(
                "service_account token_endpoint_auth_method must be "
                "client_secret_basic or client_secret_post"
            )
        self._client_id = client_id
        self._client_secret = client_secret
        self._token_url = token_url
        self._scope = " ".join(scopes)
        self._method = method
        self._buffer = max(0.0, float(token_cache_buffer_seconds))
        self._timeout = timeout
        self._clock = clock
        self._http_client_factory = http_client_factory
        self._token: str | None = None
        self._renew_at = 0.0
        self._lock: asyncio.Lock | None = None

    def __repr__(self) -> str:
        return f"{type(self).__name__}(token_url={self._token_url!r})"

    def _cached(self) -> str | None:
        if self._token is not None and self._clock() < self._renew_at:
            return self._token
        return None

    async def get_access_token(self, *, stale: str | None = None) -> str:
        """Return a valid access token, fetching one when needed.

        Pass the token a server just rejected as ``stale`` to force a new one.
        Concurrent callers share one token request.
        """

        token = self._cached()
        if token is not None and token != stale:
            return token
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            token = self._cached()
            if token is not None and token != stale:
                return token
            return await self._fetch()

    async def _fetch(self) -> str:
        import httpx

        data = {"grant_type": "client_credentials"}
        if self._scope:
            data["scope"] = self._scope
        headers = {"Accept": "application/json"}
        if self._method == CLIENT_SECRET_BASIC:
            # RFC 6749 section 2.3.1: form-encode the id and secret first.
            pair = f"{quote(self._client_id, safe='')}:{quote(self._client_secret, safe='')}"
            headers["Authorization"] = "Basic " + base64.b64encode(pair.encode()).decode()
        else:
            data["client_id"] = self._client_id
            data["client_secret"] = self._client_secret

        factory = self._http_client_factory or (
            lambda: httpx.AsyncClient(timeout=self._timeout)
        )
        try:
            async with factory() as client:
                response = await client.post(self._token_url, data=data, headers=headers)
        except httpx.HTTPError as error:
            raise ServiceAccountTokenError(
                f"token request failed ({type(error).__name__})", rejected=False
            ) from None

        if response.status_code in (400, 401):
            raise ServiceAccountTokenError(
                f"token endpoint rejected the client credentials (HTTP {response.status_code})",
                rejected=True,
            )
        if response.status_code != 200:
            raise ServiceAccountTokenError(
                f"token endpoint returned HTTP {response.status_code}", rejected=False
            )
        try:
            body = response.json()
        except ValueError:
            body = None
        token = body.get("access_token") if isinstance(body, dict) else None
        if not isinstance(token, str) or not token:
            raise ServiceAccountTokenError(
                "token response has no access_token", rejected=False
            )
        lifetime = _lifetime(body.get("expires_in"))
        # Renew ahead of expiry, but never use more than half of a short-lived
        # token's lifetime as the buffer, or every request would fetch a token.
        buffer = min(self._buffer, lifetime / 2)
        self._token = token
        self._renew_at = self._clock() + lifetime - buffer
        return token


def _lifetime(value: Any) -> float:
    try:
        lifetime = float(value)
    except (TypeError, ValueError):
        return DEFAULT_TOKEN_LIFETIME_SECONDS
    if lifetime <= 0:
        return DEFAULT_TOKEN_LIFETIME_SECONDS
    return lifetime


def service_account_auth(token_client: ServiceAccountTokenClient) -> Any:
    """Return an ``httpx.Auth`` that sends the client's bearer token.

    On a ``401`` the auth fetches a new token and retries the request once. A
    second ``401`` is returned to the caller unchanged.
    """

    import httpx

    class _ServiceAccountAuth(httpx.Auth):
        requires_request_body = True

        def __init__(self, client: ServiceAccountTokenClient) -> None:
            self.token_client = client

        def __repr__(self) -> str:
            return f"{type(self).__name__}({self.token_client!r})"

        def sync_auth_flow(self, request: httpx.Request) -> Any:
            raise RuntimeError("service_account authentication requires an async client")

        async def async_auth_flow(self, request: httpx.Request) -> Any:
            token = await self.token_client.get_access_token()
            request.headers["Authorization"] = f"Bearer {token}"
            response = yield request
            if response.status_code != 401:
                return
            token = await self.token_client.get_access_token(stale=token)
            request.headers["Authorization"] = f"Bearer {token}"
            yield request

    return _ServiceAccountAuth(token_client)
