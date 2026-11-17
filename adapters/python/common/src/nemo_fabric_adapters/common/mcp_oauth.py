# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OAuth 2.0 browser sign-in for MCP servers (``oauth2``).

An adapter whose MCP client accepts an ``httpx.Auth`` passes the result of
:func:`browser_sign_in_auth` to the client. The auth is the MCP SDK's
``OAuthClientProvider`` (authorization code with PKCE), with tokens stored in
owner-only files and the authorization redirect received on a loopback
callback.

Sign-in is off until the adapter turns it on through the auth's
:class:`BrowserSignIn` controller (``auth.sign_in.interactive``). While it is
off, a server that needs a new sign-in fails with :class:`McpSignInRequired`
instead of opening a browser. Stored tokens are used, and refreshed when they
expire, either way.

Files in the token directory, in one folder per server:

- ``<server>-<digest>/tokens.json``: tokens, with an absolute ``expires_at``.
- ``<server>-<digest>/client.json``: the registered or configured OAuth client.
- ``<server>-<digest>/metadata.json``: the authorization server metadata, so
  refresh uses its token endpoint.

``<digest>`` is a hash of the full server name, so no two servers share a folder.

Every HTTP request the auth's MCP client sends, including discovered OAuth
endpoints and redirects, must use ``https`` (``http`` only on a loopback host);
pass :func:`secure_http_client_factory` as the client factory.

This module needs ``mcp`` and ``httpx``, which it imports only when an auth is
built. Adapters that use it must depend on both.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import ipaddress
import json
import os
import re
import signal
import socket
import sys
import time
from collections.abc import Callable
from collections.abc import Awaitable
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs
from urllib.parse import urlsplit

TOKEN_DIRECTORY_NAME = "mcp-tokens"
CALLBACK_PATH = "/callback"
DEFAULT_CLIENT_NAME = "NeMo Fabric"
DEFAULT_AUTHORIZATION_TIMEOUT_SECONDS = 300.0
# Bounds reading one callback request so a stalled connection cannot hold sign-in.
CALLBACK_REQUEST_TIMEOUT_SECONDS = 10.0
_CALLBACK_PAGE = (
    b"<!doctype html><title>Signed in</title>"
    b"<p>Sign-in complete. You can close this tab.</p>"
)


class McpSignInRequired(RuntimeError):
    """The server needs a browser sign-in, and sign-in is not on."""


class McpSignInError(RuntimeError):
    """A browser sign-in was started but did not complete."""


class McpInsecureRequestError(RuntimeError):
    """An OAuth MCP client refused to send a request over plain HTTP."""


def is_sign_in_error(error: BaseException) -> bool:
    """Return whether ``error`` means the server needs a (new) sign-in."""

    if isinstance(error, (McpSignInRequired, McpSignInError)):
        return True
    # The MCP SDK's OAuth errors; matched by name so callers need not import mcp.
    return type(error).__name__ in {
        "OAuthFlowError",
        "OAuthTokenError",
        "OAuthRegistrationError",
    }


def storage_name(server_name: str) -> str:
    """Return a folder name for ``server_name`` that cannot leave the directory.

    The readable part keeps only safe characters; the digest of the full name
    keeps two different server names from sharing a folder.
    """

    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", server_name).lstrip(".")
    digest = hashlib.sha256(server_name.encode()).hexdigest()[:16]
    return f"{safe}-{digest}" if safe else digest


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _write_private_json(path: Path, data: dict[str, Any]) -> None:
    """Write ``data`` atomically to an owner-only file in an owner-only folder."""

    directory = path.parent
    directory.mkdir(parents=True, exist_ok=True)
    # The token directory and the server's folder inside it.
    for folder in (directory.parent, directory):
        with contextlib.suppress(OSError):
            os.chmod(folder, 0o700)
    temporary = directory / f".{path.name}.{os.getpid()}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            json.dump(data, file)
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(OSError):
            temporary.unlink()
        raise


class FileTokenStorage:
    """MCP SDK ``TokenStorage`` backed by owner-only JSON files."""

    def __init__(
        self,
        directory: Path,
        server_name: str,
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        # One folder per server, so no server name can collide with another
        # server's files.
        server_directory = Path(directory) / storage_name(server_name)
        self._tokens_path = server_directory / "tokens.json"
        self._client_path = server_directory / "client.json"
        self._metadata_path = server_directory / "metadata.json"
        self._clock = clock

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._tokens_path.parent!s})"

    def has_tokens(self) -> bool:
        data = _read_json(self._tokens_path)
        return bool(data and data.get("access_token"))

    async def get_tokens(self) -> Any:
        from mcp.shared.auth import OAuthToken

        data = _read_json(self._tokens_path)
        if data is None:
            return None
        expires_at = data.pop("expires_at", None)
        if isinstance(expires_at, (int, float)):
            # Report the remaining lifetime so a token that expired while no
            # runtime was running is refreshed before its first use.
            data["expires_in"] = int(max(expires_at - self._clock(), 0))
        try:
            return OAuthToken.model_validate(data)
        except ValueError:
            return None

    async def set_tokens(self, tokens: Any) -> None:
        data = tokens.model_dump(mode="json", exclude_none=True)
        if isinstance(tokens.expires_in, int):
            data["expires_at"] = self._clock() + tokens.expires_in
        _write_private_json(self._tokens_path, data)

    async def get_client_info(self) -> Any:
        from mcp.shared.auth import OAuthClientInformationFull

        data = _read_json(self._client_path)
        if data is None:
            return None
        try:
            return OAuthClientInformationFull.model_validate(data)
        except ValueError:
            return None

    async def set_client_info(self, client_info: Any) -> None:
        _write_private_json(
            self._client_path, client_info.model_dump(mode="json", exclude_none=True)
        )

    def get_metadata(self) -> Any:
        from mcp.shared.auth import OAuthMetadata

        data = _read_json(self._metadata_path)
        if data is None:
            return None
        try:
            return OAuthMetadata.model_validate(data)
        except ValueError:
            return None

    def set_metadata(self, metadata: Any) -> None:
        _write_private_json(
            self._metadata_path, metadata.model_dump(mode="json", exclude_none=True)
        )

    def cached_redirect_uri(self) -> str | None:
        """Return the loopback redirect URI of the stored client, if any."""

        data = _read_json(self._client_path) or {}
        for uri in data.get("redirect_uris") or []:
            if isinstance(uri, str) and _loopback_redirect(uri) is not None:
                return uri
        return None


def _is_loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _loopback_redirect(uri: str) -> tuple[str, int, str] | None:
    """Return ``(host, port, path)`` for an ``http`` loopback URI with a port."""

    parts = urlsplit(uri)
    host = parts.hostname or ""
    try:
        port = parts.port
    except ValueError:
        return None
    if parts.scheme != "http" or not _is_loopback(host) or not port:
        return None
    return host, port, parts.path or "/"


def validate_redirect_uri(server_name: str, redirect_uri: str) -> str:
    """Return ``redirect_uri`` if the adapter can receive the redirect on it."""

    uri = redirect_uri.strip()
    if _loopback_redirect(uri) is None:
        raise ValueError(
            f"MCP server {server_name!r} authentication.redirect_uri must be an "
            "http loopback URL with a port, such as http://127.0.0.1:8765/callback"
        )
    return uri


def _free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


class LoopbackCallback:
    """Receive one authorization redirect on a loopback HTTP listener."""

    def __init__(self, redirect_uri: str) -> None:
        parsed = _loopback_redirect(redirect_uri)
        if parsed is None:
            raise ValueError("redirect_uri must be an http loopback URL with a port")
        self._host, self._port, self._path = parsed
        self._server: asyncio.base_events.Server | None = None
        self._result: asyncio.Future[tuple[str, str | None]] | None = None

    async def start(self) -> None:
        self._result = asyncio.get_running_loop().create_future()
        # "localhost" binds every loopback address it resolves to.
        self._server = await asyncio.start_server(
            self._handle, host=self._host.strip("[]"), port=self._port
        )

    async def wait(self) -> tuple[str, str | None]:
        if self._result is None:
            raise RuntimeError("the callback listener is not started")
        return await self._result

    async def close(self) -> None:
        server, self._server = self._server, None
        if server is not None:
            server.close()
            with contextlib.suppress(Exception):
                await server.wait_closed()

    async def _handle(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            async with asyncio.timeout(CALLBACK_REQUEST_TIMEOUT_SECONDS):
                request_line = await reader.readline()
                while (await reader.readline()).strip():
                    pass
            parts = request_line.decode("latin-1").split()
            target = urlsplit(parts[1]) if len(parts) >= 2 else None
            if target is None or target.path != self._path:
                writer.write(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n")
                return
            query = parse_qs(target.query)
            result = self._result
            if result is not None and not result.done():
                if error := query.get("error"):
                    result.set_exception(
                        McpSignInError(f"authorization failed ({error[0][:64]})")
                    )
                elif code := query.get("code"):
                    state = query.get("state")
                    result.set_result((code[0], state[0] if state else None))
                else:
                    result.set_exception(
                        McpSignInError("authorization redirect has no code")
                    )
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\n"
                + f"Content-Length: {len(_CALLBACK_PAGE)}\r\n".encode()
                + b"Connection: close\r\n\r\n"
                + _CALLBACK_PAGE
            )
        except (TimeoutError, OSError, UnicodeError):
            pass
        finally:
            with contextlib.suppress(Exception):
                await writer.drain()
                writer.close()
                await writer.wait_closed()


async def open_in_browser(url: str) -> bool:
    """Open ``url`` in a process that cancellation can kill.

    ``webbrowser.open`` can block, and a worker thread cannot be cancelled. The
    child's output is discarded so it cannot reach the adapter's protocol
    streams.
    """

    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-I",
        "-c",
        "import sys, webbrowser; sys.exit(0 if webbrowser.open(sys.argv[1]) else 1)",
        url,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
        start_new_session=True,  # POSIX only: lets cancellation kill launcher children.
    )
    try:
        return await process.wait() == 0
    finally:
        if process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
            await process.wait()


class BrowserSignIn:
    """Controls when one MCP server may open a browser sign-in."""

    def __init__(
        self,
        server_name: str,
        redirect_uri: str,
        *,
        timeout: float = DEFAULT_AUTHORIZATION_TIMEOUT_SECONDS,
        open_browser: Callable[[str], Awaitable[bool]] = open_in_browser,
    ) -> None:
        self.server_name = server_name
        self.redirect_uri = redirect_uri
        self.timeout = float(timeout)
        self.interactive = False
        self._open_browser = open_browser
        self._callback: LoopbackCallback | None = None

    async def redirect(self, authorization_url: str) -> None:
        if not self.interactive:
            raise McpSignInRequired(
                f"MCP server {self.server_name!r} needs a browser sign-in"
            )
        await self._close_callback()
        callback = LoopbackCallback(self.redirect_uri)
        try:
            await callback.start()
        except OSError as error:
            raise McpSignInError(
                f"cannot listen for the sign-in redirect on {self.redirect_uri} "
                f"({type(error).__name__})"
            ) from None
        self._callback = callback
        # The URL carries only public request parameters (no secret or token).
        print(
            f"NeMo Fabric: sign in to MCP server {self.server_name!r} in your "
            f"browser. If it does not open, visit:\n  {authorization_url}",
            file=sys.stderr,
            flush=True,
        )
        try:
            await self._open_browser(authorization_url)
        except BaseException:
            # The SDK only waits for the callback, and closes it, after this
            # returns; stop listening before the failure propagates.
            await self._close_callback()
            raise

    async def callback(self) -> tuple[str, str | None]:
        callback = self._callback
        if callback is None:
            raise McpSignInError("sign-in was not started")
        try:
            async with asyncio.timeout(self.timeout):
                return await callback.wait()
        except TimeoutError:
            raise McpSignInError(
                f"sign-in was not completed within {self.timeout:g} seconds"
            ) from None
        finally:
            await self._close_callback()

    async def _close_callback(self) -> None:
        callback, self._callback = self._callback, None
        if callback is not None:
            await callback.close()


def secure_http_client_factory(server_name: str) -> Callable[..., Any]:
    """Return an MCP ``httpx_client_factory`` that refuses plaintext requests.

    The OAuth flow sends refresh tokens and client credentials to endpoints the
    server advertises, and the MCP client follows redirects. The returned
    client checks every request, redirects included, and refuses one that is
    not ``https`` unless it targets a loopback host. It also refuses any
    redirect from ``https`` to ``http``, loopback included, because a ``307``
    or ``308`` redirect resends the request body.
    """

    from mcp.shared._httpx_utils import create_mcp_http_client

    from nemo_fabric_adapters.common.mcp_service_account import is_secure_url

    async def refuse_plaintext(request: Any) -> None:
        if not is_secure_url(str(request.url)):
            raise McpInsecureRequestError(
                f"MCP server {server_name!r} refused a request to "
                f"{request.url.scheme}://{request.url.host}: OAuth requests must "
                "use https (http is accepted only for a loopback host)"
            )

    async def refuse_downgrade(response: Any) -> None:
        if not response.has_redirect_location:
            return
        source = response.request.url
        target = source.join(response.headers["Location"])
        if source.scheme == "https" and target.scheme != "https":
            raise McpInsecureRequestError(
                f"MCP server {server_name!r} refused a redirect from https to "
                f"{target.scheme}://{target.host}"
            )

    def factory(
        headers: dict[str, str] | None = None,
        timeout: Any = None,
        auth: Any = None,
    ) -> Any:
        client = create_mcp_http_client(headers=headers, timeout=timeout, auth=auth)
        hooks = client.event_hooks
        client.event_hooks = {
            "request": [*hooks.get("request", []), refuse_plaintext],
            "response": [*hooks.get("response", []), refuse_downgrade],
        }
        return client

    return factory


def browser_sign_in_auth(
    server_name: str,
    server_url: str,
    token_directory: Path,
    *,
    client_id: str | None = None,
    client_secret: str | None = None,
    client_name: str | None = None,
    scopes: list[str] | tuple[str, ...] = (),
    redirect_uri: str | None = None,
    token_endpoint_auth_method: str | None = None,
    authorization_timeout_seconds: float = DEFAULT_AUTHORIZATION_TIMEOUT_SECONDS,
    open_browser: Callable[[str], Awaitable[bool]] = open_in_browser,
) -> Any:
    """Return an ``httpx.Auth`` that signs in to ``server_url`` through a browser.

    The returned auth has a ``sign_in`` attribute, the server's
    :class:`BrowserSignIn`. Sign-in starts off.
    """

    from mcp.client.auth import OAuthClientProvider
    from mcp.shared.auth import OAuthClientInformationFull
    from mcp.shared.auth import OAuthClientMetadata

    storage = FileTokenStorage(token_directory, server_name)
    redirect = (
        validate_redirect_uri(server_name, redirect_uri)
        if redirect_uri
        else storage.cached_redirect_uri()
        or f"http://127.0.0.1:{_free_loopback_port()}{CALLBACK_PATH}"
    )
    scope = " ".join(scopes) or None
    method = token_endpoint_auth_method or (
        "client_secret_post" if client_secret else "none"
    )
    metadata = OAuthClientMetadata(
        client_name=client_name or DEFAULT_CLIENT_NAME,
        redirect_uris=[redirect],
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
        token_endpoint_auth_method=method,
        scope=scope,
    )
    if client_id:
        # A configured client replaces any earlier registration for this server.
        client = OAuthClientInformationFull(
            **metadata.model_dump(exclude_none=True),
            client_id=client_id,
            client_secret=client_secret,
        )
        _write_private_json(
            storage._client_path, client.model_dump(mode="json", exclude_none=True)
        )
    sign_in = BrowserSignIn(
        server_name,
        redirect,
        timeout=authorization_timeout_seconds,
        open_browser=open_browser,
    )

    class _BrowserSignInAuth(OAuthClientProvider):
        def __repr__(self) -> str:
            return f"{type(self).__name__}({server_name!r}, {storage!r})"

        async def _initialize(self) -> None:
            tokens = await storage.get_tokens()
            self.context.current_tokens = tokens
            if tokens is not None and tokens.expires_in is not None:
                self.context.update_token_expiry(tokens)
                if tokens.expires_in <= 0:
                    # An expiry of "now" still counts as valid within the same
                    # clock tick; mark the token expired so it is refreshed.
                    self.context.token_expiry_time = time.time() - 1
            self.context.client_info = await storage.get_client_info()
            if self.context.oauth_metadata is None:
                self.context.oauth_metadata = storage.get_metadata()
            self._initialized = True

        async def _perform_authorization_code_grant(self) -> tuple[str, str]:
            if scope:
                # The SDK replaces the requested scope with the server's
                # advertised scopes; configured scopes take precedence.
                self.context.client_metadata.scope = scope
            return await super()._perform_authorization_code_grant()

        async def _handle_token_response(self, response: Any) -> None:
            await super()._handle_token_response(response)
            if self.context.oauth_metadata is not None:
                storage.set_metadata(self.context.oauth_metadata)

    auth = _BrowserSignInAuth(
        server_url=server_url,
        client_metadata=metadata,
        storage=storage,
        redirect_handler=sign_in.redirect,
        callback_handler=sign_in.callback,
        timeout=float(authorization_timeout_seconds),
    )
    auth.sign_in = sign_in
    auth.storage = storage
    return auth
