# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One authenticated Gateway connection for a runtime's native session checks."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from typing import Any
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import InvalidHandshake, WebSocketException

from nemo_fabric_adapters.common.lifecycle import LifecycleError


# Gateway frames contain credentials, conversation history, and tool arguments.
# Keep this unregistered logger private so global DEBUG settings cannot emit frames.
_TRANSPORT_LOGGER = logging.Logger(__name__ + ".transport")
_TRANSPORT_LOGGER.disabled = True


class _DirectConnect(connect):
    def process_redirect(self, exc: Exception) -> Exception:
        # Connection material is valid only for the selected Gateway.
        return InvalidHandshake("Gateway redirects are unsupported")


class SessionClient:
    def __init__(self, gateway_url: str, token: str):
        parsed = urlsplit(gateway_url)
        self._url = urlunsplit(
            ("wss" if parsed.scheme == "https" else "ws", parsed.netloc, "/", "", "")
        )
        self._token = token
        self._connection: ClientConnection | None = None
        self._lock = asyncio.Lock()
        self._failed = False
        self._reader: asyncio.Task[None] | None = None

    async def _read_events(self, connection: ClientConnection) -> None:
        # The Gateway keeps sending events between calls, including during an
        # HTTP turn. Left unread, they fill the websockets receive buffer, which
        # then stops reading the socket, so keepalive pongs go unseen and
        # websockets closes the connection.
        try:
            while True:
                frame = json.loads(await connection.recv())
                if not isinstance(frame, dict) or frame.get("type") != "event":
                    raise ValueError("Unexpected Gateway frame")
        except (ValueError, WebSocketException):
            # The next call reports the connection as unavailable.
            self._failed = True

    async def _stop_reading(self) -> None:
        reader, self._reader = self._reader, None
        if reader is not None:
            # Cancelling recv() loses no frame; the next call reads it.
            reader.cancel()
            await asyncio.wait([reader])

    async def _response(self, identifier: str) -> dict[str, Any]:
        assert self._connection is not None
        while True:
            frame = json.loads(await self._connection.recv())
            if not isinstance(frame, dict):
                raise ValueError("Invalid Gateway frame")
            if frame.get("type") == "event":
                continue
            if (
                frame.get("type") != "res"
                or frame.get("id") != identifier
                or frame.get("ok") is not True
                or not isinstance(frame.get("payload"), dict)
            ):
                raise ValueError("Gateway operation was not confirmed")
            return frame["payload"]

    async def _request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        assert self._connection is not None
        identifier = str(uuid4())
        await self._connection.send(
            json.dumps(
                {"type": "req", "id": identifier, "method": method, "params": params}
            )
        )
        return await self._response(identifier)

    async def call(
        self, method: str, params: dict[str, Any], *, timeout: float
    ) -> dict[str, Any]:
        async with self._lock:
            try:
                # Cancellation here still closes the connection below.
                await self._stop_reading()
                if self._failed:
                    raise LifecycleError(
                        "session_incompatible",
                        "Native session connection is unavailable",
                    )
                async with asyncio.timeout(timeout):
                    if self._connection is None:
                        self._connection = await _DirectConnect(
                            self._url,
                            proxy=None,
                            logger=_TRANSPORT_LOGGER,
                            open_timeout=timeout,
                            close_timeout=0.25,
                            max_size=16 * 1024 * 1024,
                        )
                        challenge = json.loads(await self._connection.recv())
                        if (
                            not isinstance(challenge, dict)
                            or challenge.get("type") != "event"
                            or challenge.get("event") != "connect.challenge"
                        ):
                            raise ValueError("Gateway challenge is unavailable")
                        hello = await self._request(
                            "connect",
                            {
                                "minProtocol": 4,
                                "maxProtocol": 4,
                                "client": {
                                    "id": "gateway-client",
                                    "displayName": "NeMo Fabric",
                                    "version": "1",
                                    "platform": sys.platform,
                                    "mode": "backend",
                                },
                                "role": "operator",
                                "scopes": ["operator.admin"],
                                "auth": {"token": self._token},
                            },
                        )
                        if (
                            hello.get("type") != "hello-ok"
                            or hello.get("protocol") != 4
                        ):
                            raise ValueError("Unsupported Gateway protocol")
                    result = await self._request(method, params)
            except BaseException as error:
                await self.close()
                if isinstance(
                    error, (OSError, ValueError, TimeoutError, WebSocketException)
                ):
                    # Native errors can contain prompts or authentication material.
                    raise LifecycleError(
                        "session_incompatible",
                        "Native session operation could not be confirmed",
                        metadata={"method": method},
                    ) from None
                raise
            self._reader = asyncio.create_task(self._read_events(self._connection))
            return result

    async def close(self) -> None:
        self._failed = True
        connection, self._connection = self._connection, None
        try:
            await self._stop_reading()
        finally:
            if connection is not None:
                await connection.close()
