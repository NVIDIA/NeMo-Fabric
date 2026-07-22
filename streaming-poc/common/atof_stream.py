# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SDK-side loopback ATOF stream listener (Fabric streaming POC).

Stands up a loopback ndjson sink that NeMo Relay pushes ATOF records into, and
exposes them as an async queue. Handles the large records gateway harnesses
(Claude/Codex) emit — they embed the full model request/response, so a single
line can exceed aiohttp's default 512 KB readline limit; we read raw chunks and
split on newlines ourselves.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from aiohttp import web


class AtofStreamListener:
    """Receive live ndjson ATOF push; expose raw records via an async queue."""

    def __init__(
        self, host: str = "127.0.0.1", port: int = 0, maxsize: int = 0
    ) -> None:
        self._host = host
        self._port = port
        # maxsize>0 bounds memory: a full queue blocks the handler's put(), which
        # stops reading the socket -> TCP backpressure -> Relay's sender backs off.
        self._queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=maxsize)
        self._runner: web.AppRunner | None = None
        self._bound_port: int | None = None

    @property
    def url(self) -> str:
        if self._bound_port is None:
            raise RuntimeError("listener not started")
        return f"http://{self._host}:{self._bound_port}/atof"

    @property
    def records(self) -> "asyncio.Queue[Any]":
        return self._queue

    async def start(self) -> "AtofStreamListener":
        app = web.Application(client_max_size=256 * 1024 * 1024)
        app.router.add_route("*", "/{tail:.*}", self._handle)
        self._runner = web.AppRunner(app, access_log=None)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self._host, self._port)
        await site.start()
        self._bound_port = self._runner.addresses[0][1]  # discover ephemeral port
        return self

    async def _handle(self, request: web.Request) -> web.Response:
        buf = b""
        async for chunk in request.content.iter_chunked(65536):
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                await self._emit(line)
        await self._emit(buf)
        return web.Response(text="ok")

    async def _emit(self, line: bytes) -> None:
        line = line.strip()
        if not line:
            return
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            return
        await self._queue.put(rec)

    async def close(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
