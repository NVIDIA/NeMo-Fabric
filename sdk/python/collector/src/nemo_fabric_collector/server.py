# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Programmatic lifecycle management for an embedded collector server."""

from __future__ import annotations

import asyncio
import socket
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager

import uvicorn

from nemo_fabric_collector.app import create_app


class _EmbeddedServer(uvicorn.Server):
    """Run Uvicorn without disabling the default signal handling from the host application."""

    @contextmanager
    def capture_signals(self) -> Iterator[None]:
        yield


@asynccontextmanager
async def serve_collector(
    *,
    host: str = "127.0.0.1",
    port: int = 0,
) -> AsyncIterator[str]:
    """Run a collector until the asynchronous context exits."""

    application = create_app()
    config = uvicorn.Config(
        application,
        host=host,
        port=port,
        log_level="warning",
        access_log=False,
        log_config=None,
    )
    listener = socket.create_server((host, port), backlog=config.backlog)
    listener.setblocking(False)
    bound_host, bound_port = listener.getsockname()[:2]
    server = _EmbeddedServer(config)
    server_task = asyncio.create_task(server.serve(sockets=[listener]))
    try:
        while not server.started:
            if server_task.done():
                await server_task
                raise RuntimeError("ATOF collector stopped before startup completed")
            await asyncio.sleep(0.01)
        yield f"http://{bound_host}:{bound_port}"
    finally:
        server.should_exit = True
        try:
            await server_task
        finally:
            listener.close()
