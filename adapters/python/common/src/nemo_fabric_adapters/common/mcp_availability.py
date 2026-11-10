# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared report of configured MCP servers a runtime could not use.

Adapters start a runtime even when some configured MCP servers are unavailable
and report those servers in each result under ``output.mcp.unavailable_servers``.
Every entry has the same shape across adapters:

- ``name``: the configured server name.
- ``transport``: the configured transport, lowercase with ``-`` separators
  (for example ``stdio`` or ``streamable-http``).
- ``reason``: one of :data:`UNAVAILABLE_REASONS`.
- ``error_type``: the failure's exception type, when the adapter observed one.

Error messages are never reported, because they can contain URLs, headers or
other credentials.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Awaitable
from collections.abc import Iterable
from collections.abc import Mapping
from typing import Any

FAILED = "failed"
TIMED_OUT = "timed_out"
AUTHENTICATION_REQUIRED = "authentication_required"
UNAVAILABLE_REASONS = frozenset({FAILED, TIMED_OUT, AUTHENTICATION_REQUIRED})


def unavailable_server(
    name: str,
    transport: str,
    reason: str,
    error_type: str | None = None,
) -> dict[str, str]:
    """Return one report entry for a server the runtime could not use."""

    if reason not in UNAVAILABLE_REASONS:
        raise ValueError(f"unknown MCP unavailability reason {reason!r}")
    entry = {
        "name": name,
        "transport": transport.strip().lower().replace("_", "-"),
        "reason": reason,
    }
    if error_type:
        entry["error_type"] = error_type
    return entry


async def wait_bounded(
    awaitable: Awaitable[Any], *, timeout: float, grace: float
) -> tuple[asyncio.Future[Any], bool]:
    """Wait for one server's connection attempt without letting it stall startup.

    Returns the attempt's task and whether it timed out. After a timeout the task
    is cancelled and given ``grace`` seconds to close its connection; a task that
    is still closing then finishes in the background with its outcome discarded.
    Cancelling the caller cancels the attempt and propagates.
    """

    task = asyncio.ensure_future(awaitable)
    try:
        done, _ = await asyncio.wait({task}, timeout=timeout)
    except asyncio.CancelledError:
        task.cancel()
        # Cleanup can still raise; retrieve that outcome so asyncio never logs
        # its message, which may carry credentials.
        task.add_done_callback(_discard_outcome)
        raise
    if done:
        return task, False
    task.cancel()
    task.add_done_callback(_discard_outcome)
    await asyncio.wait({task}, timeout=grace)
    return task, True


def _discard_outcome(task: asyncio.Future[Any]) -> None:
    """Retrieve an abandoned task's outcome so asyncio does not log it."""
    if not task.cancelled():
        task.exception()


def exception_type(error: BaseException) -> str:
    """Name the underlying failure without its message, which may carry secrets."""

    while isinstance(error, BaseExceptionGroup) and error.exceptions:
        error = error.exceptions[0]
    return type(error).__name__


def output_block(
    unavailable: Iterable[Mapping[str, str]],
) -> dict[str, Any] | None:
    """Return the ``output.mcp`` block, or ``None`` when every server was usable."""

    servers = sorted((dict(entry) for entry in unavailable), key=lambda e: e["name"])
    if not servers:
        return None
    return {"unavailable_servers": servers}


def warn(adapter: str, entries: Iterable[Mapping[str, str]]) -> None:
    """Write one standard-error warning per unavailable server."""

    for entry in entries:
        detail = entry["reason"]
        if error_type := entry.get("error_type"):
            detail = f"{detail} ({error_type})"
        print(
            f"NeMo Fabric {adapter}: MCP server {entry['name']!r} "
            f"({entry['transport']}) is unavailable: {detail}",
            file=sys.stderr,
        )
