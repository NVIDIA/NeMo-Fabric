# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared report of unavailable MCP servers."""

from __future__ import annotations

import asyncio

import pytest
from nemo_fabric_adapters.common import mcp_availability


def test_entry_omits_unknown_error_type():
    """Native harnesses that report only a state produce no error_type."""
    assert mcp_availability.unavailable_server("jira", "sse", "failed") == {
        "name": "jira",
        "transport": "sse",
        "reason": "failed",
    }


def test_entry_rejects_unknown_reason():
    """Reasons are a closed set so consumers can branch on them."""
    with pytest.raises(ValueError, match="unknown MCP unavailability reason"):
        mcp_availability.unavailable_server("jira", "sse", "down")


def test_exception_type_unwraps_exception_groups():
    """Report the underlying failure, not the TaskGroup wrapper."""
    error = ExceptionGroup("task group", [FileNotFoundError("missing --key secret")])
    assert mcp_availability.exception_type(error) == "FileNotFoundError"


def test_output_block_is_sorted_and_absent_when_empty():
    """Clean runs keep their shape; reports are ordered by server name."""
    assert mcp_availability.output_block([]) is None
    block = mcp_availability.output_block(
        [
            mcp_availability.unavailable_server("zeta", "stdio", "failed"),
            mcp_availability.unavailable_server("alpha", "sse", "timed_out"),
        ]
    )
    assert [server["name"] for server in block["unavailable_servers"]] == [
        "alpha",
        "zeta",
    ]


def test_warn_writes_one_line_per_server(capsys):
    """Warnings name the server, transport and reason only."""
    mcp_availability.warn(
        "Example",
        [
            mcp_availability.unavailable_server(
                "jira", "streamable_http", "failed", "ConnectError"
            ),
            mcp_availability.unavailable_server(
                "docs", "stdio", "authentication_required"
            ),
        ],
    )
    assert capsys.readouterr().err.splitlines() == [
        "NeMo Fabric Example: MCP server 'jira' (streamable-http) is unavailable: "
        "failed (ConnectError)",
        "NeMo Fabric Example: MCP server 'docs' (stdio) is unavailable: "
        "authentication_required",
    ]


def test_entry_normalizes_transport_spelling():
    """Every adapter reports one spelling for the same configured transport."""
    entry = mcp_availability.unavailable_server("jira", " Streamable_HTTP ", "failed")
    assert entry["transport"] == "streamable-http"


async def test_wait_bounded_returns_finished_task():
    """A connection attempt that finishes in time is returned as done."""

    async def connect():
        return "tools"

    task, timed_out = await mcp_availability.wait_bounded(connect(), timeout=1, grace=1)
    assert timed_out is False
    assert task.result() == "tools"


async def test_wait_bounded_does_not_wait_past_grace_for_cleanup():
    """A stalled attempt with slow cleanup times out within timeout plus grace."""
    cleanup_started = asyncio.Event()

    async def stall_then_close_slowly():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cleanup_started.set()
            await asyncio.sleep(30)
            raise

    task, timed_out = await asyncio.wait_for(
        mcp_availability.wait_bounded(
            stall_then_close_slowly(), timeout=0.01, grace=0.05
        ),
        timeout=5,
    )
    assert timed_out is True
    assert cleanup_started.is_set()
    assert not task.done()
    task.cancel()


async def test_wait_bounded_propagates_caller_cancellation():
    """Cancelling the caller cancels the attempt instead of reporting it."""
    attempt_cancelled = asyncio.Event()

    async def stall():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            attempt_cancelled.set()
            raise

    waiter = asyncio.create_task(
        mcp_availability.wait_bounded(stall(), timeout=30, grace=1)
    )
    await asyncio.sleep(0.01)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    await asyncio.sleep(0)
    assert attempt_cancelled.is_set()
