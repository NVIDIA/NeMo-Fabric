# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared report of unavailable MCP servers."""

from __future__ import annotations

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
        "NeMo Fabric Example: MCP server 'jira' (streamable_http) is unavailable: "
        "failed (ConnectError)",
        "NeMo Fabric Example: MCP server 'docs' (stdio) is unavailable: "
        "authentication_required",
    ]
