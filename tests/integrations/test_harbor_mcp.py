# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the dependency-free MCP server shipped with the Harbor example."""

from __future__ import annotations

import io
import json
import runpy
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest


def load_reply() -> Callable[[dict[str, Any]], dict[str, Any] | None]:
    source = Path(__file__).parents[2] / "examples/harbor/swebench/mcp/repo_inspector.py"
    return runpy.run_path(str(source))["reply"]


def test_mcp_initialize_negotiates_supported_protocol_version():
    response = load_reply()(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2025-03-26"},
        }
    )

    assert response is not None
    assert response["result"]["protocolVersion"] == "2025-03-26"


def test_mcp_initialize_rejects_unsupported_protocol_version():
    response = load_reply()(
        {
            "jsonrpc": "2.0",
            "id": "initialize-1",
            "method": "initialize",
            "params": {"protocolVersion": "2099-01-01"},
        }
    )

    assert response == {
        "jsonrpc": "2.0",
        "id": "initialize-1",
        "error": {
            "code": -32602,
            "message": "UNSUPPORTED_PROTOCOL_VERSION",
            "data": {"requested": "2099-01-01", "supported": ["2025-03-26"]},
        },
    }


def test_mcp_tools_call_rejects_unknown_tool():
    response = load_reply()(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "write_repo", "arguments": {}},
        }
    )

    assert response is not None
    assert response["id"] == 2
    assert response["error"]["code"] == -32602
    assert response["error"]["message"] == "unknown tool"


def test_mcp_tools_call_rejects_nonempty_arguments():
    response = load_reply()(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "repo_summary", "arguments": {"path": "/etc"}},
        }
    )

    assert response is not None
    assert response["id"] == 3
    assert response["error"]["code"] == -32602


def test_mcp_dispatch_error_preserves_request_id(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    source = Path(__file__).parents[2] / "examples/harbor/swebench/mcp/repo_inspector.py"
    server = runpy.run_path(str(source))
    request = {
        "jsonrpc": "2.0",
        "id": "call-4",
        "method": "tools/call",
        "params": {"name": "repo_summary", "arguments": {}},
    }

    def fail_to_list(_: Path):
        raise OSError("testbed unavailable")

    monkeypatch.setattr(Path, "iterdir", fail_to_list)
    monkeypatch.setattr(server["sys"], "stdin", io.StringIO(json.dumps(request)))

    server["main"]()

    response = json.loads(capsys.readouterr().out)
    assert response["id"] == "call-4"
    assert response["error"]["code"] == -32603
