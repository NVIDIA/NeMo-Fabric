# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in real Kiro ACP integration test.

RUN_FABRIC_KIRO_INTEGRATION=1 uv run pytest tests/e2e/test_kiro.py
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

import pytest


async def test_kiro_live_runtime_retains_session_context(tmp_path):
    if os.environ.get("RUN_FABRIC_KIRO_INTEGRATION") != "1":
        pytest.skip("set RUN_FABRIC_KIRO_INTEGRATION=1 to run")

    from examples.code_review_agent import kiro_config
    from nemo_fabric import Fabric

    config = kiro_config()
    root = Path(__file__).resolve().parents[2]
    config.discovery.local_paths = [
        root / "adapters" / "typescript" / "kiro" / "kiro.fabric-adapter.json"
    ]
    config.environment.workspace = tmp_path
    config.environment.artifacts = tmp_path / "artifacts"
    config.runtime.artifacts = tmp_path / "artifacts"
    nonce = f"fabric-kiro-{uuid.uuid4().hex[:12]}"

    async with await Fabric().start_runtime(config, base_dir=tmp_path) as runtime:
        first = await runtime.invoke(
            input=(
                f"Remember this exact token for my next message: {nonce}. "
                "Reply with exactly ACK and do not use tools."
            )
        )
        second = await runtime.invoke(
            input="Reply with only the exact token from my previous message. Do not use tools."
        )

    assert first.status == "succeeded", first.to_mapping()
    assert second.status == "succeeded", second.to_mapping()
    assert first.output.response == "ACK"
    assert second.output.response == nonce
    assert (
        first.metadata["adapter"]["session_id"]
        == second.metadata["adapter"]["session_id"]
    )


async def test_kiro_live_runtime_has_mcp_tools_on_first_invocation(tmp_path):
    if os.environ.get("RUN_FABRIC_KIRO_INTEGRATION") != "1":
        pytest.skip("set RUN_FABRIC_KIRO_INTEGRATION=1 to run")

    from examples.code_review_agent import kiro_config
    from nemo_fabric import Fabric

    config = kiro_config()
    root = Path(__file__).resolve().parents[2]
    config.discovery.local_paths = [
        root / "adapters" / "typescript" / "kiro" / "kiro.fabric-adapter.json"
    ]
    config.environment.workspace = tmp_path
    config.environment.artifacts = tmp_path / "artifacts"
    config.runtime.artifacts = tmp_path / "artifacts"
    config.add_mcp_server(
        "time",
        transport="stdio",
        url=sys.executable,
        args=["-m", "mcp_server_time"],
    )

    async with await Fabric().start_runtime(config, base_dir=tmp_path) as runtime:
        result = await runtime.invoke(
            input=(
                "Call the time MCP tool for Asia/Tokyo on this first invocation. "
                "After the tool succeeds, include FABRIC_KIRO_MCP_OK in your response. "
                "If the tool is unavailable, reply TOOL_UNAVAILABLE instead."
            )
        )

    assert result.status == "succeeded", result.to_mapping()
    assert "FABRIC_KIRO_MCP_OK" in result.output.response
    assert "TOOL_UNAVAILABLE" not in result.output.response
