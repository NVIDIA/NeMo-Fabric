# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in real Codex SDK integration gates for Fabric runtime behavior.

RUN_FABRIC_CODEX_INTEGRATION=1 uv run pytest tests/e2e/test_codex.py
"""

from __future__ import annotations

import importlib.util
import os
import shlex
import shutil
import sys
import uuid
from pathlib import Path

import pytest
from _utils.utils import assert_semantic_relay_artifacts


def _select_codex_runtime(config):
    codex_bin = os.environ.get("FABRIC_TEST_CODEX_BIN")
    if codex_bin:
        config.harness.settings["codex_bin"] = codex_bin
    return config


async def test_codex_sdk():
    if os.environ.get("RUN_FABRIC_CODEX_INTEGRATION") != "1":
        pytest.skip("set RUN_FABRIC_CODEX_INTEGRATION=1 to run")
    if importlib.util.find_spec("openai_codex") is None:
        pytest.fail("the openai-codex Python SDK is required")
    if importlib.util.find_spec("nemo_fabric._native") is None:
        pytest.fail("the nemo_fabric native extension is required (pip install -e .)")
    await _run()


async def test_codex_blocked_shell(tmp_path):
    if os.environ.get("RUN_FABRIC_CODEX_INTEGRATION") != "1":
        pytest.skip("set RUN_FABRIC_CODEX_INTEGRATION=1 to run")
    if importlib.util.find_spec("openai_codex") is None:
        pytest.fail("the openai-codex Python SDK is required")
    if importlib.util.find_spec("nemo_fabric._native") is None:
        pytest.fail("the nemo_fabric native extension is required (pip install -e .)")
    await _run_blocked_shell(tmp_path)


async def test_codex_blocked_mcp_tool(tmp_path):
    if os.environ.get("RUN_FABRIC_CODEX_INTEGRATION") != "1":
        pytest.skip("set RUN_FABRIC_CODEX_INTEGRATION=1 to run")
    if importlib.util.find_spec("openai_codex") is None:
        pytest.fail("the openai-codex Python SDK is required")
    if importlib.util.find_spec("mcp") is None:
        pytest.fail("the MCP Python SDK is required")
    if importlib.util.find_spec("nemo_fabric._native") is None:
        pytest.fail("the nemo_fabric native extension is required (pip install -e .)")
    await _run_blocked_mcp_tool(tmp_path)


async def test_codex_sdk_with_relay():
    if os.environ.get("RUN_FABRIC_CODEX_RELAY_INTEGRATION") != "1":
        pytest.skip("set RUN_FABRIC_CODEX_RELAY_INTEGRATION=1 to run")
    if importlib.util.find_spec("openai_codex") is None:
        pytest.fail("the openai-codex Python SDK is required")
    if importlib.util.find_spec("nemo_fabric._native") is None:
        pytest.fail("the nemo_fabric native extension is required (pip install -e .)")
    relay_command = os.environ.get("FABRIC_TEST_NEMO_RELAY_COMMAND") or shutil.which(
        "nemo-relay"
    )
    if relay_command is None:
        pytest.fail("the nemo-relay CLI is required")
    await _run_relay(relay_command)


async def _run() -> None:
    from examples.code_review_agent import BASE_DIR, codex_config
    from nemo_fabric import Fabric

    config = _select_codex_runtime(codex_config())
    nonce = f"fabric-{uuid.uuid4().hex[:8]}"
    client = Fabric()
    single = await client.run(
        config,
        base_dir=BASE_DIR,
        input="Reply with exactly: FABRIC_CODEX_SINGLE_INVOCATION_OK",
    )
    assert single["status"] == "succeeded", single.to_mapping()
    assert (
        "fabric_codex_single_invocation_ok" in single["output"]["response"].lower()
    ), single.to_mapping()
    assert single["output"]["adapter"] == "sdk", single.to_mapping()
    assert "command" not in single["output"], single.to_mapping()

    async with await client.start_runtime(
        config,
        base_dir=BASE_DIR,
    ) as runtime:
        first = await runtime.invoke(input=f"Remember this value: {nonce}")
        second = await runtime.invoke(
            input="Reply with only the value I asked you to remember."
        )

    results = (first.to_mapping(), second.to_mapping())
    assert first["status"] == second["status"] == "succeeded", results
    assert first["output"]["thread_id"] == second["output"]["thread_id"], results
    assert nonce in second["output"]["response"], second.to_mapping()
    assert first["metadata"]["adapter_runner"] == "persistent_local_host", results
    assert first["metadata"]["host_pid"] == second["metadata"]["host_pid"], results
    assert first["output"]["events"], first.to_mapping()
    assert second["output"]["usage"] is not None, second.to_mapping()


async def _run_blocked_shell(tmp_path: Path) -> None:
    from examples.code_review_agent import BASE_DIR, codex_config
    from nemo_fabric import Fabric

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    client = Fabric()

    blocked = codex_config()
    blocked.environment.workspace = str(workspace)
    blocked.block_tools(
        "apps",
        "browser",
        "image_generation",
        "mcp",
        "multi_agent",
        "plugins",
        "request_user_input",
        "shell",
        "tool_suggest",
        "web_search",
        "app:fabric-test:files/delete",
    )
    first_marker = workspace / "blocked-first"
    second_marker = workspace / "blocked-second"
    async with await client.start_runtime(blocked, base_dir=BASE_DIR) as runtime:
        first = await runtime.invoke(
            input=f"Use the shell tool to run exactly: touch {first_marker}"
        )
        second = await runtime.invoke(
            input=f"Use the shell tool to run exactly: touch {second_marker}"
        )

    blocked_results = (first.to_mapping(), second.to_mapping())
    assert first["status"] == second["status"] == "succeeded", blocked_results
    assert first["output"]["thread_id"] == second["output"]["thread_id"]
    assert not first_marker.exists(), first.to_mapping()
    assert not second_marker.exists(), second.to_mapping()
    assert all(
        event["type"] != "commandExecution"
        for result in (first, second)
        for event in result["output"]["events"]
    ), blocked_results

    allowed = codex_config()
    allowed.environment.workspace = str(workspace)
    allowed_marker = workspace / "allowed"
    result = await client.run(
        allowed,
        base_dir=BASE_DIR,
        input=f"Use the shell tool to run exactly: touch {allowed_marker}",
    )

    assert result["status"] == "succeeded", result.to_mapping()
    assert allowed_marker.exists(), result.to_mapping()
    assert any(
        event["type"] == "commandExecution" for event in result["output"]["events"]
    ), result.to_mapping()


async def _run_blocked_mcp_tool(tmp_path: Path) -> None:
    from examples.code_review_agent import BASE_DIR, codex_config
    from nemo_fabric import Fabric

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    probe = Path(__file__).parents[1] / "fixtures" / "codex_mcp_probe.py"
    client = Fabric()

    blocked_marker = workspace / "blocked-mcp"
    blocked = codex_config()
    blocked.environment.workspace = str(workspace)
    blocked.add_mcp_server(
        "probe",
        transport="stdio",
        url=shlex.join([sys.executable, str(probe), str(blocked_marker)]),
        exposure="harness_native",
    )
    blocked.block_tools("shell", "mcp:probe:mark")
    blocked_result = await client.run(
        blocked,
        base_dir=BASE_DIR,
        input="Call the probe MCP tool named mark exactly once, then reply done.",
    )

    assert blocked_result["status"] == "succeeded", blocked_result.to_mapping()
    assert not blocked_marker.exists(), blocked_result.to_mapping()
    assert all(
        event["type"] != "mcpToolCall" for event in blocked_result["output"]["events"]
    ), blocked_result.to_mapping()

    allowed_marker = workspace / "allowed-mcp"
    allowed = codex_config()
    allowed.environment.workspace = str(workspace)
    allowed.add_mcp_server(
        "probe",
        transport="stdio",
        url=shlex.join([sys.executable, str(probe), str(allowed_marker)]),
        exposure="harness_native",
    )
    allowed.block_tools("shell")
    allowed_result = await client.run(
        allowed,
        base_dir=BASE_DIR,
        input="Call the probe MCP tool named mark exactly once, then reply done.",
    )

    assert allowed_result["status"] == "succeeded", allowed_result.to_mapping()
    assert allowed_marker.exists(), allowed_result.to_mapping()
    assert any(
        event["type"] == "mcpToolCall" for event in allowed_result["output"]["events"]
    ), allowed_result.to_mapping()


async def _run_relay(relay_command: str) -> None:
    from examples.code_review_agent import BASE_DIR, codex_config, with_relay
    from nemo_fabric import Fabric

    config = _select_codex_runtime(with_relay(codex_config()))
    config.harness.settings["nemo_relay_command"] = relay_command
    client = Fabric()
    result = await client.run(
        config,
        base_dir=BASE_DIR,
        input="Reply with exactly: FABRIC_CODEX_RELAY_OK",
    )

    mapping = result.to_mapping()
    assert result["status"] == "succeeded", mapping
    assert "fabric_codex_relay_ok" in result["output"]["response"].lower(), mapping
    assert result["output"]["adapter"] == "sdk", mapping
    assert result["output"]["relay_runtime"]["enabled"] is True, mapping
    assert {item["kind"] for item in result["output"]["relay_artifacts"]} >= {
        "atof",
        "atif",
    }, mapping
    assert_semantic_relay_artifacts(result["output"], "FABRIC_CODEX_RELAY_OK")

    nonce = f"fabric-relay-{uuid.uuid4().hex[:8]}"
    async with await client.start_runtime(config, base_dir=BASE_DIR) as runtime:
        first = await runtime.invoke(input=f"Remember this value: {nonce}")
        second = await runtime.invoke(
            input="Reply with only the value I asked you to remember."
        )

    results = (first.to_mapping(), second.to_mapping())
    assert first["status"] == second["status"] == "succeeded", results
    assert first["output"]["thread_id"] == second["output"]["thread_id"], results
    assert nonce in second["output"]["response"], second.to_mapping()
    assert first["metadata"]["adapter_runner"] == "persistent_local_host", results
    assert first["metadata"]["host_pid"] == second["metadata"]["host_pid"], results
    for turn in (first, second):
        assert turn["output"]["relay_runtime"]["enabled"] is True, turn.to_mapping()
        assert {item["kind"] for item in turn["output"]["relay_artifacts"]} >= {
            "atof",
            "atif",
        }, turn.to_mapping()
