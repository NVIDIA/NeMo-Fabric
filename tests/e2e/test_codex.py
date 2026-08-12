# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in real Codex SDK integration gates for Fabric runtime behavior.

RUN_FABRIC_CODEX_INTEGRATION=1 uv run pytest tests/e2e/test_codex.py
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
import uuid
import warnings

import pytest
import requests
from _utils.utils import (
    assert_atof_model,
    assert_atof_skill_selection,
    assert_semantic_relay_artifacts,
)


def _mock_codex_config(api_server, tmp_path):
    from examples.code_review_agent import codex_config, with_relay

    config = with_relay(codex_config())
    config.models["default"].provider = "fabric-test"
    config.models["default"].api_key_env = "FABRIC_TEST_API_KEY"
    config.models["default"].base_url = f"{api_server}/v1"
    config.environment.workspace = tmp_path
    config.environment.artifacts = tmp_path / "artifacts"
    config.environment.env["FABRIC_TEST_API_KEY"] = "test"
    config.runtime.artifacts = tmp_path / "artifacts"
    return config


def _skill_tool_call(selected_skill, workdir):
    if sys.platform == "win32":
        return {
            "name": "shell_command",
            "arguments": {
                "command": f'Get-Content -Raw "{selected_skill / "SKILL.md"}"',
                "workdir": str(workdir),
            },
        }
    return {
        "name": "exec_command",
        "arguments": {
            "cmd": f"cat {selected_skill / 'SKILL.md'}",
            "workdir": str(workdir),
        },
    }


def test_skill_tool_call_uses_classic_shell_on_windows(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "win32")

    tool_call = _skill_tool_call(tmp_path / "default", tmp_path)

    assert tool_call["name"] == "shell_command"
    assert set(tool_call["arguments"]) == {"command", "workdir"}


@pytest.mark.usefixtures("nemo_relay")
@pytest.mark.parametrize("skill", ["default", "alternate", None])
async def test_skill_selection(
    api_server, tmp_path, skill, default_skill, alternate_skill
):
    from nemo_fabric import Fabric

    config = _mock_codex_config(api_server, tmp_path)
    config.models["default"].model = "fabric-echo"
    config.add_skill_path(default_skill)

    if skill == "alternate" or skill is None:
        config.remove_skill_path(default_skill)
    if skill == "alternate":
        config.add_skill_path(alternate_skill)

    if skill is not None:
        selected_skill = default_skill if skill == "default" else alternate_skill
        scenario_response = requests.post(
            f"{api_server}/_scenario",
            json={"tool_call": _skill_tool_call(selected_skill, tmp_path)},
            timeout=5,
        )
        scenario_response.raise_for_status()

    result = await Fabric().run(
        config,
        base_dir=tmp_path,
        input=f"Use the {skill} skill." if skill else "Reply without using a skill.",
    )

    assert result["status"] == "succeeded", result.to_mapping()
    assert_atof_skill_selection(result["output"], skill)


@pytest.mark.usefixtures("nemo_relay")
@pytest.mark.parametrize("model", ["m1", "m2"])
async def test_model_selection(api_server, tmp_path, model):
    from nemo_fabric import Fabric

    config = _mock_codex_config(api_server, tmp_path)
    config.models["default"].model = model

    result = await Fabric().run(config, base_dir=tmp_path, input="Reply with hello.")

    assert result["status"] == "succeeded", result.to_mapping()
    assert_atof_model(result["output"], model)


@pytest.mark.parametrize("enabled", [True, False])
async def test_mcp_stdio_transport(api_server, tmp_path, enabled):
    from examples.code_review_agent import codex_config
    from nemo_fabric import Fabric

    tool_name = "get_current_time"
    scenario_response = requests.post(
        f"{api_server}/_scenario",
        json={
            "tool_call": {
                "name": tool_name,
                "namespace": "mcp__mcp_server_time",
                "arguments": {"timezone": "America/Los_Angeles"},
            }
        },
        timeout=5,
    )
    scenario_response.raise_for_status()

    config = codex_config()
    config.models["default"].provider = "fabric-test"
    config.models["default"].model = "fabric-echo"
    config.models["default"].api_key_env = "FABRIC_TEST_API_KEY"
    config.models["default"].base_url = f"{api_server}/v1"
    config.environment.workspace = tmp_path
    config.environment.artifacts = tmp_path / "artifacts"
    config.environment.env["FABRIC_TEST_API_KEY"] = "test"
    config.runtime.artifacts = tmp_path / "artifacts"
    config.add_mcp_server(
        "mcp_server_time",
        transport="stdio",
        url=sys.executable,
        args=["-m", "mcp_server_time"],
        env={"MCP_TIME_TEST": "enabled"},
    )

    if not enabled:
        config.remove_mcp_server("mcp_server_time")

    result = await Fabric().run(
        config,
        base_dir=tmp_path,
        input="Use the time MCP tool to get the current time in America/Los_Angeles.",
    )

    assert result["status"] == "succeeded", result.to_mapping()
    tool_events = [
        event for event in result["output"]["events"] if event["type"] == "mcpToolCall"
    ]

    if not enabled:
        assert not tool_events
    else:
        assert len(tool_events) == 1
        assert tool_events[0]["server"] == "mcp_server_time"
        assert tool_events[0]["tool"] == "get_current_time"
        assert tool_events[0]["status"] == "completed"
        assert "America/Los_Angeles" in str(tool_events[0]["result"])


async def test_codex_sdk():
    if os.environ.get("RUN_FABRIC_CODEX_INTEGRATION") != "1":
        pytest.skip("set RUN_FABRIC_CODEX_INTEGRATION=1 to run")
    if importlib.util.find_spec("openai_codex") is None:
        pytest.fail("the openai-codex Python SDK is required")
    if importlib.util.find_spec("nemo_fabric._native") is None:
        pytest.fail("the nemo_fabric native extension is required (pip install -e .)")
    await _run()


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
    await _run_relay(str(relay_command))


async def _run() -> None:
    from examples.code_review_agent import BASE_DIR, codex_config
    from nemo_fabric import Fabric

    config = codex_config()
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


async def _run_relay(relay_command: str) -> None:
    from examples.code_review_agent import BASE_DIR, codex_config, with_relay
    from nemo_fabric import Fabric

    config = with_relay(codex_config())
    assert config.environment is not None
    config.environment.env["FABRIC_TEST_NEMO_RELAY_COMMAND"] = relay_command
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
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        async with await client.start_runtime(
            config,
            base_dir=BASE_DIR,
            streaming=True,
        ) as runtime:
            first_stream = runtime.invoke_stream(input=f"Remember this value: {nonce}")
            first_records = [record async for record in first_stream]
            first = await first_stream.result()
            second_stream = runtime.invoke_stream(
                input="Reply with only the value I asked you to remember."
            )
            second_records = [record async for record in second_stream]
            second = await second_stream.result()

    results = (first.to_mapping(), second.to_mapping())
    assert first_records
    assert second_records
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
