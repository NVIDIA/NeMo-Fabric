# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in real Codex SDK integration gates for Fabric runtime behavior.

RUN_FABRIC_CODEX_INTEGRATION=1 uv run pytest tests/e2e/test_codex.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import uuid
from pathlib import Path

import pytest


def _assert_semantic_relay_artifacts(output, expected_response: str) -> None:
    artifacts = {
        item["kind"]: Path(item["path"]) for item in output["relay_artifacts"]
    }
    events = [
        json.loads(line)
        for line in artifacts["atof"].read_text(encoding="utf-8").splitlines()
    ]
    llm_starts = [
        event
        for event in events
        if event.get("category") == "llm" and event.get("scope_category") == "start"
    ]
    assert llm_starts, events
    assert all(
        isinstance(event.get("data", {}).get("content"), dict)
        for event in llm_starts
    )
    assert all(event["data"]["content"].get("model") for event in llm_starts)

    llm_ends = [
        event
        for event in events
        if event.get("category") == "llm" and event.get("scope_category") == "end"
    ]
    assert llm_ends, events
    assert any(
        (event.get("data", {}).get("usage") or {}).get("total_tokens", 0) > 0
        for event in llm_ends
    )

    trajectory = json.loads(artifacts["atif"].read_text(encoding="utf-8"))
    agent_messages = [
        step.get("message", "")
        for step in trajectory.get("steps", [])
        if step.get("source") == "agent"
    ]
    assert any(expected_response.lower() in message.lower() for message in agent_messages)


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
    await _run_relay(relay_command)


async def _run() -> None:
    from examples.code_review_agent import BASE_DIR, codex_config
    from nemo_fabric import Fabric

    config = codex_config()
    nonce = f"fabric-{uuid.uuid4().hex[:8]}"
    client = Fabric()
    oneshot = await client.run(
        config,
        base_dir=BASE_DIR,
        input="Reply with exactly: FABRIC_CODEX_ONESHOT_OK",
    )
    assert oneshot["status"] == "succeeded", oneshot.to_mapping()
    assert "fabric_codex_oneshot_ok" in oneshot["output"]["response"].lower(), (
        oneshot.to_mapping()
    )
    assert oneshot["output"]["adapter"] == "sdk", oneshot.to_mapping()
    assert "command" not in oneshot["output"], oneshot.to_mapping()

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
    assert first["output"]["events"], first.to_mapping()
    assert second["output"]["usage"] is not None, second.to_mapping()


async def _run_relay(relay_command: str) -> None:
    from examples.code_review_agent import BASE_DIR, codex_config, with_relay
    from nemo_fabric import Fabric

    config = with_relay(codex_config())
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
    _assert_semantic_relay_artifacts(result["output"], "FABRIC_CODEX_RELAY_OK")

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
    for turn in (first, second):
        assert turn["output"]["relay_runtime"]["enabled"] is True, turn.to_mapping()
        assert {item["kind"] for item in turn["output"]["relay_artifacts"]} >= {
            "atof",
            "atif",
        }, turn.to_mapping()
