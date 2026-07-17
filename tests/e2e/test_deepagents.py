# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in real Deep Agents smoke for Fabric one-shot and multi-turn runtimes.

RUN_FABRIC_DEEPAGENTS_INTEGRATION=1 NVIDIA_API_KEY=... \
    pytest tests/e2e/test_deepagents.py
"""

from __future__ import annotations

import importlib.util
import os
import uuid

import pytest


@pytest.mark.usefixtures("mock_nvidia_api_key")
async def test_deepagents_persistent_host_with_mock_model(api_server, tmp_path):
    pytest.importorskip("deepagents")
    from examples.code_review_agent import deepagents_config
    from nemo_fabric import EnvironmentConfig, Fabric, RuntimeConfig

    config = deepagents_config()
    config.harness.settings["base_url"] = f"{api_server}/v1"
    config.harness.settings["workspace"] = str(tmp_path)
    config.environment = EnvironmentConfig(
        provider="local",
        workspace=tmp_path,
        artifacts=tmp_path / "artifacts",
    )
    config.runtime = RuntimeConfig(
        input_schema="chat",
        output_schema="message",
        artifacts=tmp_path / "artifacts",
    )

    async with await Fabric().start_runtime(config, base_dir=tmp_path) as runtime:
        first = await runtime.invoke(input="first")
        second = await runtime.invoke(input="second")

    results = (first.to_mapping(), second.to_mapping())
    assert first["status"] == second["status"] == "succeeded", results
    assert first["metadata"]["adapter_runner"] == "persistent_local_host", results
    assert first["metadata"]["host_pid"] == second["metadata"]["host_pid"], results
    assert first["output"]["thread_id"] == second["output"]["thread_id"], results
    assert first["output"]["resumed"] is False, results
    assert second["output"]["resumed"] is True, results
    assert "user_count=2" in second["output"]["response"], results


@pytest.mark.usefixtures("mock_nvidia_api_key", "nemo_relay")
async def test_deepagents_persistent_host_with_relay_and_mock_model(
    api_server, tmp_path
):
    pytest.importorskip("deepagents")
    from examples.code_review_agent import deepagents_config, with_relay
    from nemo_fabric import EnvironmentConfig, Fabric, RuntimeConfig

    config = with_relay(deepagents_config())
    config.harness.settings["base_url"] = f"{api_server}/v1"
    config.harness.settings["workspace"] = str(tmp_path)
    config.environment = EnvironmentConfig(
        provider="local",
        workspace=tmp_path,
        artifacts=tmp_path / "artifacts",
    )
    config.runtime = RuntimeConfig(
        input_schema="chat",
        output_schema="message",
        artifacts=tmp_path / "artifacts",
    )

    async with await Fabric().start_runtime(config, base_dir=tmp_path) as runtime:
        first = await runtime.invoke(input="first")
        second = await runtime.invoke(input="second")

    results = (first.to_mapping(), second.to_mapping())
    assert first["status"] == second["status"] == "succeeded", results
    assert first["metadata"]["host_pid"] == second["metadata"]["host_pid"], results
    assert first["output"]["thread_id"] == second["output"]["thread_id"], results
    assert first["output"]["resumed"] is False, results
    assert second["output"]["resumed"] is True, results
    assert "user_count=2" in second["output"]["response"], results
    for turn in (first, second):
        assert turn.telemetry[0].provider == "relay", turn.to_mapping()
        assert {artifact["kind"] for artifact in turn["output"]["relay_artifacts"]} >= {
            "atof",
            "atif",
        }, turn.to_mapping()


@pytest.fixture(name="_require_integration")
def _require_integration_fixture() -> None:
    if os.environ.get("RUN_FABRIC_DEEPAGENTS_INTEGRATION") != "1":
        pytest.skip("set RUN_FABRIC_DEEPAGENTS_INTEGRATION=1 to run")
    if importlib.util.find_spec("deepagents") is None:
        pytest.fail(
            "the deepagents package is required (pip install -e '.[deepagents]')"
        )
    if importlib.util.find_spec("nemo_fabric._native") is None:
        pytest.fail("the nemo_fabric native extension is required (pip install -e .)")
    if not os.environ.get("NVIDIA_API_KEY"):
        pytest.fail("NVIDIA_API_KEY is required")


@pytest.mark.usefixtures("_require_integration")
async def test_deepagents_oneshot():
    from examples.code_review_agent import BASE_DIR, deepagents_config
    from nemo_fabric import Fabric

    client = Fabric()
    oneshot = await client.run(
        deepagents_config(),
        base_dir=BASE_DIR,
        input="Reply with exactly: FABRIC_DEEPAGENTS_ONESHOT_OK",
    )
    assert oneshot["status"] == "succeeded", oneshot.to_mapping()
    assert oneshot["output"]["response"], oneshot.to_mapping()
    # each one-shot run gets a fresh runtime, so it is never a resume
    assert oneshot["output"]["resumed"] is False, oneshot.to_mapping()


@pytest.mark.usefixtures("_require_integration")
async def test_deepagents_multi_turn():
    from examples.code_review_agent import BASE_DIR, deepagents_config
    from nemo_fabric import Fabric

    client = Fabric()
    nonce = f"fabric-{uuid.uuid4().hex[:8]}"

    async with await client.start_runtime(
        deepagents_config(), base_dir=BASE_DIR
    ) as runtime:
        first = await runtime.invoke(input=f"Remember this value: {nonce}")
        second = await runtime.invoke(
            input="Reply with only the value I asked you to remember."
        )

    results = (first.to_mapping(), second.to_mapping())
    assert first["status"] == second["status"] == "succeeded", results
    # one started runtime keeps a stable LangGraph thread across turns
    assert first["output"]["thread_id"] == second["output"]["thread_id"], results
    assert first["metadata"]["adapter_runner"] == "persistent_local_host", results
    assert first["metadata"]["host_pid"] == second["metadata"]["host_pid"], results
    # the first turn opens the runtime; the second resumes and recalls turn one
    assert first["output"]["resumed"] is False, results
    assert second["output"]["resumed"] is True, results
    assert nonce in second["output"]["response"], second.to_mapping()


@pytest.mark.usefixtures("_require_integration", "nemo_relay")
async def test_deepagents_multi_turn_with_relay():
    from examples.code_review_agent import BASE_DIR, deepagents_config, with_relay
    from nemo_fabric import Fabric

    client = Fabric()
    nonce = f"fabric-relay-{uuid.uuid4().hex[:8]}"
    config = with_relay(deepagents_config())

    async with await client.start_runtime(config, base_dir=BASE_DIR) as runtime:
        first = await runtime.invoke(input=f"Remember this value: {nonce}")
        second = await runtime.invoke(
            input="Reply with only the value I asked you to remember."
        )

    results = (first.to_mapping(), second.to_mapping())
    assert first["status"] == second["status"] == "succeeded", results
    assert first["output"]["thread_id"] == second["output"]["thread_id"], results
    assert first["metadata"]["host_pid"] == second["metadata"]["host_pid"], results
    assert first["output"]["resumed"] is False, results
    assert second["output"]["resumed"] is True, results
    assert nonce in second["output"]["response"], second.to_mapping()
    for turn in (first, second):
        assert turn.telemetry[0].provider == "relay", turn.to_mapping()
        assert {artifact["kind"] for artifact in turn["output"]["relay_artifacts"]} >= {
            "atof",
            "atif",
        }, turn.to_mapping()


@pytest.mark.usefixtures("_require_integration")
async def test_deepagents_subagent_delegation():
    # Exercise the real delegated-subagent path end to end: a `task`-delegating run
    # under the adapter's gating wiring. Subagents inherit the parent's model, tools,
    # workspace, and the config.tools policy, so a subagent cannot broaden capabilities.
    # Deterministic verification of the inherited tool-gating is covered by the mock
    # tests in tests/adapters/test_deepagents.py; a fuller real-subagent contract
    # (independently configured subagent tools/skills/models) is future work.
    from examples.code_review_agent import BASE_DIR, deepagents_config
    from nemo_fabric import Fabric

    config = deepagents_config()
    config.harness.settings["deepagents"] = {
        "subagents": [
            {
                "name": "echoer",
                "description": (
                    "Echoes a short phrase back verbatim. Use to delegate an echo task."
                ),
                "system_prompt": "You echo the exact phrase you are given and nothing else.",
            }
        ]
    }

    client = Fabric()
    result = await client.run(
        config,
        base_dir=BASE_DIR,
        input=(
            "Delegate to the `echoer` subagent via the task tool to echo the phrase "
            "FABRIC_DEEPAGENTS_SUBAGENT_OK, then reply with its result."
        ),
    )

    assert result["status"] == "succeeded", result.to_mapping()
    assert result["output"]["response"], result.to_mapping()
    # delegated steps are folded into this turn's usage aggregation
    assert result["output"]["resumed"] is False, result.to_mapping()


@pytest.mark.usefixtures("_require_integration")
async def test_deepagents_doctor():
    from examples.code_review_agent import BASE_DIR, deepagents_config
    from nemo_fabric import Fabric

    config = deepagents_config()
    client = Fabric()
    report = await client.doctor(config, base_dir=BASE_DIR)

    # The adapter declares no static env requirement (auth is provider-specific and
    # validated by the runtime preflight), so doctor resolves without failures.
    assert report.status == "pass", report
