# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in real Deep Agents smoke for single-invocation and multi-turn runtimes.

RUN_FABRIC_DEEPAGENTS_INTEGRATION=1 NVIDIA_API_KEY=... \
    pytest tests/e2e/test_deepagents.py
"""

from __future__ import annotations

import importlib.util
import os
import sys
import uuid
import warnings

import pytest
import requests


@pytest.mark.usefixtures("mock_nvidia_api_key")
async def test_deepagents_persistent_host_with_mock_model(api_server, tmp_path):
    pytest.importorskip("deepagents")
    from examples.code_review_agent import deepagents_config
    from nemo_fabric import EnvironmentConfig, Fabric, RuntimeConfig

    config = deepagents_config()
    config.models["default"].base_url = f"{api_server}/v1"
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
    config.models["default"].base_url = f"{api_server}/v1"
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

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        async with await Fabric().start_runtime(
            config,
            base_dir=tmp_path,
            streaming=True,
        ) as runtime:
            first_stream = runtime.invoke_stream(input="first")
            first_records = [record async for record in first_stream]
            first = await first_stream.result()

            second_stream = runtime.invoke_stream(input="second")
            second_records = [record async for record in second_stream]
            second = await second_stream.result()

    results = (first.to_mapping(), second.to_mapping())
    assert first_records
    assert second_records
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


@pytest.mark.usefixtures("mock_nvidia_api_key")
async def test_env_secrets_in_headers_are_not_expanded(api_server, tmp_path):
    pytest.importorskip("deepagents")
    from examples.code_review_agent import deepagents_config
    from nemo_fabric import EnvironmentConfig, Fabric, RuntimeConfig

    os.environ["MY_KEY"] = "XYZ"
    scenario_response = requests.post(
        f"{api_server}/_scenario",
        json={
            "tool_call": {
                "name": "get_authorization_header",
                "arguments": {},
            }
        },
        timeout=5,
    )
    scenario_response.raise_for_status()

    config = deepagents_config()
    config.models["default"].base_url = f"{api_server}/v1"
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
    config.add_mcp_server(
        "headers",
        transport="streamable-http",
        url=f"{api_server}/mcp",
        authentication=None,
        custom_headers={"Authorization": "Bearer ${MY_KEY}"},
    )

    result = await Fabric().run(
        config,
        base_dir=tmp_path,
        input="Use the MCP tool to return its Authorization header.",
    )

    assert result["status"] == "succeeded", result.to_mapping()
    response = requests.get(f"{api_server}/_mcp_authorization_headers", timeout=5)
    response.raise_for_status()
    assert set(response.json()) == {"Bearer ${MY_KEY}"}


@pytest.mark.usefixtures("mock_nvidia_api_key")
@pytest.mark.parametrize("enabled", [True, False])
async def test_mcp_stdio_transport(api_server, tmp_path, enabled):
    pytest.importorskip("deepagents")
    from examples.code_review_agent import deepagents_config
    from nemo_fabric import EnvironmentConfig, Fabric, RuntimeConfig

    tool_name = "get_current_time"
    scenario_response = requests.post(
        f"{api_server}/_scenario",
        json={
            "tool_call": {
                "name": tool_name,
                "arguments": {"timezone": "America/Los_Angeles"},
            }
        },
        timeout=5,
    )
    scenario_response.raise_for_status()

    config = deepagents_config()
    config.models["default"].base_url = f"{api_server}/v1"
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
    tool_calls = [
        call
        for message in result["output"]["messages"]
        for call in message.get("tool_calls", [])
        if call["name"] == tool_name
    ]
    tool_results = [
        message for message in result["output"]["messages"] if message["role"] == "tool"
    ]

    if not enabled:
        assert not tool_calls
        assert not tool_results
    else:
        assert tool_calls
        assert tool_results
        assert "America/Los_Angeles" in str(tool_results[0]["content"])


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
async def test_deepagents_single_invocation():
    from examples.code_review_agent import BASE_DIR, deepagents_config
    from nemo_fabric import Fabric

    client = Fabric()
    single = await client.run(
        deepagents_config(),
        base_dir=BASE_DIR,
        input="Reply with exactly: FABRIC_DEEPAGENTS_SINGLE_INVOCATION_OK",
    )
    assert single["status"] == "succeeded", single.to_mapping()
    assert single["output"]["response"], single.to_mapping()
    assert single["output"]["resumed"] is False, single.to_mapping()


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
    # The second turn continues the live runtime and recalls turn one.
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
async def test_deepagents_builtin_subagent_delegation():
    # Exercise the built-in delegated-subagent path end to end. The built-in
    # subagent inherits the parent model, tools, workspace, and tool policy, so
    # delegation cannot broaden capabilities.
    from examples.code_review_agent import BASE_DIR, deepagents_config
    from nemo_fabric import Fabric

    config = deepagents_config()

    client = Fabric()
    result = await client.run(
        config,
        base_dir=BASE_DIR,
        input=(
            "Delegate to the built-in general-purpose subagent via the task tool to "
            "echo the phrase FABRIC_DEEPAGENTS_SUBAGENT_OK, then reply with its result."
        ),
    )

    assert result["status"] == "succeeded", result.to_mapping()
    output = result["output"]
    assert "FABRIC_DEEPAGENTS_SUBAGENT_OK" in output["response"], result.to_mapping()
    task_calls = [
        call
        for message in output["messages"]
        for call in message.get("tool_calls", [])
        if call["name"] == "task"
    ]
    assert any(
        call["args"]["subagent_type"] == "general-purpose" for call in task_calls
    ), result.to_mapping()
    assert any("subgraph" in event for event in output["events"]), result.to_mapping()
    # delegated steps are folded into this turn's usage aggregation
    assert output["resumed"] is False, result.to_mapping()


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
