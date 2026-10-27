# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Health evidence from the selected compiled graph and checkpoint connection."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import aiosqlite
import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import StateSnapshot
from nemo_fabric_adapter_contract.models import AdapterCheckResult
from nemo_fabric_adapters.deepagents.adapter import DeepAgentsRuntime


@pytest.fixture(name="deepagents_health")
def deepagents_health_fixture():
    runtime = DeepAgentsRuntime()
    connection = MagicMock(spec=aiosqlite.Connection)
    connection._running = True
    connection.execute_fetchall = AsyncMock(return_value=[(1,)])
    saver = MagicMock(spec=AsyncSqliteSaver)
    saver.conn = connection
    agent = MagicMock(spec=CompiledStateGraph)
    agent.checkpointer = saver
    agent.get_input_jsonschema.return_value = {"properties": {"messages": {}}}
    agent.aget_state.return_value = StateSnapshot(
        values={},
        next=(),
        config={"configurable": {"thread_id": "thread-health"}},
        metadata=None,
        created_at=None,
        parent_config=None,
        tasks=(),
        interrupts=(),
    )
    runtime._agent = agent
    runtime._checkpointer = saver
    runtime._runtime_id = "runtime-health"
    runtime._thread_id = "thread-health"
    runtime._start_payload = {"config": {}}
    runtime._started = True
    return runtime, agent, saver, connection


@pytest.mark.parametrize("level", ["live", "active", "ready"])
async def test_deepagents_health_uses_native_evidence(deepagents_health, level):
    runtime, agent, _, connection = deepagents_health
    result = await runtime.check(level)
    AdapterCheckResult.from_mapping(result)
    assert (result["outcome"], result["activity"], result["reason_code"]) == (
        "passed",
        "idle",
        None,
    )
    assert (
        result["source"]
        == {
            "live": "fabric.adapter.liveness",
            "active": "fabric.adapter.interface",
            "ready": "fabric.adapter.readiness",
        }[level]
    )
    assert result["observed_at"] is not None
    assert runtime._completed_invocations == 0
    agent.astream.assert_not_called()
    if level == "live":
        connection.execute_fetchall.assert_awaited_once_with("SELECT 1")
    if level == "ready":
        agent.aget_state.assert_awaited_once_with(
            {"configurable": {"thread_id": "thread-health"}}
        )


@pytest.mark.parametrize("level", ["live", "active", "ready"])
async def test_deepagents_health_detects_native_connection_closure(
    deepagents_health, level
):
    runtime, agent, _, connection = deepagents_health
    connection._running = False
    result = await runtime.check(level)
    assert (result["outcome"], result["reason_code"]) == ("failed", "session_closed")
    agent.aget_state.assert_not_called()


@pytest.mark.parametrize(
    "field", ["_agent", "_checkpointer", "_runtime_id", "_thread_id", "_start_payload"]
)
async def test_deepagents_health_requires_owned_session(deepagents_health, field):
    runtime, _, _, _ = deepagents_health
    setattr(runtime, field, None)
    result = await runtime.check("live")
    assert (result["outcome"], result["reason_code"]) == (
        "unknown",
        "runtime_unavailable",
    )


@pytest.mark.parametrize("broken", ["astream", "input_schema"])
async def test_deepagents_active_requires_request_interface(deepagents_health, broken):
    runtime, agent, _, _ = deepagents_health
    if broken == "astream":
        agent.astream = None
    else:
        agent.get_input_jsonschema.return_value = {"properties": {}}
    result = await runtime.check("active")
    assert (result["outcome"], result["reason_code"]) == (
        "failed",
        "endpoint_unavailable",
    )


@pytest.mark.parametrize(
    "field", ["_agent", "_checkpointer", "_runtime_id", "_thread_id", "_start_payload"]
)
async def test_deepagents_health_invalidates_changed_owner(deepagents_health, field):
    runtime, agent, _, _ = deepagents_health

    async def changed(_config):
        setattr(runtime, field, object())
        return agent.aget_state.return_value

    agent.aget_state.side_effect = changed
    result = await runtime.check("ready")
    assert (result["outcome"], result["reason_code"]) == (
        "unknown",
        "observation_changed",
    )
    assert result["observed_at"] is None


@pytest.mark.parametrize("response", [None, {}, {"values": {}}])
async def test_deepagents_readiness_rejects_malformed_evidence(
    deepagents_health, response
):
    runtime, agent, _, _ = deepagents_health
    agent.aget_state.return_value = response
    result = await runtime.check("ready")
    assert (result["outcome"], result["reason_code"]) == ("unknown", "probe_error")


@pytest.mark.parametrize("error", [RuntimeError, KeyError, IndexError])
async def test_deepagents_health_redacts_native_errors(deepagents_health, error):
    runtime, agent, _, _ = deepagents_health
    agent.aget_state.side_effect = error("private-native-detail")
    result = await runtime.check("ready")
    assert (result["outcome"], result["reason_code"]) == ("unknown", "probe_error")
    assert "private-native-detail" not in str(result)


async def test_deepagents_health_bounds_pending_native_reads(deepagents_health):
    runtime, agent, _, _ = deepagents_health
    entered, release = asyncio.Event(), asyncio.Event()

    async def blocked(_config):
        entered.set()
        await release.wait()
        return agent.aget_state.return_value

    agent.aget_state.side_effect = blocked
    invocation = asyncio.create_task(asyncio.Event().wait())
    runtime._active_invoke_task = invocation
    try:
        for _ in range(3):
            with pytest.raises(TimeoutError):
                async with asyncio.timeout(0.03):
                    await runtime.check("ready")
        assert entered.is_set()
        assert agent.aget_state.await_count == 1
        assert not invocation.done()
        release.set()
        await asyncio.sleep(0)
        result = await runtime.check("ready")
        assert (result["outcome"], result["activity"]) == ("passed", "busy")
        assert agent.aget_state.await_count == 2
    finally:
        release.set()
        invocation.cancel()
        await asyncio.gather(invocation, return_exceptions=True)


async def test_deepagents_unsupported_level_has_no_probe(deepagents_health):
    runtime, agent, _, connection = deepagents_health
    result = await runtime.check("other")
    assert (result["outcome"], result["reason_code"]) == (
        "unsupported",
        "unsupported_level",
    )
    connection.execute_fetchall.assert_not_called()
    agent.aget_state.assert_not_called()


@pytest.mark.parametrize("change", ["close", "stop"])
async def test_deepagents_health_detects_closure_during_read(deepagents_health, change):
    runtime, agent, _, connection = deepagents_health

    async def changed(_config):
        if change == "close":
            connection._running = False
        else:
            runtime._stopping = True
        return agent.aget_state.return_value

    agent.aget_state.side_effect = changed
    result = await runtime.check("ready")
    expected = (
        ("failed", "session_closed")
        if change == "close"
        else ("unknown", "observation_changed")
    )
    assert (result["outcome"], result["reason_code"]) == expected
