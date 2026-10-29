# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Health evidence from the selected Hermes agent's native diagnostics."""

import asyncio
import threading
from unittest.mock import MagicMock

import pytest
from nemo_fabric_adapter_contract.models import AdapterCheckResult, AgentConfig
from nemo_fabric_adapters.hermes.adapter import HermesRuntime


@pytest.fixture(name="hermes_health")
def hermes_health_fixture():
    runtime = HermesRuntime()
    agent = MagicMock(
        spec=[
            "session_id",
            "get_activity_summary",
            "run_conversation",
            "close",
            "_memory_provider_shutdown",
        ]
    )
    agent.session_id = "runtime-health"
    agent._memory_provider_shutdown = False
    agent.get_activity_summary.return_value = {
        "api_call_count": 0,
        "budget_used": 0,
        "budget_max": 90,
        "max_iterations": 90,
        "current_tool": None,
        "last_activity_description": "private-conversation-detail",
    }
    config = AgentConfig.from_mapping(
        {
            "models": {"default": {"provider": "openai", "model": "local-test"}},
        }
    )
    runtime._agent = agent
    runtime._runtime_id = "runtime-health"
    runtime._agent_config = config
    runtime._model_config = config.models["default"]
    runtime._session_db = MagicMock()
    runtime._started = True
    return runtime, agent


@pytest.mark.parametrize("level", ["live", "active", "ready"])
async def test_hermes_health_uses_fresh_native_diagnostics(hermes_health, level):
    runtime, agent = hermes_health
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
    assert "private-conversation-detail" not in str(result)
    agent.get_activity_summary.assert_called_once_with()
    agent.run_conversation.assert_not_called()
    assert runtime._conversation_history is None


@pytest.mark.parametrize("level", ["live", "active", "ready"])
async def test_hermes_health_diagnostic_reply_does_not_hide_closure(
    hermes_health, level
):
    runtime, agent = hermes_health
    agent._memory_provider_shutdown = True
    result = await runtime.check(level)
    assert (result["outcome"], result["reason_code"]) == ("failed", "session_closed")
    agent.get_activity_summary.assert_not_called()


@pytest.mark.parametrize(
    "field", ["_agent", "_agent_config", "_model_config", "_runtime_id", "_session_db"]
)
async def test_hermes_health_requires_owned_session(hermes_health, field):
    runtime, agent = hermes_health
    setattr(runtime, field, None)
    result = await runtime.check("live")
    assert (result["outcome"], result["reason_code"]) == (
        "unknown",
        "runtime_unavailable",
    )
    agent.get_activity_summary.assert_not_called()


@pytest.mark.parametrize("level", ["live", "active", "ready"])
async def test_hermes_health_reports_unavailable_after_stop(hermes_health, level):
    runtime, agent = hermes_health
    await runtime.stop()

    result = await runtime.check(level)

    assert (result["outcome"], result["reason_code"]) == (
        "unknown",
        "runtime_unavailable",
    )
    assert result["observed_at"] is None
    agent.close.assert_called_once_with()
    agent.get_activity_summary.assert_not_called()


@pytest.mark.parametrize("level", ["live", "active", "ready"])
@pytest.mark.parametrize("rotation", ["before", "during"])
async def test_hermes_health_allows_native_session_rotation(
    hermes_health, level, rotation
):
    runtime, agent = hermes_health
    runtime_id = runtime._runtime_id

    def rotate():
        agent.session_id = "compressed-child-session"
        return agent.get_activity_summary.return_value

    if rotation == "before":
        rotate()
    else:
        agent.get_activity_summary.side_effect = rotate

    result = await runtime.check(level)

    assert (result["outcome"], result["reason_code"], result["activity"]) == (
        "passed",
        None,
        "idle",
    )
    assert result["observed_at"] is not None
    assert agent.session_id == "compressed-child-session"
    assert runtime._runtime_id == runtime_id
    agent.get_activity_summary.assert_called_once_with()
    agent.run_conversation.assert_not_called()


async def test_hermes_active_requires_invocation_interface(hermes_health):
    runtime, agent = hermes_health
    agent.run_conversation = None
    result = await runtime.check("active")
    assert (result["outcome"], result["reason_code"]) == (
        "failed",
        "endpoint_unavailable",
    )


@pytest.mark.parametrize(
    "response", [None, {}, {"api_call_count": True}, {"api_call_count": -1}]
)
async def test_hermes_health_rejects_malformed_diagnostics(hermes_health, response):
    runtime, agent = hermes_health
    agent.get_activity_summary.return_value = response
    result = await runtime.check("ready")
    assert (result["outcome"], result["reason_code"]) == ("unknown", "probe_error")


@pytest.mark.parametrize(
    "field", ["_agent", "_agent_config", "_model_config", "_runtime_id", "_session_db"]
)
async def test_hermes_health_invalidates_changed_owner(hermes_health, field):
    runtime, agent = hermes_health

    def changed():
        setattr(runtime, field, object())
        return agent.get_activity_summary.return_value

    agent.get_activity_summary.side_effect = changed
    result = await runtime.check("ready")
    assert (result["outcome"], result["reason_code"]) == (
        "unknown",
        "observation_changed",
    )
    assert result["observed_at"] is None


async def test_hermes_health_redacts_native_errors(hermes_health):
    runtime, agent = hermes_health
    agent.get_activity_summary.side_effect = RuntimeError("private-native-detail")
    result = await runtime.check("ready")
    assert (result["outcome"], result["reason_code"]) == ("unknown", "probe_error")
    assert "private-native-detail" not in str(result)


async def test_hermes_health_bounds_pending_native_diagnostics(hermes_health):
    runtime, agent = hermes_health
    entered, release = threading.Event(), threading.Event()

    def blocked():
        entered.set()
        assert release.wait(5)
        return agent.get_activity_summary.return_value

    agent.get_activity_summary.side_effect = blocked
    invocation = asyncio.create_task(asyncio.Event().wait())
    runtime._active_invoke_task = invocation
    try:
        for _ in range(3):
            with pytest.raises(TimeoutError):
                async with asyncio.timeout(0.03):
                    await runtime.check("ready")
        assert entered.is_set()
        assert agent.get_activity_summary.call_count == 1
        assert not invocation.done()
        release.set()
        await runtime._health_task
        result = await runtime.check("ready")
        assert (result["outcome"], result["activity"]) == ("passed", "busy")
        assert agent.get_activity_summary.call_count == 2
    finally:
        release.set()
        invocation.cancel()
        await asyncio.gather(invocation, return_exceptions=True)


async def test_hermes_unsupported_level_has_no_probe(hermes_health):
    runtime, agent = hermes_health
    result = await runtime.check("other")
    assert (result["outcome"], result["reason_code"]) == (
        "unsupported",
        "unsupported_level",
    )
    agent.get_activity_summary.assert_not_called()


@pytest.mark.parametrize("change", ["close", "stop"])
async def test_hermes_health_detects_closure_during_diagnostic(hermes_health, change):
    runtime, agent = hermes_health

    def changed():
        if change == "close":
            agent._memory_provider_shutdown = True
        else:
            runtime._stopping = True
        return agent.get_activity_summary.return_value

    agent.get_activity_summary.side_effect = changed
    result = await runtime.check("ready")
    expected = (
        ("failed", "session_closed")
        if change == "close"
        else ("unknown", "observation_changed")
    )
    assert (result["outcome"], result["reason_code"]) == expected


async def test_health_timeout_consumes_late_native_errors_without_logging(
    hermes_health,
):
    runtime, agent = hermes_health
    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()
    diagnostics = []
    entered, release = threading.Event(), threading.Event()
    pending = None

    def delayed_error():
        entered.set()
        assert release.wait(5)
        raise RuntimeError("private-native-health-detail")

    agent.get_activity_summary.side_effect = delayed_error
    loop.set_exception_handler(lambda _loop, context: diagnostics.append(context))
    try:
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(0.03):
                await runtime.check("ready")
        pending = runtime._health_task
        assert entered.is_set()
        assert pending is not None and not pending.done()
        release.set()
        await asyncio.wait({pending})
        await asyncio.sleep(0)
        assert not pending.cancelled()
        assert diagnostics == []
        agent.get_activity_summary.assert_called_once_with()
    finally:
        release.set()
        if pending is not None:
            await asyncio.gather(pending, return_exceptions=True)
        loop.set_exception_handler(previous_handler)


@pytest.mark.parametrize("completion", ["error", "cancelled"])
async def test_health_obtains_fresh_evidence_after_an_older_probe_fails(
    hermes_health, completion
):
    runtime, agent = hermes_health
    release = threading.Event()

    def diagnostic():
        if agent.get_activity_summary.call_count == 1:
            assert release.wait(5)
            if completion == "cancelled":
                raise asyncio.CancelledError
            raise RuntimeError("private-prior-probe-detail")
        return agent.get_activity_summary.return_value

    agent.get_activity_summary.side_effect = diagnostic
    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.03):
            await runtime.check("ready")
    pending = runtime._health_task
    assert pending is not None and not pending.done()
    checking = asyncio.create_task(runtime.check("live"))
    try:
        await asyncio.sleep(0)
        assert agent.get_activity_summary.call_count == 1
        assert not checking.done()
        release.set()
        result = await asyncio.wait_for(checking, 1)
        assert (result["outcome"], result["reason_code"]) == ("passed", None)
        assert agent.get_activity_summary.call_count == 2
    finally:
        release.set()
        await asyncio.gather(pending, checking, return_exceptions=True)
