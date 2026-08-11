# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration coverage for the Relay scope leak the Deep Agents adapter contains.

The Deep Agents unit tests monkeypatch a context manager that raises, which is enough to
pin result normalization but cannot reproduce the state leak: the real callback drops a
run from ``_scope_handles`` *before* its pop fails, so the scope stays on Relay's shared
stack and outlives the invocation. These tests drive the actual Relay callback and scope
so that behavior is covered rather than assumed.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid

import pytest
from nemo_fabric_adapters.deepagents import adapter

nemo_relay = pytest.importorskip("nemo_relay", reason="requires the nemo-relay extra")

from nemo_relay.integrations.langchain.callbacks import (  # noqa: E402
    NemoRelayCallbackHandler,
)


@pytest.fixture(autouse=True)
def isolated_scope_stack(monkeypatch: pytest.MonkeyPatch):
    """Give each test its own Relay scope stack.

    These tests strand a scope on purpose, and the stack is process-global, so without
    isolation the damage would leak into every test that runs afterwards. ``ContextVar``
    assignment is the only way to install a stack for the current context.
    """

    monkeypatch.setattr(
        adapter.common_utils, "reject_ambient_relay_plugin_config", lambda: None
    )
    token = nemo_relay._scope_stack_var.set(nemo_relay.create_scope_stack())
    try:
        yield
    finally:
        nemo_relay._scope_stack_var.reset(token)


async def _overlapping_chain_runs(handler: NemoRelayCallbackHandler) -> None:
    """Close two sibling chain runs out of LIFO order, as LangGraph's tasks do.

    A starts, B starts, A ends, B ends. Relay's stack requires B to close before A, so
    A's pop is rejected.
    """

    run_a, run_b = uuid.uuid4(), uuid.uuid4()
    a_started, b_started, allow_b_end = (asyncio.Event() for _ in range(3))

    async def drive_a() -> None:
        handler.on_chain_start({}, {"task": "A"}, run_id=run_a, name="A")
        a_started.set()
        await b_started.wait()
        handler.on_chain_end({"done": "A"}, run_id=run_a)
        allow_b_end.set()

    async def drive_b() -> None:
        await a_started.wait()
        handler.on_chain_start({}, {"task": "B"}, run_id=run_b, name="B")
        b_started.set()
        await allow_b_end.wait()
        handler.on_chain_end({"done": "B"}, run_id=run_b)

    await asyncio.gather(asyncio.create_task(drive_a()), asyncio.create_task(drive_b()))


async def test_overlapping_chain_runs_strand_a_scope_on_the_shared_stack(caplog):
    """The documented failure, reproduced against the installed Relay."""

    handler = NemoRelayCallbackHandler()
    baseline = nemo_relay.scope.get_handle()

    with caplog.at_level(logging.ERROR):
        with pytest.raises(RuntimeError, match="not at the top of the stack"):
            with nemo_relay.scope.scope(
                "deepagents-request", nemo_relay.ScopeType.Agent
            ):
                await _overlapping_chain_runs(handler)

    # The callback stopped tracking the run before its pop failed...
    assert handler._scope_handles == {}
    # ...but the scope is still on Relay's stack, and stays current after the turn.
    current = nemo_relay.scope.get_handle()
    assert current.uuid != baseline.uuid
    assert current.name == "A"


async def test_the_adapter_detects_the_stranded_scope():
    """``_scope_top_unchanged`` is what turns that leak into a sticky quarantine.

    Without a real stranded scope this cannot be exercised, which is why it lives here
    rather than beside the monkeypatched unit tests.
    """

    handler = NemoRelayCallbackHandler()
    baseline = adapter._current_scope_handle()
    assert baseline is not None
    assert adapter._scope_top_unchanged(baseline) is True

    with pytest.raises(RuntimeError, match="not at the top of the stack"):
        with nemo_relay.scope.scope("deepagents-request", nemo_relay.ScopeType.Agent):
            await _overlapping_chain_runs(handler)

    assert adapter._scope_top_unchanged(baseline) is False


async def test_a_clean_turn_leaves_the_stack_restored():
    """The detector must not report damage for a properly nested turn."""

    baseline = adapter._current_scope_handle()
    with nemo_relay.scope.scope("deepagents-request", nemo_relay.ScopeType.Agent):
        await asyncio.sleep(0)

    assert adapter._scope_top_unchanged(baseline) is True


class _RecordingScope:
    """The real Relay scope, with a note of which turns opened one."""

    def __init__(self) -> None:
        self.opened: list[str] = []

    @contextlib.contextmanager
    def scope(self, name: str, scope_type: object, **kwargs: object):
        self.opened.append(name)
        with nemo_relay.scope.scope(name, scope_type, **kwargs):
            yield


class _NoopPlugin:
    """Stand-in for the Relay plugin, which needs a live gateway config to start."""

    @contextlib.asynccontextmanager
    async def plugin(self, config: object):
        yield {"diagnostics": [], "runtime_diagnostics": []}


async def test_a_poisoned_runtime_quarantines_its_next_turn(monkeypatch):
    """Two ordered turns end to end: real scope, real callback, real stack.

    The unit tests prove the quarantine against a monkeypatched context manager, which
    by construction cannot show that the adapter reads the *same* state Relay actually
    leaves behind. This drives the real callback overlap through the adapter's telemetry
    path twice, so baseline capture, detection, and the sticky verdict are exercised
    against Relay rather than against a stub. Only the plugin and the agent are stubbed:
    the plugin needs a live gateway, and the agent is irrelevant to scope bookkeeping.
    """

    async def fake_invoke(agent, user_message, thread_id, callbacks=None):
        if callbacks:
            # Turn 1 runs the overlapping chain callbacks that strand a scope.
            await _overlapping_chain_runs(callbacks[0])
        return {"messages": []}, [], []

    monkeypatch.setattr(adapter, "invoke_compiled_agent", fake_invoke)

    recording_scope = _RecordingScope()
    runtime = adapter.DeepAgentsRuntime()
    runtime._agent = object()
    runtime._relay_plugin = _NoopPlugin()
    runtime._relay_plugin_config = {}
    runtime._relay_scope = recording_scope
    runtime._relay_scope_type = nemo_relay.ScopeType
    runtime._callback_handler_type = NemoRelayCallbackHandler

    first = await runtime._invoke_with_telemetry("hello", "request-1")
    second = await runtime._invoke_with_telemetry("hello again", "request-2")

    # Turn 1 completed, reported the real Relay fault, and poisoned the runtime.
    assert first.error is None
    assert "not at the top of the stack" in first.telemetry_error
    assert runtime._telemetry_quarantine is not None

    # Turn 2 is functionally fine, never opens a scope on the dirty stack, and cannot
    # report itself telemetry-clean.
    assert second.error is None
    assert second.telemetry_error == runtime._telemetry_quarantine
    assert "not at the top of the stack" not in second.telemetry_error
    assert recording_scope.opened == ["deepagents-request"]
