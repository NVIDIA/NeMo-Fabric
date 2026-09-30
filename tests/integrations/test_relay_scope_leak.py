# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration coverage for overlapping LangGraph runs on Relay's shared scope stack.

Relay restores the stack after an out-of-order sibling pop. The Deep Agents adapter
still captures a baseline and quarantines telemetry if a scope is ever stranded, so
these tests drive the actual Relay callback and scope rather than a stub.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
from nemo_fabric_adapters.deepagents import adapter

nemo_relay = pytest.importorskip("nemo_relay", reason="requires the nemo-relay extra")

from nemo_relay.integrations.langchain.callbacks import (  # noqa: E402
    NemoRelayCallbackHandler,
)


@pytest.fixture(autouse=True)
def isolated_scope_stack(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Give each test its own Relay scope stack; the stack is process-global."""

    os.environ["XDG_CONFIG_HOME"] = str(tmp_path / "xdg-config")
    monkeypatch.chdir(tmp_path)
    token = nemo_relay._scope_stack_var.set(nemo_relay.create_scope_stack())
    try:
        yield
    finally:
        nemo_relay._scope_stack_var.reset(token)


async def _overlapping_chain_runs(handler: NemoRelayCallbackHandler) -> None:
    """Close two sibling chain runs out of LIFO order, as LangGraph's tasks do."""

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


async def test_overlapping_chain_runs_leave_the_shared_stack_restored():
    handler = NemoRelayCallbackHandler()
    baseline = nemo_relay.scope.get_handle()

    with nemo_relay.scope.scope("deepagents-request", nemo_relay.ScopeType.Agent):
        await _overlapping_chain_runs(handler)

    assert handler._scope_handles == {}
    assert nemo_relay.scope.get_handle().uuid == baseline.uuid
    assert adapter._scope_top_unchanged(baseline) is True


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
        self.parent_uuids: list[str] = []
        self.metadata: list[object] = []

    @contextlib.contextmanager
    def scope(self, name: str, scope_type: object, **kwargs: object):
        self.opened.append(name)
        self.parent_uuids.append(str(nemo_relay.scope.get_handle().uuid))
        self.metadata.append(kwargs.get("metadata"))
        with nemo_relay.scope.scope(name, scope_type, **kwargs):
            yield


class _NoopPlugin:
    """Stand-in for the Relay plugin, which needs a live gateway config to start."""

    @contextlib.asynccontextmanager
    async def activate(self, config: object):
        yield SimpleNamespace(
            report={"config": {"diagnostics": [], "runtime_diagnostics": []}}
        )


async def test_uuid_request_id_seeds_real_relay_parent(monkeypatch):
    async def fake_invoke(agent, user_message, thread_id, callbacks=None):
        return {"messages": []}, [], []

    monkeypatch.setattr(adapter, "invoke_compiled_agent", fake_invoke)

    baseline = nemo_relay.scope.get_handle()
    recording_scope = _RecordingScope()
    runtime = adapter.DeepAgentsRuntime()
    runtime._agent = object()
    runtime._relay_plugin = _NoopPlugin()
    runtime._relay_plugin_config = {}
    runtime._relay_scope = recording_scope
    runtime._relay_scope_type = nemo_relay.ScopeType
    runtime._callback_handler_type = NemoRelayCallbackHandler
    request_id = "018f47a4-3af7-7d94-8e61-9f0f89b5d312"

    outcome = await runtime._invoke_with_telemetry(
        "hello",
        request_id,
        "invocation-1",
    )

    assert outcome.error is None
    assert outcome.telemetry_error is None
    assert recording_scope.parent_uuids == [request_id]
    assert recording_scope.metadata == [
        {
            "nemo_fabric_request_id": request_id,
            "nemo_fabric_invocation_id": "invocation-1",
        }
    ]
    assert nemo_relay.scope.get_handle().uuid == baseline.uuid


async def test_overlapping_turns_stay_telemetry_clean(monkeypatch):
    """Two ordered turns end to end: real scope, real callback, real stack."""

    async def fake_invoke(agent, user_message, thread_id, callbacks=None):
        if callbacks:
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

    first = await runtime._invoke_with_telemetry("hello", "request-1", "invocation-1")
    second = await runtime._invoke_with_telemetry(
        "hello again", "request-2", "invocation-2"
    )

    assert (first.error, first.telemetry_error) == (None, None)
    assert (second.error, second.telemetry_error) == (None, None)
    assert runtime._telemetry_quarantine is None
    assert recording_scope.opened == ["deepagents-request", "deepagents-request"]
