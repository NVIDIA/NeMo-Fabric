# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the SDK Session boundary: start / invoke / stream / cancel / stop.

Dependency-free: the inline adapter is monkeypatched, so these exercise the
Session orchestration (history replay, per-turn overrides, handle correlation,
lifecycle, gating, errors) without the native extension or a real harness.
"""

from __future__ import annotations

import asyncio

import pytest

from nemo_fabric import (
    FabricClient,
    FabricNativeUnavailableError,
    FabricSessionUnsupportedError,
    Session,
    SessionStatus,
)
from nemo_fabric import client as client_mod


def _plan(adapter_kind: str = "python") -> dict:
    return {
        "agent_name": "demo",
        "profile": "hermes_sdk",
        "adapter_descriptor": {
            "descriptor": {"adapter_kind": adapter_kind, "adapter_id": "test.fabric.shim"}
        },
    }


def _session(overrides: dict | None = None) -> Session:
    return Session(
        client=FabricClient(),
        plan=_plan(),
        entrypoint=("fake.module", "run"),
        overrides=overrides,
    )


@pytest.fixture(name="seen_history")
def echo_adapter_fixture(monkeypatch: pytest.MonkeyPatch) -> list[list]:
    """Patch _run_inline_adapter to echo history and emit per-turn handles.

    Returns the list of histories the adapter saw, one entry per turn.
    """

    seen: list[list] = []

    async def _fake(plan, request, entrypoint):
        history = (request.get("context") or {}).get("history") or []
        seen.append(list(history))
        turn = len(history) // 2 + 1
        transcript = list(history) + [
            {"role": "user", "content": request.get("input")},
            {"role": "assistant", "content": f"reply-{turn}"},
        ]
        return {
            "status": "succeeded",
            "request_id": request.get("request_id"),
            "runtime_id": f"runtime-{turn}",
            "invocation_id": f"invocation-{turn}",
            "events": [{"event_id": f"evt-{turn}", "kind": "log", "message": f"turn {turn}"}],
            "output": {
                "messages": transcript,
                "response": f"reply-{turn}",
                "session_id": "sess-1",
            },
        }

    monkeypatch.setattr(client_mod, "_run_inline_adapter", _fake)
    return seen


async def test_new_session_is_active_and_empty(seen_history: list[list]) -> None:
    session = _session()
    assert session.status is SessionStatus.ACTIVE
    assert session.messages == []
    assert session.invocations == []


async def test_invoke_threads_accumulated_history(seen_history: list[list]) -> None:
    session = _session()

    await session.invoke("My name is Robin.")
    assert seen_history[0] == []  # turn 1 sees no prior history
    after_turn1 = session.messages
    assert len(after_turn1) == 2

    await session.invoke("What's my name?")
    assert seen_history[1] == after_turn1  # turn 2 sees turn 1's transcript
    assert len(session.messages) == 4


async def test_invoke_records_per_turn_handles(seen_history: list[list]) -> None:
    session = _session()
    await session.invoke("a")
    await session.invoke("b")

    assert [inv["runtime_id"] for inv in session.invocations] == ["runtime-1", "runtime-2"]
    assert session.invocations[0]["invocation_id"] == "invocation-1"
    # A session may span multiple runtimes when the harness has no resumable one.
    assert session.invocations[0]["runtime_id"] != session.invocations[1]["runtime_id"]


async def test_invoke_adopts_session_id_from_output(seen_history: list[list]) -> None:
    session = _session()
    await session.invoke("hi")
    assert session.id == "sess-1"


async def test_invoke_merges_session_and_turn_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}

    async def _fake(plan, request, entrypoint):
        captured["overrides"] = request.get("overrides")
        return {"status": "succeeded", "output": {}}

    monkeypatch.setattr(client_mod, "_run_inline_adapter", _fake)
    session = _session(overrides={"model": "a"})
    await session.invoke("x", overrides={"temperature": 0.0})

    assert captured["overrides"] == {"model": "a", "temperature": 0.0}


async def test_stream_yields_events_then_result(seen_history: list[list]) -> None:
    session = _session()
    items = [item async for item in session.stream("hi")]

    assert items[-1]["status"] == "succeeded"  # RunResult is the terminal item
    events = items[:-1]
    assert events and all(event.get("kind") == "log" for event in events)
    assert len(session.messages) == 2  # the streamed turn advanced the transcript


async def test_stop_is_idempotent_and_blocks_invoke(seen_history: list[list]) -> None:
    session = _session()
    await session.stop()
    assert session.status is SessionStatus.STOPPED
    await session.stop()  # idempotent
    with pytest.raises(RuntimeError):
        await session.invoke("too late")


async def test_context_manager_auto_stops(seen_history: list[list]) -> None:
    async with _session() as session:
        await session.invoke("hi")
        assert session.status is SessionStatus.ACTIVE
    assert session.status is SessionStatus.STOPPED


async def test_cancel_when_idle_marks_cancelled(seen_history: list[list]) -> None:
    session = _session()
    await session.cancel()
    assert session.status is SessionStatus.CANCELLED
    await session.cancel()  # idempotent
    with pytest.raises(RuntimeError):
        await session.invoke("after cancel")


async def test_cancel_aborts_in_flight_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def _blocking(plan, request, entrypoint):
        started.set()
        await release.wait()  # only cancellation unblocks this
        return {"status": "succeeded", "output": {}}

    monkeypatch.setattr(client_mod, "_run_inline_adapter", _blocking)
    session = _session()
    turn = asyncio.create_task(session.invoke("long running"))
    await started.wait()

    await session.cancel()
    assert session.status is SessionStatus.CANCELLED
    with pytest.raises(asyncio.CancelledError):
        await turn


async def test_info_summarizes_the_session(seen_history: list[list]) -> None:
    session = _session()
    info = session.info
    assert info["session_id"] == session.id
    assert info["agent_name"] == "demo"
    assert info["profile"] == "hermes_sdk"
    assert info["adapter_kind"] == "python"
    assert info["harness_type"] == "test.fabric.shim"


async def test_messages_and_invocations_return_copies(seen_history: list[list]) -> None:
    session = _session()
    await session.invoke("hi")

    snapshot = session.messages
    snapshot.append({"role": "user", "content": "tampered"})
    snapshot[0]["content"] = "mutated"  # deep mutation of a returned message object
    invocations = session.invocations
    invocations.clear()

    # Mutating the returned lists or their items must not affect session state.
    assert len(session.messages) == 2
    assert session.messages[0]["content"] != "mutated"
    assert len(session.invocations) == 1


async def test_invoke_without_output_messages_keeps_transcript(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake(plan, request, entrypoint):
        return {"status": "succeeded", "runtime_id": "r1", "invocation_id": "i1", "output": {}}

    monkeypatch.setattr(client_mod, "_run_inline_adapter", _fake)
    session = _session()
    await session.invoke("hi")

    assert session.messages == []  # no echoed messages -> transcript unchanged
    assert len(session.invocations) == 1  # the turn is still recorded for correlation


async def test_session_history_is_authoritative_over_request(
    seen_history: list[list],
) -> None:
    session = _session()
    await session.invoke("turn one")
    transcript = session.messages

    # A caller-supplied request carrying stale history must not override the
    # session's accumulated transcript.
    stale = {"input": "turn two", "context": {"history": [{"role": "user", "content": "stale"}]}}
    await session.invoke(request=stale)

    assert seen_history[1] == transcript


async def test_request_level_overrides_are_merged(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    async def _fake(plan, request, entrypoint):
        captured["overrides"] = request.get("overrides")
        return {"status": "succeeded", "output": {}}

    monkeypatch.setattr(client_mod, "_run_inline_adapter", _fake)
    session = _session(overrides={"a": "session"})
    await session.invoke(
        request={"input": "x", "overrides": {"b": "request"}},
        overrides={"c": "turn"},
    )

    # session < request < per-turn, all merged (none bypassed).
    assert captured["overrides"] == {"a": "session", "b": "request", "c": "turn"}


async def test_make_session_rejects_non_session_adapter() -> None:
    with pytest.raises(FabricSessionUnsupportedError):
        client_mod._make_session(FabricClient(), _plan(adapter_kind="process"), None)


async def test_start_requires_native_extension() -> None:
    # A CLI-pinned client has no native module, so the typed session path must
    # fail loudly rather than silently degrade.
    client = FabricClient(command=("fabric",))
    with pytest.raises(FabricNativeUnavailableError):
        await client.start("any/agent")
