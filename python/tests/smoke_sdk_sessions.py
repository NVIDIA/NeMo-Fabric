# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke: the SDK Session boundary -- start / invoke / stream / cancel / stop.

Dependency-free -- no native extension and no Hermes. A fake inline adapter
stands in for ``_run_inline_adapter`` and echoes the conversation history it
received, so we can assert that turn N sees the transcript produced by turn
N-1 (the core of stateless multi-turn), plus the full lifecycle: buffered
stream, cooperative cancel (idle and in-flight), idempotent stop, and
session-support gating.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemo_fabric import (
    FabricClient,
    FabricSessionUnsupportedError,
    Session,
    SessionStatus,
)
from nemo_fabric import client as client_mod
from nemo_fabric.client import _make_session

seen_history: list[list] = []


async def _fake_inline(plan, request, entrypoint):
    """Stand-in for _run_inline_adapter: echo history, return a full transcript."""

    history = (request.get("context") or {}).get("history") or []
    seen_history.append(list(history))
    turn = len(history) // 2 + 1
    transcript = list(history) + [
        {"role": "user", "content": request.get("input")},
        {"role": "assistant", "content": f"reply-{turn}"},
    ]
    return {
        "status": "succeeded",
        "events": [{"event_id": f"evt-{turn}", "kind": "log", "message": f"turn {turn}"}],
        "output": {
            "messages": transcript,
            "response": f"reply-{turn}",
            "session_id": "sess-fake",
        },
    }


def _session() -> Session:
    return Session(
        client=FabricClient(),
        plan={"agent_name": "demo", "profile": "hermes_sdk"},
        entrypoint=("fake.module", "run"),
    )


async def multi_turn_threads_history() -> None:
    seen_history.clear()
    client_mod._run_inline_adapter = _fake_inline  # type: ignore[assignment]
    session = _session()
    assert session.status is SessionStatus.ACTIVE
    assert session.messages == []

    await session.invoke("My name is Robin.")
    assert seen_history[0] == [], seen_history[0]
    after_turn1 = session.messages
    assert len(after_turn1) == 2, after_turn1

    await session.invoke("What's my name?")
    assert seen_history[1] == after_turn1, (seen_history[1], after_turn1)
    assert len(session.messages) == 4, session.messages
    assert session.id == "sess-fake", session.id


async def stream_yields_events_then_result() -> None:
    client_mod._run_inline_adapter = _fake_inline  # type: ignore[assignment]
    session = _session()
    items = [item async for item in session.stream("hello")]
    assert items[-1]["status"] == "succeeded", items[-1]  # RunResult is the last item
    events = items[:-1]
    assert events and all(e.get("kind") == "log" for e in events), events
    assert len(session.messages) == 2, session.messages  # the streamed turn advanced it


async def stop_is_idempotent_and_blocks_invoke() -> None:
    client_mod._run_inline_adapter = _fake_inline  # type: ignore[assignment]
    session = _session()
    await session.stop()
    assert session.status is SessionStatus.STOPPED
    await session.stop()  # idempotent
    try:
        await session.invoke("too late")
    except RuntimeError:
        pass
    else:
        raise AssertionError("invoke after stop should raise")

    async with _session() as ctx:
        await ctx.invoke("hi")
        assert ctx.status is SessionStatus.ACTIVE
    assert ctx.status is SessionStatus.STOPPED  # context manager auto-stops


async def cancel_when_idle_marks_cancelled() -> None:
    client_mod._run_inline_adapter = _fake_inline  # type: ignore[assignment]
    session = _session()
    await session.cancel()
    assert session.status is SessionStatus.CANCELLED
    await session.cancel()  # idempotent
    try:
        await session.invoke("after cancel")
    except RuntimeError:
        pass
    else:
        raise AssertionError("invoke after cancel should raise")


async def cancel_aborts_in_flight_turn() -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def _blocking_inline(plan, request, entrypoint):
        started.set()
        await release.wait()  # never released; cancellation is the only way out
        return {"status": "succeeded", "output": {}}

    client_mod._run_inline_adapter = _blocking_inline  # type: ignore[assignment]
    session = _session()
    turn = asyncio.create_task(session.invoke("long running"))
    await started.wait()
    await session.cancel()
    assert session.status is SessionStatus.CANCELLED
    try:
        await turn
    except asyncio.CancelledError:
        pass
    else:
        raise AssertionError("in-flight invoke should be cancelled")


async def gating_rejects_non_session_adapter() -> None:
    try:
        _make_session(
            FabricClient(),
            {"adapter_descriptor": {"descriptor": {"adapter_kind": "process"}}},
            None,
        )
    except FabricSessionUnsupportedError:
        pass
    else:
        raise AssertionError("process adapter should not be session-capable")


async def main() -> None:
    original = client_mod._run_inline_adapter
    try:
        await multi_turn_threads_history()
        await stream_yields_events_then_result()
        await stop_is_idempotent_and_blocks_invoke()
        await cancel_when_idle_marks_cancelled()
        await cancel_aborts_in_flight_turn()
        await gating_rejects_non_session_adapter()
    finally:
        client_mod._run_inline_adapter = original  # type: ignore[assignment]
    print("smoke_sdk_sessions ok")


if __name__ == "__main__":
    asyncio.run(main())
