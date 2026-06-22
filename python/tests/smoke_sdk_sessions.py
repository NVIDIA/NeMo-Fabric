# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke: the SDK Session threads accumulated history across turns.

Dependency-free — no native extension and no Hermes. A fake inline adapter
stands in for ``_run_inline_adapter`` and echoes the conversation history it
received, so we can assert that turn N sees the transcript produced by turn
N-1 (the core of stateless multi-turn), plus lifecycle behavior.
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
    """Stand-in for _run_inline_adapter: echo history, return full transcript."""
    history = (request.get("context") or {}).get("history") or []
    seen_history.append(list(history))
    turn = len(history) // 2 + 1
    transcript = list(history) + [
        {"role": "user", "content": request.get("input")},
        {"role": "assistant", "content": f"reply-{turn}"},
    ]
    return {
        "status": "succeeded",
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


async def main() -> None:
    client_mod._run_inline_adapter = _fake_inline  # type: ignore[assignment]

    s = _session()
    assert s.status is SessionStatus.ACTIVE
    assert s.messages == []

    # Turn 1: no prior history.
    await s.invoke("My name is Robin.")
    assert seen_history[0] == [], seen_history[0]
    after_turn1 = s.messages
    assert len(after_turn1) == 2, after_turn1

    # Turn 2: must see turn-1's transcript as history.
    await s.invoke("What's my name?")
    assert seen_history[1] == after_turn1, (seen_history[1], after_turn1)
    assert len(s.messages) == 4, s.messages

    # Harness session id is adopted from adapter output.
    assert s.id == "sess-fake", s.id

    # Lifecycle: stop is idempotent; invoke after stop is rejected.
    await s.stop()
    assert s.status is SessionStatus.STOPPED
    await s.stop()  # idempotent
    try:
        await s.invoke("too late")
    except RuntimeError:
        pass
    else:
        raise AssertionError("invoke after stop should raise")

    # Context manager auto-stops.
    async with _session() as s2:
        await s2.invoke("hi")
        assert s2.status is SessionStatus.ACTIVE
    assert s2.status is SessionStatus.STOPPED

    # Follow-up verbs are explicit NotImplementedError, not silent.
    for coro in (_session().cancel(), _session().stream("x")):
        try:
            await coro
        except NotImplementedError:
            pass
        else:
            raise AssertionError("cancel/stream should raise NotImplementedError")

    # Gating: a non-python adapter cannot open a session.
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

    print("smoke_sdk_sessions ok")


if __name__ == "__main__":
    asyncio.run(main())
