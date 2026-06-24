# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke: the SDK Session boundary over the native RuntimeHandle lifecycle."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemo_fabric import FabricClient, Session, SessionStatus


def _plan() -> dict[str, Any]:
    return {
        "agent_name": "demo",
        "profile": "hermes_sdk",
        "adapter_descriptor": {
            "descriptor": {"adapter_kind": "python", "adapter_id": "test.fabric.shim"}
        },
    }


def _runtime() -> dict[str, Any]:
    return {
        "runtime_id": "runtime-1",
        "agent_name": "demo",
        "harness_type": "test.fabric.shim",
        "mode": "session",
        "adapter_kind": "python",
        "adapter_id": "test.fabric.shim",
        "environment": {
            "environment_id": "environment-1",
            "provider": "local",
            "control_location": "external_control",
            "ownership": "caller_owned",
        },
    }


class FakeNative:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.stopped = 0

    def invoke_runtime(self, plan_json: str, runtime_json: str, request_json: str) -> str:
        request = json.loads(request_json)
        self.requests.append(request)
        turn = len(self.requests)
        return json.dumps(
            {
                "status": "succeeded",
                "request_id": request["request_id"],
                "runtime_id": json.loads(runtime_json)["runtime_id"],
                "invocation_id": f"invocation-{turn}",
                "events": [{"event_id": f"evt-{turn}", "kind": "log", "message": "ok"}],
                "output": {
                    "messages": [
                        {"role": "user", "content": request.get("input")},
                        {"role": "assistant", "content": f"reply-{turn}"},
                    ],
                },
            }
        )

    def stop_runtime(self, plan_json: str, runtime_json: str) -> str:
        self.stopped += 1
        return "[]"


class NativeClient(FabricClient):
    def __init__(self, native: FakeNative) -> None:
        super().__init__()
        self.native = native

    def _require_native_module(self, method: str) -> FakeNative:
        return self.native


def _session(native: FakeNative) -> Session:
    return Session(client=NativeClient(native), plan=_plan(), runtime=_runtime())


async def stable_runtime_across_turns() -> None:
    native = FakeNative()
    session = _session(native)
    assert session.status is SessionStatus.ACTIVE
    assert session.runtime_id == "runtime-1"
    assert "session_id" not in session.info
    assert not hasattr(session, "id")

    await session.invoke("My name is Robin.")
    await session.invoke("What's my name?")

    assert [inv["runtime_id"] for inv in session.invocations] == ["runtime-1", "runtime-1"]
    assert "history" not in native.requests[0]["context"]
    assert "history" not in native.requests[1]["context"]
    assert session.runtime_id == "runtime-1"


async def stream_and_lifecycle() -> None:
    native = FakeNative()
    session = _session(native)
    items = [item async for item in session.stream("hello")]
    assert items[-1]["status"] == "succeeded"
    assert items[:-1] and all(e.get("kind") == "log" for e in items[:-1])

    await session.stop()
    await session.stop()
    assert session.status is SessionStatus.STOPPED
    assert native.stopped == 1
    try:
        await session.invoke("too late")
    except RuntimeError:
        pass
    else:
        raise AssertionError("invoke after stop should raise")


async def cancel_when_idle_marks_cancelled() -> None:
    session = _session(FakeNative())
    await session.cancel()
    assert session.status is SessionStatus.CANCELLED
    try:
        await session.invoke("after cancel")
    except RuntimeError:
        pass
    else:
        raise AssertionError("invoke after cancel should raise")


async def main() -> None:
    await stable_runtime_across_turns()
    await stream_and_lifecycle()
    await cancel_when_idle_marks_cancelled()
    print("smoke_sdk_sessions ok")


if __name__ == "__main__":
    asyncio.run(main())
