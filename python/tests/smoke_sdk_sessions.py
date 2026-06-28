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

from nemo_fabric import (
    FabricCapabilityError,
    FabricClient,
    FabricStateError,
    RunRequest,
    RunResult,
    Session,
    SessionStatus,
)


def _plan() -> dict[str, Any]:
    config = {
        "metadata": {"name": "demo"},
        "harness": {"adapter_id": "test.fabric.shim"},
        "runtime": {
            "mode": "session",
            "transport": "library",
            "input_schema": "chat",
            "output_schema": "message",
        },
    }
    return {
        "agent_name": "demo",
        "profiles": ["hermes_sdk"],
        "effective_config": {
            "agent_name": "demo",
            "profiles": ["hermes_sdk"],
            "agent_root": ".",
            "config_path": "agent.yaml",
            "config_root": ".",
            "config": config,
        },
        "config": config,
        "adapter_descriptor": {
            "descriptor": {
                "adapter_kind": "python",
                "adapter_id": "test.fabric.shim",
                "harness": "hermes",
            }
        },
        "capabilities": {
            "session": True,
            "service": False,
            "streaming": False,
            "updates": False,
            "cancellation": False,
            "concurrent_invocations": False,
        },
    }


def _runtime() -> dict[str, Any]:
    return {
        "runtime_id": "runtime-1",
        "runtime_binding": "fabric-runtime-binding-test",
        "agent_name": "demo",
        "harness": "hermes",
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


class MockNative:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.stopped = 0

    def invoke_runtime(self, plan_json: str, runtime_json: str, request_json: str) -> str:
        request = json.loads(request_json)
        self.requests.append(request)
        turn = len(self.requests)
        return json.dumps(
            {
                "agent_name": "demo",
                "profiles": ["hermes_sdk"],
                "harness": "hermes",
                "adapter_kind": "python",
                "adapter_id": "test.fabric.shim",
                "status": "failed" if request.get("input") == "fail" else "succeeded",
                "request_id": request["request_id"],
                "runtime_id": json.loads(runtime_json)["runtime_id"],
                "invocation_id": f"invocation-{turn}",
                "events": [
                    {
                        "event_id": f"evt-{turn}",
                        "timestamp_millis": turn,
                        "kind": "log",
                        "message": "ok",
                    }
                ],
                "artifacts": {"artifacts": []},
                "output": {
                    "messages": [
                        {"role": "user", "content": request.get("input")},
                        {"role": "assistant", "content": f"reply-{turn}"},
                    ],
                },
                "error": {
                    "stage": "invoke",
                    "code": "adapter_failed",
                    "message": "adapter failed",
                    "retryable": False,
                }
                if request.get("input") == "fail"
                else None,
            }
        )

    def stop_runtime(self, plan_json: str, runtime_json: str) -> str:
        self.stopped += 1
        return "[]"


class NativeClient(FabricClient):
    def __init__(self, native: MockNative) -> None:
        super().__init__()
        self.native = native

    def _require_native_module(self, method: str) -> MockNative:
        return self.native


def _session(native: MockNative) -> Session:
    return Session(client=NativeClient(native), plan=_plan(), runtime=_runtime())


async def stable_runtime_across_turns() -> None:
    native = MockNative()
    session = _session(native)
    assert session.status is SessionStatus.ACTIVE
    assert session.runtime_id == "runtime-1"
    assert session.session_id == "runtime-1"
    assert session.info["session_id"] == "runtime-1"
    assert not hasattr(session, "id")

    first = await session.invoke(
        request=RunRequest(
            input="My name is Robin.",
            request_id="session-request-1",
            context={"job_id": "job-1", "turn_id": "turn-1"},
        ),
    )
    await session.invoke(input="What's my name?")

    assert isinstance(first, RunResult)
    assert first.request_id == "session-request-1"
    assert [inv["runtime_id"] for inv in session.invocations] == ["runtime-1", "runtime-1"]
    assert native.requests[0]["context"]["job_id"] == "job-1"
    assert native.requests[0]["context"]["turn_id"] == "turn-1"
    assert native.requests[0]["context"]["session_id"] == "runtime-1"
    assert native.requests[1]["context"]["session_id"] == "runtime-1"
    assert "history" not in native.requests[0]["context"]
    assert "history" not in native.requests[1]["context"]
    assert session.runtime_id == "runtime-1"


async def stream_and_lifecycle() -> None:
    native = MockNative()
    session = _session(native)
    items = [item async for item in session.stream(input="hello")]
    assert items[-1].status == "succeeded"
    assert items[:-1] and all(event.kind == "log" for event in items[:-1])

    await session.stop()
    await session.stop()
    assert session.status is SessionStatus.STOPPED
    assert native.stopped == 1
    try:
        await session.invoke(input="too late")
    except FabricStateError:
        pass
    else:
        raise AssertionError("invoke after stop should raise")


async def unsupported_cancel_leaves_session_active() -> None:
    native = MockNative()
    session = _session(native)
    try:
        await session.cancel()
    except FabricCapabilityError:
        pass
    else:
        raise AssertionError("unsupported cancellation should raise")
    assert session.status is SessionStatus.ACTIVE
    await session.stop()


async def failed_result_exposes_structured_error() -> None:
    native = MockNative()
    session = _session(native)
    result = await session.invoke(input="fail")

    assert isinstance(result, RunResult)
    assert result.status == "failed"
    assert result.error.stage == "invoke"
    assert result.error.code == "adapter_failed"
    assert result.error.retryable is False
    await session.stop()
    assert session.status is SessionStatus.STOPPED
    assert native.stopped == 1


async def main() -> None:
    await stable_runtime_across_turns()
    await stream_and_lifecycle()
    await unsupported_cancel_leaves_session_active()
    await failed_result_exposes_structured_error()
    print("smoke_sdk_sessions ok")


if __name__ == "__main__":
    asyncio.run(main())
