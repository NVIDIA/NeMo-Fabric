# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke: the SDK Runtime boundary over the native runtime lifecycle."""

from __future__ import annotations

import json
from typing import Any

from nemo_fabric import Fabric, FabricStateError, RunRequest, RunResult, Runtime, RuntimeStatus


def _plan() -> dict[str, Any]:
    config = {
        "metadata": {"name": "demo"},
        "harness": {"adapter_id": "test.fabric.shim"},
        "runtime": {
            "input_schema": "chat",
            "output_schema": "message",
        },
    }
    return {
        "agent_name": "demo",
        "profiles": ["hermes"],
        "effective_config": {
            "agent_name": "demo",
            "profiles": ["hermes"],
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
            "service": False,
            "streaming": False,
            "updates": False,
            "cancellation": False,
        },
    }


def _runtime() -> dict[str, Any]:
    return {
        "runtime_id": "runtime-1",
        "runtime_binding": "fabric-runtime-binding-test",
        "agent_name": "demo",
        "harness": "hermes",
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

    def invoke_runtime(
        self, plan_json: str, runtime_json: str, request_json: str
    ) -> str:
        assert json.loads(plan_json)["agent_name"] == "demo"
        request = json.loads(request_json)
        self.requests.append(request)
        turn = len(self.requests)
        return json.dumps(
            {
                "agent_name": "demo",
                "profiles": ["hermes"],
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


class NativeClient(Fabric):
    def __init__(self, native: MockNative) -> None:
        super().__init__()
        self.native = native

    def _require_native_module(self, method: str) -> MockNative:
        return self.native


def _runtime_wrapper(native: MockNative) -> Runtime:
    return Runtime(client=NativeClient(native), plan=_plan(), runtime=_runtime())


async def stable_runtime_across_turns() -> None:
    native = MockNative()
    runtime = _runtime_wrapper(native)
    assert runtime.status is RuntimeStatus.ACTIVE
    assert runtime.runtime_id == "runtime-1"
    assert runtime.handle.runtime_id == "runtime-1"

    first = await runtime.invoke(
        request=RunRequest(
            input="My name is Robin.",
            request_id="runtime-request-1",
            context={"job_id": "job-1", "turn_id": "turn-1"},
        ),
    )
    await runtime.invoke(input="What's my name?")

    assert isinstance(first, RunResult)
    assert first.request_id == "runtime-request-1"
    assert [inv["runtime_id"] for inv in runtime.invocations] == ["runtime-1", "runtime-1"]
    assert native.requests[0]["context"]["job_id"] == "job-1"
    assert native.requests[0]["context"]["turn_id"] == "turn-1"
    assert "history" not in native.requests[0]["context"]
    assert "history" not in native.requests[1]["context"]
    assert runtime.runtime_id == "runtime-1"


async def runtime_lifecycle() -> None:
    native = MockNative()
    runtime = _runtime_wrapper(native)
    result = await runtime.invoke(input="hello")
    assert result.status == "succeeded"
    assert result.events and all(event.kind == "log" for event in result.events)

    await runtime.stop()
    await runtime.stop()
    assert runtime.status is RuntimeStatus.STOPPED
    assert native.stopped == 1
    try:
        await runtime.invoke(input="too late")
    except FabricStateError:
        pass
    else:
        raise AssertionError("invoke after stop should raise")


async def failed_result_exposes_structured_error() -> None:
    native = MockNative()
    runtime = _runtime_wrapper(native)
    result = await runtime.invoke(input="fail")

    assert isinstance(result, RunResult)
    assert result.status == "failed"
    assert result.error.stage == "invoke"
    assert result.error.code == "adapter_failed"
    assert result.error.retryable is False
    await runtime.stop()
    assert runtime.status is RuntimeStatus.STOPPED
    assert native.stopped == 1


async def test_sdk_runtimes():
    await stable_runtime_across_turns()
    await runtime_lifecycle()
    await failed_result_exposes_structured_error()
