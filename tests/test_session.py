# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the SDK Session boundary: start / invoke / stream / cancel / stop.

Dependency-free: a fake native lifecycle module stands in for the Rust binding,
so these exercise the Python orchestration without Hermes or a built extension.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import pytest

from nemo_fabric import FabricClient, FabricNativeUnavailableError, Session, SessionStatus
from nemo_fabric import client as client_mod


def _plan(adapter_kind: str = "python", runtime_mode: str = "session") -> dict[str, Any]:
    return {
        "agent_name": "demo",
        "profile": "hermes_sdk",
        "config": {
            "runtime": {
                "mode": runtime_mode,
                "transport": "library",
                "input_schema": "chat",
                "output_schema": "message",
            },
        },
        "adapter_descriptor": {
            "descriptor": {
                "adapter_kind": adapter_kind,
                "adapter_id": "test.fabric.shim",
                "runner": {"module": "fake.module", "callable": "run"},
            }
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
    def __init__(self, runtime_mode: str = "session") -> None:
        self.runtime_mode = runtime_mode
        self.plans: list[dict[str, Any]] = []
        self.requests: list[dict[str, Any]] = []
        self.stopped = 0
        self.block_invoke = False
        self.fail_invoke = False
        self.fail_stop = False

    def plan(self, path: str, profile: Any = None) -> str:
        self.plans.append({"path": path, "profile": profile})
        assert path == "agent"
        if profile is not None:
            assert profile == "hermes_sdk"
        return json.dumps(_plan(runtime_mode=self.runtime_mode))

    def plan_config(
        self,
        config_json: str,
        profiles_json: str | None = None,
        base_dir: str | None = None,
    ) -> str:
        assert json.loads(config_json)["metadata"]["name"] == "demo"
        return json.dumps(_plan(runtime_mode=self.runtime_mode))

    def start_runtime(self, plan_json: str) -> str:
        assert json.loads(plan_json)["agent_name"] == "demo"
        return json.dumps(_runtime())

    def invoke_runtime(
        self, plan_json: str, runtime_json: str, request_json: str
    ) -> str:
        if self.block_invoke:
            time.sleep(0.2)
        if self.fail_invoke:
            raise RuntimeError("invoke failed")
        plan = json.loads(plan_json)
        runtime = json.loads(runtime_json)
        request = json.loads(request_json)
        self.requests.append(request)
        turn = len(self.requests)
        return json.dumps(
            {
                "agent_name": plan["agent_name"],
                "profile": plan.get("profile"),
                "harness_type": "test.fabric.shim",
                "adapter_kind": "python",
                "adapter_id": "test.fabric.shim",
                "runtime_id": runtime["runtime_id"],
                "invocation_id": f"invocation-{turn}",
                "request_id": request["request_id"],
                "status": "succeeded",
                "events": [
                    {
                        "event_id": f"evt-{turn}",
                        "kind": "log",
                        "message": f"turn {turn}",
                    }
                ],
                "output": {
                    "messages": [
                        {"role": "user", "content": request.get("input")},
                        {"role": "assistant", "content": f"reply-{turn}"},
                    ],
                    "response": f"reply-{turn}",
                },
                "artifacts": {"artifacts": []},
            }
        )

    def stop_runtime(self, plan_json: str, runtime_json: str) -> str:
        assert json.loads(plan_json)["agent_name"] == "demo"
        assert json.loads(runtime_json)["runtime_id"] == "runtime-1"
        self.stopped += 1
        if self.fail_stop:
            raise RuntimeError("stop failed")
        return json.dumps([])


class NativeClient(FabricClient):
    def __init__(self, native: FakeNative) -> None:
        super().__init__()
        self.native = native

    def plan(self, path, *, profile=None):  # type: ignore[no-untyped-def,override]
        return json.loads(self.native.plan(str(path), profile))

    def _native_module(self) -> FakeNative:
        return self.native

    def _require_native_module(self, method: str) -> FakeNative:
        return self.native


def _session(native: FakeNative | None = None, overrides: dict | None = None) -> Session:
    return Session(
        client=NativeClient(native or FakeNative()),
        plan=_plan(),
        runtime=_runtime(),
        overrides=overrides,
    )


def test_session_constructor_rejects_non_session_runtime_mode():
    with pytest.raises(RuntimeError, match="requires runtime.mode=session"):
        Session(
            client=NativeClient(FakeNative()),
            plan=_plan(runtime_mode="oneshot"),
            runtime=_runtime(),
        )


async def test_start_creates_session_from_core_runtime_handle() -> None:
    native = FakeNative()
    session = await NativeClient(native).start("agent", profile="hermes_sdk")

    assert native.plans == [{"path": "agent", "profile": "hermes_sdk"}]
    assert session.status is SessionStatus.ACTIVE
    assert session.runtime_id == "runtime-1"
    assert session.runtime["runtime_id"] == "runtime-1"
    assert session.info["runtime_id"] == "runtime-1"
    assert "session_id" not in session.info
    assert not hasattr(session, "id")


async def test_start_accepts_caller_session_id_and_propagates_to_turn_context():
    native = FakeNative()
    session = await NativeClient(native).start(
        "agent",
        profile="hermes_sdk",
        session_id="caller-session-123",
    )

    result = await session.invoke("hello session")

    assert session.session_id == "caller-session-123"
    assert result["runtime_id"] == "runtime-1"
    assert native.requests[0]["context"]["session_id"] == "caller-session-123"


async def test_start_rejects_non_session_runtime_mode():
    native = FakeNative(runtime_mode="oneshot")

    with pytest.raises(RuntimeError, match="requires runtime.mode=session"):
        await NativeClient(native).start("agent", profile="hermes_sdk")

    assert native.stopped == 0
    assert native.requests == []


async def test_session_id_defaults_to_runtime_id_for_adapter_context():
    native = FakeNative()
    session = _session(native)

    await session.invoke("hello default session")

    assert session.session_id == "runtime-1"
    assert native.requests[0]["context"]["session_id"] == "runtime-1"


async def test_invoke_uses_stable_runtime_and_does_not_replay_history() -> None:
    native = FakeNative()
    session = _session(native)

    await session.invoke("My name is Robin.")
    await session.invoke("What's my name?")

    assert [inv["runtime_id"] for inv in session.invocations] == [
        "runtime-1",
        "runtime-1",
    ]
    assert "history" not in native.requests[0]["context"]
    assert "history" not in native.requests[1]["context"]
    assert session.runtime_id == "runtime-1"
    assert len(session.messages) == 2


async def test_request_level_overrides_are_merged() -> None:
    native = FakeNative()
    session = _session(native, overrides={"a": "session"})
    await session.invoke(
        request={"input": "x", "overrides": {"b": "request"}},
        overrides={"c": "turn"},
    )

    assert native.requests[0]["overrides"] == {
        "a": "session",
        "b": "request",
        "c": "turn",
    }


async def test_stream_yields_events_then_result() -> None:
    session = _session()
    items = [item async for item in session.stream("hi")]

    assert items[-1]["status"] == "succeeded"
    assert items[:-1] and all(event.get("kind") == "log" for event in items[:-1])


async def test_stop_is_idempotent_and_blocks_invoke() -> None:
    native = FakeNative()
    session = _session(native)

    await session.stop()
    await session.stop()

    assert session.status is SessionStatus.STOPPED
    assert native.stopped == 1
    with pytest.raises(RuntimeError):
        await session.invoke("too late")


async def test_stop_rejects_in_flight_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def _blocking(func):  # type: ignore[no-untyped-def]
        started.set()
        await release.wait()
        return func()

    monkeypatch.setattr(client_mod, "_call_blocking", _blocking)
    native = FakeNative()
    session = _session(native)
    first = asyncio.create_task(session.invoke("turn one"))
    await started.wait()

    with pytest.raises(RuntimeError, match="turn is in flight"):
        await session.stop()
    assert session.status is SessionStatus.ACTIVE
    assert native.stopped == 0

    release.set()
    await first


async def test_stop_blocks_new_turns_while_shutdown_is_in_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def _blocking(func):  # type: ignore[no-untyped-def]
        if session._closing:  # noqa: SLF001 - state-machine regression test
            started.set()
            await release.wait()
        return func()

    monkeypatch.setattr(client_mod, "_call_blocking", _blocking)
    native = FakeNative()
    session = _session(native)
    stop_task = asyncio.create_task(session.stop())
    await started.wait()

    with pytest.raises(RuntimeError, match="shutdown is in progress"):
        await session.invoke("too late")

    release.set()
    await stop_task
    assert session.status is SessionStatus.STOPPED
    assert native.stopped == 1


async def test_stop_failure_clears_shutdown_guard_for_retry() -> None:
    native = FakeNative()
    native.fail_stop = True
    session = _session(native)

    with pytest.raises(RuntimeError, match="stop failed"):
        await session.stop()

    assert session.status is SessionStatus.ACTIVE
    assert session._closing is False  # noqa: SLF001 - state-machine regression test

    native.fail_stop = False
    await session.stop()

    assert session.status is SessionStatus.STOPPED
    assert native.stopped == 2


async def test_context_manager_auto_stops() -> None:
    native = FakeNative()
    async with _session(native) as session:
        await session.invoke("hi")
        assert session.status is SessionStatus.ACTIVE

    assert session.status is SessionStatus.STOPPED
    assert native.stopped == 1


async def test_cancel_when_idle_marks_cancelled() -> None:
    native = FakeNative()
    session = _session(native)
    await session.cancel()
    await session.cancel()

    assert session.status is SessionStatus.CANCELLED
    assert native.stopped == 1
    with pytest.raises(RuntimeError):
        await session.invoke("after cancel")


async def test_cancel_stop_failure_keeps_session_retryable() -> None:
    native = FakeNative()
    native.fail_stop = True
    session = _session(native)

    with pytest.raises(RuntimeError, match="stop failed"):
        await session.cancel()

    assert session.status is SessionStatus.ACTIVE
    assert native.stopped == 1

    native.fail_stop = False
    await session.cancel()

    assert session.status is SessionStatus.CANCELLED
    assert native.stopped == 2


async def test_cancel_blocks_new_turns_while_shutdown_is_in_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def _blocking(func):  # type: ignore[no-untyped-def]
        if session._closing:  # noqa: SLF001 - state-machine regression test
            started.set()
            await release.wait()
        return func()

    monkeypatch.setattr(client_mod, "_call_blocking", _blocking)
    native = FakeNative()
    session = _session(native)
    cancel_task = asyncio.create_task(session.cancel())
    await started.wait()

    with pytest.raises(RuntimeError, match="shutdown is in progress"):
        await session.invoke("too late")

    release.set()
    await cancel_task
    assert session.status is SessionStatus.CANCELLED
    assert native.stopped == 1


async def test_cancel_aborts_in_flight_turn() -> None:
    native = FakeNative()
    native.block_invoke = True
    session = _session(native)
    turn = asyncio.create_task(session.invoke("long running"))
    await asyncio.sleep(0)

    await session.cancel()

    assert session.status is SessionStatus.CANCELLED
    assert native.stopped == 1
    with pytest.raises(asyncio.CancelledError):
        await turn


async def test_info_summarizes_the_session() -> None:
    session = _session()
    info = session.info

    assert info["runtime_id"] == "runtime-1"
    assert info["agent_name"] == "demo"
    assert info["profile"] == "hermes_sdk"
    assert info["adapter_kind"] == "python"
    assert info["harness_type"] == "test.fabric.shim"


async def test_messages_invocations_and_runtime_return_copies() -> None:
    session = _session()
    await session.invoke("hi")

    messages = session.messages
    messages[0]["content"] = "mutated"
    invocations = session.invocations
    invocations.clear()
    runtime = session.runtime
    runtime["runtime_id"] = "mutated"

    assert session.messages[0]["content"] == "hi"
    assert len(session.invocations) == 1
    assert session.runtime["runtime_id"] == "runtime-1"


async def test_invoke_without_output_messages_keeps_transcript() -> None:
    class NoMessageNative(FakeNative):
        def invoke_runtime(self, plan_json, runtime_json, request_json):  # type: ignore[no-untyped-def]
            self.requests.append(json.loads(request_json))
            return json.dumps(
                {
                    "status": "succeeded",
                    "runtime_id": "runtime-1",
                    "invocation_id": "invocation-1",
                    "request_id": self.requests[-1]["request_id"],
                    "output": {},
                }
            )

    session = _session(NoMessageNative())
    await session.invoke("hi")

    assert session.messages == []
    assert len(session.invocations) == 1


async def test_empty_output_messages_replaces_existing_transcript() -> None:
    class EmptyMessageNative(FakeNative):
        def invoke_runtime(self, plan_json, runtime_json, request_json):  # type: ignore[no-untyped-def]
            if not self.requests:
                return super().invoke_runtime(plan_json, runtime_json, request_json)
            request = json.loads(request_json)
            self.requests.append(request)
            return json.dumps(
                {
                    "status": "succeeded",
                    "runtime_id": "runtime-1",
                    "invocation_id": f"invocation-{len(self.requests)}",
                    "request_id": request["request_id"],
                    "output": {"messages": []},
                }
            )

    native = EmptyMessageNative()
    session = _session(native)
    await session.invoke("hi")
    assert session.messages

    await session.invoke("reset")
    assert session.messages == []


async def test_concurrent_invokes_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def _blocking(func):  # type: ignore[no-untyped-def]
        started.set()
        await release.wait()
        return func()

    monkeypatch.setattr(client_mod, "_call_blocking", _blocking)
    session = _session()
    first = asyncio.create_task(session.invoke("turn one"))
    await started.wait()

    with pytest.raises(RuntimeError):
        await session.invoke("turn two")

    release.set()
    await first


async def test_start_requires_native_extension() -> None:
    client = FabricClient(command=("fabric",))
    with pytest.raises(FabricNativeUnavailableError):
        await client.start("any/agent")


async def test_run_collapses_through_core_runtime_lifecycle() -> None:
    native = FakeNative()

    result = await NativeClient(native).run("agent", input_text="hello")

    assert result["status"] == "succeeded"
    assert result["runtime_id"] == "runtime-1"
    assert native.requests[0]["input"] == "hello"
    assert native.stopped == 1


async def test_run_stops_runtime_when_invoke_raises() -> None:
    native = FakeNative()
    native.fail_invoke = True

    with pytest.raises(RuntimeError, match="invoke failed"):
        await NativeClient(native).run("agent", input_text="hello")

    assert native.stopped == 1


async def test_run_preserves_invoke_error_when_stop_also_raises() -> None:
    native = FakeNative()
    native.fail_invoke = True
    native.fail_stop = True

    with pytest.raises(RuntimeError, match="invoke failed"):
        await NativeClient(native).run("agent", input_text="hello")

    assert native.stopped == 1


async def test_run_surfaces_stop_error_after_successful_invoke() -> None:
    native = FakeNative()
    native.fail_stop = True

    with pytest.raises(RuntimeError, match="stop failed"):
        await NativeClient(native).run("agent", input_text="hello")

    assert native.stopped == 1


async def test_run_config_collapses_through_core_runtime_lifecycle() -> None:
    native = FakeNative()
    config = {"schema_version": "fabric.agent/v1alpha1", "metadata": {"name": "demo"}}

    result = await NativeClient(native).run_config(config, input_text="hello typed")

    assert result["status"] == "succeeded"
    assert result["runtime_id"] == "runtime-1"
    assert native.requests[0]["input"] == "hello typed"
    assert native.stopped == 1


async def test_start_config_creates_session_from_core_runtime_handle() -> None:
    native = FakeNative()
    config = {"schema_version": "fabric.agent/v1alpha1", "metadata": {"name": "demo"}}

    session = await NativeClient(native).start_config(config)
    result = await session.invoke("hello typed session")

    assert session.status is SessionStatus.ACTIVE
    assert session.runtime_id == "runtime-1"
    assert result["runtime_id"] == "runtime-1"
    assert native.requests[0]["input"] == "hello typed session"


async def test_start_config_accepts_caller_session_id():
    native = FakeNative()
    config = {"schema_version": "fabric.agent/v1alpha1", "metadata": {"name": "demo"}}

    session = await NativeClient(native).start_config(
        config,
        session_id="typed-session-123",
    )
    await session.invoke("hello typed session")

    assert session.session_id == "typed-session-123"
    assert native.requests[0]["context"]["session_id"] == "typed-session-123"


async def test_start_config_rejects_non_session_runtime_mode():
    native = FakeNative(runtime_mode="oneshot")
    config = {"schema_version": "fabric.agent/v1alpha1", "metadata": {"name": "demo"}}

    with pytest.raises(RuntimeError, match="requires runtime.mode=session"):
        await NativeClient(native).start_config(config)

    assert native.stopped == 0
    assert native.requests == []
