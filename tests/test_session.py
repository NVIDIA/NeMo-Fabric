# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the public Session lifecycle."""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Any
from unittest.mock import MagicMock

import pytest

from nemo_fabric import (
    FabricCapabilityError,
    FabricClient,
    FabricConfig,
    FabricConfigError,
    FabricNativeUnavailableError,
    FabricRuntimeError,
    FabricStateError,
    HarnessConfig,
    MetadataConfig,
    RunResult,
    RuntimeConfig,
    Session,
    SessionStatus,
)
from nemo_fabric import client as client_mod
from nemo_fabric import session as session_mod


def _plan(runtime_mode: str = "session") -> dict[str, Any]:
    config = {
        "metadata": {"name": "demo"},
        "harness": {"adapter_id": "test.fabric.shim"},
        "runtime": {"mode": runtime_mode, "transport": "library"},
    }
    return {
        "agent_name": "demo",
        "profiles": ["typed"],
        "effective_config": {
            "agent_name": "demo",
            "profiles": ["typed"],
            "agent_root": ".",
            "config_path": "agent.yaml",
            "config_root": ".",
            "config": config,
        },
        "config": config,
        "adapter_descriptor": {
            "descriptor": {
                "adapter_id": "test.fabric.shim",
                "harness": "hermes",
                "adapter_kind": "python",
            }
        },
        "capabilities": {
            "session": runtime_mode == "session",
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


def _config(mode: str = "session") -> FabricConfig:
    return FabricConfig(
        metadata=MetadataConfig(name="demo"),
        harness=HarnessConfig(adapter_id="test.fabric.shim"),
        runtime=RuntimeConfig(mode=mode),
    )


@pytest.fixture(name="mock_native")
def mock_native_fixture() -> MagicMock:
    mock_native = MagicMock()
    mock_native.requests = []
    mock_native.plan.side_effect = lambda path, profiles: json.dumps(_plan())
    mock_native.plan_config.side_effect = (
        lambda config_json, profiles_json, base_dir: json.dumps(_plan())
    )
    mock_native.start_runtime.return_value = json.dumps(_runtime())

    def invoke(plan_json: str, runtime_json: str, request_json: str) -> str:
        request = json.loads(request_json)
        mock_native.requests.append(request)
        turn = len(mock_native.requests)
        return json.dumps(
            {
                "agent_name": "demo",
                "profiles": ["typed"],
                "harness": "hermes",
                "adapter_kind": "python",
                "adapter_id": "test.fabric.shim",
                "runtime_id": "runtime-1",
                "invocation_id": f"invocation-{turn}",
                "request_id": request["request_id"],
                "status": "succeeded",
                "output": {
                    "messages": [
                        {"role": "user", "content": request["input"]},
                        {"role": "assistant", "content": f"reply-{turn}"},
                    ]
                },
                "artifacts": {"artifacts": []},
                "events": [],
            }
        )

    mock_native.invoke_runtime.side_effect = invoke
    mock_native.stop_runtime.return_value = json.dumps([])
    return mock_native


@pytest.fixture(name="native_client")
def native_client_fixture(
    monkeypatch: pytest.MonkeyPatch,
    mock_native: MagicMock,
) -> FabricClient:
    monkeypatch.setattr(client_mod, "_native", mock_native)
    return FabricClient()


def _session(mock_native: MagicMock, *, overrides: dict[str, Any] | None = None) -> Session:
    client = FabricClient()
    client._native_module = lambda: mock_native  # type: ignore[method-assign]
    return Session(
        client=client,
        plan=_plan(),
        runtime=_runtime(),
        overrides=overrides,
    )


async def test_start_session_supports_path_and_typed_sources(
    native_client: FabricClient,
    mock_native: MagicMock,
):
    path_session = await native_client.start_session("agent", profiles=["typed"])
    typed_session = await native_client.start_session(
        _config(),
        profiles=[],
        base_dir=".",
        session_id="caller-session",
    )

    assert path_session.runtime_id == "runtime-1"
    assert typed_session.session_id == "caller-session"
    assert mock_native.plan.call_args.args == ("agent", ["typed"])
    assert mock_native.plan_config.called


async def test_start_session_rejects_non_session_capability(
    native_client: FabricClient,
    mock_native: MagicMock,
):
    mock_native.plan.side_effect = lambda path, profiles: json.dumps(_plan("oneshot"))

    with pytest.raises(FabricCapabilityError, match="session capability"):
        await native_client.start_session("agent")


async def test_start_session_preserves_start_stage(
    native_client: FabricClient,
    mock_native: MagicMock,
):
    mock_native.start_runtime.side_effect = RuntimeError("start failed")

    with pytest.raises(FabricRuntimeError, match="start failed") as caught:
        await native_client.start_session("agent")

    assert caught.value.stage == "start"


async def test_start_session_rejects_invalid_overrides_before_start(
    native_client: FabricClient,
    mock_native: MagicMock,
):
    with pytest.raises(FabricConfigError, match="keys must be strings"):
        await native_client.start_session(
            "agent",
            overrides={"nested": {1: "invalid"}},  # type: ignore[dict-item]
        )

    mock_native.start_runtime.assert_not_called()


async def test_start_session_rejects_cyclic_overrides_before_start(
    native_client: FabricClient,
    mock_native: MagicMock,
):
    overrides: dict[str, Any] = {}
    overrides["cycle"] = overrides

    with pytest.raises(FabricConfigError, match="JSON-compatible"):
        await native_client.start_session("agent", overrides=overrides)

    mock_native.start_runtime.assert_not_called()


async def test_session_reuses_runtime_and_orders_turns(mock_native: MagicMock):
    session = _session(mock_native)

    first = await session.invoke(input="one")
    second = await session.invoke(input="two")

    assert isinstance(first, RunResult)
    assert first.runtime_id == second.runtime_id == "runtime-1"
    assert [request["input"] for request in mock_native.requests] == ["one", "two"]
    assert session.messages[-1]["content"] == "reply-2"
    assert len(session.invocations) == 2


async def test_session_recursively_merges_overrides(mock_native: MagicMock):
    session = _session(
        mock_native,
        overrides={"limits": {"turns": 2, "tokens": 10}, "mode": "session"},
    )

    await session.invoke(
        input="hello",
        overrides={"limits": {"tokens": 20}, "mode": None},
    )

    assert mock_native.requests[0]["overrides"] == {
        "limits": {"turns": 2, "tokens": 20},
        "mode": None,
    }


async def test_stream_yields_terminal_result(mock_native: MagicMock):
    items = [item async for item in _session(mock_native).stream(input="hello")]

    assert len(items) == 1
    assert isinstance(items[0], RunResult)


async def test_stop_is_idempotent_and_blocks_future_invokes(mock_native: MagicMock):
    session = _session(mock_native)

    await session.stop()
    await session.stop()

    assert session.status is SessionStatus.STOPPED
    assert mock_native.stop_runtime.call_count == 1
    with pytest.raises(FabricStateError, match="stopped"):
        await session.invoke(input="hello")


async def test_stop_rejects_in_flight_turn(
    monkeypatch: pytest.MonkeyPatch,
    mock_native: MagicMock,
):
    started = asyncio.Event()
    release = asyncio.Event()

    async def blocking(func):  # type: ignore[no-untyped-def]
        started.set()
        await release.wait()
        return func()

    monkeypatch.setattr(session_mod, "_call_blocking", blocking)
    session = _session(mock_native)
    turn = asyncio.create_task(session.invoke(input="hello"))
    await started.wait()

    with pytest.raises(FabricStateError, match="in flight"):
        await session.stop()

    release.set()
    await turn


async def test_concurrent_invokes_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
    mock_native: MagicMock,
):
    started = asyncio.Event()
    release = asyncio.Event()

    async def blocking(func):  # type: ignore[no-untyped-def]
        started.set()
        await release.wait()
        return func()

    monkeypatch.setattr(session_mod, "_call_blocking", blocking)
    session = _session(mock_native)
    first = asyncio.create_task(session.invoke(input="one"))
    await started.wait()

    with pytest.raises(FabricStateError, match="already running"):
        await session.invoke(input="two")

    release.set()
    await first


async def test_run_stops_runtime_after_success_and_failure(
    native_client: FabricClient,
    mock_native: MagicMock,
):
    result = await native_client.run("agent", input="hello")
    assert result.status == "succeeded"
    assert mock_native.stop_runtime.call_count == 1

    mock_native.invoke_runtime.side_effect = RuntimeError("invoke failed")
    with pytest.raises(FabricRuntimeError, match="invoke failed"):
        await native_client.run("agent", input="hello")
    assert mock_native.stop_runtime.call_count == 2


async def test_run_surfaces_cleanup_failure_after_success(
    native_client: FabricClient,
    mock_native: MagicMock,
):
    mock_native.stop_runtime.side_effect = RuntimeError("stop failed")

    with pytest.raises(FabricRuntimeError, match="stop failed") as caught:
        await native_client.run("agent", input="hello")

    assert caught.value.stage == "run"
    assert mock_native.stop_runtime.call_count == 1


async def test_run_cancellation_keeps_event_loop_responsive_until_cleanup(
    native_client: FabricClient,
    mock_native: MagicMock,
):
    started = threading.Event()
    release = threading.Event()
    invoke = mock_native.invoke_runtime.side_effect

    def blocking_invoke(*args: str) -> str:
        started.set()
        release.wait(timeout=1)
        return invoke(*args)

    mock_native.invoke_runtime.side_effect = blocking_invoke
    run = asyncio.create_task(native_client.run("agent", input="hello"))
    await asyncio.to_thread(started.wait, 1)
    fallback_release = threading.Timer(1, release.set)
    fallback_release.start()

    run.cancel()
    await asyncio.sleep(0.01)
    assert not run.done()

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await run
    fallback_release.cancel()
    assert mock_native.stop_runtime.call_count == 1


async def test_context_manager_stops_runtime(mock_native: MagicMock):
    session = _session(mock_native)

    async with session:
        await session.invoke(input="hello")

    assert session.status is SessionStatus.STOPPED


async def test_native_unavailable_uses_typed_error(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(client_mod, "_native", None)

    with pytest.raises(FabricNativeUnavailableError, match="native extension"):
        FabricClient().plan("agent")
