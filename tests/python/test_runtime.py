# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the public Runtime lifecycle."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from nemo_fabric import (
    Fabric,
    FabricConfig,
    FabricConfigError,
    FabricNativeUnavailableError,
    FabricRuntimeError,
    FabricStateError,
    HarnessConfig,
    MetadataConfig,
    RunResult,
    RuntimeConfig,
    Runtime,
    RuntimeStatus,
)
from nemo_fabric import client as client_mod
from nemo_fabric import runtime as runtime_mod


def _plan() -> dict[str, Any]:
    config = {
        "metadata": {"name": "demo"},
        "harness": {"adapter_id": "test.fabric.shim"},
        "runtime": {},
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
            "service": False,
            "streaming": False,
            "updates": False,
            "cancellation": False,
        },
    }


def _runtime(runtime_id: str = "runtime-1") -> dict[str, Any]:
    return {
        "runtime_id": runtime_id,
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


def _config() -> FabricConfig:
    return FabricConfig(
        metadata=MetadataConfig(name="demo"),
        harness=HarnessConfig(adapter_id="test.fabric.shim"),
        runtime=RuntimeConfig(),
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
        runtime = json.loads(runtime_json)
        mock_native.requests.append(request)
        turn = len(mock_native.requests)
        return json.dumps(
            {
                "agent_name": "demo",
                "profiles": ["typed"],
                "harness": "hermes",
                "adapter_kind": "python",
                "adapter_id": "test.fabric.shim",
                "runtime_id": runtime["runtime_id"],
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
) -> Fabric:
    monkeypatch.setattr(client_mod, "_native", mock_native)
    return Fabric()


def _runtime_wrapper(
    mock_native: MagicMock,
    *,
    runtime_id: str = "runtime-1",
    overrides: dict[str, Any] | None = None,
) -> Runtime:
    client = Fabric()
    client._native_module = lambda: mock_native  # type: ignore[method-assign]
    return Runtime(
        client=client,
        plan=_plan(),
        runtime=_runtime(runtime_id),
        overrides=overrides,
    )


async def test_start_runtime_supports_path_and_typed_sources(
    native_client: Fabric,
    mock_native: MagicMock,
):
    path_runtime = await native_client.start_runtime("agent", profiles=["typed"])
    typed_runtime = await native_client.start_runtime(
        _config(),
        profiles=[],
        base_dir=".",
    )

    assert path_runtime.runtime_id == "runtime-1"
    assert typed_runtime.runtime_id == "runtime-1"
    assert mock_native.plan.call_args.args == ("agent", ["typed"])
    assert mock_native.plan_config.called


async def test_start_runtime_preserves_start_stage(
    native_client: Fabric,
    mock_native: MagicMock,
):
    mock_native.start_runtime.side_effect = RuntimeError("start failed")

    with pytest.raises(FabricRuntimeError, match="start failed") as caught:
        await native_client.start_runtime("agent")

    assert caught.value.stage == "start"


async def test_start_runtime_rejects_invalid_overrides_before_start(
    native_client: Fabric,
    mock_native: MagicMock,
):
    with pytest.raises(FabricConfigError, match="keys must be strings"):
        await native_client.start_runtime(
            "agent",
            overrides={"nested": {1: "invalid"}},  # type: ignore[dict-item]
        )

    mock_native.start_runtime.assert_not_called()


async def test_start_runtime_rejects_cyclic_overrides_before_start(
    native_client: Fabric,
    mock_native: MagicMock,
):
    overrides: dict[str, Any] = {}
    overrides["cycle"] = overrides

    with pytest.raises(FabricConfigError, match="JSON-compatible"):
        await native_client.start_runtime("agent", overrides=overrides)

    mock_native.start_runtime.assert_not_called()


async def test_runtime_reuses_runtime_and_orders_turns(mock_native: MagicMock):
    runtime = _runtime_wrapper(mock_native)

    first = await runtime.invoke(input="one")
    second = await runtime.invoke(input="two")

    assert isinstance(first, RunResult)
    assert first.runtime_id == second.runtime_id == "runtime-1"
    assert [request["input"] for request in mock_native.requests] == ["one", "two"]
    assert runtime.messages[-1]["content"] == "reply-2"
    assert len(runtime.invocations) == 2


async def test_native_invoke_failure_marks_runtime_failed(mock_native: MagicMock):
    mock_native.invoke_runtime.side_effect = RuntimeError("invoke failed")
    runtime = _runtime_wrapper(mock_native)

    with pytest.raises(FabricRuntimeError, match="invoke failed"):
        await runtime.invoke(input="hello")

    assert runtime.status is RuntimeStatus.FAILED
    with pytest.raises(FabricStateError, match="failed"):
        await runtime.invoke(input="too late")


async def test_failed_invoke_is_not_masked_by_context_cleanup(mock_native: MagicMock):
    mock_native.invoke_runtime.side_effect = RuntimeError("invoke failed")
    runtime = _runtime_wrapper(mock_native)

    with pytest.raises(FabricRuntimeError, match="invoke failed"):
        async with runtime:
            await runtime.invoke(input="hello")

    assert runtime.status is RuntimeStatus.FAILED
    mock_native.stop_runtime.assert_not_called()


async def test_runtime_preserves_non_mapping_message_values(mock_native: MagicMock):
    result = json.loads(
        mock_native.invoke_runtime.side_effect(
            "",
            json.dumps(_runtime()),
            json.dumps({"input": "hello", "request_id": "request-1"}),
        )
    )
    result["output"]["messages"] = ["notice", {"role": "assistant", "content": "ok"}, 1]
    mock_native.invoke_runtime.side_effect = None
    mock_native.invoke_runtime.return_value = json.dumps(result)
    runtime = _runtime_wrapper(mock_native)

    await runtime.invoke(input="hello")

    assert runtime.messages == ["notice", {"role": "assistant", "content": "ok"}, 1]


async def test_runtime_recursively_merges_overrides(mock_native: MagicMock):
    runtime = _runtime_wrapper(
        mock_native,
        overrides={"limits": {"turns": 2, "tokens": 10}, "phase": "runtime"},
    )

    await runtime.invoke(
        input="hello",
        overrides={"limits": {"tokens": 20}, "mode": None},
    )

    assert mock_native.requests[0]["overrides"] == {
        "limits": {"turns": 2, "tokens": 20},
        "phase": "runtime",
        "mode": None,
    }


async def test_stop_is_idempotent_and_blocks_future_invokes(mock_native: MagicMock):
    runtime = _runtime_wrapper(mock_native)

    await runtime.stop()
    await runtime.stop()

    assert runtime.status is RuntimeStatus.STOPPED
    assert mock_native.stop_runtime.call_count == 1
    with pytest.raises(FabricStateError, match="stopped"):
        await runtime.invoke(input="hello")


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

    monkeypatch.setattr(runtime_mod, "_call_blocking", blocking)
    runtime = _runtime_wrapper(mock_native)
    turn = asyncio.create_task(runtime.invoke(input="hello"))
    await started.wait()

    with pytest.raises(FabricStateError, match="in flight"):
        await runtime.stop()

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

    monkeypatch.setattr(runtime_mod, "_call_blocking", blocking)
    runtime = _runtime_wrapper(mock_native)
    first = asyncio.create_task(runtime.invoke(input="one"))
    await started.wait()

    with pytest.raises(FabricStateError, match="already running"):
        await runtime.invoke(input="two")

    release.set()
    await first


async def test_independent_runtimes_can_invoke_concurrently(
    monkeypatch: pytest.MonkeyPatch,
    mock_native: MagicMock,
):
    both_started = asyncio.Event()
    release = asyncio.Event()
    started = 0

    async def blocking(func):  # type: ignore[no-untyped-def]
        nonlocal started
        started += 1
        if started == 2:
            both_started.set()
        await release.wait()
        return func()

    monkeypatch.setattr(runtime_mod, "_call_blocking", blocking)
    first_runtime = _runtime_wrapper(mock_native)
    second_runtime = _runtime_wrapper(mock_native, runtime_id="runtime-2")

    first = asyncio.create_task(first_runtime.invoke(input="one"))
    second = asyncio.create_task(second_runtime.invoke(input="two"))
    await both_started.wait()

    assert not first.done()
    assert not second.done()
    release.set()
    first_result, second_result = await asyncio.gather(first, second)
    assert {first_result.runtime_id, second_result.runtime_id} == {
        "runtime-1",
        "runtime-2",
    }


async def test_run_stops_runtime_after_success_and_failure(
    native_client: Fabric,
    mock_native: MagicMock,
):
    result = await native_client.run("agent", input="hello")
    assert result.status == "succeeded"
    assert mock_native.stop_runtime.call_count == 1

    mock_native.invoke_runtime.side_effect = RuntimeError("invoke failed")
    with pytest.raises(FabricRuntimeError, match="invoke failed"):
        await native_client.run("agent", input="hello")
    assert mock_native.stop_runtime.call_count == 2


async def test_async_lifecycle_methods_resolve_plans(
    native_client: Fabric,
    monkeypatch: pytest.MonkeyPatch,
):
    planning_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    original_plan = native_client.plan

    def record_plan(*args: Any, **kwargs: Any):
        planning_calls.append((args, kwargs))
        return original_plan(*args, **kwargs)

    monkeypatch.setattr(native_client, "plan", record_plan)

    await native_client.run("agent", input="hello")
    runtime = await native_client.start_runtime("agent")
    await runtime.stop()

    assert len(planning_calls) == 2


async def test_run_surfaces_cleanup_failure_after_success(
    native_client: Fabric,
    mock_native: MagicMock,
):
    mock_native.stop_runtime.side_effect = RuntimeError("stop failed")

    with pytest.raises(FabricRuntimeError, match="stop failed") as caught:
        await native_client.run("agent", input="hello")

    assert caught.value.stage == "run"
    assert mock_native.stop_runtime.call_count == 1


async def test_context_manager_stops_runtime(mock_native: MagicMock):
    runtime = _runtime_wrapper(mock_native)

    async with runtime:
        await runtime.invoke(input="hello")

    assert runtime.status is RuntimeStatus.STOPPED


async def test_native_unavailable_uses_typed_error(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(client_mod, "_native", None)

    with pytest.raises(FabricNativeUnavailableError, match="native extension"):
        Fabric().plan("agent")
