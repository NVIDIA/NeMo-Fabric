# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the public Runtime lifecycle."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from contextlib import AsyncExitStack
from typing import Any, Literal
from unittest.mock import AsyncMock, MagicMock

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
    RunRequest,
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
        "base_dir": ".",
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


async def _wait_for(event: threading.Event, timeout: float = 2.0) -> bool:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not event.is_set() and loop.time() < deadline:
        await asyncio.sleep(0.001)
    return event.is_set()


@pytest.fixture(name="mock_native")
def mock_native_fixture() -> MagicMock:
    mock_native = MagicMock()
    mock_native.requests = []
    mock_native.plan_config.side_effect = lambda config_json, base_dir: json.dumps(
        _plan()
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


async def test_start_runtime_supports_typed_source_and_base_dir(
    native_client: Fabric,
    mock_native: MagicMock,
):
    typed_runtime = await native_client.start_runtime(
        _config(),
        base_dir=".",
    )

    assert typed_runtime.runtime_id == "runtime-1"
    assert mock_native.plan_config.call_args.args[1] == "."


async def test_start_runtime_preserves_start_stage(
    native_client: Fabric,
    mock_native: MagicMock,
):
    mock_native.start_runtime.side_effect = RuntimeError("start failed")

    with pytest.raises(FabricRuntimeError, match="start failed") as caught:
        await native_client.start_runtime(_config())

    assert caught.value.stage == "start"


async def test_start_runtime_rejects_invalid_overrides_before_start(
    native_client: Fabric,
    mock_native: MagicMock,
):
    with pytest.raises(FabricConfigError, match="keys must be strings"):
        await native_client.start_runtime(
            _config(),
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
        await native_client.start_runtime(_config(), overrides=overrides)

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

    await runtime.stop()
    assert runtime.status is RuntimeStatus.STOPPED
    mock_native.stop_runtime.assert_called_once()


async def test_failed_invoke_is_not_masked_by_context_cleanup(mock_native: MagicMock):
    mock_native.invoke_runtime.side_effect = RuntimeError("invoke failed")
    runtime = _runtime_wrapper(mock_native)

    with pytest.raises(FabricRuntimeError, match="invoke failed"):
        async with runtime:
            await runtime.invoke(input="hello")

    assert runtime.status is RuntimeStatus.STOPPED
    mock_native.stop_runtime.assert_called_once()


async def test_failed_cleanup_does_not_mask_invoke_failure(mock_native: MagicMock):
    mock_native.invoke_runtime.side_effect = RuntimeError("invoke failed")
    mock_native.stop_runtime.side_effect = RuntimeError("stop failed")
    runtime = _runtime_wrapper(mock_native)

    with pytest.raises(FabricRuntimeError, match="invoke failed") as caught:
        async with runtime:
            await runtime.invoke(input="hello")

    assert runtime.status is RuntimeStatus.FAILED
    assert caught.value.__notes__ == ["runtime cleanup failed: stop failed"]
    mock_native.stop_runtime.assert_called_once()


async def test_stop_logs_collector_deregistration_failure(
    mock_native: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    runtime = _runtime_wrapper(mock_native)
    mock_deregister = AsyncMock(side_effect=RuntimeError("collector unavailable"))
    mock_close = AsyncMock()
    monkeypatch.setattr(runtime, "_deregister_requests", mock_deregister)
    monkeypatch.setattr(runtime, "_close_streaming_resources", mock_close)

    with caplog.at_level(logging.WARNING, logger=runtime_mod.logger.name):
        await runtime.stop()

    assert runtime.status is RuntimeStatus.STOPPED
    assert "ATOF collector deregistration failed during runtime shutdown" in caplog.text
    mock_native.stop_runtime.assert_called_once()
    mock_deregister.assert_awaited_once()
    mock_close.assert_awaited_once()


async def test_stop_preserves_native_failure_when_collector_deregistration_fails(
    mock_native: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
):
    mock_native.stop_runtime.side_effect = RuntimeError("native stop failed")
    runtime = _runtime_wrapper(mock_native)
    mock_deregister = AsyncMock(side_effect=RuntimeError("collector unavailable"))
    mock_close = AsyncMock()
    monkeypatch.setattr(runtime, "_deregister_requests", mock_deregister)
    monkeypatch.setattr(runtime, "_close_streaming_resources", mock_close)

    with pytest.raises(FabricRuntimeError, match="native stop failed") as caught:
        await runtime.stop()

    assert caught.value.__notes__ == ["runtime cleanup failed: collector unavailable"]
    mock_deregister.assert_awaited_once()
    mock_close.assert_awaited_once()


async def test_deregister_requests_attempts_all_registered_requests(
    mock_native: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
):
    runtime = _runtime_wrapper(mock_native)
    runtime._registered_requests = {"request-1", "request-2", "request-3"}
    first_error = RuntimeError("first collector failure")
    third_error = RuntimeError("third collector failure")
    mock_deregister = AsyncMock(side_effect=[first_error, None, third_error])
    monkeypatch.setattr(runtime, "_deregister_request", mock_deregister)

    with pytest.raises(ExceptionGroup) as caught:
        await runtime._deregister_requests()

    assert mock_deregister.await_count == 3
    assert {call.args[0] for call in mock_deregister.await_args_list} == {
        "request-1",
        "request-2",
        "request-3",
    }
    assert caught.value.exceptions == (first_error, third_error)


async def test_cancelled_registration_is_deregistered_during_shutdown(
    mock_native: MagicMock,
):
    registration_committed = asyncio.Event()
    wait_for_response = asyncio.Event()
    collector_registrations: set[str] = set()

    async def register(request_id: str) -> None:
        collector_registrations.add(request_id)
        registration_committed.set()
        await wait_for_response.wait()

    async def deregister(request_id: str, *, remove_queue: bool) -> None:
        assert remove_queue is True
        collector_registrations.discard(request_id)

    mock_collector = MagicMock()
    mock_collector.register = AsyncMock(side_effect=register)
    mock_collector.deregister = AsyncMock(side_effect=deregister)
    mock_collector.aclose = AsyncMock()
    runtime = _runtime_wrapper(mock_native)
    runtime._collector_client = mock_collector

    runtime._reserve_request_registration("request-1")
    registration = asyncio.create_task(runtime._register_request("request-1"))
    await registration_committed.wait()
    registration.cancel()
    with pytest.raises(asyncio.CancelledError):
        await registration

    assert runtime._registered_requests == {"request-1"}
    assert collector_registrations == {"request-1"}

    await runtime.stop()

    assert runtime._registered_requests == set()
    assert collector_registrations == set()
    mock_collector.deregister.assert_awaited_once_with(
        "request-1",
        remove_queue=True,
    )


async def test_cancelled_invocation_waits_for_registration_before_cleanup(
    mock_native: MagicMock,
):
    registration_started = asyncio.Event()
    finish_registration = asyncio.Event()
    events: list[str] = []

    async def register(request_id: str) -> None:
        events.append(f"register-start:{request_id}")
        registration_started.set()
        await finish_registration.wait()
        events.append(f"register-commit:{request_id}")

    async def deregister(request_id: str, *, remove_queue: bool) -> None:
        assert remove_queue is True
        events.append(f"deregister:{request_id}")

    mock_collector = MagicMock()
    mock_collector.register = AsyncMock(side_effect=register)
    mock_collector.deregister = AsyncMock(side_effect=deregister)
    runtime = _runtime_wrapper(mock_native)
    runtime._collector_client = mock_collector
    invocation = asyncio.create_task(
        runtime._invoke_registered_payload(
            {"input": "hello", "request_id": "request-1"},
            None,
            capture_records=False,
        )
    )
    await registration_started.wait()

    invocation.cancel()
    await asyncio.sleep(0)
    invocation.cancel()
    await asyncio.sleep(0)

    assert not invocation.done()
    mock_collector.deregister.assert_not_awaited()

    finish_registration.set()
    with pytest.raises(asyncio.CancelledError):
        await invocation

    assert events == [
        "register-start:request-1",
        "register-commit:request-1",
        "deregister:request-1",
    ]
    assert runtime._registered_requests == set()
    mock_native.invoke_runtime.assert_not_called()


@pytest.mark.parametrize(
    "phases",
    [("outcome", "finalizer"), ("finalizer", "outcome")],
)
async def test_stream_registration_retires_after_both_cleanup_phases(
    mock_native: MagicMock,
    phases: tuple[
        Literal["outcome", "finalizer"],
        Literal["outcome", "finalizer"],
    ],
):
    mock_collector = MagicMock()
    mock_collector.deregister = AsyncMock()
    runtime = _runtime_wrapper(mock_native)
    runtime._collector_client = mock_collector
    runtime._registered_requests.add("request-1")

    first, second = phases
    await runtime._finish_registered_request(
        "request-1",
        remove_queue=first == "finalizer",
        pi_boundary="preserve" if first == "finalizer" else None,
        stream_phase=first,
    )
    assert runtime._registered_requests == {"request-1"}

    await runtime._finish_registered_request(
        "request-1",
        remove_queue=second == "finalizer",
        pi_boundary="preserve" if second == "finalizer" else None,
        stream_phase=second,
    )

    assert runtime._registered_requests == set()
    assert runtime._stream_outcomes_finished == set()
    assert runtime._stream_finalizers_finished == set()
    assert mock_collector.deregister.await_count == 2


async def test_failed_stream_cleanup_still_retires_outcome_phase(
    mock_native: MagicMock,
):
    mock_collector = MagicMock()
    mock_collector.deregister = AsyncMock(
        side_effect=RuntimeError("collector unavailable")
    )
    runtime = _runtime_wrapper(mock_native)
    runtime._collector_client = mock_collector
    runtime._registered_requests.add("request-1")
    runtime._stream_finalizers_finished.add("request-1")

    with pytest.raises(RuntimeError, match="collector unavailable"):
        await runtime._finish_registered_request(
            "request-1",
            remove_queue=False,
            pi_boundary=None,
            stream_phase="outcome",
        )

    assert runtime._registered_requests == set()
    assert runtime._stream_outcomes_finished == set()
    assert runtime._stream_finalizers_finished == set()


async def test_cancelled_outcome_cleanup_finishes_before_reraising(
    mock_native: MagicMock,
):
    deregistration_started = asyncio.Event()
    finish_deregistration = asyncio.Event()

    async def deregister(request_id: str, *, remove_queue: bool) -> None:
        assert request_id == "request-1"
        assert remove_queue is False
        deregistration_started.set()
        await finish_deregistration.wait()

    mock_collector = MagicMock()
    mock_collector.deregister = AsyncMock(side_effect=deregister)
    runtime = _runtime_wrapper(mock_native)
    runtime._collector_client = mock_collector
    runtime._registered_requests.add("request-1")
    cleanup = asyncio.create_task(
        runtime._finish_registered_request(
            "request-1",
            remove_queue=False,
            pi_boundary=None,
            stream_phase="outcome",
        )
    )
    await deregistration_started.wait()

    cleanup.cancel()
    await asyncio.sleep(0)
    cleanup.cancel()
    await asyncio.sleep(0)

    assert not cleanup.done()
    finish_deregistration.set()
    with pytest.raises(asyncio.CancelledError):
        await cleanup

    assert runtime._registered_requests == {"request-1"}
    assert runtime._stream_outcomes_finished == {"request-1"}
    assert runtime._stream_finalizers_finished == set()

    mock_collector.deregister.side_effect = None
    await runtime._finish_registered_request(
        "request-1",
        remove_queue=True,
        pi_boundary="preserve",
        stream_phase="finalizer",
    )
    assert runtime._registered_requests == set()


async def test_failed_registration_remains_tracked_until_cleanup_succeeds(
    mock_native: MagicMock,
):
    registration_error = FabricRuntimeError("collector response lost")
    mock_collector = MagicMock()
    mock_collector.register = AsyncMock(side_effect=registration_error)
    mock_collector.deregister = AsyncMock(
        side_effect=RuntimeError("collector unavailable")
    )
    runtime = _runtime_wrapper(mock_native)
    runtime._collector_client = mock_collector

    runtime._reserve_request_registration("request-1")
    with pytest.raises(FabricRuntimeError) as caught:
        await runtime._register_request("request-1")

    assert caught.value is registration_error
    assert runtime._registered_requests == {"request-1"}
    with pytest.raises(FabricStateError, match="collector cleanup is still pending"):
        await runtime._invoke_registered_payload(
            {"input": "hello", "request_id": "request-1"},
            None,
            capture_records=False,
        )
    assert mock_collector.register.await_count == 1
    mock_collector.deregister.assert_not_awaited()

    with pytest.raises(ExceptionGroup, match="deregistration failed"):
        await runtime._deregister_requests()

    assert runtime._registered_requests == {"request-1"}

    mock_collector.deregister.side_effect = None
    await runtime._deregister_requests()

    assert runtime._registered_requests == set()


def test_stream_rejects_request_id_with_pending_collector_cleanup(
    mock_native: MagicMock,
):
    mock_collector = MagicMock()
    mock_collector.deregister = AsyncMock()
    runtime = _runtime_wrapper(mock_native)
    runtime._collector_client = mock_collector
    runtime._registered_requests.add("request-1")

    with pytest.raises(FabricStateError, match="collector cleanup is still pending"):
        runtime.invoke_stream(request=RunRequest(input="hello", request_id="request-1"))

    mock_collector.register.assert_not_called()
    mock_collector.deregister.assert_not_called()
    assert runtime._registered_requests == {"request-1"}


@pytest.mark.parametrize(
    ("error_code", "metadata", "expected"),
    [
        ("pi_aborted", {"adapter": {"pi_turn_started": False}}, False),
        ("pi_aborted", {"adapter": {"pi_turn_started": True}}, True),
        ("pi_extension_shutdown", {"adapter": {"pi_turn_started": False}}, False),
        ("pi_extension_shutdown", {"adapter": {"pi_turn_started": True}}, True),
        ("pi_extension_shutdown", {}, True),
        (
            "pi_extension_shutdown",
            {"adapter": {"pi_turn_started": "false"}},
            True,
        ),
    ],
)
def test_pi_boundary_uses_explicit_turn_start_signal(
    mock_native: MagicMock,
    error_code: str,
    metadata: dict[str, Any],
    expected: bool,
):
    plan = _plan()
    plan["config"]["harness"]["adapter_id"] = "nvidia.fabric.pi"
    plan["adapter_descriptor"]["descriptor"].update(
        {
            "adapter_id": "nvidia.fabric.pi",
            "harness": "pi",
            "adapter_kind": "typescript",
        }
    )
    runtime = Runtime(
        client=MagicMock(),
        plan=plan,
        runtime=_runtime(),
        collector_client=MagicMock(),
    )
    result = json.loads(
        mock_native.invoke_runtime.side_effect(
            "",
            json.dumps(_runtime()),
            json.dumps({"input": "hello", "request_id": "request-1"}),
        )
    )
    result["metadata"] = metadata
    result.update(
        {
            "status": "cancelled",
            "output": None,
            "error": {
                "code": error_code,
                "message": "Pi invocation was cancelled",
                "retryable": False,
            },
        }
    )

    assert runtime._pi_result_started_turn(RunResult.from_mapping(result)) is expected


async def test_pi_stream_reserves_one_token_for_registration_and_cleanup(
    mock_native: MagicMock,
):
    plan = _plan()
    plan["config"]["harness"]["adapter_id"] = "nvidia.fabric.pi"
    plan["adapter_descriptor"]["descriptor"].update(
        {
            "adapter_id": "nvidia.fabric.pi",
            "harness": "pi",
            "adapter_kind": "typescript",
        }
    )

    streamed_tokens: list[str | None] = []

    async def no_records(
        _: str,
        *,
        registration_token: str | None = None,
    ):
        streamed_tokens.append(registration_token)
        if False:
            yield {}

    mock_collector = MagicMock()
    mock_collector.register = AsyncMock()
    mock_collector.deregister = AsyncMock()
    mock_collector.stream = no_records
    client = Fabric()
    client._native_module = lambda: mock_native  # type: ignore[method-assign]
    runtime = Runtime(
        client=client,
        plan=plan,
        runtime=_runtime(),
        collector_client=mock_collector,
    )

    stream = runtime.invoke_stream(
        request=RunRequest(input="hello", request_id="request-1")
    )

    assert runtime._registered_requests == {"request-1"}
    token = runtime._registration_tokens["request-1"]
    assert token

    assert [record async for record in stream] == []
    assert (await stream.result()).status == "succeeded"

    mock_collector.register.assert_awaited_once_with(
        "request-1",
        correlation_mode="pi_turn_window",
        capture_records=True,
        registration_token=token,
    )
    assert mock_collector.deregister.await_count == 2
    for call in mock_collector.deregister.await_args_list:
        assert call.kwargs["registration_token"] == token
    assert streamed_tokens == [token]
    assert runtime._registered_requests == set()
    assert runtime._registration_tokens == {}


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
        request=RunRequest(
            input="hello",
            overrides={"limits": {"tokens": 20}, "mode": None},
        ),
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


async def test_stop_closes_runtime_owned_collector(mock_native: MagicMock):
    mock_collector = MagicMock(spec=AsyncExitStack)
    client = Fabric()
    client._native_module = lambda: mock_native  # type: ignore[method-assign]
    runtime = Runtime(
        client=client,
        plan=_plan(),
        runtime=_runtime(),
        collector=mock_collector,
    )

    await runtime.stop()

    mock_collector.aclose.assert_awaited_once()


async def test_stop_rejects_in_flight_turn(
    mock_native: MagicMock,
):
    started = threading.Event()
    release = threading.Event()
    invoke = mock_native.invoke_runtime.side_effect

    def blocking_invoke(*args: Any) -> str:
        started.set()
        assert release.wait(timeout=5)
        return invoke(*args)

    mock_native.invoke_runtime.side_effect = blocking_invoke
    runtime = _runtime_wrapper(mock_native)
    turn = asyncio.create_task(runtime.invoke(input="hello"))
    assert await _wait_for(started)

    with pytest.raises(FabricStateError, match="in flight"):
        await runtime.stop()

    release.set()
    await turn


async def test_concurrent_invokes_are_rejected(
    mock_native: MagicMock,
):
    started = threading.Event()
    release = threading.Event()
    invoke = mock_native.invoke_runtime.side_effect

    def blocking_invoke(*args: Any) -> str:
        started.set()
        assert release.wait(timeout=5)
        return invoke(*args)

    mock_native.invoke_runtime.side_effect = blocking_invoke
    runtime = _runtime_wrapper(mock_native)
    first = asyncio.create_task(runtime.invoke(input="one"))
    assert await _wait_for(started)

    with pytest.raises(FabricStateError, match="already running"):
        await runtime.invoke(input="two")

    release.set()
    await first


async def test_independent_runtimes_can_invoke_concurrently(
    mock_native: MagicMock,
):
    both_started = threading.Event()
    release = threading.Event()
    invoke = mock_native.invoke_runtime.side_effect
    lock = threading.Lock()
    started = 0

    def blocking_invoke(*args: Any) -> str:
        nonlocal started
        with lock:
            started += 1
            if started == 2:
                both_started.set()
        assert release.wait(timeout=5)
        return invoke(*args)

    mock_native.invoke_runtime.side_effect = blocking_invoke
    first_runtime = _runtime_wrapper(mock_native)
    second_runtime = _runtime_wrapper(mock_native, runtime_id="runtime-2")

    first = asyncio.create_task(first_runtime.invoke(input="one"))
    second = asyncio.create_task(second_runtime.invoke(input="two"))
    assert await _wait_for(both_started)

    assert not first.done()
    assert not second.done()
    release.set()
    first_result, second_result = await asyncio.gather(first, second)
    assert {first_result.runtime_id, second_result.runtime_id} == {
        "runtime-1",
        "runtime-2",
    }


async def test_blocking_native_calls_run_off_the_event_loop():
    event_loop_thread = threading.get_ident()

    worker_thread = await runtime_mod._call_blocking(threading.get_ident)

    assert worker_thread != event_loop_thread


async def test_cancelling_invoke_waits_for_native_work_and_stops_runtime(
    mock_native: MagicMock,
):
    started = threading.Event()
    release = threading.Event()
    invoke = mock_native.invoke_runtime.side_effect

    def blocking_invoke(*args: Any) -> str:
        started.set()
        assert release.wait(timeout=5)
        return invoke(*args)

    mock_native.invoke_runtime.side_effect = blocking_invoke
    runtime = _runtime_wrapper(mock_native)
    turn = asyncio.create_task(runtime.invoke(input="hello"))
    assert await _wait_for(started)

    turn.cancel()
    await asyncio.sleep(0)
    assert not turn.done()
    turn.cancel()
    await asyncio.sleep(0)
    assert not turn.done()

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await turn

    assert runtime.status is RuntimeStatus.STOPPED
    mock_native.stop_runtime.assert_called_once()


async def test_cancelling_start_stops_the_completed_native_runtime(
    native_client: Fabric,
    mock_native: MagicMock,
):
    started = threading.Event()
    release = threading.Event()

    def blocking_start(*args: Any) -> str:
        started.set()
        assert release.wait(timeout=5)
        return json.dumps(_runtime())

    mock_native.start_runtime.side_effect = blocking_start
    start = asyncio.create_task(native_client.start_runtime(_config()))
    assert await _wait_for(started)

    start.cancel()
    await asyncio.sleep(0)
    assert not start.done()

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await start

    mock_native.stop_runtime.assert_called_once()


async def test_cancelling_one_shot_run_waits_for_stop(
    native_client: Fabric,
    mock_native: MagicMock,
):
    started = threading.Event()
    release = threading.Event()
    invoke = mock_native.invoke_runtime.side_effect

    def blocking_invoke(*args: Any) -> str:
        started.set()
        assert release.wait(timeout=5)
        return invoke(*args)

    mock_native.invoke_runtime.side_effect = blocking_invoke
    run = asyncio.create_task(native_client.run(_config(), input="hello"))
    assert await _wait_for(started)

    run.cancel()
    await asyncio.sleep(0)
    assert not run.done()

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await run

    mock_native.stop_runtime.assert_called_once()


async def test_run_stops_runtime_after_success_and_failure(
    native_client: Fabric,
    mock_native: MagicMock,
):
    result = await native_client.run(_config(), input="hello")
    assert result.status == "succeeded"
    assert mock_native.stop_runtime.call_count == 1

    mock_native.invoke_runtime.side_effect = RuntimeError("invoke failed")
    with pytest.raises(FabricRuntimeError, match="invoke failed"):
        await native_client.run(_config(), input="hello")
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

    await native_client.run(_config(), input="hello")
    runtime = await native_client.start_runtime(_config())
    await runtime.stop()

    assert len(planning_calls) == 2


@pytest.mark.parametrize(
    ("stop_error", "expected_message"),
    [
        (RuntimeError("stop failed"), "stop failed"),
        (RuntimeError(), "runtime shutdown failed"),
    ],
)
async def test_run_surfaces_cleanup_failure_after_success(
    native_client: Fabric,
    mock_native: MagicMock,
    stop_error: RuntimeError,
    expected_message: str,
):
    mock_native.stop_runtime.side_effect = stop_error

    result = await native_client.run(_config(), input="hello")

    assert result.status == "failed"
    assert result.output["messages"][-1]["content"] == "reply-1"
    assert result.error is not None
    assert result.error.stage == "stop"
    assert result.error.code == "runtime_stop_failed"
    assert result.error.message == expected_message
    assert mock_native.stop_runtime.call_count == 1


async def test_context_manager_stops_runtime(mock_native: MagicMock):
    runtime = _runtime_wrapper(mock_native)

    async with runtime:
        await runtime.invoke(input="hello")

    assert runtime.status is RuntimeStatus.STOPPED


async def test_native_unavailable_uses_typed_error(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(client_mod, "_native", None)

    with pytest.raises(FabricNativeUnavailableError, match="native extension"):
        Fabric().plan(_config())
