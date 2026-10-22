# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recording activation delegates finalized one-shot evidence to shared core."""

import asyncio
import json
from threading import Event
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from nemo_fabric import RecordConfig
from nemo_fabric.errors import FabricRuntimeError
from nemo_fabric.runtime import _run_native_lifecycle


@pytest.fixture(name="recorded_plan")
def recorded_plan_fixture(tmp_path):
    return {
        "base_dir": str(tmp_path),
        "config": {"runtime": {"record": {"enabled": True}, "artifacts": "runs"}},
    }


async def test_recorded_one_shot_uses_shared_core_finalization(recorded_plan):
    returned = {"request_id": "req", "status": "failed", "error": {"stage": "stop"}}
    native = MagicMock(
        spec=["run_plan", "start_runtime", "invoke_runtime", "stop_runtime"]
    )
    native.run_plan.return_value = json.dumps(returned)
    request = {"request_id": "req", "input": "custom task"}
    assert await _run_native_lifecycle(native, recorded_plan, request) == returned
    plan_json, request_json = native.run_plan.call_args.args
    assert json.loads(plan_json) == recorded_plan
    assert json.loads(request_json) == request
    native.start_runtime.assert_not_called()
    native.invoke_runtime.assert_not_called()
    native.stop_runtime.assert_not_called()


async def test_recorded_start_failure_exposes_evidence_location(
    recorded_plan, tmp_path
):
    native = MagicMock(spec=["run_plan"])
    native.run_plan.side_effect = RuntimeError("original startup error")
    with pytest.raises(FabricRuntimeError) as caught:
        await _run_native_lifecycle(native, recorded_plan, {"request_id": "req"})
    assert str(caught.value) == "original startup error"
    assert caught.value.details["record_root"] == str(tmp_path / "runs/.fabric")
    assert caught.value.__notes__ == [
        f"Run evidence location: {tmp_path / 'runs/.fabric'}"
    ]
    assert caught.value.__cause__ is native.run_plan.side_effect


async def test_recorded_failure_preserves_existing_structured_error_identity(
    recorded_plan,
):
    original = FabricRuntimeError("original", stage="start", code="host_error")
    native = MagicMock(spec=["run_plan"])
    native.run_plan.side_effect = original
    with pytest.raises(FabricRuntimeError) as caught:
        await _run_native_lifecycle(native, recorded_plan, {"request_id": "req"})
    assert caught.value is original
    assert caught.value.stage == "start"
    assert caught.value.code == "host_error"
    assert caught.value.__cause__ is None


@pytest.mark.parametrize("value", [1, "true", None])
def test_recording_enabled_requires_boolean(value):
    with pytest.raises(ValidationError):
        RecordConfig(enabled=value)


def test_recording_rejects_unrecognized_settings():
    with pytest.raises(ValidationError):
        RecordConfig(enabled=True, retries=2)


async def test_recorded_failure_preserves_native_lifecycle_stage(recorded_plan):
    original = RuntimeError("native start error")
    original.fabric_stage = "start"
    native = MagicMock(spec=["run_plan"])
    native.run_plan.side_effect = original
    with pytest.raises(FabricRuntimeError) as caught:
        await _run_native_lifecycle(native, recorded_plan, {"request_id": "req"})
    assert caught.value.stage == "start"
    assert caught.value.__cause__ is original


async def test_recorded_cancellation_keeps_cancellation_and_evidence_location(
    recorded_plan, tmp_path
):
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    release = Event()
    native = MagicMock(spec=["run_plan"])

    def run(*args):
        loop.call_soon_threadsafe(started.set)
        assert release.wait(timeout=5), "test did not release the native worker"
        return json.dumps({"status": "succeeded"})

    native.run_plan.side_effect = run
    task = asyncio.create_task(
        _run_native_lifecycle(native, recorded_plan, {"request_id": "req"})
    )
    await started.wait()
    task.cancel("caller cancellation")
    release.set()
    with pytest.raises(asyncio.CancelledError) as caught:
        await task
    assert caught.value.args == ("caller cancellation",)
    assert caught.value.__notes__ == [
        f"Run evidence location: {tmp_path / 'runs/.fabric'}"
    ]
    native.run_plan.assert_called_once()
