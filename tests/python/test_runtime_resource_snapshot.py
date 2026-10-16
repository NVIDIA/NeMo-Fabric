# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for runtime resource snapshots."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import nemo_fabric._native as native
import pytest
from nemo_fabric import Fabric
from nemo_fabric import FabricConfig
from nemo_fabric import FabricConfigError
from nemo_fabric import FabricRuntimeError
from nemo_fabric import ResourceObservationUnavailableError
from nemo_fabric import ResourceObservationUnsupportedError
from nemo_fabric import RunRequest
from nemo_fabric import RunResult
from nemo_fabric import Runtime
from nemo_fabric import RuntimeResourceSnapshot

_SNAPSHOT = {
    "runtime_id": "runtime-1",
    "started_at": "2026-10-15T09:00:00.000Z",
    "observed_at": "2026-10-15T09:00:05.250Z",
    "resident_memory_bytes": 52_428_800,
    "process_count": 3,
    "cpu_time_seconds": 1.25,
}
_SUPPORTED_PLATFORM = sys.platform in {"linux", "darwin"}


def _shim_config() -> FabricConfig:
    return FabricConfig.from_mapping(
        {
            "metadata": {"name": "resource-snapshot-probe"},
            "harness": {
                "adapter_id": "test.fabric.hermes_shim",
                "resolution": "preinstalled",
            },
            "discovery": {"local_paths": ["adapters"]},
        }
    )


def _runtime_with_native(native_module: Any) -> Runtime:
    client = Fabric()
    client._native_module = lambda: native_module  # type: ignore[method-assign]
    return Runtime(
        client=client,
        plan={
            "agent_name": "demo",
            "base_dir": ".",
            "config": {
                "metadata": {"name": "demo"},
                "harness": {"adapter_id": "test.fabric.shim"},
            },
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
        },
        runtime={
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
        },
    )


def test_runtime_resource_snapshot_round_trips_and_rejects_invalid_fields():
    snapshot = RuntimeResourceSnapshot.from_mapping(_SNAPSHOT)
    assert snapshot.cpu_time_seconds == 1.25
    assert snapshot.to_mapping() == _SNAPSHOT

    for field, value in (
        ("resident_memory_bytes", -1),
        ("process_count", True),
        ("cpu_time_seconds", float("nan")),
        ("observed_at", ""),
    ):
        with pytest.raises(FabricConfigError):
            RuntimeResourceSnapshot.from_mapping({**_SNAPSHOT, field: value})


def test_run_result_resource_snapshot_is_typed_or_none():
    base = {
        "agent_name": "demo",
        "harness": "hermes",
        "adapter_kind": "python",
        "runtime_id": "runtime-1",
        "invocation_id": "invocation-1",
        "request_id": "request-1",
        "status": "succeeded",
    }

    with_snapshot = RunResult.from_mapping({**base, "resource_snapshot": _SNAPSHOT})
    without_snapshot = RunResult.from_mapping(base)

    assert isinstance(with_snapshot.resource_snapshot, RuntimeResourceSnapshot)
    assert with_snapshot.resource_snapshot.process_count == 3
    assert without_snapshot.resource_snapshot is None


@pytest.mark.parametrize(
    ("native_name", "sdk_type", "code"),
    [
        (
            "ResourceObservationUnavailableError",
            ResourceObservationUnavailableError,
            "resource_observation_unavailable",
        ),
        (
            "ResourceObservationUnsupportedError",
            ResourceObservationUnsupportedError,
            "resource_observation_unsupported",
        ),
    ],
)
async def test_resource_snapshot_native_error_maps_to_typed_sdk_error(
    native_name, sdk_type, code
):
    native_error = getattr(native, native_name)
    fake_native = MagicMock()
    fake_native.ResourceObservationUnavailableError = (
        native.ResourceObservationUnavailableError
    )
    fake_native.ResourceObservationUnsupportedError = (
        native.ResourceObservationUnsupportedError
    )
    fake_native.runtime_resource_snapshot.side_effect = native_error("no reader")
    runtime = _runtime_with_native(fake_native)

    with pytest.raises(sdk_type) as caught:
        await runtime.resource_snapshot()

    assert isinstance(caught.value, FabricRuntimeError)
    assert caught.value.code == code
    assert caught.value.stage == "resource_snapshot"


@pytest.mark.skipif(not _SUPPORTED_PLATFORM, reason="needs a process reader")
async def test_resource_snapshot_reads_active_runtime_and_final_result(
    hermes_shim_agent_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("ADAPTER_PYTHON", sys.executable)
    runtime = await Fabric().start_runtime(
        _shim_config(), base_dir=hermes_shim_agent_dir
    )
    try:
        invocation = asyncio.create_task(
            runtime.invoke(
                request=RunRequest(input="slow", context={"delay_seconds": 2})
            )
        )
        await asyncio.sleep(0.3)
        live = await runtime.resource_snapshot()
        assert not invocation.done(), "snapshot waited for the invocation"
        result = await invocation
    finally:
        await runtime.stop()

    assert live.runtime_id == runtime.runtime_id
    assert live.process_count >= 1
    assert live.resident_memory_bytes > 0
    assert isinstance(result.resource_snapshot, RuntimeResourceSnapshot)
    assert result.resource_snapshot.runtime_id == runtime.runtime_id
    assert result.resource_snapshot.started_at == live.started_at
    with pytest.raises(ResourceObservationUnavailableError):
        await runtime.resource_snapshot()


@pytest.mark.skipif(_SUPPORTED_PLATFORM, reason="platform has a process reader")
async def test_resource_snapshot_on_unsupported_platform_raises_unsupported(
    hermes_shim_agent_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("ADAPTER_PYTHON", sys.executable)
    runtime = await Fabric().start_runtime(
        _shim_config(), base_dir=hermes_shim_agent_dir
    )
    try:
        with pytest.raises(ResourceObservationUnsupportedError):
            await runtime.resource_snapshot()
        result = await runtime.invoke(input="hello")
    finally:
        await runtime.stop()

    assert result.resource_snapshot is None
