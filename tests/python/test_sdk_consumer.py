# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public SDK lifecycle and descriptor validation work without a control host."""

import json
import os
import subprocess
import sys
from unittest.mock import MagicMock

import pytest
from nemo_fabric import Fabric, FabricConfigError, RecordConfig, RunPlan

from _utils.configs import hermes_shim_config


def test_public_runtime_health_api_is_removed():
    import nemo_fabric
    from nemo_fabric import Runtime, types

    with pytest.raises(AttributeError):
        Runtime.check
    with pytest.raises(AttributeError):
        nemo_fabric.RuntimeHealth
    with pytest.raises(AttributeError):
        types.RuntimeHealth


def test_native_runtime_health_api_is_removed():
    from nemo_fabric import _native

    with pytest.raises(AttributeError):
        _native.check_runtime


def test_run_plan_fingerprint_api_is_removed():
    with pytest.raises(AttributeError):
        RunPlan.descriptor_fingerprint


async def test_start_rejects_removed_fingerprint_argument(
    hermes_shim_agent_dir, monkeypatch
):
    from nemo_fabric import _native

    mock_start = MagicMock()
    monkeypatch.setattr(_native, "start_runtime", mock_start)
    with pytest.raises(TypeError):
        await Fabric().start_runtime(
            hermes_shim_config(),
            base_dir=hermes_shim_agent_dir,
            expected_descriptor_fingerprint="0" * 64,
        )
    mock_start.assert_not_called()


def test_existing_sdk_lifecycle_works_without_openshell_or_control_host(
    hermes_shim_agent_dir, tmp_path
):
    control_socket = tmp_path / "fabric.sock"
    control_socket.write_text("consumer-owned file")
    environment = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(("OPENSHELL_", "NEMOCLAW_", "FABRIC_HOST_"))
    }
    environment["FABRIC_HOST_SOCKET"] = str(control_socket)
    # A fresh consumer interpreter cannot import OpenShell or the outer host.
    # Fabric's normal adapter subprocess remains available through the SDK.
    consumer = """
import asyncio
import sys

for module in ("openshell", "nemoclaw", "nemo_fabric.runtime_server"):
    sys.modules[module] = None

from nemo_fabric import Fabric, FabricConfig, RuntimeStatus

async def main():
    client = Fabric()
    config = FabricConfig.model_validate_json(sys.stdin.read())
    directory = sys.argv[1]
    plan = client.plan(config, base_dir=directory)
    descriptor = plan.get("adapter_descriptor")
    assert descriptor["descriptor"]["adapter_id"] == config.harness.adapter_id
    assert (await client.doctor(config, base_dir=directory)).agent_name == config.metadata.name
    async with await client.start_runtime(config, base_dir=directory) as first:
        async with await client.start_runtime(config, base_dir=directory) as second:
            assert first.runtime_id != second.runtime_id
            for runtime in (first, second):
                assert runtime.invocations == []
                result = await runtime.invoke(input={"message": runtime.runtime_id})
                assert result.status == "succeeded"
                assert result.runtime_id == runtime.runtime_id
                assert result.output["received"] == {"message": runtime.runtime_id}
                assert len(runtime.invocations) == 1

            assert first.status is RuntimeStatus.ACTIVE
            assert second.status is RuntimeStatus.ACTIVE

            await first.stop()
            assert first.status is RuntimeStatus.STOPPED
            assert second.status is RuntimeStatus.ACTIVE
            result = await second.invoke(input="second runtime remains usable")
            assert result.output["received"] == "second runtime remains usable"

    assert second.status is RuntimeStatus.STOPPED
    assert sys.modules["nemo_fabric.runtime_server"] is None

asyncio.run(main())
"""
    subprocess.run(
        [
            sys.executable,
            "-c",
            consumer,
            str(hermes_shim_agent_dir),
        ],
        input=json.dumps(hermes_shim_config().to_mapping()),
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
        env=environment,
    )
    assert control_socket.read_text() == "consumer-owned file"


@pytest.mark.parametrize("invalid_config", ["missing_adapter", "unsupported_setting"])
async def test_installed_descriptors_reject_invalid_config_before_native_start(
    hermes_shim_agent_dir,
    monkeypatch,
    invalid_config,
):
    from nemo_fabric import _native

    config = hermes_shim_config()
    if invalid_config == "missing_adapter":
        config.harness.adapter_id = "test.missing"
    else:
        config.harness.settings["unsupported_setting"] = True
    mock_start = MagicMock()
    monkeypatch.setattr(_native, "start_runtime", mock_start)
    with pytest.raises(FabricConfigError):
        await Fabric().start_runtime(
            config,
            base_dir=hermes_shim_agent_dir,
        )
    mock_start.assert_not_called()


def test_record_without_an_artifact_root_fails_native_planning(hermes_shim_agent_dir):
    config = hermes_shim_config()
    config.runtime.artifacts = None
    config.environment.artifacts = None
    config.runtime.record = RecordConfig(enabled=True)

    with pytest.raises(FabricConfigError, match="runtime.record.enabled"):
        Fabric().plan(config, base_dir=hermes_shim_agent_dir)

    config.runtime.artifacts = "./artifacts"
    assert Fabric().plan(config, base_dir=hermes_shim_agent_dir).config.runtime.record == {"enabled": True}
