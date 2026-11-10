# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qualify invocation deadlines through a live OpenShell gateway and task SDK.

Build examples/openshell_echo/runtime-image.Dockerfile with the current SDK.
Set OPENSHELL_TIMEOUT_TEST_GATEWAY and OPENSHELL_TIMEOUT_TEST_IMAGE (an immutable
image ID/digest), and NEMO_FABRIC_OPEN_SHELL_PROVIDER to the rebuilt provider.
The gateway, supervisor and sandbox must use the pinned OpenShell release.
"""

import asyncio
import os
from pathlib import Path
import time

import pytest
from nemo_fabric import (
    DiscoveryConfig,
    EnvironmentConfig,
    Fabric,
    FabricConfig,
    FabricRuntimeError,
    HarnessConfig,
    MetadataConfig,
    RunRequest,
    RuntimeConfig,
)

ROOT = Path(__file__).resolve().parents[2]
GATEWAY = os.environ.get("OPENSHELL_TIMEOUT_TEST_GATEWAY")
IMAGE = os.environ.get("OPENSHELL_TIMEOUT_TEST_IMAGE")
pytestmark = pytest.mark.skipif(
    not GATEWAY or not IMAGE, reason="live OpenShell timeout qualification is opt-in"
)


@pytest.fixture(name="timeout_config")
def timeout_config_fixture(tmp_path):
    return FabricConfig(
        metadata=MetadataConfig(name="openshell-timeout"),
        harness=HarnessConfig(
            adapter_id="nvidia.fabric.example.openshell.echo",
            resolution="image_provided",
        ),
        discovery=DiscoveryConfig(
            local_paths=[ROOT / "examples/openshell_echo/adapter"]
        ),
        runtime=RuntimeConfig(artifacts=tmp_path / "artifacts"),
        environment=EnvironmentConfig(
            provider="openshell",
            ownership="fabric_owned",
            control_location="in_env_control",
            workspace="/sandbox/workspace",
            artifacts="/sandbox/artifacts",
            connection={"gateway": GATEWAY, "workspace": "default"},
            settings={
                "image": IMAGE,
                "command": ["fabric-agent-ctl", "serve"],
                "policy_yaml": (
                    ROOT / "examples/openshell_echo/policy.yaml"
                ).read_text(),
                "task_discovery_paths": ["/opt/fabric/agents/echo"],
            },
        ),
    )


async def test_live_invocations_outlast_300_seconds_with_finite_and_no_timeout(
    timeout_config,
):
    # Separate sandboxes run concurrently to keep the qualification near 5 minutes.
    async def run(timeout):
        config = timeout_config.model_copy(deep=True)
        config.runtime.timeout_seconds = timeout
        fabric = Fabric()
        environment = await fabric.prepare_environment(config)
        try:
            async with await fabric.start_runtime_in(config, environment) as runtime:
                started = time.monotonic()
                result = await runtime.invoke(
                    request=RunRequest(
                        input="long task", context={"delay_seconds": 325}
                    )
                )
                elapsed = time.monotonic() - started
                assert elapsed >= 325
                assert result.status == "succeeded"
                assert result.output == {"first_input": "long task", "turn": 1}
                assert result.error is None
                print(
                    f"OpenShell timeout={timeout}: elapsed={elapsed:.3f}s status={result.status}"
                )
        finally:
            await fabric.release_environment(environment)

    await asyncio.wait_for(asyncio.gather(run(None), run(360)), timeout=600)


async def test_live_finite_deadline_keeps_timeout_reason(timeout_config):
    timeout_config.runtime.timeout_seconds = 0.2
    fabric = Fabric()
    environment = await fabric.prepare_environment(timeout_config)
    try:
        runtime = await fabric.start_runtime_in(timeout_config, environment)
        try:
            started = time.monotonic()
            with pytest.raises(FabricRuntimeError) as caught:
                await runtime.invoke(
                    request=RunRequest(input="deadline", context={"delay_seconds": 30})
                )
            assert time.monotonic() - started < 20
            assert caught.value.code == "timeout"
            assert caught.value.stage == "invoke"
        finally:
            await runtime.stop()
    finally:
        await fabric.release_environment(environment)
