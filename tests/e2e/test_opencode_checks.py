# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native opencode health through the public SDK during a blocked model call."""

import asyncio
import json
import os
import shutil
from pathlib import Path

import pytest

from e2e.test_checks import blocking_model_fixture as blocking_model_fixture
from nemo_fabric import (
    CheckRequest,
    DiscoveryConfig,
    EnvironmentConfig,
    Fabric,
    FabricConfig,
    HarnessConfig,
    MetadataConfig,
    ModelConfig,
)

ROOT = Path(__file__).resolve().parents[2]


async def test_opencode_sdk_health_during_native_invocation(blocking_model, tmp_path):
    if shutil.which("bun") is None:
        pytest.skip("Install bun to qualify native opencode health")
    endpoint, entered, release, requests = blocking_model
    config = FabricConfig(
        metadata=MetadataConfig(name="opencode-health"),
        discovery=DiscoveryConfig(
            local_paths=[
                ROOT / "adapters/typescript/opencode/opencode.fabric-adapter.json"
            ]
        ),
        harness=HarnessConfig(adapter_id="nvidia.fabric.opencode"),
        models={
            "default": ModelConfig(
                provider="openai",
                model="gpt-4.1-mini",
                api_key_env="FABRIC_NATIVE_HEALTH_TEST_KEY",
                base_url=endpoint,
            )
        },
        environment=EnvironmentConfig(
            provider="local",
            workspace=tmp_path,
            env={
                "FABRIC_NATIVE_HEALTH_TEST_KEY": "local-test-only",
                "PATH": os.environ["PATH"],
            },
        ),
    )
    fabric = Fabric()
    environment = await fabric.prepare_environment(config, base_dir=ROOT)
    runtime = await fabric.start_runtime_in(config, environment, base_dir=ROOT)
    invocation = None
    try:
        request = CheckRequest(
            scope="runtime", runtime_id=runtime.runtime_id, level="ready"
        )
        report = await fabric.check(environment=environment, request=request)
        assert report.outcome == "passed", (report.reason_code, report.checks)
        assert requests == []
        invocation = asyncio.create_task(runtime.invoke(input="PRIVATE prompt"))
        await asyncio.wait_for(entered.wait(), 20)
        report = await fabric.check(environment=environment, request=request)
        assert report.outcome == "passed", (report.reason_code, report.checks)
        assert report.runtime["activity"] == "busy"
        assert not invocation.done()
        assert "PRIVATE" not in json.dumps(report.to_mapping())
        timeout = await fabric.check(
            environment=environment,
            request=CheckRequest(
                scope="runtime", runtime_id=runtime.runtime_id, timeout_seconds=1e-7
            ),
        )
        assert timeout.reason_code == "probe_timeout"
        assert not invocation.done()
        release.set()
        result = await asyncio.wait_for(invocation, 20)
        assert result.status == "succeeded", result.to_mapping()
        report = await fabric.check(environment=environment, request=request)
        assert report.outcome == "passed", (report.reason_code, report.checks)
        assert report.runtime["activity"] == "idle"
    finally:
        release.set()
        if invocation is not None:
            await asyncio.gather(invocation, return_exceptions=True)
        await runtime.stop()
        await fabric.release_environment(environment)
