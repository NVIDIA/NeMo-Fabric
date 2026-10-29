# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native strands health through the public SDK during a blocked model call."""

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


async def test_strands_sdk_health_during_native_invocation(blocking_model, tmp_path):
    if shutil.which("strands") is None:
        pytest.skip("Install strands to qualify native strands health")
    endpoint, entered, release, requests = blocking_model
    config = FabricConfig(
        metadata=MetadataConfig(name="strands-health"),
        discovery=DiscoveryConfig(
            local_paths=[
                ROOT / "adapters/typescript/strands/strands.fabric-adapter.json"
            ]
        ),
        harness=HarnessConfig(
            adapter_id="nvidia.fabric.strands", settings={"builtin_tools": []}
        ),
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
            scope="runtime", runtime_id=runtime.runtime_id, level="live"
        )
        report = await fabric.check(environment=environment, request=request)
        assert report.outcome == "passed", report.to_mapping()
        assert requests == []
        invocation = asyncio.create_task(runtime.invoke(input="PRIVATE prompt"))
        await asyncio.wait_for(entered.wait(), 20)
        report = await fabric.check(environment=environment, request=request)
        assert report.outcome == "passed", report.to_mapping()
        assert report.runtime["activity"] == "busy"
        assert not invocation.done()
        assert len(requests) == 1
        assert "PRIVATE" not in json.dumps(report.to_mapping())
        timeout = await fabric.check(
            environment=environment,
            request=CheckRequest(
                scope="runtime", runtime_id=runtime.runtime_id, timeout_seconds=1e-7
            ),
        )
        assert timeout.reason_code == "probe_timeout"
        assert not invocation.done()
        assert len(requests) == 1
        release.set()
        result = await asyncio.wait_for(invocation, 20)
        assert result.status == "succeeded", result.to_mapping()
        report = await fabric.check(environment=environment, request=request)
        assert report.outcome == "passed", report.to_mapping()
        assert report.runtime["activity"] == "idle"
    finally:
        release.set()
        if invocation is not None:
            await asyncio.gather(invocation, return_exceptions=True)
        await runtime.stop()
        await fabric.release_environment(environment)
