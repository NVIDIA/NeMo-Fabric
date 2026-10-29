# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public SDK health with the pinned native kilo harness and a local model."""

import asyncio
import json
import os
from pathlib import Path

import pytest

from nemo_fabric import (
    CheckRequest,
    DiscoveryConfig,
    EnvironmentConfig,
    Fabric,
    FabricConfig,
    HarnessConfig,
    MetadataConfig,
    ModelConfig,
    RuntimeStatus,
    ToolsConfig,
)
from tests.e2e.test_checks import blocking_model_fixture

ROOT = Path(__file__).resolve().parents[2]
NATIVE = os.environ.get("FABRIC_KILO_EXECUTABLE")


@pytest.mark.skipif(
    NATIVE is None,
    reason="Set FABRIC_KILO_EXECUTABLE and install the pinned native harness",
)
async def test_kilo_sdk_health_during_native_invocation(tmp_path, blocking_model):
    endpoint, entered, release, requests = blocking_model
    environment_values = {
        "PATH": str(Path(NATIVE).parent) + os.pathsep + os.environ["PATH"],
        "TEST_NATIVE_KEY": "PRIVATE-test-key",
    }
    config = FabricConfig(
        metadata=MetadataConfig(name="kilo-health-e2e"),
        harness=HarnessConfig(adapter_id="nvidia.fabric.kilo"),
        discovery=DiscoveryConfig(
            local_paths=[ROOT / "adapters/typescript/kilo/kilo.fabric-adapter.json"]
        ),
        environment=EnvironmentConfig(
            provider="local", workspace=tmp_path, env=environment_values
        ),
        models={
            "default": ModelConfig(
                provider="native-health",
                model="test-model",
                api_key_env="TEST_NATIVE_KEY",
                base_url=endpoint,
            )
        },
        tools=ToolsConfig(enabled=[]),
    )
    fabric = Fabric()
    environment = await fabric.prepare_environment(config, base_dir=ROOT)
    runtime = await fabric.start_runtime_in(config, environment, base_dir=ROOT)
    invocation = None
    try:
        request = CheckRequest(
            scope="runtime", runtime_id=runtime.runtime_id, level="ready"
        )
        before = await fabric.check(environment=environment, request=request)
        assert before.outcome == "passed", f"{before.reason_code}: {before.checks}"
        assert [check["name"] for check in before.checks] == ["live", "active", "ready"]
        assert not requests
        invocation = asyncio.create_task(runtime.invoke(input="PRIVATE prompt"))
        await asyncio.wait_for(entered.wait(), 15)
        during = await fabric.check(environment=environment, request=request)
        assert during.outcome == "passed"
        assert not invocation.done()
        assert len(requests) == 1
        assert "PRIVATE" not in json.dumps(during.to_mapping())
        timeout = await fabric.check(
            environment=environment,
            request=CheckRequest(
                scope="runtime",
                runtime_id=runtime.runtime_id,
                timeout_seconds=1e-7,
            ),
        )
        assert timeout.reason_code == "probe_timeout"
        assert runtime.status is RuntimeStatus.ACTIVE
        assert not invocation.done()
        release.set()
        result = await asyncio.wait_for(invocation, 15)
        assert result.status == "succeeded"
        assert (
            await fabric.check(environment=environment, request=request)
        ).outcome == "passed"
        assert len(requests) == 1
    finally:
        release.set()
        if invocation is not None:
            await asyncio.wait_for(invocation, 15)
        await runtime.stop()
        await fabric.release_environment(environment)
