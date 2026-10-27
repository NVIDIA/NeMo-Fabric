# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public SDK health for an owned Kiro-compatible ACP process fixture."""

import asyncio
import json
import os
import shutil
import sys
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
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(sys.platform == "win32", reason="Unix executable fixture")
async def test_kiro_sdk_health_during_blocked_acp_invocation(tmp_path):
    node = shutil.which("node")
    assert node is not None
    executable = tmp_path / "kiro-cli"
    fixture = ROOT / "adapters/typescript/common/test/fixtures/acp-agent.mjs"
    executable.write_text(f'#!/bin/sh\nexec "{node}" "{fixture}"\n')
    executable.chmod(0o755)
    marker = tmp_path / "entered"
    gate = tmp_path / "release"
    config = FabricConfig(
        metadata=MetadataConfig(name="kiro-health-e2e"),
        harness=HarnessConfig(adapter_id="nvidia.fabric.kiro"),
        discovery=DiscoveryConfig(
            local_paths=[ROOT / "adapters/typescript/kiro/kiro.fabric-adapter.json"]
        ),
        environment=EnvironmentConfig(
            provider="local",
            workspace=tmp_path,
            env={
                "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
                "TEST_PROMPT_STARTED": str(marker),
                "TEST_PROMPT_GATE": str(gate),
            },
        ),
        models={"default": ModelConfig(provider="kiro", model="model-a")},
    )
    fabric = Fabric()
    environment = await fabric.prepare_environment(config, base_dir=ROOT)
    runtime = await fabric.start_runtime_in(config, environment, base_dir=ROOT)
    invocation = None
    try:
        request = CheckRequest(scope="runtime", runtime_id=runtime.runtime_id)
        observation = await fabric.check(environment=environment, request=request)
        assert observation.outcome == "passed", (
            f"{observation.reason_code}: {observation.checks}"
        )
        for level in ["active", "ready"]:
            unsupported = await fabric.check(
                environment=environment,
                request=CheckRequest(
                    scope="runtime", runtime_id=runtime.runtime_id, level=level
                ),
            )
            assert unsupported.outcome == "unsupported"
        invocation = asyncio.create_task(runtime.invoke(input="PRIVATE prompt"))
        async with asyncio.timeout(5):
            while not marker.exists():
                await asyncio.sleep(0.01)
        during = await fabric.check(environment=environment, request=request)
        assert during.outcome == "passed"
        assert not invocation.done()
        assert "PRIVATE" not in json.dumps(during.to_mapping())
        timeout = await fabric.check(
            environment=environment,
            request=CheckRequest(
                scope="runtime", runtime_id=runtime.runtime_id, timeout_seconds=1e-7
            ),
        )
        assert timeout.reason_code == "probe_timeout"
        assert runtime.status is RuntimeStatus.ACTIVE
        assert not invocation.done()
        gate.write_text("release")
        assert (await asyncio.wait_for(invocation, 5)).status == "succeeded"
        observation = await fabric.check(environment=environment, request=request)
        assert observation.outcome == "passed", (
            f"{observation.reason_code}: {observation.checks}"
        )
    finally:
        gate.write_text("release")
        try:
            if invocation is not None:
                await asyncio.wait_for(invocation, 5)
        finally:
            try:
                await runtime.stop()
            finally:
                await fabric.release_environment(environment)
