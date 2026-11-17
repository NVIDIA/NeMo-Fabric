# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise configured budgets through real host and OpenShell SDK processes.

Run through response-limits-workload.sh with examples/openshell/run-demo.sh.
No inference is needed for byte-exact artifact production. Live inference uses
sibling test_code_review_response_limits.py with two native harnesses.
"""

import asyncio
import hashlib
import json
import os
from pathlib import Path
import sys

import pytest
from nemo_fabric import (
    CheckRequest,
    DiscoveryConfig,
    EnvironmentConfig,
    Fabric,
    FabricConfig,
    HarnessConfig,
    MetadataConfig,
    ResponseLimits,
    RunRequest,
    RuntimeConfig,
)

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(
    os.environ.get("RESPONSE_LIMIT_TEST_LIVE") != "1",
    reason="live response-limit qualification is opt-in",
)


@pytest.mark.parametrize("location", ["host", "openshell"])
async def test_response_limit_matrix(tmp_path, location):
    descriptor_path = ROOT / "tests/e2e/response-limits.fabric-adapter.json"
    descriptor = json.loads(descriptor_path.read_text())
    if location == "host":
        descriptor["runner"] = {
            "module": "response_limits_adapter",
            "python": sys.executable,
            "cwd": str(ROOT / "tests/e2e"),
        }
        descriptor_path = tmp_path / "limits.fabric-adapter.json"
        descriptor_path.write_text(json.dumps(descriptor))
    config = FabricConfig(
        metadata=MetadataConfig(name="response-limits"),
        harness=HarnessConfig(
            adapter_id=descriptor["adapter_id"],
            resolution="image_provided" if location == "openshell" else "preinstalled",
        ),
        discovery=DiscoveryConfig(local_paths=[descriptor_path]),
        runtime=RuntimeConfig(artifacts=tmp_path / "artifacts", timeout_seconds=60),
        environment=EnvironmentConfig(
            provider="local", workspace=tmp_path, artifacts=tmp_path / "artifacts"
        ),
    )
    fabric = Fabric()
    environment = None
    cases = []
    for field, boundary, sizes in [
        ("max_artifact_file_bytes", 64, [64]),
        ("max_artifact_bytes", 64, [32, 32]),
        ("max_artifact_files", 3, [1, 1, 1]),
    ]:
        for delta in (-1, 0, 1):
            cases.append(
                (field, {field: boundary + delta}, {"artifact_sizes": sizes}, delta < 0)
            )
    cases.extend(
        [
            ("max_response_bytes", {"max_response_bytes": 1}, {}, True),
            (
                "max_response_bytes",
                {"max_response_bytes": 8 * 1024 * 1024},
                {"response_bytes": 5 * 1024 * 1024},
                False,
            ),
            (
                "max_artifact_file_bytes",
                {},
                {"artifact_sizes": [1024 * 1024 + 1]},
                True,
            ),
            (
                "max_artifact_file_bytes",
                {"max_artifact_file_bytes": 2 * 1024 * 1024},
                {"artifact_sizes": [1024 * 1024 + 1]},
                False,
            ),
            (
                "max_artifact_bytes",
                {
                    "max_artifact_bytes": 4 * 1024 * 1024,
                    "max_artifact_file_bytes": 2 * 1024 * 1024,
                },
                {"artifact_sizes": [1024 * 1024 + 1] * 3},
                False,
            ),
            (
                "max_artifact_files",
                {"max_artifact_files": 20},
                {"artifact_sizes": [1] * 17},
                False,
            ),
        ]
    )
    if location == "openshell":
        config.environment = EnvironmentConfig(
            provider="openshell",
            ownership="fabric_owned",
            control_location="in_env_control",
            workspace="/sandbox/workspace",
            artifacts="/sandbox/artifacts",
            connection={
                "gateway": os.environ["RESPONSE_LIMIT_TEST_GATEWAY"],
                "workspace": "default",
            },
            settings={
                "image": os.environ["RESPONSE_LIMIT_TEST_IMAGE"],
                "command": ["fabric-agent-ctl", "serve"],
                "policy_yaml": (
                    ROOT / "examples/openshell_echo/policy.yaml"
                ).read_text(),
                "task_discovery_paths": ["/opt/fabric/agents/limits"],
            },
        )
        environment = await fabric.prepare_environment(config)
    try:
        if environment is not None:
            async with asyncio.timeout(60):
                while (
                    await fabric.check(
                        environment=environment, request=CheckRequest(scope="server")
                    )
                ).outcome != "passed":
                    await asyncio.sleep(0.2)
        for field, limits, context, rejected in cases:
            config.runtime.response_limits = ResponseLimits(**limits)
            runtime = (
                await fabric.start_runtime_in(config, environment)
                if environment is not None
                else await fabric.start_runtime(config)
            )
            try:
                result = await runtime.invoke(
                    request=RunRequest(input="limits", context=context)
                )
                if rejected:
                    assert result.status == "failed", (
                        field,
                        limits,
                        result.to_mapping(),
                    )
                    assert result.error.code == "response_limit_exceeded"
                    assert result.error.metadata["limit"] == field
                    assert result.error.metadata["asset"] in result.error.message
                    assert result.output is None
                    assert not result.artifacts.artifacts
                else:
                    assert result.status == "succeeded", result.to_mapping()
                    assert result.output["response"] == "x" * context.get(
                        "response_bytes", 16
                    )
                    paths = [
                        Path(artifact.path)
                        for artifact in result.artifacts.artifacts
                        if artifact.name.startswith("screenshot-")
                    ]
                    assert [path.stat().st_size for path in paths] == context.get(
                        "artifact_sizes", []
                    )
                    assert [
                        hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
                    ] == result.output["sha256"]
                print(
                    f"{location}/{field}/{limits}: {'rejected' if rejected else 'complete'}",
                    flush=True,
                )
            finally:
                await runtime.stop()
    finally:
        if environment is not None:
            await fabric.release_environment(environment)
