# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke test for the POC Python SDK."""

from __future__ import annotations

import asyncio
import sys
import tempfile
from shutil import copytree
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "src"))

from nemo_fabric import FabricClient


async def main() -> None:
    async with FabricClient(
        command=("cargo", "run", "-q", "-p", "fabric-cli", "--"),
        cwd=ROOT,
    ) as client:
        await smoke(client)


async def smoke(client: FabricClient) -> None:
    example_agent = ROOT / "examples" / "code-review-agent"
    fixture_agent = ROOT / "tests" / "fixtures" / "hermes-shim-agent"

    assert client.validate(example_agent).startswith("validated")

    plan = client.plan(example_agent, profile="env_local")
    assert plan["agent_name"] == "code-review-agent"
    assert plan["adapter_descriptor"]["source"] == "repository"
    assert plan["adapter_descriptor"]["descriptor"]["adapter_id"] == "nvidia.fabric.hermes.sdk"
    assert plan["environment_plan"]["provider"] == "local"

    report = await client.doctor(fixture_agent, profile="env_local")
    assert report["agent_name"] == "hermes-shim-agent"
    assert report["checks"]

    multi_plan = client.plan(fixture_agent, profile=("env_local", "mcp_github"))
    assert multi_plan["profiles"] == ["env_local", "mcp_github"]
    assert "profile" not in multi_plan
    assert multi_plan["telemetry_plan"]["relay_enabled"] is True

    with tempfile.TemporaryDirectory(prefix="fabric-python-sdk-") as tmpdir:
        temp_agent = Path(tmpdir) / "hermes-shim-agent"
        copytree(fixture_agent, temp_agent)
        hermes_result = await client.run(
            temp_agent,
            profile="env_local",
            input_text="hello hermes",
        )
        structured = await client.run(
            temp_agent,
            profile="env_local",
            request={
                "request_id": "sdk-structured-request",
                "input": "hello structured sdk",
                "context": {"task": {"source": "sdk-smoke"}},
            },
        )

    assert hermes_result["status"] == "succeeded"
    assert hermes_result["adapter_kind"] == "python"
    assert hermes_result["output"]["harness"] == "hermes"
    assert hermes_result["output"]["received"] == "hello hermes"
    assert hermes_result["output"]["native_skill_paths"]
    assert hermes_result["output"]["native_mcp_servers"] == ["github"]
    assert hermes_result["output"]["managed_skill_paths"] == []
    assert hermes_result["output"]["managed_mcp_servers"] == []

    assert structured["request_id"] == "sdk-structured-request"
    assert structured["output"]["received"] == "hello structured sdk"


if __name__ == "__main__":
    asyncio.run(main())
