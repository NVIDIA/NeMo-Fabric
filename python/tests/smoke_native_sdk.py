"""Smoke test for the installed native Python SDK."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from shutil import copytree

import nemo_fabric._native as native
from nemo_fabric import FabricClient

ROOT = Path(__file__).resolve().parents[2]


async def main() -> None:
    assert native.version()

    async with FabricClient() as client:
        await smoke(client)


async def smoke(client: FabricClient) -> None:
    example_agent = ROOT / "examples" / "code-review-agent"
    fixture_agent = ROOT / "tests" / "fixtures" / "hermes-shim-agent"

    assert client.validate(example_agent).startswith("validated")

    plan = client.plan(example_agent, profile="env_local")
    assert plan["agent_name"] == "code-review-agent"
    assert plan["adapter_descriptor"]["descriptor"]["adapter_id"] == "nvidia.fabric.hermes.sdk"
    assert plan["capability_plan"]["native"]["mcp_servers"]["github"]
    assert plan["capability_plan"]["native"]["skill_paths"]

    multi_plan = client.plan(fixture_agent, profile=["env_local", "mcp_github"])
    assert multi_plan["profiles"] == ["env_local", "mcp_github"]
    assert multi_plan["telemetry_plan"]["relay_enabled"] is True

    with tempfile.TemporaryDirectory(prefix="fabric-native-sdk-") as tmpdir:
        temp_agent = Path(tmpdir) / "hermes-shim-agent"
        copytree(fixture_agent, temp_agent)
        result = await client.run(temp_agent, profile="env_local", input_text="hello native")

    assert result["status"] == "succeeded"
    assert result["adapter_kind"] == "python"
    assert result["output"]["received"] == "hello native"
    assert result["output"]["native_mcp_servers"] == ["github"]


if __name__ == "__main__":
    asyncio.run(main())
