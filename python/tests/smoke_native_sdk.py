# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke test for the installed native Python SDK."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from shutil import copytree

import nemo_fabric._native as native
from nemo_fabric import FabricClient

ROOT = Path(__file__).resolve().parents[2]


class ModelDumpLike:
    def __init__(self, value: dict) -> None:
        self.value = value

    def model_dump(self, *, mode: str, exclude_none: bool) -> dict:
        assert mode == "json"
        assert exclude_none is True
        return self.value


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

    typed_config = ModelDumpLike(
        {
            "schema_version": "fabric.agent/v1alpha1",
            "metadata": {"name": "typed-hermes-shim-agent"},
            "harness": {
                "adapter_id": "test.fabric.hermes_shim",
                "resolution": "preinstalled",
                "settings": {"workspace": "./repos/my-service"},
            },
            "models": {
                "default": {
                    "provider": "test",
                    "model": "test-model",
                    "temperature": 0.0,
                }
            },
            "runtime": {
                "mode": "session",
                "transport": "library",
                "input_schema": "chat",
                "output_schema": "message",
                "artifacts": "./artifacts",
            },
            "environment": {
                "provider": "local",
                "workspace": "./repos/my-service",
                "artifacts": "./artifacts/local",
            },
            "mcp": {
                "servers": {
                    "github": {
                        "transport": "streamable-http",
                        "url": "${GITHUB_MCP_URL}",
                        "expose_as": "native",
                    }
                }
            },
            "telemetry": {"enabled": False},
        }
    )
    typed_profile = {
        "name": "typed_relay",
        "telemetry": {"enabled": True, "output_dir": "./artifacts/relay"},
    }
    typed_plan = client.plan_config(
        typed_config,
        profile_configs=[typed_profile],
        base_dir=fixture_agent,
    )
    assert typed_plan["agent_name"] == "typed-hermes-shim-agent"
    assert typed_plan["profile"] == "typed_relay"
    assert typed_plan["adapter_descriptor"]["source"] == "local"
    assert typed_plan["telemetry_plan"]["relay_enabled"] is True

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
