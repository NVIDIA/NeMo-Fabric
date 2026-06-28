# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke test for the installed native Python SDK."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from shutil import copytree

import nemo_fabric._native as native
from nemo_fabric import FabricClient, FabricConfig, FabricProfileConfig

ROOT = Path(__file__).resolve().parents[2]


async def main() -> None:
    assert native.version()

    async with FabricClient() as client:
        await smoke(client)


async def smoke(client: FabricClient) -> None:
    example_agent = ROOT / "examples" / "code-review-agent"
    fixture_agent = ROOT / "tests" / "fixtures" / "hermes-shim-agent"

    inspected = client.resolve(example_agent, profiles=["env_local"])
    assert inspected["agent_name"] == "code-review-agent"
    assert inspected.profiles == ("env_local",)
    assert inspected["config"]["metadata"]["name"] == "code-review-agent"

    plan = client.plan(example_agent, profiles=["env_local"])
    assert plan["agent_name"] == "code-review-agent"
    assert plan["adapter_descriptor"]["descriptor"]["adapter_id"] == "nvidia.fabric.hermes.sdk"
    assert plan["capability_plan"]["native"]["mcp_servers"]["github"]
    assert plan["capability_plan"]["native"]["skill_paths"]

    multi_plan = client.plan(fixture_agent, profiles=["env_local", "mcp_github"])
    assert multi_plan.profiles == ("env_local", "mcp_github")
    assert multi_plan["telemetry_plan"]["relay_enabled"] is True

    minimal = FabricConfig.from_mapping(
        {
            "metadata": {"name": "minimal-typed-agent"},
            "harness": {"adapter_id": "nvidia.fabric.hermes.sdk"},
        }
    )
    minimal_resolved = client.resolve(minimal)
    assert minimal_resolved.config.runtime.mode == "oneshot"
    assert minimal_resolved.config.runtime.transport == "library"
    assert minimal_resolved.config.runtime.input_schema == "text"
    assert minimal_resolved.config.runtime.output_schema == "text"

    typed_config = FabricConfig.from_mapping(
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
                        "exposure": "harness_native",
                    }
                }
            },
            "telemetry": {"enabled": False},
            "consumer_extension": {
                "base": True,
                "nested": {"first": 1},
            },
        }
    )
    typed_profile = FabricProfileConfig.from_mapping(
        {
            "name": "typed_relay",
            "harness": {"settings": {"timeout_seconds": 30}},
            "telemetry": {"enabled": True, "output_dir": "./artifacts/relay"},
            "consumer_extension": {
                "profile": True,
                "nested": {"second": 2},
            },
        }
    )
    typed_config_resolved = client.resolve(
        typed_config,
        profiles=[typed_profile],
        base_dir=fixture_agent,
    )
    typed_plan = client.plan(
        typed_config,
        profiles=[typed_profile],
        base_dir=fixture_agent,
    )
    assert typed_config_resolved.agent_name == "typed-hermes-shim-agent"
    assert typed_plan["agent_name"] == "typed-hermes-shim-agent"
    assert typed_plan.profiles == ("typed_relay",)
    assert typed_plan["adapter_descriptor"]["source"] == "local"
    assert typed_plan["telemetry_plan"]["relay_enabled"] is True
    resolved_config = typed_config_resolved.config.to_mapping()
    assert resolved_config["harness"]["adapter_id"] == "test.fabric.hermes_shim"
    assert resolved_config["harness"]["settings"]["workspace"] == "./repos/my-service"
    assert resolved_config["harness"]["settings"]["timeout_seconds"] == 30
    assert resolved_config["consumer_extension"] == {
        "base": True,
        "profile": True,
        "nested": {"first": 1, "second": 2},
    }

    with tempfile.TemporaryDirectory(prefix="fabric-native-sdk-") as tmpdir:
        temp_agent = Path(tmpdir) / "hermes-shim-agent"
        copytree(fixture_agent, temp_agent)
        result = await client.run(
            temp_agent,
            profiles=["env_local"],
            input="hello native",
        )
        async with await client.start_session(
            temp_agent,
            profiles=["env_local"],
        ) as session:
            first = await session.invoke(input="hello session one")
            second = await session.invoke(input="hello session two")

    assert result["status"] == "succeeded"
    assert result["adapter_kind"] == "python"
    assert result["metadata"]["adapter_runner"] == "python"
    assert result["output"]["received"] == "hello native"
    assert result.output["native_mcp_servers"] == ("github",)
    assert any(artifact.name == "stdout" for artifact in result.artifacts.artifacts)
    assert first["status"] == "succeeded"
    assert second["status"] == "succeeded"
    assert first["runtime_id"] == second["runtime_id"]
    assert session.runtime["runtime_id"] == first["runtime_id"]


if __name__ == "__main__":
    asyncio.run(main())
