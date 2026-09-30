# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic local-provider E2E for the direct Qwen Code adapter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import requests

from nemo_fabric import DiscoveryConfig
from nemo_fabric import EnvironmentConfig
from nemo_fabric import Fabric
from nemo_fabric import FabricConfig
from nemo_fabric import HarnessConfig
from nemo_fabric import MetadataConfig
from nemo_fabric import McpConfig
from nemo_fabric import McpServerConfig
from nemo_fabric import ModelConfig
from nemo_fabric import RuntimeConfig

ROOT = Path(__file__).resolve().parents[2]
DESCRIPTOR = ROOT / "adapters/typescript/qwen/qwen.fabric-adapter.json"

pytestmark = pytest.mark.skipif(
    not (ROOT / "adapters/typescript/qwen/dist/cli.js").is_file(),
    reason="Build the Qwen TypeScript adapter before its E2E test",
)


def qwen_config(
    api_server: str, workspace: Path, mcp_server: McpServerConfig
) -> FabricConfig:
    return FabricConfig(
        metadata=MetadataConfig(name="qwen-e2e"),
        discovery=DiscoveryConfig(local_paths=[DESCRIPTOR]),
        harness=HarnessConfig(
            adapter_id="nvidia.fabric.qwen", settings={"permission_mode": "yolo"}
        ),
        models={
            "default": ModelConfig(
                provider="openai",
                model="fabric-test-model",
                api_key_env="QWEN_E2E_KEY",
                base_url=f"{api_server}/v1",
            )
        },
        mcp=McpConfig(servers={"probe": mcp_server}),
        environment=EnvironmentConfig(
            provider="local", workspace=workspace, env={"QWEN_E2E_KEY": "test-key"}
        ),
        runtime=RuntimeConfig(input_schema="text", output_schema="message"),
    )


async def test_qwen_doctor_and_run_against_local_provider(
    api_server: str, tmp_path: Path
):
    config = qwen_config(
        api_server,
        tmp_path,
        McpServerConfig(
            transport="stdio",
            url="node",
            args=[str(ROOT / "adapters/typescript/qwen/test/fixtures/mcp-server.mjs")],
            allowed_tools=["echo"],
        ),
    )
    fabric = Fabric()
    report = await fabric.doctor(config, base_dir=ROOT)
    assert report.status in {"pass", "warn"}

    scenario = requests.post(
        f"{api_server}/_scenario",
        json={
            "tool_calls": [
                {
                    "name": "tool_search",
                    "arguments": {"query": "select:mcp__probe__echo"},
                },
                {
                    "name": "tool_call",
                    "arguments": {
                        "name": "mcp__probe__echo",
                        "arguments": {"text": "hello"},
                    },
                },
            ]
        },
        timeout=5,
    )
    scenario.raise_for_status()

    single = await fabric.run(config, base_dir=ROOT, input="single")
    assert single["status"] == "succeeded", single.get("error")
    assert "single" in single["output"]["response"]

    async with await fabric.start_runtime(config, base_dir=ROOT) as runtime:
        first = await runtime.invoke(input="first")
        second = await runtime.invoke(input="second")
    assert first["status"] == second["status"] == "succeeded"
    assert "first" in first["output"]["response"]
    assert "user_count=1" in first["output"]["response"]
    assert "second" in second["output"]["response"]
    assert "user_count=2" in second["output"]["response"]
    assert second["usage"]["input_tokens"] >= 0

    captured = requests.get(f"{api_server}/_requests", timeout=5).json()
    evidence = [
        {
            "messages": payload.get("messages", [])[-4:],
            "tools": [
                tool.get("function", {}).get("name")
                for tool in payload.get("tools", [])
            ],
        }
        for payload in captured
    ]
    assert "echo:hello" in json.dumps(evidence), json.dumps(evidence, indent=2)


async def test_qwen_reports_an_unavailable_configured_mcp_server(
    api_server: str, tmp_path: Path
):
    config = qwen_config(
        api_server,
        tmp_path,
        McpServerConfig(transport="stdio", url="node", args=["-e", "process.exit(1)"]),
    )

    result = await Fabric().run(config, base_dir=ROOT, input="use MCP")

    assert result["status"] == "failed"
    assert result["error"]["code"] == "qwen_mcp_unavailable"
