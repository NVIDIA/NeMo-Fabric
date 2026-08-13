# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Credential-free mini-SWE-agent adapter lifecycle coverage."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import requests
from nemo_fabric import EnvironmentConfig, Fabric, FabricConfig, HarnessConfig
from nemo_fabric import InstructionConfig, InstructionsConfig, MetadataConfig
from nemo_fabric import ModelConfig, RuntimeConfig

ROOT = Path(__file__).resolve().parents[2]


def mini_config(api_server: str, workspace: Path) -> FabricConfig:
    return FabricConfig(
        metadata=MetadataConfig(name="mini-swe-agent-test"),
        harness=HarnessConfig(
            adapter_id="nvidia.fabric.mini-swe-agent",
            resolution="preinstalled",
            settings={"timeout_seconds": 30},
        ),
        models={
            "default": ModelConfig(
                provider="openai",
                model="fabric-echo",
                api_key_env="FABRIC_TEST_API_KEY",
                base_url=f"{api_server}/v1",
                temperature=0.0,
            )
        },
        instructions=InstructionsConfig(
            system=InstructionConfig(content="Make the requested change.")
        ),
        runtime=RuntimeConfig(max_turns=2, timeout_seconds=60),
        environment=EnvironmentConfig(
            provider="local",
            workspace=workspace,
            env={"FABRIC_TEST_API_KEY": "test-key"},
        ),
    )


async def test_mini_swe_agent_plans_diagnoses_and_runs(
    api_server: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    response = await asyncio.to_thread(
        requests.post,
        f"{api_server}/_scenario",
        json={
            "tool_call": {
                "name": "bash",
                "arguments": {
                    "command": "printf 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\ndone\\n'"
                },
            }
        },
        timeout=5,
    )
    response.raise_for_status()
    monkeypatch.setenv("MSWEA_SILENT_STARTUP", "1")
    monkeypatch.setenv("MSWEA_GLOBAL_CONFIG_DIR", str(tmp_path / "mini-config"))
    config = mini_config(api_server, tmp_path)
    client = Fabric()

    plan = client.plan(config, base_dir=ROOT)
    report = await client.doctor(config, base_dir=ROOT)
    result = await client.run(config, base_dir=ROOT, input="Finish the task.")

    assert plan["adapter_descriptor"]["descriptor"]["adapter_id"] == (
        "nvidia.fabric.mini-swe-agent"
    )
    assert report.status == "pass", report
    assert result.status == "succeeded", result.to_mapping()
    assert result.output["output"] == "done\n"
    assert result.output["usage"]["api_calls"] == 1
