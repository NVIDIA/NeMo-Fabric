# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Credential-free mini-SWE-agent adapter lifecycle coverage."""

from __future__ import annotations

import asyncio
import sys
import warnings
from pathlib import Path

import pytest
import requests
from _utils.utils import atof_records
from nemo_fabric import EnvironmentConfig, Fabric, FabricConfig, HarnessConfig
from nemo_fabric import InstructionConfig, InstructionsConfig, MetadataConfig
from nemo_fabric import ModelConfig, RuntimeConfig
from nemo_fabric import RelayAtifConfig, RelayAtofConfig, RelayAtofFileSinkConfig
from nemo_fabric import RelayObservabilityConfig


def mini_config(api_server: str, workspace: Path) -> FabricConfig:
    return FabricConfig(
        metadata=MetadataConfig(name="mini-swe-agent-test"),
        harness=HarnessConfig(
            adapter_id="nvidia.fabric.mini-swe-agent",
            resolution="preinstalled",
            settings={"timeout": 30},
        ),
        models={
            "default": ModelConfig(
                provider="openai",
                model="gpt-4o-mini",
                api_key_env="FABRIC_TEST_API_KEY",
                base_url=f"{api_server}/v1",
                temperature=0.0,
            )
        },
        instructions=InstructionsConfig(
            system=InstructionConfig(content="Make the requested change.")
        ),
        runtime=RuntimeConfig(
            max_turns=2,
            timeout_seconds=60,
            artifacts=workspace / "artifacts",
        ),
        environment=EnvironmentConfig(
            provider="local",
            workspace=workspace,
            artifacts=workspace / "artifacts",
            env={"FABRIC_TEST_API_KEY": "test-key"},
        ),
    )


async def test_mini_swe_agent_plans_diagnoses_and_runs(
    api_server: str,
    repo_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
    monkeypatch.setenv("MSWEA_COST_TRACKING", "ignore_errors")
    monkeypatch.setenv("MSWEA_GLOBAL_CONFIG_DIR", str(tmp_path / "mini-config"))
    monkeypatch.setenv("ADAPTER_PYTHON", sys.executable)
    config = mini_config(api_server, tmp_path)
    client = Fabric()

    plan = client.plan(config, base_dir=repo_root)
    report = await client.doctor(config, base_dir=repo_root)
    result = await client.run(config, base_dir=repo_root, input="Finish the task.")

    assert plan["adapter_descriptor"]["descriptor"]["adapter_id"] == (
        "nvidia.fabric.mini-swe-agent"
    )
    assert report.status == "pass", report
    assert result.status == "succeeded", result.to_mapping()
    assert result.output["output"] == "done\n"
    assert result.output["usage"]["api_calls"] == 1


@pytest.mark.usefixtures("nemo_relay")
async def test_mini_swe_agent_streams_relay_telemetry(
    api_server: str,
    repo_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
    monkeypatch.setenv("MSWEA_COST_TRACKING", "ignore_errors")
    monkeypatch.setenv("MSWEA_GLOBAL_CONFIG_DIR", str(tmp_path / "mini-config"))
    monkeypatch.setenv("ADAPTER_PYTHON", sys.executable)
    config = mini_config(api_server, tmp_path)
    relay_output = tmp_path / "artifacts" / "relay"
    config.enable_relay(
        output_dir=relay_output,
        observability=RelayObservabilityConfig(
            atif=RelayAtifConfig(
                enabled=True,
                output_directory=relay_output,
                agent_name="mini-swe-agent-test",
            ),
            atof=RelayAtofConfig(
                enabled=True,
                sinks=[
                    RelayAtofFileSinkConfig(
                        output_directory=relay_output,
                        filename="events.atof.jsonl",
                        mode="overwrite",
                    )
                ],
            ),
        ),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        async with await Fabric().start_runtime(
            config,
            base_dir=repo_root,
            streaming=True,
        ) as runtime:
            stream = runtime.invoke_stream(input="Finish the task.")
            records = [record async for record in stream]
            result = await stream.result()

    assert records
    assert result.status == "succeeded", result.to_mapping()
    assert result.output["output"] == "done\n"
    assert result.telemetry[0].provider == "relay"
    assert {artifact["kind"] for artifact in result.output["relay_artifacts"]} >= {
        "atof",
        "atif",
    }
    categories = {record["category"] for record in records}
    assert {"agent", "function", "llm", "tool"} <= categories
    persisted_records = atof_records(result.output)
    llm_start = next(
        record
        for record in persisted_records
        if record["category"] == "llm" and record["scope_category"] == "start"
    )
    tool_start = next(
        record
        for record in persisted_records
        if record["category"] == "tool" and record["scope_category"] == "start"
    )
    assert llm_start["data"]["content"]["model"] == "openai/gpt-4o-mini"
    assert tool_start["name"] == "bash"
    assert tool_start["data"]["command"].startswith("printf ")
