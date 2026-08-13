# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end remote MCP coverage shared by all supported adapters."""

from __future__ import annotations

import json
import os
from importlib.metadata import version as distribution_version
from pathlib import Path
import sys

from packaging.version import Version
import pytest
import requests
from _utils.utils import atof_records

from nemo_fabric import (
    EnvironmentConfig,
    Fabric,
    FabricConfig,
    HarnessConfig,
    MetadataConfig,
    ModelConfig,
    RelayAtofConfig,
    RelayAtofFileSinkConfig,
    RelayObservabilityConfig,
    RuntimeConfig,
)

def _tool_call(adapter: str) -> dict[str, object]:
    if adapter == "codex":
        return {"name": "ping", "namespace": "mcp__logging", "arguments": {}}
    if adapter == "deepagents":
        return {"name": "ping", "arguments": {}}

    tool_name = "mcp__logging__ping"
    if adapter == "hermes" and Version(distribution_version("hermes-agent")) >= Version(
        "0.20"
    ):
        return {
            "name": "tool_call",
            "arguments": {"name": tool_name, "arguments": {}},
        }
    return {"name": tool_name, "arguments": {}}

async def _test_e2e_remote_mcp(
    adapter: str,
    api_server: str,
    mcp_server: tuple[str, Path],
    tmp_path: Path,
    adapter_ids: dict[str, str],
):
    mcp_url, request_log = mcp_server

    model_config = ModelConfig(
        provider="nvidia", 
        model="fabric-test-model",
        api_key_env="FABRIC_TEST_API_KEY",
        base_url=f"{api_server}/v1",
    )

    config = FabricConfig(
        metadata=MetadataConfig(name="quickstart-agent"),
        harness=HarnessConfig(adapter_id=adapter_ids[adapter]),
        models={"default": model_config},
        runtime=RuntimeConfig(artifacts=tmp_path / "artifacts"),
        environment=EnvironmentConfig(
            workspace=tmp_path,
            artifacts=tmp_path / "artifacts",
            env={"FABRIC_TEST_API_KEY": "test"},
        ),
    )

    config.enable_relay(
        output_dir="./artifacts/relay",
        observability=RelayObservabilityConfig(
            atof=RelayAtofConfig(
                enabled=True,
                sinks=[
                    RelayAtofFileSinkConfig(
                        output_directory="./artifacts/relay",
                        filename="events.atof.jsonl",
                        mode="overwrite",
                    )
                ],
            ),
        ),
    )

    config.add_mcp_server(
        "logging",
        transport="streamable-http",
        url=mcp_url,
        authentication=None,
        custom_headers={"X-API-Key": "Bearer ${TEST_SECRET_KEY}"},
    )

    secret_key = "TEST_ABC123"
    os.environ.update({
        "ADAPTER_PYTHON": sys.executable,
        "TEST_SECRET_KEY": secret_key,
    })

    scenario_response = requests.post(
        f"{api_server}/_scenario",
        json={"tool_call": _tool_call(adapter)},
        timeout=5,
    )
    scenario_response.raise_for_status()

    result = await Fabric().run(
        config,
        base_dir=tmp_path,
        input="Call the ping MCP tool.",
    )

    assert result["status"] == "succeeded", result.to_mapping()
    tool_records = [
        record
        for record in atof_records(result["output"])
        if record.get("category") == "tool" and "ping" in record.get("name", "")
    ]
    assert tool_records, result.to_mapping()

    requests_logged = [
        json.loads(line)
        for line in request_log.read_text(encoding="utf-8").splitlines()
    ]
    assert any(
        record["path"] == "/mcp"
        and record["headers"].get("x-api-key") == f"Bearer {secret_key}"
        for record in requests_logged
    ), requests_logged


@pytest.mark.usefixtures("mock_nvidia_api_key", "nemo_relay")
async def test_e2e_remote_mcp_claude(
    api_server: str,
    mcp_server: tuple[str, Path],
    tmp_path: Path,
    adapter_ids: dict[str, str],
):
    pytest.importorskip("claude_agent_sdk")
    await _test_e2e_remote_mcp(
        "claude", api_server, mcp_server, tmp_path, adapter_ids
    )

@pytest.mark.usefixtures("mock_nvidia_api_key", "nemo_relay")
async def test_e2e_remote_mcp_codex(
    api_server: str,
    mcp_server: tuple[str, Path],
    tmp_path: Path,
    adapter_ids: dict[str, str],
):
    pytest.importorskip("openai_codex")
    await _test_e2e_remote_mcp(
        "codex", api_server, mcp_server, tmp_path, adapter_ids
    )

@pytest.mark.usefixtures("mock_nvidia_api_key", "nemo_relay")
async def test_e2e_remote_mcp_deepagents(
    api_server: str,
    mcp_server: tuple[str, Path],
    tmp_path: Path,
    adapter_ids: dict[str, str],
):
    pytest.importorskip("deepagents")
    await _test_e2e_remote_mcp(
        "deepagents", api_server, mcp_server, tmp_path, adapter_ids
    )

@pytest.mark.usefixtures("mock_nvidia_api_key", "nemo_relay", "requires_hermes_agent")
async def test_e2e_remote_mcp_hermes(
    api_server: str,
    mcp_server: tuple[str, Path],
    tmp_path: Path,
    adapter_ids: dict[str, str],
):
    await _test_e2e_remote_mcp(
        "hermes", api_server, mcp_server, tmp_path, adapter_ids
    )
