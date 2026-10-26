# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in tests against an installed Strands CLI and a real model.

RUN_FABRIC_STRANDS_INTEGRATION=1 STRANDS_TEST_MODEL=<model> uv run pytest tests/e2e/test_strands.py
Set STRANDS_TEST_BASE_URL for Chat Completions; otherwise the model uses Bedrock.
Set STRANDS_TEST_API_KEY_ENV to name an existing credential, or omit it for a keyless endpoint.
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path
from urllib.parse import urlsplit

import pytest


@pytest.fixture(name="strands_config")
def strands_config_fixture(tmp_path):
    if os.environ.get("RUN_FABRIC_STRANDS_INTEGRATION") != "1":
        pytest.skip("set RUN_FABRIC_STRANDS_INTEGRATION=1 to run")
    from nemo_fabric import (
        DiscoveryConfig,
        EnvironmentConfig,
        FabricConfig,
        HarnessConfig,
        MetadataConfig,
        ModelConfig,
        RuntimeConfig,
    )

    root = Path(__file__).resolve().parents[2]
    endpoint = os.environ.get("STRANDS_TEST_BASE_URL")
    model = os.environ["STRANDS_TEST_MODEL"]
    return FabricConfig(
        metadata=MetadataConfig(name="strands-e2e"),
        discovery=DiscoveryConfig(
            local_paths=[
                root / "adapters/typescript/strands/strands.fabric-adapter.json"
            ]
        ),
        harness=HarnessConfig(
            adapter_id="nvidia.fabric.strands",
            settings={
                "builtin_tools": [],
                "insecure_http_model_hosts": [urlsplit(endpoint).hostname]
                if endpoint
                else [],
            },
        ),
        models={
            "default": ModelConfig(
                provider="openai" if endpoint else "bedrock",
                model=model,
                base_url=endpoint,
                api_key_env=os.environ.get("STRANDS_TEST_API_KEY_ENV"),
            )
        },
        environment=EnvironmentConfig(
            provider="local", workspace=tmp_path, artifacts=tmp_path / "artifacts"
        ),
        runtime=RuntimeConfig(
            timeout_seconds=60,
            input_schema="text",
            output_schema="message",
            artifacts=tmp_path / "artifacts",
        ),
    )


async def test_strands_live_conversation_and_runtime_isolation(
    strands_config, tmp_path
):
    from nemo_fabric import Fabric

    token = uuid.uuid4().hex
    async with await Fabric().start_runtime(strands_config, base_dir=tmp_path) as first:
        remembered = await first.invoke(
            input=f"Remember this exact token: {token}. Reply only ACK."
        )
        recalled = await first.invoke(
            input="Reply only with the exact token from my previous message."
        )
        async with await Fabric().start_runtime(
            strands_config, base_dir=tmp_path
        ) as second:
            isolated = await second.invoke(
                input="What token did I previously give you? If none, reply only NONE."
            )
    assert remembered.status == "succeeded", remembered.to_mapping()
    assert recalled.status == "succeeded", recalled.to_mapping()
    assert recalled.output.response.strip() == token
    assert recalled.usage.input_tokens > 0
    assert (
        recalled.usage.total_tokens
        == recalled.usage.input_tokens + recalled.usage.output_tokens
    )
    assert (
        recalled.usage.metadata["nemo_fabric.detailed_token_usage.v1"][
            "input_tokens_include_cache"
        ]
        is True
    )
    assert isolated.status == "succeeded", isolated.to_mapping()
    assert isolated.output.response.strip() == "NONE"
    assert (
        remembered.metadata["adapter"]["session_id"]
        == recalled.metadata["adapter"]["session_id"]
    )
    assert (
        isolated.metadata["adapter"]["session_id"]
        != recalled.metadata["adapter"]["session_id"]
    )
    assert not (tmp_path / ".agent" / "sessions").exists()


async def test_strands_live_mcp_available_on_first_turn(strands_config, tmp_path):
    from nemo_fabric import Fabric

    marker = tmp_path / "probe-called"
    token = uuid.uuid4().hex
    root = Path(__file__).resolve().parents[2]
    strands_config.add_mcp_server(
        "probe",
        transport="stdio",
        url=sys.executable,
        args=[
            str(root / "tests/_utils/probe_mcp_server.py"),
            "--marker",
            str(marker),
            "--token",
            token,
        ],
    )
    async with await Fabric().start_runtime(
        strands_config, base_dir=tmp_path
    ) as runtime:
        result = await runtime.invoke(
            input="Call the probe MCP tool now. Reply only with the token it returns."
        )
    assert result.status == "succeeded", result.to_mapping()
    assert marker.read_text(encoding="utf-8") == token
    assert token in result.output.response


async def test_strands_live_harness_reads_workspace(strands_config, tmp_path):
    from nemo_fabric import Fabric

    token = uuid.uuid4().hex
    (tmp_path / "review-note.txt").write_text(token, encoding="utf-8")
    strands_config.harness.settings["builtin_tools"] = ["read"]
    async with await Fabric().start_runtime(
        strands_config, base_dir=tmp_path
    ) as runtime:
        result = await runtime.invoke(
            input="Read review-note.txt in the workspace. Reply only with its contents."
        )
    assert result.status == "succeeded", result.to_mapping()
    assert result.output.response.strip() == token
