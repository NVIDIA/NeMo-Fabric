# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Goose CLI E2E coverage.

The local test runs when `goose` is on PATH. Run the live test with:
RUN_FABRIC_GOOSE_INTEGRATION=1 \\
GOOSE_LIVE_PROVIDER=nvidia \\
GOOSE_LIVE_MODEL=nvidia/nemotron-3-nano-omni-30b-a3b-reasoning \\
GOOSE_LIVE_API_KEY_ENV=NVIDIA_API_KEY \\
GOOSE_LIVE_BASE_URL=https://integrate.api.nvidia.com/v1 \\
pytest tests/e2e/test_goose.py
"""

from __future__ import annotations

import os
import shutil
import sys
import uuid
from pathlib import Path

import pytest

from nemo_fabric import Fabric
from nemo_fabric import FabricConfig

ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.skipif(
    shutil.which("goose") is None, reason="The Goose CLI is required"
)


def goose_config(workspace: Path, model: dict[str, str], **fields) -> FabricConfig:
    return FabricConfig.from_mapping(
        {
            "metadata": {"name": "goose-e2e"},
            "harness": {"adapter_id": "nvidia.fabric.goose"},
            "models": {"default": model},
            "environment": {"provider": "local", "workspace": str(workspace)},
            **fields,
        }
    )


async def test_goose_resumes_one_session_against_a_local_endpoint(api_server, tmp_path):
    model = {
        "provider": "openai",
        "model": "fabric-echo",
        "base_url": f"{api_server}/v1",
    }
    fabric = Fabric()

    async with await fabric.start_runtime(
        goose_config(tmp_path, model), base_dir=tmp_path
    ) as runtime:
        first = await runtime.invoke(input="first")
        second = await runtime.invoke(input="second")
    isolated = await fabric.run(
        goose_config(tmp_path, model), input="isolated", base_dir=tmp_path
    )

    assert first.status == second.status == isolated.status == "succeeded"
    assert first.output.response.startswith("echo user_count=1 ")
    assert second.output.response.startswith("echo user_count=2 ")
    assert isolated.output.response.startswith("echo user_count=1 ")


async def test_goose_live_provider_runtime(tmp_path):
    if os.environ.get("RUN_FABRIC_GOOSE_INTEGRATION") != "1":
        pytest.skip("set RUN_FABRIC_GOOSE_INTEGRATION=1 to run")
    model = {
        "provider": os.environ.get("GOOSE_LIVE_PROVIDER", "openai"),
        "model": os.environ.get("GOOSE_LIVE_MODEL", ""),
    }
    for field, name in (
        ("base_url", "GOOSE_LIVE_BASE_URL"),
        ("api_key_env", "GOOSE_LIVE_API_KEY_ENV"),
    ):
        if os.environ.get(name):
            model[field] = os.environ[name]
    if not model["model"]:
        pytest.fail(
            "set GOOSE_LIVE_MODEL and, for a custom endpoint, GOOSE_LIVE_BASE_URL"
        )
    marker = tmp_path / "probe-called"
    token = uuid.uuid4().hex
    nonce = f"fabric-goose-{uuid.uuid4().hex[:8]}"
    config = goose_config(
        tmp_path,
        model,
        mcp={
            "servers": {
                "probe": {
                    "transport": "stdio",
                    "url": sys.executable,
                    "args": [
                        str(ROOT / "tests" / "_utils" / "probe_mcp_server.py"),
                        "--marker",
                        str(marker),
                        "--token",
                        token,
                    ],
                }
            }
        },
    )

    async with await Fabric().start_runtime(config, base_dir=tmp_path) as runtime:
        remembered = await runtime.invoke(
            input=f"Remember this exact token: {nonce}. Reply with exactly ACK."
        )
        recalled = await runtime.invoke(
            input="Reply with only the exact token from my previous message."
        )
        probed = await runtime.invoke(
            input="Call the probe tool, then reply with only the token it returns."
        )

    for result in (remembered, recalled, probed):
        assert result.status == "succeeded", result.to_mapping()
    assert nonce in recalled.output.response
    assert marker.read_text(encoding="utf-8") == token
    assert token in probed.output.response
