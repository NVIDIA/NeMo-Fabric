# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke test for independent concurrent Fabric SDK runs."""

from __future__ import annotations

import asyncio
from pathlib import Path
from shutil import copytree

from nemo_fabric import FabricClient

ROOT = Path(__file__).resolve().parents[2]


async def run_copy(
    client: FabricClient, fixture_agent: Path, root: Path, name: str
) -> dict:
    agent = root / name
    copytree(fixture_agent, agent)
    return await client.run(agent, profiles=["env_local"], input=f"hello from {name}")


async def test_sdk_concurrency(tmp_path: Path):
    fixture_agent = ROOT / "tests" / "fixtures" / "hermes-shim-agent"

    async with FabricClient() as client:
        first, second = await asyncio.gather(
            run_copy(client, fixture_agent, tmp_path, "agent-one"),
            run_copy(client, fixture_agent, tmp_path, "agent-two"),
        )

    assert first["status"] == "succeeded"
    assert second["status"] == "succeeded"
    assert first["runtime_id"] != second["runtime_id"]
    assert first["invocation_id"] != second["invocation_id"]
    assert first["output"]["received"] == "hello from agent-one"
    assert second["output"]["received"] == "hello from agent-two"
    assert first["artifacts"]["root"] != second["artifacts"]["root"]
