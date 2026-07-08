# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke test for independent concurrent Fabric SDK runs."""

from __future__ import annotations

import asyncio
from pathlib import Path
from shutil import copytree

from nemo_fabric import Fabric


async def run_copy(client: Fabric, fixture_agent: Path, root: Path, name: str) -> dict:
    agent = root / name
    copytree(fixture_agent, agent)
    return await client.run(agent, profiles=["env_local"], input=f"hello from {name}")


async def test_sdk_concurrency(hermes_shim_agent_dir_src: Path, tmp_path: Path):
    async with Fabric() as client:
        first, second = await asyncio.gather(
            run_copy(client, hermes_shim_agent_dir_src, tmp_path, "agent-one"),
            run_copy(client, hermes_shim_agent_dir_src, tmp_path, "agent-two"),
        )

    assert first["status"] == "succeeded"
    assert second["status"] == "succeeded"
    assert first["runtime_id"] != second["runtime_id"]
    assert first["invocation_id"] != second["invocation_id"]
    assert first["output"]["received"] == "hello from agent-one"
    assert second["output"]["received"] == "hello from agent-two"
    assert first["artifacts"]["root"] != second["artifacts"]["root"]
