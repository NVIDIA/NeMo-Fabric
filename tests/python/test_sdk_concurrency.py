# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke test for independent concurrent Fabric SDK runs."""

from __future__ import annotations

import asyncio
from pathlib import Path
from shutil import copytree

from nemo_fabric import Fabric


async def run_runtime(client: Fabric, agent: Path, name: str) -> dict:
    async with await client.start_runtime(agent, profiles=["env_local"]) as runtime:
        return await runtime.invoke(input=f"hello from {name}")


async def run_copy(client: Fabric, fixture_agent: Path, root: Path, name: str) -> dict:
    agent = root / name
    copytree(fixture_agent, agent)
    return await client.run(agent, profiles=["env_local"], input=f"hello from {name}")


async def test_sdk_concurrency(hermes_shim_agent_dir_src: Path, tmp_path: Path):
    client = Fabric()
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


async def test_independent_runtimes_isolate_files_in_shared_artifact_root(
    hermes_shim_agent_dir: Path,
):
    client = Fabric()
    first, second = await asyncio.gather(
        run_runtime(client, hermes_shim_agent_dir, "runtime-one"),
        run_runtime(client, hermes_shim_agent_dir, "runtime-two"),
    )

    assert first["status"] == "succeeded"
    assert second["status"] == "succeeded"
    assert first["runtime_id"] != second["runtime_id"]
    assert first["invocation_id"] != second["invocation_id"]
    assert first["output"]["received"] == "hello from runtime-one"
    assert second["output"]["received"] == "hello from runtime-two"
    assert first["artifacts"]["root"] == second["artifacts"]["root"]
    first_paths = {artifact["path"] for artifact in first["artifacts"]["artifacts"]}
    second_paths = {artifact["path"] for artifact in second["artifacts"]["artifacts"]}
    assert first_paths.isdisjoint(second_paths)
