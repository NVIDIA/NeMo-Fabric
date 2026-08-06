# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from nemo_fabric_adapters.common import relay_artifacts


def atif_plugin_config(
    output_directory: Path,
    *,
    component_enabled: bool = True,
    storage: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    atif: dict[str, Any] = {
        "enabled": True,
        "output_directory": str(output_directory),
        "filename_template": "trajectory-{session_id}.atif.json",
    }
    if storage is not None:
        atif["storage"] = storage
    return {
        "version": 1,
        "components": [
            {
                "kind": "observability",
                "enabled": component_enabled,
                "config": {"atif": atif},
            }
        ],
    }


def test_expects_local_atif_requires_enabled_local_output(tmp_path):
    local = atif_plugin_config(tmp_path / "local")
    disabled = atif_plugin_config(tmp_path / "disabled", component_enabled=False)
    remote = atif_plugin_config(
        tmp_path / "remote",
        storage=[{"type": "http", "endpoint": "https://example.test/atif"}],
    )

    assert relay_artifacts.expects_local_atif(local) is True
    assert relay_artifacts.expects_local_atif(disabled) is False
    assert relay_artifacts.expects_local_atif(remote) is False
    assert relay_artifacts.expects_local_atif({"components": []}) is False


async def test_wait_for_finalized_atif_ignores_unchanged_and_partial_files(
    tmp_path, monkeypatch
):
    atif_dir = tmp_path / "atif"
    atif_dir.mkdir()
    existing = atif_dir / "trajectory-existing.atif.json"
    candidate = atif_dir / "trajectory-current.atif.json"
    existing.write_text('{"schema_version":"ATIF-v1.7"}', encoding="utf-8")
    plugin_config = atif_plugin_config(atif_dir)
    before = relay_artifacts.snapshot_atif_files(plugin_config)
    reads: list[Path] = []
    read_bytes = Path.read_bytes

    def record_read(path: Path) -> bytes:
        reads.append(path)
        return read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", record_read)

    async def finish_candidate():
        candidate.write_text("{", encoding="utf-8")
        await asyncio.sleep(0.02)
        candidate.write_text(
            json.dumps({"schema_version": "ATIF-v1.7", "steps": []}),
            encoding="utf-8",
        )

    writer = asyncio.create_task(finish_candidate())
    finalized = await relay_artifacts.wait_for_finalized_atif(
        plugin_config,
        before,
        timeout_seconds=0.5,
        poll_interval_seconds=0.001,
    )
    await writer

    assert set(before) == {existing.resolve()}
    assert finalized == candidate.resolve()
    assert set(reads) == {candidate.resolve()}


async def test_wait_for_finalized_atif_accepts_modified_existing_path(tmp_path):
    atif_dir = tmp_path / "atif"
    atif_dir.mkdir()
    existing = atif_dir / "trajectory-existing.atif.json"
    existing.write_text('{"schema_version":"ATIF-v1.7"}', encoding="utf-8")
    plugin_config = atif_plugin_config(atif_dir)
    before = relay_artifacts.snapshot_atif_files(plugin_config)
    existing.write_text(
        json.dumps({"schema_version": "ATIF-v1.7", "steps": [{"step_id": 1}]}),
        encoding="utf-8",
    )

    finalized = await relay_artifacts.wait_for_finalized_atif(
        plugin_config,
        before,
        timeout_seconds=0.5,
        poll_interval_seconds=0.001,
    )

    assert finalized == existing.resolve()


async def test_wait_for_finalized_atif_has_a_hard_deadline(tmp_path):
    atif_dir = tmp_path / "atif"
    atif_dir.mkdir()
    plugin_config = atif_plugin_config(atif_dir)

    finalized = await relay_artifacts.wait_for_finalized_atif(
        plugin_config,
        relay_artifacts.snapshot_atif_files(plugin_config),
        timeout_seconds=0.01,
        poll_interval_seconds=0.001,
    )

    assert finalized is None


async def test_wait_for_finalized_atif_isolated_by_runtime_directory(tmp_path):
    first_dir = tmp_path / "runtime-1"
    second_dir = tmp_path / "runtime-2"
    first_dir.mkdir()
    second_dir.mkdir()
    first_config = atif_plugin_config(first_dir)
    second_config = atif_plugin_config(second_dir)

    async def write_atif(directory: Path, session_id: str, delay: float):
        await asyncio.sleep(delay)
        path = directory / f"trajectory-{session_id}.atif.json"
        path.write_text(
            json.dumps({"schema_version": "ATIF-v1.7", "steps": []}),
            encoding="utf-8",
        )
        return path.resolve()

    first_writer = asyncio.create_task(write_atif(first_dir, "first", 0.02))
    second_writer = asyncio.create_task(write_atif(second_dir, "second", 0.01))
    first_wait = relay_artifacts.wait_for_finalized_atif(
        first_config,
        relay_artifacts.snapshot_atif_files(first_config),
        timeout_seconds=0.5,
        poll_interval_seconds=0.001,
    )
    second_wait = relay_artifacts.wait_for_finalized_atif(
        second_config,
        relay_artifacts.snapshot_atif_files(second_config),
        timeout_seconds=0.5,
        poll_interval_seconds=0.001,
    )

    first_finalized, second_finalized, first_path, second_path = await asyncio.gather(
        first_wait, second_wait, first_writer, second_writer
    )

    assert first_finalized == first_path
    assert second_finalized == second_path
