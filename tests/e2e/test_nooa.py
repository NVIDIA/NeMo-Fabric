# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Relay-backed subprocess E2E for the OO Agents adapter."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from nemo_fabric import DiscoveryConfig
from nemo_fabric import EnvironmentConfig
from nemo_fabric import Fabric
from nemo_fabric import FabricConfig
from nemo_fabric import MetadataConfig
from nemo_fabric import RelayAtifConfig
from nemo_fabric import RelayAtofConfig
from nemo_fabric import RelayAtofFileSinkConfig
from nemo_fabric import RelayObservabilityConfig
from nemo_fabric import RuntimeConfig
from nemo_fabric import WorkflowConfig

ROOT = Path(__file__).parents[2]
NOOA_ROOT = ROOT / "external" / "nooa"
FIXTURE_ROOT = ROOT / "tests" / "fixtures" / "nooa"
FIXTURE_SOURCE = FIXTURE_ROOT / "src"


@pytest.mark.usefixtures("nemo_relay")
async def test_nooa_relay_streams_correlated_atof_and_returns_once(
    tmp_path: Path,
):
    current_pythonpath = os.environ.get("PYTHONPATH")
    pythonpath = os.pathsep.join(
        [
            str(NOOA_ROOT / "src"),
            str(FIXTURE_SOURCE),
            *([current_pythonpath] if current_pythonpath else []),
        ]
    )
    os.environ["PYTHONPATH"] = pythonpath
    os.environ["ADAPTER_PYTHON"] = sys.executable

    config = FabricConfig(
        metadata=MetadataConfig(name="nooa-relay-e2e"),
        discovery=DiscoveryConfig(local_paths=[NOOA_ROOT, FIXTURE_ROOT]),
        workflow=WorkflowConfig(target_id="nvidia.tests.nooa.echo", settings={}),
        runtime=RuntimeConfig(
            input_schema="text",
            output_schema="message",
            artifacts=tmp_path / "artifacts",
        ),
        environment=EnvironmentConfig(
            provider="local",
            workspace=tmp_path,
            artifacts=tmp_path / "artifacts",
        ),
    )
    config.enable_relay(
        output_dir=tmp_path / "relay",
        observability=RelayObservabilityConfig(
            atif=RelayAtifConfig(
                enabled=True,
                output_directory=tmp_path / "relay",
                filename_template="trajectory-{session_id}.atif.json",
            ),
            atof=RelayAtofConfig(
                enabled=True,
                sinks=[
                    RelayAtofFileSinkConfig(
                        output_directory=tmp_path / "relay",
                        filename="events.atof.jsonl",
                        mode="overwrite",
                    )
                ],
            ),
        ),
    )

    async with await Fabric().start_runtime(
        config,
        base_dir=tmp_path,
        streaming=True,
    ) as runtime:
        first_stream = runtime.invoke_stream(input="first")
        first_records = [record async for record in first_stream]
        first = await first_stream.result()

        second_stream = runtime.invoke_stream(input="second")
        second_records = [record async for record in second_stream]
        second = await second_stream.result()

    results = (first.to_mapping(), second.to_mapping())
    assert first_records and second_records, results
    assert first.status == second.status == "succeeded", results
    assert first.output["response"] == "reply-1: first", results
    assert second.output["response"] == "reply-2: second", results
    assert first.metadata["host_pid"] == second.metadata["host_pid"], results
    first_atof = json.dumps(first_records, sort_keys=True)
    second_atof = json.dumps(second_records, sort_keys=True)
    assert first.request_id in first_atof
    assert second.request_id in second_atof
    assert second.request_id not in first_atof
    assert first.request_id not in second_atof
    assert "fixture-model" in first_atof
    assert "execute_python" in first_atof
    for result in (first, second):
        assert result.telemetry[0].provider == "relay", result.to_mapping()
        assert {item["kind"] for item in result.output["relay_artifacts"]} == {
            "atof",
            "atif",
        }
