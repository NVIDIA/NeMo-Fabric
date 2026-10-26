# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native sampling, typed results, and explicit measurement availability."""

import json
import sys

import pytest

from nemo_fabric import Fabric
from nemo_fabric import FabricConfig
from nemo_fabric import FabricConfigError
from nemo_fabric import GaugeSummary
from nemo_fabric import Measurement
from nemo_fabric import RunRequest
from nemo_fabric import RunResult
from nemo_fabric import SampleSummary
from nemo_fabric import SystemStatsConfig
from nemo_fabric import SystemStatsSamplingConfig
from nemo_fabric.integrations.harbor.models import FabricRunPayload


@pytest.mark.parametrize(
    "mapping",
    [
        {"interval_ms": True},
        {"interval_ms": 99},
        {"interval_ms": 60001},
        {"max_samples": 1},
        {"max_samples": 100001},
        {"unknown": True},
    ],
)
def test_sampling_options_enforce_cadence_and_retention_bounds(mapping):
    with pytest.raises(ValueError):
        SystemStatsSamplingConfig.from_mapping(mapping)


@pytest.mark.parametrize(
    "mapping",
    [
        {"status": "available", "value": True},
        {"status": "available", "value": -1},
        {"status": "available", "value": float("nan")},
        {"status": "unavailable", "reason": ""},
        {"status": "unlimited", "value": 0},
    ],
)
def test_measurement_rejects_invalid_numeric_and_availability_states(mapping):
    with pytest.raises(FabricConfigError):
        Measurement.from_mapping(mapping)


@pytest.mark.skipif(
    sys.platform not in {"linux", "darwin"}, reason="requires a native process reader"
)
async def test_sdk_returns_typed_samples_and_artifacts_without_relay_or_recording(
    hermes_shim_agent_dir, tmp_path, monkeypatch
):
    monkeypatch.setenv("ADAPTER_PYTHON", sys.executable)
    config = FabricConfig.from_mapping(
        {
            "metadata": {"name": "native-stats-test"},
            "harness": {
                "adapter_id": "test.fabric.hermes_shim",
                "resolution": "preinstalled",
            },
            "discovery": {"local_paths": ["adapters"]},
            "runtime": {"artifacts": tmp_path / "artifacts"},
            "system_stats": {"sampling": {"interval_ms": 100, "max_samples": 100}},
        }
    )
    result = await Fabric().run(
        config,
        base_dir=hermes_shim_agent_dir,
        request=RunRequest(input="sample", context={"delay_seconds": 0.5}),
    )
    assert result.status == "succeeded"
    sampled = result.system_stats.sampled
    assert isinstance(sampled, SampleSummary)
    assert isinstance(sampled.resident_memory_bytes, GaugeSummary)
    assert sampled.sample_count >= 3
    assert (
        sampled.resident_memory_bytes.sampled_peak
        >= sampled.resident_memory_bytes.final_value
    )
    # Gaps above twice the sampling interval do not contribute coverage.
    assert 0 <= sampled.resident_memory_bytes.covered_fraction <= 1
    assert (
        RunResult.from_mapping(result.to_mapping()).system_stats == result.system_stats
    )
    refs = {reference.kind: reference for reference in result.artifacts.artifacts}
    assert sampled.samples.path == refs["system_stats_samples"].path
    rows = [json.loads(line) for line in sampled.samples.path.read_text().splitlines()]
    assert len(rows) == sampled.sample_count
    assert all(row["invocation_id"] == result.invocation_id for row in rows)
    assert rows[-1]["offset_seconds"] == result.system_stats.elapsed_seconds
    saved = json.loads(refs["system_stats_summary"].path.read_text())
    assert saved["invocation_id"] == result.invocation_id
    assert saved["system_stats"]["sampled"]["sample_count"] == sampled.sample_count


@pytest.mark.parametrize(
    "collect", [[], ["unknown"], ["process", "process"], "process"]
)
def test_metric_selection_rejects_invalid_groups(collect):
    with pytest.raises(ValueError):
        SystemStatsConfig(collect=collect)


@pytest.mark.parametrize(
    "selection",
    [
        None,
        {},
        {"collect": ["process"]},
        {"collect": ["cgroup.cpu"], "sampling": {"interval_ms": 100}},
    ],
)
@pytest.mark.skipif(
    sys.platform not in {"linux", "darwin"}, reason="requires a native process reader"
)
async def test_stats_controls_survive_sdk_plan_and_runner_mapping(
    selection, hermes_shim_agent_dir, tmp_path, monkeypatch
):
    monkeypatch.setenv("ADAPTER_PYTHON", sys.executable)
    mapping = {
        "metadata": {"name": "stats-controls"},
        "harness": {
            "adapter_id": "test.fabric.hermes_shim",
            "resolution": "preinstalled",
        },
        "discovery": {"local_paths": ["adapters"]},
        "runtime": {"artifacts": str(tmp_path / "artifacts")},
    }
    default = FabricConfig.from_mapping(mapping)
    assert default.system_stats == SystemStatsConfig()
    default.system_stats = None
    assert default.to_mapping()["system_stats"] is None
    config = FabricConfig.from_mapping({**mapping, "system_stats": selection})
    encoded = config.to_mapping()
    assert (encoded["system_stats"] is None) == (selection is None)
    assert FabricConfig.from_mapping(encoded) == config
    assert (
        config.model_dump(mode="json", exclude_none=True)["system_stats"]
        == encoded["system_stats"]
    )
    payload = FabricRunPayload(
        config=config, config_base_dir="/task", request=RunRequest(input="controls")
    )
    wire = payload.model_dump(mode="json", exclude_none=True)
    assert wire["config"]["system_stats"] == encoded["system_stats"]
    assert (
        FabricRunPayload.model_validate(wire).config.system_stats == config.system_stats
    )
    fabric = Fabric()
    plan = fabric.plan(config, base_dir=hermes_shim_agent_dir)
    assert plan.config.to_mapping()["system_stats"] == encoded["system_stats"]
    result = await fabric.run(config, base_dir=hermes_shim_agent_dir, input="controls")
    assert result.resource_snapshot is not None
    stats = result.system_stats
    refs = [a for a in result.artifacts.artifacts if a.kind.startswith("system_stats")]
    if selection is None:
        assert stats is None
        assert refs == []
        assert list((tmp_path / "artifacts").rglob("system-stats*")) == []
    elif "sampling" not in selection:
        assert stats.sampled is None
        assert refs == []
        assert stats.before.status == stats.after.status == "available"
        if selection.get("collect") == ["process"]:
            assert stats.cgroup is None
    else:
        assert stats.before.status == stats.after.status == "not_collected"
        assert stats.sampled.resident_memory_bytes is None
        assert stats.sampled.process_count is None
        assert stats.sampled.cpu_time_seconds_estimate.status == "not_collected"
        assert stats.cgroup.oom_kills.status == "not_collected"
        rows = [
            json.loads(line)
            for line in stats.sampled.samples.path.read_text().splitlines()
        ]
        assert all(row["adapter_tree"]["status"] == "not_collected" for row in rows)
        if sys.platform == "linux" and stats.cgroup.after.status == "available":
            snapshot = stats.cgroup.after.snapshot
            assert snapshot.memory_current_bytes.status == "not_collected"
            assert all(
                m.status == "not_collected"
                for m in snapshot.pressure_stall_microseconds.values()
            )
    assert RunResult.from_mapping(result.to_mapping()).system_stats == stats
