# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SDK software control and result compatibility."""

import json
from pathlib import Path

from nemo_fabric import FabricConfig
from nemo_fabric import RunResult


def test_disabled_provenance_survives_config_and_native_plan_round_trip():
    from nemo_fabric import Fabric

    config = FabricConfig.from_mapping(
        {
            "metadata": {"name": "evidence"},
            "harness": {"adapter_id": "nvidia.fabric.codex"},
            "provenance": {"enabled": False},
        }
    )
    assert config.to_mapping()["provenance"]["enabled"] is False
    plan = Fabric().plan(config, base_dir=Path(__file__).parents[2])
    assert plan.to_mapping()["config"]["provenance"]["enabled"] is False
    assert "provenance" not in plan.config.extra_fields


def test_plan_snapshot_normalizes_declared_provenance_settings():
    from nemo_fabric.types import _FabricConfigSnapshot

    snapshot = _FabricConfigSnapshot.from_mapping(
        {
            "metadata": {"name": "evidence"},
            "harness": {"adapter_id": "nvidia.fabric.codex"},
            "provenance": {},
            "future_extension": {"enabled": True},
        }
    )
    assert snapshot.provenance == {"enabled": True}
    assert snapshot.extra_fields == {"future_extension": {"enabled": True}}
    assert snapshot.to_mapping()["provenance"] == {"enabled": True}


def test_old_result_decodes_and_optional_evidence_round_trips():
    result = {
        "agent_name": "evidence",
        "harness": "example",
        "adapter_kind": "python",
        "runtime_id": "r",
        "invocation_id": "i",
        "request_id": "q",
        "status": "succeeded",
        "output": "ok",
    }
    assert RunResult.from_mapping(result).execution is None
    report = json.loads(
        (Path(__file__).parents[1] / "fixtures/provenance/report.json").read_text()
    )
    decoded = RunResult.from_mapping({**result, "execution": report})
    assert decoded.to_mapping()["execution"] == report
    assert (
        decoded.execution.software_environments[0]
        .entries[0]
        .version.value.endswith("build")
    )
    broken = RunResult.from_mapping({**result, "execution": {"broken": True}})
    assert broken.status == "succeeded"
    assert broken.output == "ok"
    assert not broken.execution.software_environments[0].complete


async def test_runtime_reuses_initialization_snapshots_across_invocations(
    hermes_shim_agent_dir, tmp_path, monkeypatch
):
    import sys
    from nemo_fabric import Fabric

    monkeypatch.setenv("ADAPTER_PYTHON", sys.executable)
    config = FabricConfig.from_mapping(
        {
            "metadata": {"name": "software-runtime"},
            "harness": {
                "adapter_id": "test.fabric.hermes_shim",
                "resolution": "preinstalled",
            },
            "discovery": {"local_paths": ["adapters"]},
            "runtime": {"artifacts": str(tmp_path / "artifacts")},
        }
    )
    fabric = Fabric()
    async with await fabric.start_runtime(
        config, base_dir=hermes_shim_agent_dir
    ) as runtime:
        first = await runtime.invoke(input="first")
        second = await runtime.invoke(input="second")
    assert first.status == second.status == "succeeded"
    assert first.invocation_id != second.invocation_id
    assert first.execution.to_mapping() == second.execution.to_mapping()
    participants = {
        p for env in first.execution.software_environments for p in env.participants
    }
    assert {"caller_sdk", "core", "adapter"} <= participants
    core = next(
        env
        for env in first.execution.software_environments
        if "core" in env.participants
    )
    assert core.entries[0].version.value


async def test_disable_skips_host_and_adapter_metadata_work(
    hermes_shim_agent_dir, tmp_path, monkeypatch
):
    import sys
    from nemo_fabric import Fabric
    import nemo_fabric.provenance as sdk_provenance

    def unexpected(participant):
        raise AssertionError("host collector ran while disabled")

    monkeypatch.setenv("ADAPTER_PYTHON", sys.executable)
    marker = tmp_path / "lookups"
    adapter = next(
        hermes_shim_agent_dir.glob(
            "adapters/*/src/nemo_fabric_test_adapters/*/adapter.py"
        )
    )
    source = adapter.read_text()
    assert source.count("    lifecycle.serve(ShimRuntime)") == 1
    adapter.write_text(
        source.replace(
            "    lifecycle.serve(ShimRuntime)",
            """    import importlib.metadata
    original = importlib.metadata.distribution
    def lookup(name):
        Path("""
            + repr(str(marker))
            + """).write_text(name)
        return original(name)
    importlib.metadata.distribution = lookup
    lifecycle.serve(ShimRuntime)""",
        )
    )
    config = FabricConfig.from_mapping(
        {
            "metadata": {"name": "disabled-software"},
            "harness": {
                "adapter_id": "test.fabric.hermes_shim",
                "resolution": "preinstalled",
            },
            "discovery": {"local_paths": ["adapters"]},
            "runtime": {"artifacts": str(tmp_path / "artifacts")},
            "provenance": {"enabled": True},
        }
    )
    enabled = await Fabric().run(
        config, base_dir=hermes_shim_agent_dir, input="positive control"
    )
    assert enabled.status == "succeeded"
    assert marker.read_text()
    marker.unlink()
    monkeypatch.setattr(sdk_provenance, "sdk_snapshot", unexpected)
    config.provenance.enabled = False
    result = await Fabric().run(
        config, base_dir=hermes_shim_agent_dir, input="still executes"
    )
    assert result.status == "succeeded"
    assert result.execution is None
    assert not marker.exists()
