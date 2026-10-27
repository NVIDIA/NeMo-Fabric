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
