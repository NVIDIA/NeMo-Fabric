# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest
from nemo_fabric import RunResult
from nemo_fabric.integrations.harbor.telemetry import TelemetryValidationError
from nemo_fabric.integrations.harbor.telemetry import publish_telemetry_evidence


def make_result(*artifacts: dict[str, str]) -> RunResult:
    return RunResult.from_mapping(
        {
            "agent_name": "harbor-demo",
            "profiles": [],
            "harness": "hermes",
            "adapter_kind": "python",
            "adapter_id": "nvidia.fabric.hermes.cli",
            "status": "succeeded",
            "runtime_id": "runtime-1",
            "invocation_id": "invocation-1",
            "request_id": "request-1",
            "output": {"response": "done"},
            "error": None,
            "artifacts": {"root": None, "artifacts": list(artifacts)},
            "telemetry": [],
            "events": [],
            "metadata": {},
        }
    )


def artifact(name: str, kind: str, path: Path) -> dict[str, str]:
    return {
        "name": name,
        "kind": kind,
        "path": str(path),
        "media_type": "application/json",
    }


def test_publish_telemetry_validates_and_promotes_atif(tmp_path: Path):
    atof = tmp_path / "events.atof.jsonl"
    atof.write_text(
        json.dumps(
            {
                "atof_version": "1.0",
                "kind": "event",
                "name": "agent.end",
                "timestamp": "2026-07-13T00:00:00Z",
                "uuid": "event-1",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    atif = tmp_path / "relay" / "trajectory-run.atif.json"
    atif.parent.mkdir()
    atif.write_text(
        json.dumps(
            {
                "schema_version": "ATIF-v1.7",
                "session_id": "runtime-1",
                "agent": {"name": "fabric", "version": "0.1.0"},
                "steps": [{"step_id": 1, "source": "agent", "message": "done"}],
                "final_metrics": {
                    "total_prompt_tokens": 12,
                    "total_completion_tokens": 4,
                    "total_steps": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    logs = tmp_path / "agent"
    result = make_result(
        artifact("atof", "atof", atof),
        artifact("atif", "atif", atif),
    )

    summary = publish_telemetry_evidence(result, logs, strict=True)

    assert summary["status"] == "succeeded"
    assert summary["atof"]["records"] == 1
    assert summary["atif"]["steps"] == 1
    assert (logs / "trajectory.json").read_bytes() == atif.read_bytes()
    assert json.loads((logs / "telemetry-validation.json").read_text())["status"] == ("succeeded")


def test_publish_telemetry_records_quality_failure_without_changing_run(tmp_path: Path):
    malformed = tmp_path / "events.atof.jsonl"
    malformed.write_text("{}\n", encoding="utf-8")
    result = make_result(artifact("atof", "atof", malformed))

    summary = publish_telemetry_evidence(result, tmp_path / "agent")

    assert summary["status"] == "failed"
    assert "ATOF record missing" in summary["error"]

    with pytest.raises(TelemetryValidationError, match="ATOF record missing"):
        publish_telemetry_evidence(result, tmp_path / "strict", strict=True)


def test_publish_telemetry_rejects_ambiguous_atif(tmp_path: Path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text("{}", encoding="utf-8")
    second.write_text("{}", encoding="utf-8")
    result = make_result(
        artifact("atif-1", "atif", first),
        artifact("atif-2", "atif", second),
    )

    with pytest.raises(TelemetryValidationError, match="at most one ATIF"):
        publish_telemetry_evidence(result, tmp_path / "agent", strict=True)


def test_publish_telemetry_rejects_obvious_credentials(tmp_path: Path):
    atof = tmp_path / "events.atof.jsonl"
    atof.write_text(
        json.dumps(
            {
                "atof_version": "1.0",
                "kind": "event",
                "name": "agent.end",
                "timestamp": "2026-07-13T00:00:00Z",
                "uuid": "event-1",
                "data": {"api_key": "sk-thismustneverbeleak123456"},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(TelemetryValidationError, match="resembles a credential"):
        publish_telemetry_evidence(
            make_result(artifact("atof", "atof", atof)),
            tmp_path / "agent",
            strict=True,
        )
