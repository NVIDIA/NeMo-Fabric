# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the OO Agents BenchAgent Harbor walkthrough verifier."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from examples.harbor.nooa_bench.verify_run import verify


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def make_job(
    tmp_path: Path,
    *,
    relay: bool,
    reward: float = 1.0,
    trial_name: str = "task__fixture",
) -> Path:
    job_dir = tmp_path / "job"
    trial_dir = job_dir / trial_name
    write_json(
        job_dir / "result.json",
        {"stats": {"n_completed_trials": 1, "n_errored_trials": 0}},
    )
    reward_path = trial_dir / "verifier" / "reward.txt"
    reward_path.parent.mkdir(parents=True)
    reward_path.write_text(f"{reward}\n", encoding="utf-8")
    write_json(
        trial_dir / "agent" / "fabric-result-fixture.json",
        {
            "status": "succeeded",
            "adapter_id": "nvidia.fabric.nooa.bench-agent",
            "output": {
                "harness": "nooa-bench",
                "mode": "bench_agent",
                "completed": True,
            },
            "telemetry": [],
            "usage": {
                "input_tokens": 5,
                "output_tokens": 3,
                "total_tokens": 8,
                "metadata": {},
            },
        },
    )
    if not relay:
        return job_dir

    result_path = trial_dir / "agent" / "fabric-result-fixture.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["output"]["relay_artifacts"] = [
        {"kind": "atof", "path": "/logs/events.atof.jsonl"},
        {"kind": "atif", "path": "/logs/trajectory.atif.json"},
    ]
    result["telemetry"] = [
        {
            "provider": "relay",
            "metadata": {
                "relay_config": {
                    "version": 1,
                    "components": [
                        {
                            "kind": "observability",
                            "enabled": True,
                            "config": {"version": 3},
                        }
                    ],
                }
            },
        }
    ]
    write_json(result_path, result)

    scopes = [
        ("nooa-bench-agent-request", "agent"),
        ("BenchAgent._run_evaluation", "function"),
        ("BenchAgent._solve_task", "function"),
        ("fixture-model", "llm"),
        ("execute_python", "tool"),
    ]
    records = [
        {
            "atof_version": "0.1",
            "name": name,
            "category": category,
            "scope_category": boundary,
            "uuid": f"scope-{index}",
        }
        for index, (name, category) in enumerate(scopes)
        for boundary in ("start", "end")
    ]
    artifact_dir = trial_dir / "agent" / "fabric-artifacts" / "relay"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "events.atof.jsonl").write_text(
        "".join(f"{json.dumps(record)}\n" for record in records),
        encoding="utf-8",
    )
    atif = {
        "schema_version": "ATIF-v1.7",
        "steps": [{"step_id": 1}],
    }
    write_json(artifact_dir / "trajectory-fixture.atif.json", atif)
    write_json(trial_dir / "agent" / "trajectory.json", atif)
    write_json(
        trial_dir / "agent" / "telemetry-validation.json",
        {
            "status": "succeeded",
            "atof": {"records": len(records)},
            "atif": {"schema_version": "ATIF-v1.7", "steps": 1},
        },
    )
    return job_dir


def test_verify_accepts_rewarded_relay_run(tmp_path: Path):
    summary = verify(make_job(tmp_path, relay=True), require_relay=True)

    assert summary["reward"] == 1.0
    assert summary["relay"] == {
        "atof_records": 10,
        "categories": {"agent": 2, "function": 4, "llm": 2, "tool": 2},
        "atif_schema_version": "ATIF-v1.7",
        "atif_steps": 1,
        "relay_config_version": 1,
        "observability_config_version": 3,
        "root_invocations": 1,
    }


def test_verify_accepts_zero_reward_swebench_run(tmp_path: Path):
    job_dir = make_job(
        tmp_path,
        relay=True,
        reward=0.0,
        trial_name="prepared-django__django-13741__fixture",
    )

    summary = verify(
        job_dir,
        require_relay=True,
        allow_zero_reward=True,
    )

    assert summary["reward"] == 0.0
