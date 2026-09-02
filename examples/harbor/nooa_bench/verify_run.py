# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate a BenchAgent Harbor job and its optional Relay artifacts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def one(root: Path, pattern: str) -> Path:
    matches = list(root.glob(pattern))
    if len(matches) != 1:
        raise AssertionError(
            f"expected one {pattern!r} beneath {root}; found {len(matches)}"
        )
    return matches[0]


def verify(
    job_dir: Path,
    *,
    require_relay: bool,
    allow_zero_reward: bool = False,
) -> dict[str, Any]:
    job = load_json(job_dir / "result.json")
    stats = job["stats"]
    assert stats["n_completed_trials"] == 1
    assert stats["n_errored_trials"] == 0

    reward_path = one(job_dir, "*/verifier/reward.txt")
    trial_dir = reward_path.parents[1]
    reward = float(reward_path.read_text().strip())
    if allow_zero_reward:
        assert 0.0 <= reward <= 1.0
    else:
        assert reward == 1.0

    result = load_json(one(trial_dir, "agent/fabric-result-*.json"))
    assert result["status"] == "succeeded"
    assert result["adapter_id"] == "nvidia.fabric.nooa.bench-agent"
    assert result["output"]["harness"] == "nooa-bench"
    assert result["output"]["mode"] == "bench_agent"
    assert result["output"]["completed"] is True

    summary: dict[str, Any] = {
        "status": result["status"],
        "reward": reward,
        "adapter_id": result["adapter_id"],
        "usage": result["usage"],
    }

    if require_relay:
        relay_artifacts = result["output"]["relay_artifacts"]
        assert {artifact["kind"] for artifact in relay_artifacts} == {"atof", "atif"}
        relay_references = [
            reference
            for reference in result["telemetry"]
            if reference["provider"] == "relay"
        ]
        assert len(relay_references) == 1
        relay_config = relay_references[0]["metadata"]["relay_config"]
        assert relay_config["version"] == 1
        observability = [
            component
            for component in relay_config["components"]
            if component["kind"] == "observability" and component["enabled"] is True
        ]
        assert len(observability) == 1
        assert observability[0]["config"]["version"] == 3

        validation = load_json(trial_dir / "agent" / "telemetry-validation.json")
        assert validation["status"] == "succeeded"
        assert validation["atof"]["records"] > 0
        assert validation["atif"]["schema_version"].startswith("ATIF-v1.")

        atof_path = one(trial_dir, "agent/fabric-artifacts/**/events.atof.jsonl")
        records = [
            json.loads(line)
            for line in atof_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert {record["atof_version"] for record in records} == {"0.1"}
        assert Counter(record["scope_category"] for record in records) == {
            "start": len(records) // 2,
            "end": len(records) // 2,
        }
        starts = Counter(
            record["uuid"] for record in records if record["scope_category"] == "start"
        )
        ends = Counter(
            record["uuid"] for record in records if record["scope_category"] == "end"
        )
        assert starts == ends
        assert set(starts.values()) == {1}
        assert (
            sum(
                record["name"] == "nooa-bench-agent-request"
                and record["scope_category"] == "start"
                for record in records
            )
            == 1
        )
        assert (
            sum(
                record["name"] == "nooa-bench-agent-request"
                and record["scope_category"] == "end"
                for record in records
            )
            == 1
        )
        categories = Counter(record["category"] for record in records)
        assert categories["agent"] == 2
        assert categories["function"] >= 4
        assert categories["llm"] >= 2
        assert categories["tool"] >= 2

        atif_path = one(trial_dir, "agent/fabric-artifacts/**/*.atif.json")
        promoted_path = trial_dir / "agent" / "trajectory.json"
        atif = load_json(atif_path)
        assert atif["schema_version"] == validation["atif"]["schema_version"]
        assert len(atif["steps"]) == validation["atif"]["steps"]
        assert atif_path.read_bytes() == promoted_path.read_bytes()

        summary["relay"] = {
            "atof_records": len(records),
            "categories": dict(categories),
            "atif_schema_version": atif["schema_version"],
            "atif_steps": len(atif["steps"]),
            "relay_config_version": 1,
            "observability_config_version": 3,
            "root_invocations": 1,
        }

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("job_dir", type=Path)
    parser.add_argument("--require-relay", action="store_true")
    parser.add_argument(
        "--allow-zero-reward",
        action="store_true",
        help="accept a completed run whose verifier reward is 0.0",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            verify(
                args.job_dir,
                require_relay=args.require_relay,
                allow_zero_reward=args.allow_zero_reward,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
