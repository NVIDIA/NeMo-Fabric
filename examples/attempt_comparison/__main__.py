# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create synthetic task comparisons without harness or model calls."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Any

from nemo_fabric.comparisons import read_comparison_manifest, render_comparison_summary


def _usage(partial: bool) -> dict[str, Any]:
    return {
        "input_tokens": None if partial else 100,
        "cached_input_tokens": 0,
        "cache_write_input_tokens": 0,
        "output_tokens": 20,
        "total_tokens": None if partial else 120,
        "observed": {"input_tokens": 80} if partial else {},
        "models_exhaustive": False,
        "models": [],
        "gaps": [
            {
                "dimensions": ["reported_cost", "estimated_cost"],
                "reason": "missing_usage",
            },
            *(
                [
                    {
                        "dimensions": ["input_tokens", "total_tokens"],
                        "reason": "missing_usage",
                    }
                ]
                if partial
                else []
            ),
        ],
        "cost": {"reported": {}, "estimated": {}},
    }


def _write_log(path: Path, scenario: str) -> None:
    # Every time and count in this file is an illustrative fixture.
    started = datetime(2026, 10, 25, 19, 1, 10, tzinfo=timezone.utc)
    rows: list[dict[str, Any]] = []

    def timestamp(seconds: float) -> str:
        return (
            (started + timedelta(seconds=seconds))
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z")
        )

    def append(
        kind: str, seconds: float, data: dict[str, Any], *, has_invocation: bool = False
    ) -> None:
        rows.append(
            {
                "schema": "nemo.fabric.run",
                "version": "v1alpha1",
                "recording_id": f"recording-{scenario}",
                "request_id": scenario,
                "sequence": len(rows),
                "time": timestamp(seconds),
                "elapsed_seconds": seconds,
                "source": "fabric",
                "kind": kind,
                "data": data,
                **(
                    {"runtime_id": f"runtime-{scenario}"}
                    if kind != "request_started"
                    else {}
                ),
                **(
                    {"invocation_id": f"invocation-{scenario}"}
                    if has_invocation
                    else {}
                ),
            }
        )

    append(
        "request_started",
        0,
        {
            "harness": "example.fixture",
            "agent_name": "example",
            "models": {},
            "context": {},
            "override_keys": [],
            "outcome_scope": "one_shot_lifecycle",
        },
    )
    if scenario != "missing":
        append(
            "runtime_ready",
            1,
            {"runtime_id": f"runtime-{scenario}"},
        )
        append("invocation_started", 1.2, {"timeout_seconds": 300}, has_invocation=True)
        result = {
            "request_id": scenario,
            "agent_name": "synthetic-example",
            "harness": "example.fixture",
            "adapter_kind": "python",
            "runtime_id": f"runtime-{scenario}",
            "invocation_id": f"invocation-{scenario}",
            "status": "succeeded",
            "model_usage": _usage(scenario == "partial"),
            "artifacts": {"artifacts": []},
        }
        append(
            "invocation_finished",
            2.5,
            {
                "status": result["status"],
                "model_usage": result["model_usage"],
                "elapsed_seconds": 1.3,
                "adapter_exchange_elapsed_seconds": 1.3,
            },
            has_invocation=True,
        )
        if scenario != "interrupted":
            append(
                "runtime_stopped",
                2.8,
                {"succeeded": True, "error": None},
                has_invocation=True,
            )
            append(
                "summary",
                3,
                {
                    "outcome_scope": "one_shot_lifecycle",
                    "result": result,
                    "error": None,
                    "request_started_at": timestamp(0),
                    "request_finished_at": timestamp(3),
                    "elapsed_seconds": 3,
                    "adapter_exchange_elapsed_seconds": 1.3,
                },
                has_invocation=True,
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="New or empty output directory")
    args = parser.parse_args()
    root = args.directory
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        parser.error(
            "use a new or empty directory; existing evidence is never overwritten"
        )
    attempts = []
    for scenario in ("normal", "partial", "interrupted", "missing"):
        evidence = f"{scenario}/run.jsonl"
        _write_log(root / evidence, scenario)
        attempts.append(
            {
                "attempt_id": scenario,
                "task_id": "review-my-change",
                "task_revision": "v1",
                "configuration_id": "baseline" if scenario == "normal" else "candidate",
                "requests_complete": True,
                "requests": [{"request_id": scenario, "evidence": evidence}],
            }
        )
    manifest = {
        "schema": "nemo.fabric.comparison_manifest",
        "version": "v1alpha1",
        "inventory_complete": True,
        "attempts": attempts,
    }
    path = root / "manifest.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
    print(
        "Synthetic example with fictional timestamps and counts; no model calls or benchmark measurements."
    )
    print(render_comparison_summary(read_comparison_manifest(path)))


if __name__ == "__main__":
    main()
