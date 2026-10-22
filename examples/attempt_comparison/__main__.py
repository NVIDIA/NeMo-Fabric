# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create a synthetic custom-task comparison; no harness or model calls."""

from __future__ import annotations

import argparse
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
    for name, configuration in [
        ("normal", "baseline"),
        ("partial", "candidate"),
        ("missing", "candidate"),
    ]:
        filename = f"{name}.json"
        if name != "missing":
            result = {
                "request_id": name,
                "runtime_id": f"runtime-{name}",
                "invocation_id": f"invocation-{name}",
                "status": "succeeded",
                "model_usage": _usage(name == "partial"),
                "artifacts": {"artifacts": []},
            }
            with (root / filename).open("x", encoding="utf-8") as stream:
                json.dump(result, stream, indent=2)
        attempts.append(
            {
                "attempt_id": name,
                "task_id": "review-my-change",
                "task_revision": "v1",
                "configuration_id": configuration,
                "requests_complete": True,
                "requests": [{"request_id": name, "evidence": filename}],
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
    print("Synthetic evidence; no model calls, measured latency or evaluator verdicts.")
    print(render_comparison_summary(read_comparison_manifest(path)))


if __name__ == "__main__":
    main()
