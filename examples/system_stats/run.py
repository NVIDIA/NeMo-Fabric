#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capture native invocation stats for a credential-free resource probe."""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from nemo_fabric import Fabric, FabricConfig, RunResult


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/system-stats"))
    parser.add_argument(
        "--harbor-runner",
        action="store_true",
        help="Exercise the existing task-side Harbor runner transport",
    )
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    config = FabricConfig.from_mapping(
        {
            "metadata": {"name": "native-system-stats"},
            "harness": {
                "adapter_id": "example.stats_probe",
                "resolution": "preinstalled",
            },
            "discovery": {"local_paths": ["adapters"]},
            "runtime": {"artifacts": str(root / "artifacts")},
            "system_stats": {"sampling": {"interval_ms": 100, "max_samples": 1000}},
        }
    )
    path = root / "run-result.json"
    if args.harbor_runner:
        spec = root / "runner-spec.json"
        spec.write_text(
            json.dumps(
                {
                    "config": config.to_mapping(),
                    "config_base_dir": str(Path(__file__).parent.resolve()),
                    "logs_dir": str(root),
                    "request": {"input": "memory spike"},
                }
            )
        )
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "nemo_fabric.integrations.harbor.runner",
            "--spec",
            str(spec),
            "--result",
            str(path),
        )
        if await process.wait():
            raise RuntimeError(f"task runner failed; inspect {path}")
        result = RunResult.from_mapping(json.loads(path.read_text()))
    else:
        result = await Fabric().run(
            config, base_dir=Path(__file__).parent, input="memory spike"
        )
        path.write_text(json.dumps(result.to_mapping(), indent=2) + "\n")
    stats = result.system_stats
    if stats is None or stats.sampled is None:
        raise RuntimeError(f"no sampled system stats in {path}; status={result.status}")
    sampled = stats.sampled
    print(
        json.dumps(
            {
                "result": str(path),
                "sampled": sampled.to_mapping(),
                "cgroup": None if stats.cgroup is None else stats.cgroup.to_mapping(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
