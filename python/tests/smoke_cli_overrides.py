# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke test for ephemeral CLI config overrides."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COMMAND = ("cargo", "run", "-q", "-p", "fabric-cli", "--")
AGENT = ROOT / "examples" / "code-review-agent"


def main() -> None:
    inspected = _json_command(
        "inspect",
        str(AGENT),
        "--profile",
        "hermes_sdk",
        "--set",
        "telemetry.enabled=true",
        "--set",
        "telemetry.output_dir=./artifacts/cli-set",
        "--set",
        'mcp.servers.github.exposure="fabric_managed"',
    )
    config = inspected["config"]
    assert config["telemetry"]["enabled"] is True
    assert config["telemetry"]["output_dir"] == "./artifacts/cli-set"
    assert config["mcp"]["servers"]["github"]["exposure"] == "fabric_managed"

    plan = _json_command(
        "plan",
        str(AGENT),
        "--profile",
        "hermes_sdk",
        "--set",
        "telemetry.enabled=true",
        "--set",
        "telemetry.output_dir=./artifacts/cli-set",
    )
    assert plan["telemetry_plan"]["relay_enabled"] is True
    assert plan["telemetry_plan"]["relay_output_dir"].endswith("artifacts/cli-set")
    print("smoke_cli_overrides ok")


def _json_command(*args: str) -> dict:
    completed = subprocess.run(
        [*COMMAND, *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise AssertionError(completed.stderr)
    return json.loads(completed.stdout)


if __name__ == "__main__":
    main()
