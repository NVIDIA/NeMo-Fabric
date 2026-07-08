# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke test for ephemeral CLI config overrides."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COMMAND = ("cargo", "run", "-q", "-p", "fabric-cli", "--")
def test_cli_overrides(code_review_agent_dir: Path):
    inspected = _json_command(
        "inspect",
        str(code_review_agent_dir),
        "--profile",
        "hermes_sdk",
        "--set",
        "telemetry.enabled=true",
        "--set",
        "telemetry.output_dir=./artifacts/cli-set",
        "--set",
        'mcp.servers.github.exposure="fabric_managed"',
        "--set",
        'metadata.name="overridden-agent"',
    )
    config = inspected["config"]
    assert inspected["agent_name"] == "overridden-agent"
    assert config["metadata"]["name"] == "overridden-agent"
    assert config["telemetry"]["enabled"] is True
    assert config["telemetry"]["output_dir"] == "./artifacts/cli-set"
    assert config["mcp"]["servers"]["github"]["exposure"] == "fabric_managed"

    plan = _json_command(
        "plan",
        str(code_review_agent_dir),
        "--profile",
        "hermes_sdk",
        "--set",
        "telemetry.enabled=true",
        "--set",
        "telemetry.output_dir=./artifacts/cli-set",
    )
    assert plan["telemetry_plan"]["relay_enabled"] is True
    assert plan["telemetry_plan"]["relay_output_dir"].endswith("artifacts/cli-set")


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
