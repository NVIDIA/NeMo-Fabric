#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in smoke: langchain-react + Relay ATIF against platform Qwen IGW."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMAND = ("cargo", "run", "-q", "-p", "fabric-cli", "--")
AGENT = ROOT / "examples" / "react-optimize-agent"


def main() -> None:
    if os.environ.get("RUN_FABRIC_LANGCHAIN_REACT_ATIF_E2E") != "1":
        print("skipping: set RUN_FABRIC_LANGCHAIN_REACT_ATIF_E2E=1 to run")
        return

    env = _python_env()
    result = call_json(
        "run",
        AGENT,
        "--profile",
        "qwen-igw-local",
        "--profile",
        "calculator-native-tools",
        "--profile",
        "relay",
        "--input",
        "What is 12 multiplied by 8? Use the calculator tool and give the final numeric answer.",
        env=env,
    )

    assert result["status"] == "succeeded", result
    assert result.get("telemetry", {}).get("relay_enabled") is True, result.get("telemetry")
    output = result.get("output") or {}
    assert output.get("failed") is False, output
    assert output.get("harness") == "langchain-react", output
    assert "96" in (output.get("response") or "").lower(), output

    relay_artifacts = output.get("relay_artifacts") or []
    kinds = {artifact["kind"] for artifact in relay_artifacts}
    assert {"atof", "atif"} <= kinds, relay_artifacts

    manifest_kinds = {
        artifact["kind"]
        for artifact in (result.get("artifacts") or {}).get("artifacts", [])
        if str(artifact.get("name", "")).startswith("relay_")
    }
    assert {"atof", "atif"} <= manifest_kinds, result.get("artifacts")

    atif_paths = [Path(a["path"]) for a in relay_artifacts if a["kind"] == "atif"]
    assert atif_paths and all(path.exists() for path in atif_paths), atif_paths

    trajectory = json.loads(atif_paths[0].read_text(encoding="utf-8"))
    assert trajectory.get("steps"), trajectory
    print(f"ATIF steps: {len(trajectory['steps'])}")
    print(f"ATIF path: {atif_paths[0]}")
    print("langchain-react qwen ATIF smoke passed")


def _python_env() -> dict[str, str]:
    env = os.environ.copy()
    venv_python = ROOT / ".venv" / "bin" / "python"
    if venv_python.is_file():
        env["FABRIC_LANGCHAIN_PYTHON"] = str(venv_python)
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(ROOT / "adapters" / "langchain-react" / "src"),
            str(ROOT / "adapters" / "common" / "src"),
            env.get("PYTHONPATH", ""),
        ]
    ).strip(os.pathsep)
    return env


def call_json(*args: str, env: dict[str, str] | None = None) -> dict:
    completed = subprocess.run(
        [*COMMAND, *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"fabric command failed ({completed.returncode}): {completed.stderr.strip() or completed.stdout}"
        )
    return json.loads(completed.stdout)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"FAILED: {error}", file=sys.stderr)
        raise
