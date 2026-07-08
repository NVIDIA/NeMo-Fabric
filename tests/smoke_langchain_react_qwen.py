#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in E2E smoke for langchain-react against a platform Qwen IGW deployment."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMAND = ("cargo", "run", "-q", "-p", "fabric-cli", "--")
AGENT = ROOT / "examples" / "react-optimize-agent"
PROFILE = "qwen-igw-local"


def main() -> None:
    if os.environ.get("RUN_FABRIC_LANGCHAIN_REACT_E2E") != "1":
        print("skipping: set RUN_FABRIC_LANGCHAIN_REACT_E2E=1 to run")
        return

    base_url = os.environ.get(
        "FABRIC_QWEN_BASE_URL",
        "http://10.0.0.51:8080/apis/inference-gateway/v2/workspaces/default/openai/-/v1",
    )
    model = os.environ.get("FABRIC_QWEN_MODEL", "default/qwen3-8b")

  # Direct adapter smoke (no wiki network): calculator profile + simple math question.
    calc_result = call_json(
        "run",
        AGENT,
        "--profile",
        PROFILE,
        "--profile",
        "qwen-calculator-text",
        "--input",
        "What is 12 multiplied by 8? Use the calculator tool and give the final numeric answer.",
        env=_python_env(),
    )
    _assert_success(calc_result, expect_substrings=["96"])

    # Wiki + datetime agent smoke with a factual question (wikipedia may be slow/unavailable).
    react_result = call_json(
        "run",
        AGENT,
        "--profile",
        PROFILE,
        "--input",
        "In one short sentence, what is the capital of France?",
        env=_python_env(),
    )
    _assert_success(react_result, expect_substrings=["paris"])

    print("langchain-react qwen e2e smoke passed")


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


def _assert_success(result: dict, *, expect_substrings: list[str]) -> None:
    assert result["status"] == "succeeded", result
    assert result.get("adapter_kind") == "python", result
    output = result.get("output") or {}
    assert output.get("failed") is False, output
    assert output.get("harness") == "langchain-react", output
    response = (output.get("response") or "").lower()
    for needle in expect_substrings:
        assert needle in response, {"response": response, "expected": needle, "full": result}


def call_json(*args: str, env: dict[str, str] | None = None) -> dict:
    command = [*COMMAND, *args]
    completed = subprocess.run(
        command,
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
