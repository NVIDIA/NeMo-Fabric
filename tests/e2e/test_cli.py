# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end coverage for the SDK-backed ``nemo-fabric`` CLI."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def test_discovery_commands() -> None:
    presets = run("preset", "list")
    examples = run("example", "list")

    assert presets.returncode == 0, presets.stderr
    assert {"hermes", "claude", "codex", "deepagents"}.issubset(presets.stdout.splitlines())
    assert examples.returncode == 0, examples.stderr
    assert "examples.code_review_agent" in examples.stdout.splitlines()

    shown = call_json("example", "show", "examples.code_review_agent")
    assert shown["default_variant"] == "hermes"
    assert set(shown["variants"]) == {"hermes", "claude", "codex", "deepagents"}


def test_preset_and_example_reach_the_same_sdk_plan_path() -> None:
    preset = call_json("plan", "--preset", "hermes")
    example = call_json(
        "plan",
        "--example",
        "examples.code_review_agent",
        "--variant",
        "hermes",
    )

    assert preset["adapter_descriptor"]["descriptor"]["adapter_id"] == "nvidia.fabric.hermes"
    assert example["adapter_descriptor"]["descriptor"]["adapter_id"] == "nvidia.fabric.hermes"
    assert preset["effective_config"]["base_dir"].endswith("nemo_fabric/_bundled")
    assert example["capability_plan"]["native"]["skill_paths"]


def test_factory_plan_doctor_and_run(hermes_shim_agent_dir: Path) -> None:
    selector = (
        "--factory",
        "_utils.configs:hermes_shim_config",
        "--base-dir",
        hermes_shim_agent_dir,
    )
    plan = call_json("plan", *selector)
    doctor = call_json("doctor", *selector)
    result = call_json("run", *selector, "--input", "hello factory")

    assert plan["agent_name"] == "hermes-shim-agent"
    assert plan["adapter_descriptor"]["descriptor"]["adapter_id"] == "test.fabric.hermes_shim"
    assert doctor["agent_name"] == "hermes-shim-agent"
    assert doctor["checks"]
    assert result["status"] == "succeeded"
    assert result["output"]["received"] == "hello factory"


def test_factory_request_json_and_output_file(hermes_shim_agent_dir: Path, tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    request = json.dumps(
        {
            "input": "hello request",
            "request_id": "cli-request-1",
            "context": {"source": "test"},
        }
    )
    completed = run(
        "run",
        "--factory",
        "_utils.configs:hermes_shim_config",
        "--base-dir",
        hermes_shim_agent_dir,
        "--request-json",
        request,
        "--output",
        output,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["request_id"] == "cli-request-1"
    assert result["output"]["received"] == "hello request"


def test_selector_errors_are_actionable() -> None:
    unknown = run("plan", "--preset", "missing")
    malformed = run("plan", "--factory", "missing-factory")
    conflict = run("plan", "--preset", "hermes", "--factory", "mod:factory")

    assert unknown.returncode == 2
    assert "available:" in unknown.stderr
    assert malformed.returncode == 2
    assert "module:callable" in malformed.stderr
    assert conflict.returncode == 2
    assert "not allowed with argument" in conflict.stderr


def call_json(*args: Any) -> dict[str, Any]:
    completed = run(*args)
    assert completed.returncode == 0, completed.stderr
    value = json.loads(completed.stdout)
    assert isinstance(value, dict)
    return value


def run(*args: Any) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    python_path = [str(ROOT / "python" / "src"), str(ROOT / "tests")]
    if env.get("PYTHONPATH"):
        python_path.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_path)
    return subprocess.run(
        [sys.executable, "-m", "nemo_fabric.cli", *(str(arg) for arg in args)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
