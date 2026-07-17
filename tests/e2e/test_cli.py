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
    assert {"scripted", "hermes", "claude", "codex", "deepagents"}.issubset(
        presets.stdout.splitlines()
    )
    assert examples.returncode == 0, examples.stderr
    assert "examples.code_review_agent" in examples.stdout.splitlines()

    shown = call_json("example", "show", "examples.code_review_agent")
    assert shown["default_variant"] == "hermes"
    assert set(shown["variants"]) == {"hermes", "claude", "codex", "deepagents"}
    assert shown["init_supported"] is True

    scripted = call_json("preset", "show", "scripted")
    assert scripted["available"] is True
    assert scripted["required_env"] == []
    assert scripted["missing_env"] == []
    assert scripted["install"] == "pip install nemo-fabric"


def test_scripted_preset_runs_without_credentials(tmp_path: Path) -> None:
    input_file = tmp_path / "input.txt"
    input_file.write_text("credential-free smoke", encoding="utf-8")

    result = call_json("run", "--preset", "scripted", "--input-file", input_file)

    assert result["status"] == "succeeded"
    assert result["adapter_id"] == "nvidia.fabric.scripted"
    assert result["output"]["response"] == "credential-free smoke"


def test_help_and_version_are_available() -> None:
    help_result = run()
    version = run("version")

    assert help_result.returncode == 0
    assert "experimentation interface" in help_result.stdout
    assert version.returncode == 0
    assert version.stdout.strip()


def test_example_can_be_copied_and_run_as_a_factory(tmp_path: Path) -> None:
    destination = tmp_path / "my_agent"
    initialized = run("example", "init", "examples.code_review_agent", destination)

    assert initialized.returncode == 0, initialized.stderr
    assert (destination / "config.py").is_file()
    assert (destination / "adapters" / "scripted" / "fabric-adapter.json").is_file()

    result = call_json_from(
        tmp_path,
        "run",
        "--factory",
        "my_agent.config:build_config",
        "--base-dir",
        destination,
        "--input",
        "review this",
    )
    assert result["status"] == "succeeded"
    assert result["output"]["response"] == "review this"


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
    misplaced_variant = run("plan", "--preset", "scripted", "--variant", "hermes")

    assert unknown.returncode == 2
    assert "available:" in unknown.stderr
    assert malformed.returncode == 2
    assert "module:callable" in malformed.stderr
    assert conflict.returncode == 2
    assert "not allowed with argument" in conflict.stderr
    assert misplaced_variant.returncode == 2
    assert "--variant requires --example" in misplaced_variant.stderr


def call_json(*args: Any) -> dict[str, Any]:
    return call_json_from(ROOT, *args)


def call_json_from(cwd: Path, *args: Any) -> dict[str, Any]:
    completed = run_from(cwd, *args)
    assert completed.returncode == 0, completed.stderr
    value = json.loads(completed.stdout)
    assert isinstance(value, dict)
    return value


def run(*args: Any) -> subprocess.CompletedProcess[str]:
    return run_from(ROOT, *args)


def run_from(cwd: Path, *args: Any) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    python_path = [str(ROOT / "python" / "src"), str(ROOT / "tests")]
    if env.get("PYTHONPATH"):
        python_path.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_path)
    return subprocess.run(
        [sys.executable, "-m", "nemo_fabric.cli", *(str(arg) for arg in args)],
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
