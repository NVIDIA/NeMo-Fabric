# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Installed-package bridge coverage for the Rust experimentation CLI."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "nemo_fabric.cli", *args],
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )


def test_package_bridge_lists_rust_catalogs():
    presets = run("preset", "list")
    examples = run("example", "show", "code-review")

    assert presets.returncode == 0, presets.stderr
    assert "scripted" in presets.stdout
    assert "hermes" in presets.stdout
    assert examples.returncode == 0, examples.stderr
    assert "variants: scripted, hermes, claude, codex, deepagents" in examples.stdout


def test_package_bridge_plans_preset_and_example():
    preset = run("plan", "--preset", "scripted")
    example = run("plan", "--example", "code-review", "--variant", "hermes")

    assert preset.returncode == 0, preset.stderr
    assert json.loads(preset.stdout)["agent_name"] == "scripted-agent"
    assert example.returncode == 0, example.stderr
    plan = json.loads(example.stdout)
    assert plan["agent_name"] == "code-review-agent"
    assert plan["config"]["harness"]["adapter_id"] == "nvidia.fabric.hermes"


def test_package_bridge_runs_the_deterministic_preset():
    completed = run("run", "--preset", "scripted", "--input", "package-smoke")

    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["status"] == "succeeded"
    assert result["output"]["response"] == "package-smoke"


def test_package_bridge_generates_editable_languages(tmp_path: Path):
    for language in ("python", "rust"):
        destination = tmp_path / language
        completed = run(
            "example",
            "init",
            "code-review",
            str(destination),
            "--language",
            language,
            "--variant",
            "hermes",
        )
        assert completed.returncode == 0, completed.stderr
        assert (destination / "repo" / "calculator.py").is_file()
        assert (destination / "skills" / "code-review.md").is_file()
        launcher = destination / ("main.py" if language == "python" else "src/main.rs")
        assert "nvidia.fabric.hermes" in launcher.read_text(encoding="utf-8")


def test_cli_rejects_removed_configuration_sources():
    for removed in ("--config", "--profile", "--factory"):
        completed = run("plan", removed, "value")
        assert completed.returncode != 0
        assert "unexpected argument" in completed.stderr
