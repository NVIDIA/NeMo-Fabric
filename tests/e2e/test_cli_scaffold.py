# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).parents[2]
SUBPROCESS_TIMEOUT_SECONDS = 120


def generate_scaffold(destination: Path, language: str) -> None:
    subprocess.run(
        [
            "cargo",
            "run",
            "--quiet",
            "-p",
            "nemo-fabric-cli",
            "--",
            "example",
            "init",
            "code-review",
            str(destination),
            "--language",
            language,
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT_SECONDS,
    )


def plan_preset(
    preset: str,
    frontier_endpoint: str | None = None,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["NVIDIA_API_KEY"] = "test-key"
    if frontier_endpoint is None:
        environment.pop("NVIDIA_FRONTIER_BASE_URL", None)
    else:
        environment["NVIDIA_FRONTIER_BASE_URL"] = frontier_endpoint
    return subprocess.run(
        [
            "cargo",
            "run",
            "--quiet",
            "-p",
            "nemo-fabric-cli",
            "--",
            "plan",
            "--preset",
            preset,
        ],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT_SECONDS,
    )


def planned_model_endpoint(result: subprocess.CompletedProcess[str]) -> str:
    model = json.loads(result.stdout)["config"]["models"]["default"]
    return model.get("base_url") or model["settings"]["base_url"]


def test_generated_python_scaffold_installs_editable(tmp_path: Path):
    destination = tmp_path / "python-agent"
    generate_scaffold(destination, "python")
    venv = tmp_path / "venv"
    subprocess.run(
        ["uv", "venv", "--seed", "--python", sys.executable, str(venv)],
        check=True,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT_SECONDS,
    )
    python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    environment = os.environ.copy()
    environment["PIP_NO_BUILD_ISOLATION"] = "1"
    environment["PIP_NO_DEPS"] = "1"

    subprocess.run(
        [str(python), "-m", "pip", "install", "-e", "."],
        cwd=destination,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT_SECONDS,
    )


def test_generated_rust_scaffold_builds(tmp_path: Path):
    destination = tmp_path / "rust-agent"
    generate_scaffold(destination, "rust")
    assert "path =" not in (destination / "Cargo.toml").read_text()
    environment = os.environ.copy()
    environment["CARGO_TARGET_DIR"] = str(ROOT / "target")
    core_path = json.dumps(str(ROOT / "crates/fabric-core"))

    subprocess.run(
        [
            "cargo",
            "build",
            "--config",
            f"patch.crates-io.nemo-fabric-core.path={core_path}",
            "--manifest-path",
            str(destination / "Cargo.toml"),
        ],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT_SECONDS,
    )


def test_failed_cli_run_returns_nonzero_status():
    environment = os.environ.copy()
    environment.pop("NVIDIA_API_KEY", None)
    result = subprocess.run(
        [
            "cargo",
            "run",
            "--quiet",
            "-p",
            "nemo-fabric-cli",
            "--",
            "run",
            "--preset",
            "hermes",
            "--input",
            "Say hello",
        ],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT_SECONDS,
    )

    assert result.returncode != 0
    failure = json.loads(result.stdout)
    assert failure["status"] == "failed"
    assert failure["error"]["code"] == "runtime_error"
    assert "adapter lifecycle start failed" in result.stderr


@pytest.mark.parametrize("preset", ["claude", "codex"])
def test_frontier_presets_require_endpoint_before_planning(preset: str):
    result = plan_preset(preset)

    assert result.returncode != 0
    assert not result.stdout
    assert preset in result.stderr
    assert "NVIDIA_FRONTIER_BASE_URL" in result.stderr


@pytest.mark.parametrize("preset", ["claude", "codex"])
def test_frontier_presets_plan_with_explicit_endpoint(preset: str):
    endpoint = "https://frontier.example/v1"
    result = plan_preset(preset, endpoint)

    assert result.returncode == 0, result.stderr
    assert planned_model_endpoint(result) == endpoint


@pytest.mark.parametrize("preset", ["hermes", "deepagents"])
def test_catalog_presets_keep_default_endpoint_without_frontier(preset: str):
    result = plan_preset(preset)

    assert result.returncode == 0, result.stderr
    assert planned_model_endpoint(result) == "https://integrate.api.nvidia.com/v1"
