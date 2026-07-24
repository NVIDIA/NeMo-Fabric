# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
from pathlib import Path
import subprocess
import sys


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
