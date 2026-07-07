# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in smoke test for Hermes adapter Relay ATOF/ATIF emission."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from shutil import copytree

import yaml

ROOT = Path(__file__).resolve().parents[1]
COMMAND = ("cargo", "run", "-q", "-p", "fabric-cli", "--")


def main() -> None:
    if os.environ.get("RUN_FABRIC_RELAY_INTEGRATION") != "1":
        print("skipped: set RUN_FABRIC_RELAY_INTEGRATION=1 to run Relay integration smoke")
        return
    if not os.environ.get("NVIDIA_API_KEY"):
        raise SystemExit("NVIDIA_API_KEY is required")

    env = os.environ.copy()
    env["HERMES_PYTHON"] = resolve_hermes_python(env)

    agent = ROOT / "examples" / "code-review-agent"
    with tempfile.TemporaryDirectory(prefix="fabric-relay-smoke-") as tmpdir:
        temp_agent = Path(tmpdir) / "code-review-agent"
        copytree(agent, temp_agent)

        result = call_json(
            "run",
            temp_agent,
            "--profile",
            "hermes_sdk",
            "--profile",
            "relay",
            "--input",
            "Reply with exactly: relay ok",
            env=env,
        )
        assert result["status"] == "succeeded"
        assert result["telemetry"]["relay_enabled"] is True
        assert "relay_mode" not in result["telemetry"]["metadata"]
        assert result["output"]["mode"] == "hermes_sdk"
        assert result["output"]["relay_runtime"]["emitter"] == "hermes.observability/nemo_relay"
        assert_hermes_config_mapping(result["output"])

        relay_config_artifacts = [
            artifact
            for artifact in result["artifacts"]["artifacts"]
            if artifact["name"] == "relay_config"
        ]
        assert len(relay_config_artifacts) == 1
        relay_config_path = Path(relay_config_artifacts[0]["path"])
        relay_config = json.loads(relay_config_path.read_text())
        assert relay_config["schema_version"] == "fabric.relay/v1alpha1"
        assert relay_config["relay"]["enabled"] is True

        relay_artifacts = result["output"]["relay_artifacts"]
        kinds = {artifact["kind"] for artifact in relay_artifacts}
        assert {"atof", "atif"} <= kinds
        manifest_relay_kinds = {
            artifact["kind"]
            for artifact in result["artifacts"]["artifacts"]
            if artifact["name"].startswith("relay_")
        }
        assert {"atof", "atif"} <= manifest_relay_kinds

        atof_paths = [Path(artifact["path"]) for artifact in relay_artifacts if artifact["kind"] == "atof"]
        atif_paths = [Path(artifact["path"]) for artifact in relay_artifacts if artifact["kind"] == "atif"]
        assert atof_paths and atif_paths
        assert all(path.exists() for path in atof_paths + atif_paths)

        atof_lines = atof_paths[0].read_text().strip().splitlines()
        assert len(atof_lines) >= 3
        assert any("hermes" in json.loads(line).get("name", "") for line in atof_lines)

        trajectory = json.loads(atif_paths[0].read_text())
        assert trajectory["agent"]["name"] in {"code-review-agent", "Hermes Agent"}
        assert trajectory["steps"]


def assert_hermes_config_mapping(output: dict) -> None:
    config_path = Path(output["hermes_config_path"])
    assert config_path.is_file()
    config = yaml.safe_load(config_path.read_text())
    assert config["model"]["provider"] == "nvidia"
    assert config["model"]["default"] == "nvidia/nemotron-3-nano-30b-a3b"
    assert config["skills"]["external_dirs"]
    assert "github" in config["mcp_servers"]
    assert config["platform_toolsets"]["cli"] == []
    assert config["plugins"]["enabled"] == ["observability/nemo_relay"]
    assert output["hermes_native_config"]["plugins"] == ["observability/nemo_relay"]


def resolve_hermes_python(env: dict[str, str]) -> str:
    configured = env.get("HERMES_PYTHON")
    if configured:
        check_hermes_relay_imports(configured)
        return configured
    try:
        check_hermes_relay_imports(sys.executable)
        return sys.executable
    except RuntimeError:
        check_hermes_relay_imports("python3")
        return "python3"


def check_hermes_relay_imports(python: str) -> None:
    env = os.environ.copy()
    with tempfile.TemporaryDirectory(prefix="fabric-hermes-import-") as tmpdir:
        env["HERMES_HOME"] = tmpdir
        completed = subprocess.run(
            [python, "-c", "import nemo_relay, run_agent"],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Hermes + Relay import failed for `{python}`. Set HERMES_PYTHON to "
            "a Python environment with local Hermes and NeMo Relay installed.\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )


def call_json(*args: object, env: dict[str, str] | None = None) -> dict:
    completed = subprocess.run(
        [*COMMAND, *(str(arg) for arg in args)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"command failed: {completed.args}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return json.loads(completed.stdout)


if __name__ == "__main__":
    main()
