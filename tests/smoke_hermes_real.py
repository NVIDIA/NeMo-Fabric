"""Opt-in smoke test for the real Hermes SDK adapter path."""

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
    if os.environ.get("RUN_FABRIC_HERMES_INTEGRATION") != "1":
        print("skipping: set RUN_FABRIC_HERMES_INTEGRATION=1 to run")
        return
    if not os.environ.get("NVIDIA_API_KEY"):
        raise SystemExit("NVIDIA_API_KEY is required")
    env = os.environ.copy()
    env["HERMES_PYTHON"] = resolve_hermes_python(env)

    agent = ROOT / "examples" / "code-review-agent"
    with tempfile.TemporaryDirectory(prefix="fabric-hermes-real-") as tmpdir:
        temp_agent = Path(tmpdir) / "code-review-agent"
        copytree(agent, temp_agent)
        result = call_json(
            "run",
            temp_agent,
            "--profile",
            "hermes_real",
            "--input",
            "Reply with exactly: hermes ok",
            env=env,
        )
        assert_hermes_config_mapping(result["output"])

    assert result["status"] == "succeeded", result
    assert result["adapter_kind"] == "python"
    assert result["output"]["mode"] == "hermes_sdk"
    assert result["output"]["failed"] is False
    response = (result["output"].get("response") or "").lower()
    assert "hermes ok" in response, response


def assert_hermes_config_mapping(output: dict) -> None:
    config_path = Path(output["hermes_config_path"])
    assert config_path.is_file()
    config = yaml.safe_load(config_path.read_text())
    assert config["model"]["provider"] == "nvidia"
    assert config["model"]["default"] == "nvidia/nemotron-3-nano-30b-a3b"
    assert config["skills"]["external_dirs"]
    assert "github" in config["mcp_servers"]
    assert config["platform_toolsets"]["cli"] == []
    assert output["hermes_native_config"]["mcp_servers"] == ["github"]


def resolve_hermes_python(env: dict[str, str]) -> str:
    configured = env.get("HERMES_PYTHON")
    if configured:
        check_run_agent_import(configured)
        return configured
    try:
        check_run_agent_import(sys.executable)
        return sys.executable
    except RuntimeError:
        check_run_agent_import("python3")
        return "python3"


def check_run_agent_import(python: str) -> None:
    completed = subprocess.run(
        [python, "-c", "import run_agent"],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Hermes SDK import failed for `{python}`. Set HERMES_PYTHON to a Python "
            "environment where Hermes is installed.\n"
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
