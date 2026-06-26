# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke test for the maintained Hermes CLI adapter."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from shutil import copytree

ROOT = Path(__file__).resolve().parents[1]
COMMAND = ("cargo", "run", "-q", "-p", "fabric-cli", "--")


def main() -> None:
    fixture_agent = ROOT / "tests" / "fixtures" / "hermes-cli-agent"
    with tempfile.TemporaryDirectory(prefix="fabric-hermes-cli-") as tmpdir:
        temp_agent = Path(tmpdir) / "hermes-cli-agent"
        copytree(fixture_agent, temp_agent)

        plan = call_json("plan", temp_agent, "--profile", "env_local")
        assert plan["adapter_descriptor"]["descriptor"]["adapter_id"] == "nvidia.fabric.hermes.cli"
        assert plan["adapter_descriptor"]["descriptor"]["adapter_kind"] == "process"
        assert plan["adapter_descriptor"]["source"] == "repository"

        result = call_json("run", temp_agent, "--profile", "env_local", "--input", "hello cli")
        assert result["status"] == "succeeded"
        assert result["adapter_kind"] == "process"
        assert result["metadata"]["adapter_runner"] == "process"
        assert result["output"]["harness"] == "hermes"
        assert result["output"]["adapter"] == "cli"
        assert result["output"]["mode"] == "hermes_cli_oneshot"
        assert Path(result["output"]["fabric_invocation"]).is_file()
        assert result["output"]["hermes_native_config"]["mcp_servers"] == ["github"]
        assert result["output"]["hermes_native_config"]["skill_dirs"]

        response = json.loads(result["output"]["response"])
        assert response["fake_hermes"] is True
        assert response["prompt"] == "hello cli"
        assert "-z" in response["argv"]
        assert "--model" in response["argv"]
        assert "test-model" in response["argv"]

        config_path = Path(result["output"]["hermes_config_path"])
        assert config_path.is_file()
        assert_relay_disabled_native_observability(result)


def call_json(*args: object) -> dict:
    completed = subprocess.run(
        [*COMMAND, *(str(arg) for arg in args)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"command failed: {completed.args}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return json.loads(completed.stdout)


def assert_relay_disabled_native_observability(result: dict) -> None:
    artifact_by_name = {
        artifact["name"]: artifact
        for artifact in result["artifacts"]["artifacts"]
    }
    assert "stdout" in artifact_by_name
    assert "relay_config" not in artifact_by_name
    assert not any(name.startswith("relay_") for name in artifact_by_name)

    stdout_path = Path(artifact_by_name["stdout"]["path"])
    assert stdout_path.is_file()
    assert stdout_path.read_text(encoding="utf-8").strip()

    event_kinds = {event["kind"] for event in result["events"]}
    assert {"runtime_start", "invocation_start", "invocation_end"} <= event_kinds

    assert result["telemetry"]["relay_enabled"] is False
    assert result["output"]["returncode"] == 0
    assert result["output"]["stderr"] == ""


if __name__ == "__main__":
    main()
