# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke test for the maintained Hermes CLI adapter."""

from __future__ import annotations

import json
from pathlib import Path

from _utils.utils import assert_process_adapter_native_observability, run_fabric_cli


def test_hermes_cli(hermes_agent_dir: Path):
    plan = call_json("plan", hermes_agent_dir, "--profile", "env_local")
    assert (
        plan["adapter_descriptor"]["descriptor"]["adapter_id"]
        == "nvidia.fabric.hermes.cli"
    )
    assert plan["adapter_descriptor"]["descriptor"]["adapter_kind"] == "process"
    assert plan["adapter_descriptor"]["source"] == "repository"

    result = call_json(
        "run", hermes_agent_dir, "--profile", "env_local", "--input", "hello cli"
    )
    assert result["status"] == "succeeded"
    assert result["adapter_kind"] == "process"
    assert result["metadata"]["adapter_runner"] == "process"
    assert result["output"]["harness"] == "hermes"
    assert result["output"]["adapter"] == "cli"
    assert result["output"]["mode"] == "hermes_cli_runtime"
    assert Path(result["output"]["fabric_invocation"]).is_file()
    assert result["output"]["hermes_native_config"]["mcp_servers"] == ["github"]
    assert result["output"]["hermes_native_config"]["skill_dirs"]

    response = json.loads(result["output"]["response"])
    assert response["fake_hermes"] is True
    assert response["prompt"] == "hello cli"
    assert "chat" in response["argv"]
    assert "--quiet" in response["argv"]
    assert "--query" in response["argv"]
    assert "--model" in response["argv"]
    assert "test-model" in response["argv"]

    config_path = Path(result["output"]["hermes_config_path"])
    assert config_path.is_file()
    assert_process_adapter_native_observability(result)


def call_json(*args: object) -> dict:
    completed = run_fabric_cli(*args)
    if completed.returncode != 0:
        raise AssertionError(
            f"command failed: {completed.args}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return json.loads(completed.stdout)
