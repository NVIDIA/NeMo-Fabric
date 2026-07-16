# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free local environment e2e smoke."""

from __future__ import annotations

import json
from pathlib import Path

from _utils.utils import run_fabric_cli


def test_local_env_e2e(hermes_shim_agent_dir: Path):
    selector = (
        "--factory",
        "_utils.configs:hermes_shim_config",
        "--base-dir",
        hermes_shim_agent_dir,
    )
    plan = call_json("plan", *selector)
    assert plan["environment_plan"]["provider"] == "local"
    assert plan["environment_plan"]["workspace"].endswith("repos/my-service")
    assert plan["adapter_descriptor"]["source"] == "local"

    result = call_json(
        "run",
        *selector,
        "--request-json",
        json.dumps(
            {
                "request_id": "local-env-e2e",
                "input": "review local workspace",
                "context": {"source": "local-e2e"},
            }
        ),
    )

    assert result["status"] == "succeeded"
    assert result["request_id"] == "local-env-e2e"
    assert result["output"]["received"] == "review local workspace"
    assert result["output"]["workspace"].endswith("repos/my-service")
    assert result["artifacts"]["root"].endswith("artifacts")

    stdout = read_artifact(result, "stdout")
    assert "review local workspace" in stdout
    assert any(
        event["kind"] == "runtime_start"
        and event["metadata"]["environment_provider"] == "local"
        for event in result["events"]
    )


def call_json(*args: object) -> dict:
    completed = run_fabric_cli(*args)
    if completed.returncode != 0:
        raise AssertionError(
            f"command failed: {completed.args}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return json.loads(completed.stdout)


def read_artifact(result: dict, name: str) -> str:
    matches = [
        artifact
        for artifact in result["artifacts"]["artifacts"]
        if artifact["name"] == name
    ]
    assert len(matches) == 1, result["artifacts"]
    return Path(matches[0]["path"]).read_text(encoding="utf-8")
