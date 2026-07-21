# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free local environment e2e smoke."""

from __future__ import annotations

from pathlib import Path

from _utils.configs import hermes_shim_config
from nemo_fabric import Fabric, RunRequest


async def test_local_env_e2e(hermes_shim_agent_dir: Path):
    fabric = Fabric()
    config = hermes_shim_config()
    plan = fabric.plan(config, base_dir=hermes_shim_agent_dir).to_mapping()
    assert plan["environment_plan"]["provider"] == "local"
    assert plan["environment_plan"]["workspace"].endswith("repos/my-service")
    assert plan["adapter_descriptor"]["source"] == "local"

    result = (
        await fabric.run(
            config,
            base_dir=hermes_shim_agent_dir,
            request=RunRequest(
                request_id="local-env-e2e",
                input="review local workspace",
                context={"source": "local-e2e"},
            ),
        )
    ).to_mapping()

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
def read_artifact(result: dict, name: str) -> str:
    matches = [
        artifact
        for artifact in result["artifacts"]["artifacts"]
        if artifact["name"] == name
    ]
    assert len(matches) == 1, result["artifacts"]
    return Path(matches[0]["path"]).read_text(encoding="utf-8")
