# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke test for a Hermes SWE-Bench-style run and patch artifact capture."""

from __future__ import annotations

import subprocess
from pathlib import Path

from _utils.configs import swebench_shim_config
from nemo_fabric import Fabric


async def test_swebench_style(hermes_shim_agent_dir: Path):
    workspace = hermes_shim_agent_dir / "repos" / "my-service"
    run_command(workspace, "git", "init", "-q")
    run_command(workspace, "git", "add", "calculator.py")

    result = (
        await Fabric().run(
            swebench_shim_config(),
            base_dir=hermes_shim_agent_dir,
            input="Fix the bug so answer() returns 42.",
        )
    ).to_mapping()

    assert result["status"] == "succeeded", result
    assert result["adapter_kind"] == "python"
    assert result["output"]["mode"] == "swebench_shim"
    assert result["output"]["changed"] is True

    artifacts = result["artifacts"]["artifacts"]
    patch_artifacts = [
        artifact for artifact in artifacts if artifact["name"] == "workspace_patch"
    ]
    assert len(patch_artifacts) == 1
    patch = Path(patch_artifacts[0]["path"]).read_text()
    assert "-    return 41" in patch
    assert "+    return 42" in patch
    assert "generated/fix-notes.txt" in patch
    assert "+patched by Fabric" in patch

    status_artifacts = [
        artifact for artifact in artifacts if artifact["name"] == "workspace_status"
    ]
    assert len(status_artifacts) == 1
    assert "calculator.py" in Path(status_artifacts[0]["path"]).read_text()
def run_command(cwd: Path, *command: str) -> None:
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"command failed: {command}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
