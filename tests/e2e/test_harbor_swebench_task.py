# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke a Fabric run against a Harbor-generated SWE-Bench task directory.

This smoke is opt-in because it needs Docker and a local SWE-Bench image. It
keeps Harbor responsible for task materialization and verification while Fabric
is responsible for invoking the selected typed harness config and collecting a patch.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from _utils.configs import harbor_swebench_config
from nemo_fabric import (
    EnvironmentConfig,
    Fabric,
    FabricConfig,
    HarnessConfig,
    MetadataConfig,
    ModelConfig,
    RunRequest,
    RuntimeConfig,
)

ROOT = Path(__file__).resolve().parents[2]
HARBOR_ROOT = ROOT.parent / "harbor"
DEFAULT_TASK = (
    HARBOR_ROOT / "datasets" / "swebench-opencode-smoke" / "django__django-13741"
)
IMAGE = "swebench/sweb.eval.x86_64.django_1776_django-13741:latest"
RUN_ENV = "RUN_FABRIC_HARBOR_SWEBENCH_DOCKER"
VERIFY_ENV = "RUN_FABRIC_HARBOR_SWEBENCH_VERIFY"
MINI_SWE_AGENT_RUN_ENV = "RUN_FABRIC_MINI_SWE_AGENT_HARBOR_SWEBENCH"


@pytest.mark.usefixtures("requires_harbor")
async def test_harbor_swebench_task(hermes_shim_agent_dir: Path):
    if os.environ.get(RUN_ENV) != "1":
        pytest.skip(f"set {RUN_ENV}=1 to run the Docker-backed SWE-Bench test")

    task_dir = Path(os.environ.get("FABRIC_HARBOR_SWEBENCH_TASK", DEFAULT_TASK))
    if not task_dir.exists():
        raise AssertionError(f"Harbor SWE-Bench task directory not found: {task_dir}")

    assert_docker_image()

    workspace = hermes_shim_agent_dir / "repos" / "swebench-django-13741"
    workspace.mkdir(parents=True)
    copy_testbed_from_image(workspace)
    assert_clean_workspace(workspace)

    result = (
        await Fabric().run(
            harbor_swebench_config(),
            base_dir=hermes_shim_agent_dir,
            request=RunRequest.from_mapping(build_request(task_dir)),
        )
    ).to_mapping()

    assert result["status"] == "succeeded", result
    assert result["output"]["task"]["instance_id"] == "django__django-13741"
    assert result["output"]["changed"] is True

    patch = read_artifact(result, "workspace_patch")
    assert "django/contrib/auth/forms.py" in patch
    assert "kwargs.setdefault" in patch and "disabled" in patch, patch

    status = read_artifact(result, "workspace_status")
    assert "django/contrib/auth/forms.py" in status

    if os.environ.get(VERIFY_ENV) == "1":
        verify_with_harbor_task(
            task_dir,
            workspace,
            hermes_shim_agent_dir / "artifacts" / "verifier",
        )


@pytest.mark.usefixtures("requires_harbor")
async def test_mini_swe_agent_harbor_swebench_task(hermes_shim_agent_dir: Path):
    if os.environ.get(MINI_SWE_AGENT_RUN_ENV) != "1":
        pytest.skip(f"set {MINI_SWE_AGENT_RUN_ENV}=1 to run the Harbor SWE-Bench smoke")
    if not os.environ.get("NVIDIA_API_KEY"):
        pytest.fail("NVIDIA_API_KEY is required for the mini-SWE-agent Harbor smoke")

    task_dir = Path(os.environ.get("FABRIC_HARBOR_SWEBENCH_TASK", DEFAULT_TASK))
    if not task_dir.exists():
        raise AssertionError(f"Harbor SWE-Bench task directory not found: {task_dir}")
    assert_docker_image()

    workspace = hermes_shim_agent_dir / "repos" / "swebench-django-13741"
    workspace.mkdir(parents=True)
    copy_testbed_from_image(workspace)
    assert_clean_workspace(workspace)
    model = os.environ.get(
        "FABRIC_MINI_SWE_AGENT_MODEL",
        "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
    )
    result = (
        await Fabric().run(
            FabricConfig(
                metadata=MetadataConfig(name="harbor-mini-swe-agent"),
                harness=HarnessConfig(
                    adapter_id="nvidia.fabric.mini-swe-agent",
                    resolution="preinstalled",
                    settings={"timeout_seconds": 180},
                ),
                models={
                    "default": ModelConfig(
                        provider=model.split("/", maxsplit=1)[0],
                        model=model,
                        api_key_env="NVIDIA_API_KEY",
                        base_url="https://integrate.api.nvidia.com/v1",
                    )
                },
                runtime=RuntimeConfig(
                    input_schema="harbor_swe_bench_task",
                    output_schema="message",
                    artifacts="./artifacts/mini-swe-agent",
                    max_turns=75,
                    timeout_seconds=1800,
                ),
                environment=EnvironmentConfig(
                    provider="local",
                    workspace=workspace,
                    artifacts="./artifacts/mini-swe-agent",
                ),
            ),
            base_dir=ROOT,
            request=RunRequest.from_mapping(build_request(task_dir)),
        )
    ).to_mapping()

    assert result["status"] == "succeeded", result


def assert_docker_image() -> None:
    run("docker", "image", "inspect", IMAGE)


def copy_testbed_from_image(workspace: Path) -> None:
    run(
        "docker",
        "run",
        "--rm",
        "-v",
        f"{workspace}:/workspace",
        IMAGE,
        "/bin/bash",
        "-lc",
        f"cp -R /testbed/. /workspace/ && chown -R {os.getuid()}:{os.getgid()} /workspace",
    )


def assert_clean_workspace(workspace: Path) -> None:
    completed = run("git", "-C", workspace, "status", "--short")
    assert completed.stdout.strip() == "", completed.stdout


def build_request(task_dir: Path) -> dict:
    config = json.loads(
        (task_dir / "tests" / "config.json").read_text(encoding="utf-8")
    )
    return {
        "request_id": config["instance_id"],
        "input": (task_dir / "instruction.md").read_text(encoding="utf-8"),
        "context": {
            "task": {
                "source": "harbor_swebench",
                "task_dir": str(task_dir),
                "instance_id": config["instance_id"],
                "repo": config["repo"],
                "base_commit": config["base_commit"],
                "difficulty": config.get("difficulty"),
            },
            "swebench": {
                "FAIL_TO_PASS": config["FAIL_TO_PASS"],
                "PASS_TO_PASS": config["PASS_TO_PASS"],
            },
        },
    }


def read_artifact(result: dict, name: str) -> str:
    matches = [
        artifact
        for artifact in result["artifacts"]["artifacts"]
        if artifact["name"] == name
    ]
    assert len(matches) == 1, result["artifacts"]
    return Path(matches[0]["path"]).read_text(encoding="utf-8")


def verify_with_harbor_task(task_dir: Path, workspace: Path, logs: Path) -> None:
    (logs / "verifier").mkdir(parents=True, exist_ok=True)
    try:
        run(
            "docker",
            "run",
            "--rm",
            "-v",
            f"{workspace}:/testbed",
            "-v",
            f"{task_dir / 'tests'}:/tests:ro",
            "-v",
            f"{logs}:/logs",
            IMAGE,
            "/bin/bash",
            "-lc",
            "git config --global --add safe.directory /testbed && /bin/bash /tests/test.sh",
        )
    finally:
        run(
            "docker",
            "run",
            "--rm",
            "-v",
            f"{workspace}:/workspace",
            "-v",
            f"{logs}:/logs",
            IMAGE,
            "/bin/bash",
            "-lc",
            f"chown -R {os.getuid()}:{os.getgid()} /workspace /logs",
        )
    reward = (logs / "verifier" / "reward.txt").read_text(encoding="utf-8").strip()
    assert reward == "1", (logs / "verifier" / "report.json").read_text(
        encoding="utf-8"
    )


def run(*command: object, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        [str(part) for part in command],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"command failed: {completed.args}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed
