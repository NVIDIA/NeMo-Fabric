"""Smoke a Fabric run against a Harbor-generated SWE-Bench task directory.

This smoke is opt-in because it needs Docker and a local SWE-Bench image. It
keeps Harbor responsible for task materialization and verification while Fabric
is responsible for invoking the selected harness profile and collecting a patch.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from shutil import copytree, rmtree

ROOT = Path(__file__).resolve().parents[1]
HARBOR_ROOT = ROOT.parent / "harbor"
DEFAULT_TASK = HARBOR_ROOT / "datasets" / "swebench-opencode-smoke" / "django__django-13741"
IMAGE = "swebench/sweb.eval.x86_64.django_1776_django-13741:latest"
COMMAND = ("cargo", "run", "-q", "-p", "fabric-cli", "--")
RUN_ENV = "RUN_FABRIC_HARBOR_SWEBENCH_DOCKER"
VERIFY_ENV = "RUN_FABRIC_HARBOR_SWEBENCH_VERIFY"


def main() -> None:
    if os.environ.get(RUN_ENV) != "1":
        print(f"skipped; set {RUN_ENV}=1 to run the Docker-backed SWE-Bench smoke")
        return

    task_dir = Path(os.environ.get("FABRIC_HARBOR_SWEBENCH_TASK", DEFAULT_TASK))
    if not task_dir.exists():
        raise AssertionError(f"Harbor SWE-Bench task directory not found: {task_dir}")

    assert_docker_image()
    scratch_root = ROOT / ".tmp"
    scratch_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fabric-harbor-swebench-", dir=scratch_root) as tmpdir:
        temp_agent = Path(tmpdir) / "hermes-shim-agent"
        copytree(ROOT / "tests" / "fixtures" / "hermes-shim-agent", temp_agent)
        rmtree(temp_agent / "artifacts", ignore_errors=True)

        workspace = temp_agent / "repos" / "swebench-django-13741"
        workspace.mkdir(parents=True)
        copy_testbed_from_image(workspace)
        assert_clean_workspace(workspace)

        request_file = temp_agent / "django-13741.request.json"
        request_file.write_text(json.dumps(build_request(task_dir), indent=2), encoding="utf-8")

        result = call_json(
            "run",
            temp_agent,
            "--profile",
            "harbor_swebench_django_13741",
            "--request-file",
            request_file,
        )

        assert result["status"] == "succeeded", result
        assert result["output"]["task"]["instance_id"] == "django__django-13741"
        assert result["output"]["changed"] is True

        patch = read_artifact(result, "workspace_patch")
        assert "django/contrib/auth/forms.py" in patch
        assert "kwargs.setdefault" in patch and "disabled" in patch, patch

        status = read_artifact(result, "workspace_status")
        assert "django/contrib/auth/forms.py" in status

        if os.environ.get(VERIFY_ENV) == "1":
            verify_with_harbor_task(task_dir, workspace, temp_agent / "artifacts" / "verifier")


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
    config = json.loads((task_dir / "tests" / "config.json").read_text(encoding="utf-8"))
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
    assert reward == "1", (logs / "verifier" / "report.json").read_text(encoding="utf-8")


def call_json(*args: object) -> dict:
    completed = run(*COMMAND, *(str(arg) for arg in args), cwd=ROOT)
    return json.loads(completed.stdout)


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


if __name__ == "__main__":
    main()
