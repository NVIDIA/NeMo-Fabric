# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from fnmatch import fnmatchcase
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
PUBLISH_WORKFLOW = REPO_ROOT / ".github/workflows/publish_typescript.yml"
TYPESCRIPT_CI_WORKFLOW = REPO_ROOT / ".github/workflows/ci_typescript.yml"
NIGHTLY_WORKFLOW = REPO_ROOT / ".github/workflows/nightly-alpha-tag.yml"


def _load_workflow(path: Path) -> dict:
    workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
    # PyYAML uses YAML 1.1 and parses GitHub's YAML 1.2 `on` key as `True`.
    workflow["on"] = workflow.pop(True)
    return workflow


def _tag_triggers_workflow(tag: str, patterns: list[str]) -> bool:
    matches = False
    for pattern in patterns:
        excluded = pattern.startswith("!")
        if fnmatchcase(tag, pattern.removeprefix("!")):
            matches = not excluded
    return matches


def test_publisher_only_triggers_for_public_release_channels():
    workflow = _load_workflow(PUBLISH_WORKFLOW)
    assert "workflow_call" not in workflow["on"]
    assert workflow["concurrency"] == {
        "group": "publish-typescript",
        "queue": "max",
        "cancel-in-progress": False,
    }

    tag_patterns = workflow["on"]["push"]["tags"]
    for tag in ("v0.3.0", "v0.3.0-beta.1", "v0.3.0-rc.1"):
        assert _tag_triggers_workflow(tag, tag_patterns)
    assert not _tag_triggers_workflow("v0.3.0-alpha.20260817", tag_patterns)
    for package in ("nemo-fabric-adapters-common", "nemo-fabric-adapters-pi"):
        for version in ("0.3.0", "0.3.0-beta.1", "0.3.0-rc.1"):
            assert _tag_triggers_workflow(f"npm/{package}/v{version}", tag_patterns)
        assert not _tag_triggers_workflow(
            f"npm/{package}/v0.3.0-alpha.20260817", tag_patterns
        )

    steps = {
        step["name"]: step
        for step in workflow["jobs"]["publish-typescript"]["steps"]
    }
    assert steps["Verify release tag"]["env"]["RELEASE_REF"] == "${{ github.ref }}"
    release_metadata = steps["Resolve package release"]["run"]
    assert '*-alpha*) dist_tag="alpha"' not in release_metadata
    assert '*-beta*|*-rc*) dist_tag="next"' in release_metadata
    assert '*) dist_tag="latest"' in release_metadata


def test_publisher_routes_packages_and_checks_adapter_dependencies():
    workflow = _load_workflow(PUBLISH_WORKFLOW)
    steps = {
        step["name"]: step
        for step in workflow["jobs"]["publish-typescript"]["steps"]
    }
    release = steps["Resolve package release"]["run"]
    assert 'package_name="nemo-fabric-adapter-contract"' in release
    assert 'package_directory="adapter-contract/typescript"' in release
    assert 'package_name="nemo-fabric-adapters-common"' in release
    assert 'package_directory="adapters/typescript/common"' in release
    assert 'package_name="nemo-fabric-adapters-pi"' in release
    assert 'package_directory="adapters/typescript/pi"' in release
    assert 'test_recipe="test-typescript"' in release
    assert 'test_recipe="test-typescript-adapters"' in release
    assert "set_typescript_project_version.py" in release
    assert "set_typescript_adapter_version.py" not in release
    assert steps["Test package"]["run"] == 'just "$RELEASE_TEST_RECIPE"'
    dependency_check = steps["Verify published dependencies"]["run"]
    assert "nemo-fabric-adapter-contract@${RELEASE_VERSION}" in dependency_check
    assert "nemo-fabric-adapters-common@${RELEASE_VERSION}" in dependency_check
    assert steps["Publish package"]["env"]["PACKAGE_DIRECTORY"] == (
        "${{ steps.release.outputs.package_directory }}"
    )


def test_nightly_alpha_runs_typescript_ci_without_npm_permissions():
    workflow = _load_workflow(NIGHTLY_WORKFLOW)
    assert "publish-typescript" not in workflow["jobs"]
    job = workflow["jobs"]["typescript-ci"]
    assert job["needs"] == "tag-nightly-alpha"
    assert job["uses"] == "./.github/workflows/ci_typescript.yml"
    assert job["permissions"] == {"contents": "read"}
    assert job["with"] == {
        "ref": "refs/tags/${{ needs.tag-nightly-alpha.outputs.tag }}",
    }


def test_typescript_ci_accepts_the_created_alpha_tag_ref():
    workflow = _load_workflow(TYPESCRIPT_CI_WORKFLOW)
    inputs = workflow["on"]["workflow_call"]["inputs"]
    assert set(inputs) == {"ref"}
    assert inputs["ref"]["required"] is True
    assert workflow["concurrency"]["group"] == (
        "ci-typescript-${{ inputs.ref || github.ref }}"
    )
    for job_name in ("test", "test-adapters"):
        checkout = workflow["jobs"][job_name]["steps"][0]
        assert checkout["name"] == "Checkout"
        assert checkout["with"]["ref"] == "${{ inputs.ref || github.ref }}"
