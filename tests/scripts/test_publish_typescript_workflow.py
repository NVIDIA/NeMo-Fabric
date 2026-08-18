# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from fnmatch import fnmatchcase
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
PUBLISH_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "publish_typescript.yml"
NIGHTLY_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "nightly-alpha-tag.yml"


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


def test_typescript_publisher_accepts_an_explicit_tag_ref():
    workflow = _load_workflow(PUBLISH_WORKFLOW)
    inputs = workflow["on"]["workflow_call"]["inputs"]
    assert set(inputs) == {"ref"}
    assert inputs["ref"]["required"] is True

    tag_patterns = workflow["on"]["push"]["tags"]
    for tag in ("v0.3.0", "v0.3.0-beta.1", "v0.3.0-rc.1"):
        assert _tag_triggers_workflow(tag, tag_patterns)
    assert not _tag_triggers_workflow("v0.3.0-alpha.20260817", tag_patterns)

    job = workflow["jobs"]["publish-typescript"]
    assert "if" not in job

    steps = {step["name"]: step for step in job["steps"]}
    assert steps["Checkout"]["with"]["ref"] == "${{ inputs.ref || github.ref }}"
    assert steps["Verify release tag"]["env"]["RELEASE_REF"] == (
        "${{ inputs.ref || github.ref }}"
    )
    verify_tag = steps["Verify release tag"]["run"]
    assert 'git rev-parse --verify "refs/tags/${release_tag}^{commit}"' in verify_tag
    assert "git rev-parse HEAD" in verify_tag
    assert steps["Prepare release metadata"]["env"]["RELEASE_TAG"] == (
        "${{ steps.source.outputs.tag }}"
    )
    release_metadata = steps["Prepare release metadata"]["run"]
    assert '*-alpha*) dist_tag="alpha"' in release_metadata
    assert '*-beta*) dist_tag="next"' in release_metadata
    assert '*-rc*) dist_tag="rc"' in release_metadata


def test_nightly_alpha_calls_typescript_publisher_for_created_tag():
    workflow = _load_workflow(NIGHTLY_WORKFLOW)
    job = workflow["jobs"]["publish-typescript"]

    assert job["needs"] == "tag-nightly-alpha"
    assert job["uses"] == "./.github/workflows/publish_typescript.yml"
    assert job["permissions"] == {"contents": "read", "id-token": "write"}
    assert job["with"] == {
        "ref": "refs/tags/${{ needs.tag-nightly-alpha.outputs.tag }}",
    }
