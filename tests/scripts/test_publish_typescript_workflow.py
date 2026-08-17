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
    return yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def test_typescript_publisher_accepts_an_explicit_tag_ref():
    workflow = _load_workflow(PUBLISH_WORKFLOW)
    inputs = workflow["on"]["workflow_call"]["inputs"]
    assert set(inputs) >= {"ref", "ref_name", "ref_type"}

    tag_patterns = workflow["on"]["push"]["tags"]
    assert any(fnmatchcase("v0.3.0-rc.1", pattern) for pattern in tag_patterns)
    assert any(
        fnmatchcase("v0.3.0-alpha.20260817", pattern) for pattern in tag_patterns
    )

    job = workflow["jobs"]["publish-typescript"]
    assert job["if"] == "${{ (inputs.ref_type || github.ref_type) == 'tag' }}"

    steps = {step["name"]: step for step in job["steps"]}
    assert steps["Checkout"]["with"]["ref"] == "${{ inputs.ref || github.sha }}"
    assert steps["Prepare release metadata"]["env"]["RELEASE_TAG"] == (
        "${{ inputs.ref_name || github.ref_name }}"
    )


def test_nightly_alpha_calls_typescript_publisher_for_created_tag():
    workflow = _load_workflow(NIGHTLY_WORKFLOW)
    job = workflow["jobs"]["publish-typescript"]

    assert job["needs"] == "tag-nightly-alpha"
    assert job["uses"] == "./.github/workflows/publish_typescript.yml"
    assert job["permissions"] == {"contents": "read", "id-token": "write"}
    assert job["with"] == {
        "ref": "refs/tags/${{ needs.tag-nightly-alpha.outputs.tag }}",
        "ref_name": "${{ needs.tag-nightly-alpha.outputs.tag }}",
        "ref_type": "tag",
    }
