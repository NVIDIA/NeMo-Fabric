# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_CI_WORKFLOW = REPO_ROOT / ".github/workflows/ci_python.yml"


def test_python_ci_builds_the_packaged_pi_adapter_before_pytest():
    workflow = yaml.safe_load(PYTHON_CI_WORKFLOW.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["test"]["steps"]
    step_by_name = {step["name"]: step for step in steps}

    assert step_by_name["Set up Node.js"]["with"]["node-version"] == "22.19.0"
    assert step_by_name["Build packaged Pi adapter"]["run"] == (
        "just install-typescript-pi\n"
        "npm run build --prefix adapter-contract/typescript\n"
        "npm run build --prefix adapters/typescript\n"
    )
    assert [step["name"] for step in steps].index("Build packaged Pi adapter") < [
        step["name"] for step in steps
    ].index("Run pytest")
