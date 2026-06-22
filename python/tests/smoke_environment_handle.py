# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression: the inline-path environment handle absolutizes the workspace.

A relative workspace (config-root-relative) must be resolved to an absolute path
so an adapter does not re-join it onto the absolute config_root and double it
(e.g. examples/agent/examples/agent/...). Dependency-free; no native extension.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemo_fabric.client import _environment_handle


def main() -> None:
    plan = {
        "environment_plan": {"workspace": "examples/code-review-agent/repos/my-service"},
        "config": {"runtime": {}},
        "agent_root": "examples/code-review-agent",
    }
    workspace = _environment_handle(plan)["workspace"]
    assert os.path.isabs(workspace), f"workspace not absolute: {workspace}"
    assert (
        "code-review-agent/examples/code-review-agent" not in workspace
    ), f"workspace path is doubled: {workspace}"
    assert workspace.endswith("repos/my-service"), workspace
    print("smoke_environment_handle ok")


if __name__ == "__main__":
    main()
