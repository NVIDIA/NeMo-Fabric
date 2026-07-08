# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression: started runtime handles absolutize the workspace path."""

from __future__ import annotations

import os
from pathlib import Path

from nemo_fabric import Fabric

ROOT = Path(__file__).resolve().parents[2]


async def test_environment_handle():
    async with Fabric() as client:
        runtime = await client.start_runtime(
            ROOT / "examples" / "code-review-agent",
            profiles=["env_local"],
        )
        try:
            workspace = runtime.handle["environment"]["workspace"]
        finally:
            await runtime.stop()

    assert os.path.isabs(workspace), f"workspace not absolute: {workspace}"
    assert "code-review-agent/examples/code-review-agent" not in workspace, (
        f"workspace path is doubled: {workspace}"
    )
    assert workspace.endswith("repos/my-service"), workspace
