# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression: started runtime handles absolutize the workspace path."""

from __future__ import annotations

import os
from pathlib import Path

from nemo_fabric import FabricClient

ROOT = Path(__file__).resolve().parents[2]


async def test_environment_handle():
    async with FabricClient() as client:
        session = await client.start_session(
            ROOT / "examples" / "code-review-agent",
            profiles=["env_local"],
        )
        try:
            workspace = session.runtime["environment"]["workspace"]
        finally:
            await session.stop()

    assert os.path.isabs(workspace), f"workspace not absolute: {workspace}"
    assert "code-review-agent/examples/code-review-agent" not in workspace, (
        f"workspace path is doubled: {workspace}"
    )
    assert workspace.endswith("repos/my-service"), workspace
