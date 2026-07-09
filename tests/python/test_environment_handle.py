# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression: started runtime handles absolutize the workspace path."""

from __future__ import annotations

import os

from examples.code_review_agent import BASE_DIR, base_config
from nemo_fabric import Fabric


async def test_environment_handle():
    runtime = await Fabric().start_runtime(
        base_config(),
        base_dir=BASE_DIR,
    )
    try:
        workspace = runtime.handle["environment"]["workspace"]
    finally:
        await runtime.stop()

    assert os.path.isabs(workspace), f"workspace not absolute: {workspace}"
    assert workspace == str((BASE_DIR / "repos" / "my-service").resolve())
