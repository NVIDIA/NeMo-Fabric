# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression: started runtime handles absolutize the workspace path."""

from __future__ import annotations

import os
from pathlib import Path

from examples.code_review_agent import base_config
from nemo_fabric import Fabric


async def test_environment_handle(hermes_shim_agent_dir: Path):
    config = base_config()
    config.harness.adapter_id = "test.fabric.hermes_shim"
    runtime = await Fabric().start_runtime(
        config,
        base_dir=hermes_shim_agent_dir,
    )
    try:
        workspace = runtime.handle["environment"]["workspace"]
    finally:
        await runtime.stop()

    assert os.path.isabs(workspace), f"workspace not absolute: {workspace}"
    assert workspace == str((hermes_shim_agent_dir / "repos" / "my-service").resolve())
