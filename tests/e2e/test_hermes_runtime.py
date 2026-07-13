# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in integration smoke for the SDK multi-turn Runtime path (real Hermes).

Drives ``Fabric.start -> invoke -> invoke -> stop`` against the Hermes SDK
adapter and asserts the runtime carries conversation memory across turns
through the same Fabric runtime handle.

This test must run in an interpreter that has both the nemo_fabric native
extension and Hermes importable:

    RUN_FABRIC_HERMES_INTEGRATION=1 NVIDIA_API_KEY=... \\
        <hermes-venv>/bin/python -m pytest tests/e2e/test_hermes_runtime.py
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

async def test_hermes_runtime():
    if os.environ.get("RUN_FABRIC_HERMES_INTEGRATION") != "1":
        pytest.skip("set RUN_FABRIC_HERMES_INTEGRATION=1 to run")
    if not os.environ.get("NVIDIA_API_KEY"):
        pytest.fail("NVIDIA_API_KEY is required")
    if importlib.util.find_spec("nemo_fabric._native") is None:
        pytest.skip(
            "skipping: the SDK runtime path needs the nemo_fabric native extension "
            "(pip install -e . into this interpreter)"
        )
    if importlib.util.find_spec("run_agent") is None:
        pytest.skip(
            "skipping: Hermes (run_agent) is not importable; run with the Hermes "
            "venv python (set ADAPTER_PYTHON or invoke it directly)"
        )
    hermes_state_spec = importlib.util.find_spec("hermes_state")
    if hermes_state_spec is None:
        pytest.skip(
            "skipping: Hermes session state (hermes_state) is not importable; run "
            "with the Hermes venv python"
        )
    hermes_state_origin = hermes_state_spec.origin
    if hermes_state_origin:
        hermes_site_packages = str(Path(hermes_state_origin).resolve().parent)
        os.environ["PYTHONPATH"] = (
            f"{hermes_site_packages}{os.pathsep}{os.environ['PYTHONPATH']}"
            if os.environ.get("PYTHONPATH")
            else hermes_site_packages
        )
    python_bin = Path(sys.executable).resolve().parent
    os.environ["PATH"] = f"{python_bin}{os.pathsep}{os.environ.get('PATH', '')}"
    await _run()


async def _run() -> None:
    await _run_sdk_runtime()


async def _run_sdk_runtime() -> None:
    from examples.code_review_agent import BASE_DIR, hermes_sdk_config
    from nemo_fabric import Fabric, RuntimeStatus

    async with await Fabric().start_runtime(
        hermes_sdk_config(),
        base_dir=BASE_DIR,
    ) as runtime:
        assert runtime.status is RuntimeStatus.ACTIVE, runtime.status

        r1 = await runtime.invoke(input="My name is Robin. Please remember it for later.")
        assert r1["status"] == "succeeded", r1
        after_turn1 = runtime.messages
        assert len(after_turn1) >= 2, after_turn1

        r2 = await runtime.invoke(input="What is my name? Reply with just the name.")
        assert r2["status"] == "succeeded", r2
        assert r2["runtime_id"] == r1["runtime_id"], (r1, r2)
        # Hermes should return a transcript that includes the prior turn.
        assert len(runtime.messages) > len(after_turn1), runtime.messages
        # And the model must recall the name supplied in turn 1.
        response = (r2["output"].get("response") or "").lower()
        assert "robin" in response, response

    assert runtime.status is RuntimeStatus.STOPPED, runtime.status
