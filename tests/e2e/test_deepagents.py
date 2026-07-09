# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in real Deep Agents smoke for Fabric one-shot and multi-turn runtimes.

RUN_FABRIC_DEEPAGENTS_INTEGRATION=1 NVIDIA_API_KEY=... \
    pytest tests/e2e/test_deepagents.py
"""

from __future__ import annotations

import importlib.util
import os
import uuid

import pytest


def _require_integration() -> None:
    if os.environ.get("RUN_FABRIC_DEEPAGENTS_INTEGRATION") != "1":
        pytest.skip("set RUN_FABRIC_DEEPAGENTS_INTEGRATION=1 to run")
    if importlib.util.find_spec("deepagents") is None:
        pytest.fail("the deepagents package is required (pip install -e '.[deepagents]')")
    if importlib.util.find_spec("nemo_fabric._native") is None:
        pytest.fail("the nemo_fabric native extension is required (pip install -e .)")
    if not os.environ.get("NVIDIA_API_KEY"):
        pytest.fail("NVIDIA_API_KEY is required")


async def test_deepagents_oneshot_and_runtime() -> None:
    _require_integration()

    from examples.code_review_agent import BASE_DIR, deepagents_config
    from nemo_fabric import Fabric

    config = deepagents_config()
    nonce = f"fabric-{uuid.uuid4().hex[:8]}"
    client = Fabric()

    oneshot = await client.run(
        config,
        base_dir=BASE_DIR,
        input="Reply with exactly: FABRIC_DEEPAGENTS_ONESHOT_OK",
    )
    assert oneshot["status"] == "succeeded", oneshot.to_mapping()
    assert oneshot["output"]["response"], oneshot.to_mapping()
    assert oneshot["output"]["resumed"] is False, oneshot.to_mapping()

    async with await client.start_runtime(config, base_dir=BASE_DIR) as runtime:
        first = await runtime.invoke(input=f"Remember this value: {nonce}")
        second = await runtime.invoke(
            input="Reply with only the value I asked you to remember."
        )

    results = (first.to_mapping(), second.to_mapping())
    assert first["status"] == second["status"] == "succeeded", results
    # the LangGraph thread id is stable across turns in the same runtime
    assert first["output"]["thread_id"] == second["output"]["thread_id"], results
    assert second["output"]["resumed"] is True, results
    assert nonce in second["output"]["response"], second.to_mapping()


async def test_deepagents_doctor() -> None:
    _require_integration()

    from examples.code_review_agent import BASE_DIR, deepagents_config
    from nemo_fabric import Fabric

    config = deepagents_config()
    client = Fabric()
    report = await client.doctor(config, base_dir=BASE_DIR)

    # The adapter declares no static env requirement (auth is provider-specific and
    # validated by the runtime preflight), so doctor resolves without failures.
    assert report.status == "pass", report
