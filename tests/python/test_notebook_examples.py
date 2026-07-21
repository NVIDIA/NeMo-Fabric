# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the Python notebook examples."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from examples.code_review_agent import BASE_DIR, base_config
from nemo_fabric import Fabric, HarnessConfig, ModelConfig


ROOT = Path(__file__).resolve().parents[2]
VARIATIONS_NOTEBOOK = ROOT / "examples" / "notebooks" / "02_variations.ipynb"


def _variation_harness_definitions():
    notebook = json.loads(VARIATIONS_NOTEBOOK.read_text(encoding="utf-8"))
    source = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if "HARNESSES =" in "".join(cell["source"])
    )
    namespace = {
        "base_config": base_config,
        "HarnessConfig": HarnessConfig,
        "ModelConfig": ModelConfig,
        "HERMES_PY": sys.executable,
        "FABRIC_PY": sys.executable,
        "INSTRUCTION": "Test instruction.",
        "WORKSPACE": "./repos/my-service",
    }
    # Execute only the checked-in notebook source controlled by this repository.
    exec(compile(source, str(VARIATIONS_NOTEBOOK), "exec"), namespace)  # noqa: S102
    return namespace["HARNESSES"], namespace["for_harness"]


def test_variations_notebook_harnesses_plan_with_current_adapters():
    harnesses, for_harness = _variation_harness_definitions()
    client = Fabric()

    plans = {
        harness["name"]: client.plan(for_harness(harness), base_dir=BASE_DIR)
        for harness in harnesses
    }

    assert {name: plan.adapter.adapter_id for name, plan in plans.items()} == {
        "Hermes": "nvidia.fabric.hermes",
        "Deep Agents": "nvidia.fabric.langchain.deepagents",
        "Codex": "nvidia.fabric.codex",
        "Claude": "nvidia.fabric.claude",
    }
    codex = next(harness for harness in harnesses if harness["name"] == "Codex")
    assert "binary" not in codex
    assert "key" not in codex
    assert "skip_git_repo_check" not in codex["settings"]
    assert "validated when the adapter starts" in codex["needs"]
    assert plans["Codex"].config.harness.settings["sandbox"] == "workspace-write"
    assert plans["Codex"].config.harness.settings["reasoning_effort"] == "high"
    assert plans["Codex"].config.runtime.input_schema == "text"
