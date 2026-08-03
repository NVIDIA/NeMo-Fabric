# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the Python notebook examples."""

from __future__ import annotations

import ast
import json
import os
import shutil
import sys
from pathlib import Path
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from examples.code_review_agent import BASE_DIR, base_config
from nemo_fabric import (
    Fabric,
    HarnessConfig,
    InstructionConfig,
    InstructionsConfig,
    ModelConfig,
)


ROOT = Path(__file__).resolve().parents[2]
QUICKSTART_NOTEBOOK = ROOT / "examples" / "notebooks" / "01_quickstart.ipynb"
VARIATIONS_NOTEBOOK = ROOT / "examples" / "notebooks" / "02_variations.ipynb"


def test_quickstart_notebook_hermes_config_plans_without_adapter_settings():
    notebook = json.loads(QUICKSTART_NOTEBOOK.read_text(encoding="utf-8"))
    source = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if "config = FabricConfig(" in "".join(cell["source"])
    )
    namespace = {"REPO_ROOT": ROOT}
    # Execute only the checked-in notebook source controlled by this repository.
    exec(compile(source, str(QUICKSTART_NOTEBOOK), "exec"), namespace)  # noqa: S102
    config = namespace["config"]

    plan = Fabric().plan(config, base_dir=ROOT)

    assert plan.adapter.adapter_id == "nvidia.fabric.hermes"
    assert plan.config.harness.settings == {}


def _variation_harness_definitions(base_dir=BASE_DIR):
    notebook = json.loads(VARIATIONS_NOTEBOOK.read_text(encoding="utf-8"))
    source = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if "HARNESSES =" in "".join(cell["source"])
    )
    namespace = {
        "base_config": base_config,
        "BASE_DIR": base_dir,
        "HarnessConfig": HarnessConfig,
        "InstructionConfig": InstructionConfig,
        "InstructionsConfig": InstructionsConfig,
        "ModelConfig": ModelConfig,
        "os": os,
        "Path": Path,
        "shutil": shutil,
        "HERMES_PY": sys.executable,
        "FABRIC_PY": sys.executable,
        "INSTRUCTION": "Test instruction.",
        "WORKSPACE": "./repos/my-service",
        "IN_COLAB": False,
    }
    # Execute only the checked-in notebook source controlled by this repository.
    exec(compile(source, str(VARIATIONS_NOTEBOOK), "exec"), namespace)  # noqa: S102
    return namespace["HARNESSES"], namespace["for_harness"], namespace["blocker"]


def test_variations_notebook_harnesses_plan_with_current_adapters():
    harnesses, for_harness, _ = _variation_harness_definitions()
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
    assert plans["Hermes"].config.harness.settings == {
        "max_tokens": 512,
        "reasoning_config": {"effort": "none"},
    }
    assert plans["Deep Agents"].config.harness.settings == {}
    assert (
        plans["Deep Agents"].config.models["default"]["base_url"]
        == "https://integrate.api.nvidia.com/v1"
    )
    codex = next(harness for harness in harnesses if harness["name"] == "Codex")
    assert "binary" not in codex
    assert "key" not in codex
    assert "skip_git_repo_check" not in codex["settings"]
    assert "validated when the adapter starts" in codex["needs"]
    assert codex["settings"] == {
        "sandbox": "workspace-write",
        "reasoning_effort": "high",
    }
    assert plans["Codex"].config.harness.settings == codex["settings"]
    assert plans["Codex"].config.runtime.input_schema == "text"


def test_variations_notebook_accepts_adapter_commands_and_relative_paths(
    monkeypatch, tmp_path
):
    adapter_python = tmp_path / "adapter" / "python"
    adapter_python.parent.mkdir()
    adapter_python.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    adapter_python.chmod(0o755)
    _, _, blocker = _variation_harness_definitions(base_dir=tmp_path)
    monkeypatch.setattr(shutil, "which", lambda _command: "/resolved/python3")

    assert blocker({"python": "python3"}) is None
    assert blocker({"python": "adapter/python"}) is None
    assert blocker({"python": "adapter/missing"}) == "adapter interpreter not found"


async def test_variations_notebook_skips_blocked_harness_before_planning(capsys):
    notebook = json.loads(VARIATIONS_NOTEBOOK.read_text(encoding="utf-8"))
    source = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if "run_failures = []" in "".join(cell["source"])
    )
    harness = {
        "name": "Unavailable",
        "needs": "install the adapter interpreter",
    }
    blocker = MagicMock(return_value="adapter interpreter not found")
    fabric = MagicMock()
    fabric.run = AsyncMock()
    for_harness = MagicMock()
    namespace = {
        "BASE_DIR": BASE_DIR,
        "HARNESSES": [harness],
        "blocker": blocker,
        "fabric": fabric,
        "failure_detail": MagicMock(),
        "for_harness": for_harness,
        "json": json,
        "oneline": MagicMock(),
        "os": os,
    }
    code = compile(
        source,
        str(VARIATIONS_NOTEBOOK),
        "exec",
        flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT,
    )

    await eval(code, namespace)  # noqa: S307

    blocker.assert_called_once_with(harness)
    for_harness.assert_not_called()
    fabric.plan.assert_not_called()
    fabric.run.assert_not_awaited()
    assert "NOT RUN here (adapter interpreter not found)." in capsys.readouterr().out


async def test_variations_notebook_continues_after_planning_failure_and_restores_env(
    capsys,
):
    notebook = json.loads(VARIATIONS_NOTEBOOK.read_text(encoding="utf-8"))
    source = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if "run_failures = []" in "".join(cell["source"])
    )
    harnesses = [
        {"name": "Broken", "python": "/broken/python"},
        {"name": "Runnable", "python": "/runnable/python"},
    ]
    plan = MagicMock()
    plan.adapter.adapter_id = "test.adapter"
    plan.config.runtime.input_schema = "chat"
    plan.config.models = {"default": {"model": "test-model"}}
    plan_adapter_python = []

    def plan_side_effect(*_args: object, **_kwargs: object) -> MagicMock:
        plan_adapter_python.append(os.environ.get("ADAPTER_PYTHON"))
        if len(plan_adapter_python) == 1:
            raise RuntimeError("plan boom")
        return plan

    fabric = MagicMock()
    fabric.plan.side_effect = plan_side_effect
    fabric.run = AsyncMock(
        return_value=MagicMock(
            status="succeeded",
            output=MagicMock(response="ok"),
        )
    )
    namespace = {
        "BASE_DIR": BASE_DIR,
        "HARNESSES": harnesses,
        "blocker": MagicMock(return_value=None),
        "fabric": fabric,
        "failure_detail": MagicMock(),
        "for_harness": MagicMock(return_value=MagicMock(name="config")),
        "json": json,
        "oneline": lambda value, _limit=200: str(value),
        "os": os,
    }
    code = compile(
        source,
        str(VARIATIONS_NOTEBOOK),
        "exec",
        flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT,
    )
    os.environ["ADAPTER_PYTHON"] = "original-python"

    await eval(code, namespace)  # noqa: S307

    assert plan_adapter_python == ["/broken/python", "/runnable/python"]
    assert os.environ["ADAPTER_PYTHON"] == "original-python"
    fabric.run.assert_awaited_once()
    assert namespace["run_failures"] == ["Broken: RuntimeError: plan boom"]
    assert (
        "Attempted planning failed; continuing with remaining variants."
        in capsys.readouterr().out
    )


async def test_variations_notebook_relay_failure_preserves_prior_failures(tmp_path):
    notebook = json.loads(VARIATIONS_NOTEBOOK.read_text(encoding="utf-8"))
    source = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if "relay_dir =" in "".join(cell["source"])
    )
    traced = MagicMock()

    def relay_failure(*_args: object, **_kwargs: object) -> None:
        assert os.environ["ADAPTER_PYTHON"] == "/deepagents/python"
        raise RuntimeError("relay boom")

    fabric = MagicMock()
    fabric.run = AsyncMock(side_effect=relay_failure)
    namespace = {
        "BASE_DIR": BASE_DIR,
        "HARNESSES": [
            {},
            {"name": "Deep Agents", "python": "/deepagents/python"},
        ],
        "PROMPT": "test",
        "RELAY_AVAILABLE": True,
        "REPO_ROOT": tmp_path,
        "RelayAtofConfig": MagicMock(),
        "RelayAtofFileSinkConfig": MagicMock(),
        "RelayObservabilityConfig": MagicMock(),
        "blocker": lambda _harness: None,
        "fabric": fabric,
        "failure_detail": MagicMock(),
        "for_harness": MagicMock(return_value=traced),
        "json": json,
        "oneline": lambda error, _limit: str(error),
        "os": os,
        "run_failures": ["Codex: prior authentication failure"],
        "shutil": shutil,
    }
    code = compile(
        source,
        str(VARIATIONS_NOTEBOOK),
        "exec",
        flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT,
    )
    os.environ["ADAPTER_PYTHON"] = "original-python"

    with pytest.raises(RuntimeError) as caught:
        await eval(code, namespace)  # noqa: S307

    assert os.environ["ADAPTER_PYTHON"] == "original-python"
    assert "Codex: prior authentication failure" in str(caught.value)
    assert "Deep Agents Relay: RuntimeError: relay boom" in str(caught.value)


def test_variations_notebook_uses_runnable_capabilities_and_checks_relay_status():
    notebook = json.loads(VARIATIONS_NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )

    assert "./skills/style-guide" not in source
    assert "${DOCS_MCP_URL}" not in source
    assert "path.is_file() and os.access(path, os.X_OK)" in source
    assert 'recomposed.add_skill_path("./skills/code-review")' in source
    assert 'exposure="harness_native"' in source
    assert 'exposure="fabric_managed"' not in source
    assert 'if result.status != "succeeded":' in source
    assert "detail = failure_detail(result)" in source
    assert 'print("    error:", json.dumps(detail, indent=2))' in source
    assert "Attempted run failed; continuing with remaining variants." in source
    assert "run_failures.append" in source
    assert "Notebook execution failures" in source
    assert "returned no Relay telemetry reference" in source
    assert "sinks=[RelayAtofFileSinkConfig(" in source
    assert "if not atof_paths:" in source
    assert "produced no ATOF trace" in source
    assert "produced an empty ATOF trace" in source
    assert "for line in atof_lines:\n                json.loads(line)" in source
    assert "# Print the full Relay trace: every ATOF event.\n        for path in atof_paths:" in source
    assert 'run_failures.append(f"Deep Agents Relay:' in source
