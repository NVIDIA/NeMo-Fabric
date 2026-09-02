# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Harbor ``FabricAgent`` to OO Agents BenchAgent end-to-end contract test."""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.usefixtures("requires_harbor")

ROOT = Path(__file__).parents[2]


async def test_harbor_fabric_agent_runs_nooa_bench_adapter(
    tmp_path: Path,
):
    from harbor.environments.base import BaseEnvironment
    from harbor.models.agent.context import AgentContext
    from nemo_fabric.integrations.harbor import FabricAgent
    from nemo_fabric.integrations.harbor.models import FabricRunPayload

    workspace = tmp_path / "testbed"
    workspace.mkdir()
    artifacts = tmp_path / "artifacts"
    bundle = tmp_path / "bundle"
    bundle_adapters = bundle / "adapters"
    bundle_adapters.mkdir(parents=True)
    shutil.copyfile(
        ROOT / "external" / "nooa" / "nooa-bench.fabric-adapter.json",
        bundle_adapters / "nooa-bench.fabric-adapter.json",
    )

    fixture_source = ROOT / "tests" / "fixtures" / "nooa_bench" / "src"
    adapter_source = ROOT / "external" / "nooa" / "src"
    python_path = os.pathsep.join((str(fixture_source), str(adapter_source)))
    agent = FabricAgent(
        logs_dir=tmp_path / "logs",
        fabric_adapter_id="nvidia.fabric.nooa.bench-agent",
        fabric_config_bundle=bundle,
        fabric_environment_env={
            "ADAPTER_PYTHON": sys.executable,
            "OPENAI_API_KEY": "fixture-key",
            "PYTHONPATH": python_path,
        },
        fabric_python=sys.executable,
        fabric_system_instruction="Return verifiable evidence.",
        model_name="openai/fixture-model",
        extra_env={
            "ADAPTER_PYTHON": sys.executable,
            "OPENAI_API_KEY": "fixture-key",
            "PYTHONPATH": python_path,
        },
    )

    remote_files: dict[str, str] = {}
    mock_environment = MagicMock(spec=BaseEnvironment)

    async def upload_file(source: Path, destination: str) -> None:
        remote_files[destination] = source.read_text(encoding="utf-8")

    async def download_file(source: str, destination: Path) -> None:
        destination.write_text(remote_files[source], encoding="utf-8")

    async def execute(command: str, **kwargs: Any) -> SimpleNamespace:
        if "nemo_fabric.integrations.harbor.runner" not in command:
            return SimpleNamespace(return_code=0, stdout="", stderr="")
        arguments = shlex.split(command)
        spec_path = arguments[arguments.index("--spec") + 1]
        result_path = arguments[arguments.index("--result") + 1]
        payload = FabricRunPayload.model_validate_json(remote_files[spec_path])
        payload.config_base_dir = bundle
        assert payload.config.environment is not None
        payload.config.environment.workspace = str(workspace)
        payload.config.environment.artifacts = str(artifacts)
        payload.config.runtime.artifacts = str(artifacts)
        payload.logs_dir = tmp_path / "logs"
        exchange = tmp_path / "exchange"
        exchange.mkdir()
        local_spec = exchange / "spec.json"
        local_result = exchange / "result.json"
        local_spec.write_text(payload.model_dump_json(), encoding="utf-8")
        process_environment = os.environ.copy()
        process_environment.update(kwargs.get("env") or {})
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "nemo_fabric.integrations.harbor.runner",
            "--spec",
            str(local_spec),
            "--result",
            str(local_result),
            env=process_environment,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=20)
        except TimeoutError:
            process.kill()
            stdout, stderr = await process.communicate()
            return SimpleNamespace(
                return_code=124,
                stdout=stdout.decode(),
                stderr=("Harbor runner timed out\n" + stderr.decode()),
            )
        if process.returncode == 0:
            remote_files[result_path] = local_result.read_text(encoding="utf-8")
        return SimpleNamespace(
            return_code=process.returncode,
            stdout=stdout.decode(),
            stderr=stderr.decode(),
        )

    mock_environment.upload_dir = AsyncMock()
    mock_environment.upload_file = AsyncMock(side_effect=upload_file)
    mock_environment.download_file = AsyncMock(side_effect=download_file)
    mock_environment.exec = AsyncMock(side_effect=execute)
    context = AgentContext()

    await agent.setup(mock_environment)
    await agent.run(
        "Create a task completion artifact.",
        mock_environment,
        context,
    )
    agent.populate_context_post_run(context)

    result_paths = list((tmp_path / "logs").glob("fabric-result-*.json"))
    assert len(result_paths) == 1
    result = json.loads(result_paths[0].read_text(encoding="utf-8"))
    assert result["status"] == "succeeded"
    assert result["adapter_id"] == "nvidia.fabric.nooa.bench-agent"
    assert result["output"] == {
        "harness": "nooa-bench",
        "adapter": "python",
        "mode": "bench_agent",
        "response": "test -f bench-agent-result.txt",
        "completed": True,
        "result": {
            "solution_description": "Created the requested task artifact.",
            "evidence": "bench-agent-result.txt exists in the task workspace",
            "command_to_verify": "test -f bench-agent-result.txt",
        },
    }
    assert result["usage"] == {
        "input_tokens": 12,
        "output_tokens": 4,
        "total_tokens": 16,
        "metadata": {},
    }
    assert (workspace / "bench-agent-result.txt").read_text(encoding="utf-8") == (
        "Create a task completion artifact.\nReturn verifiable evidence.\n"
    )
    assert context.metadata["fabric"]["status"] == "succeeded"
    assert context.metadata["fabric"]["adapter_id"] == (
        "nvidia.fabric.nooa.bench-agent"
    )
