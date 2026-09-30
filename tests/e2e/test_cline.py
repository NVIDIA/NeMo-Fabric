# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cline adapter end-to-end coverage.

Run the credentialed smoke test with:

RUN_FABRIC_CLINE_INTEGRATION=1 NVIDIA_API_KEY=... \
    pytest tests/e2e/test_cline.py
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import uuid

import pytest
import requests

from nemo_fabric import DiscoveryConfig
from nemo_fabric import EnvironmentConfig
from nemo_fabric import Fabric
from nemo_fabric import FabricConfig
from nemo_fabric import HarnessConfig
from nemo_fabric import InstructionConfig
from nemo_fabric import InstructionsConfig
from nemo_fabric import MetadataConfig
from nemo_fabric import ModelConfig
from nemo_fabric import RuntimeConfig
from nemo_fabric import ToolsConfig

ROOT = Path(__file__).resolve().parents[2]
ADAPTER_ROOT = ROOT / "adapters" / "typescript" / "cline"
DESCRIPTOR = ADAPTER_ROOT / "cline.fabric-adapter.json"
ENTRYPOINT = ADAPTER_ROOT / "dist" / "cli.js"
SDK_PACKAGE = ROOT / "adapters" / "typescript" / "node_modules" / "@cline" / "sdk"
NVIDIA_MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None or not ENTRYPOINT.exists() or not SDK_PACKAGE.exists(),
    reason="build the Cline TypeScript adapter and install its development dependencies",
)


def cline_config(
    *,
    workspace: Path,
    api_key_env: str,
    credential: str,
    base_url: str,
    model: str = NVIDIA_MODEL,
) -> FabricConfig:
    return FabricConfig(
        metadata=MetadataConfig(name="cline-e2e"),
        harness=HarnessConfig(
            adapter_id="nvidia.fabric.cline",
            resolution="preinstalled",
            settings={},
        ),
        discovery=DiscoveryConfig(local_paths=[DESCRIPTOR]),
        models={
            "default": ModelConfig(
                provider="nvidia",
                model=model,
                api_key_env=api_key_env,
                base_url=base_url,
            )
        },
        instructions=InstructionsConfig(
            system=InstructionConfig(
                content="Follow the user's response-format instructions exactly.",
                mode="replace",
            )
        ),
        tools=ToolsConfig(enabled=[]),
        runtime=RuntimeConfig(input_schema="text", output_schema="message"),
        environment=EnvironmentConfig(
            provider="local",
            workspace=workspace,
            env={api_key_env: credential},
        ),
    )


def result_error_summary(result) -> str:
    if result.error is None:
        return "no structured error"
    return f"{result.error.code}: {result.error.message}"


async def test_cline_plans_diagnoses_runs_and_isolates_sessions(api_server, tmp_path):
    config = cline_config(
        workspace=tmp_path,
        api_key_env="CLINE_E2E_KEY",
        credential="test-key",
        base_url=f"{api_server}/v1",
    )
    fabric = Fabric()

    plan = fabric.plan(config, base_dir=ROOT)
    report = await fabric.doctor(config, base_dir=ROOT)

    assert plan.adapter_descriptor["descriptor"]["adapter_id"] == "nvidia.fabric.cline"
    assert plan.agent_config["models"]["default"]["base_url"] == f"{api_server}/v1"
    assert report.status == "pass", report

    async with await fabric.start_runtime(config, base_dir=ROOT) as runtime:
        first = await runtime.invoke(
            input="Remember the word alpha. Reply only with ACK."
        )
        second = await runtime.invoke(
            input="Reply only with the word I asked you to remember."
        )

    async with await fabric.start_runtime(config, base_dir=ROOT) as isolated_runtime:
        isolated = await isolated_runtime.invoke(
            input="Reply only with the word isolated."
        )

    assert first.status == "succeeded", result_error_summary(first)
    assert second.status == "succeeded", result_error_summary(second)
    assert isolated.status == "succeeded", result_error_summary(isolated)
    assert "user_count=1" in first.output["response"]
    assert "user_count=2" in second.output["response"]
    assert "user_count=1" in isolated.output["response"]
    assert (
        first.metadata["adapter"]["session_id"]
        == second.metadata["adapter"]["session_id"]
    )
    assert (
        isolated.metadata["adapter"]["session_id"]
        != first.metadata["adapter"]["session_id"]
    )
    assert first.metadata["adapter"]["finish_reason"] == "completed"
    assert second.metadata["adapter"]["finish_reason"] == "completed"
    assert isolated.metadata["adapter"]["finish_reason"] == "completed"

    captured = requests.get(f"{api_server}/_requests", timeout=5).json()
    assert len(captured) == 3
    assert all(payload["model"] == NVIDIA_MODEL for payload in captured)


async def test_cline_live_provider_retains_context(tmp_path):
    if os.environ.get("RUN_FABRIC_CLINE_INTEGRATION") != "1":
        pytest.skip("set RUN_FABRIC_CLINE_INTEGRATION=1 to run")
    credential = os.environ.get("NVIDIA_API_KEY")
    if not credential:
        pytest.fail("NVIDIA_API_KEY is required")

    config = cline_config(
        workspace=tmp_path,
        api_key_env="NVIDIA_API_KEY",
        credential=credential,
        base_url="https://integrate.api.nvidia.com/v1",
    )
    fabric = Fabric()
    plan = fabric.plan(config, base_dir=ROOT)
    report = await fabric.doctor(config, base_dir=ROOT)
    nonce = f"fabric-cline-{uuid.uuid4().hex[:8]}"

    assert plan.adapter_descriptor["descriptor"]["adapter_id"] == "nvidia.fabric.cline"
    assert report.status == "pass", report

    async with await fabric.start_runtime(config, base_dir=ROOT) as runtime:
        first = await runtime.invoke(
            input=f"Remember this token exactly: {nonce}. Reply only with ACK."
        )
        second = await runtime.invoke(
            input="Reply only with the token I asked you to remember."
        )

    assert first.status == "succeeded", result_error_summary(first)
    assert second.status == "succeeded", result_error_summary(second)
    assert (
        first.metadata["adapter"]["session_id"]
        == second.metadata["adapter"]["session_id"]
    )
    assert second.metadata["adapter"]["finish_reason"] == "completed"
    assert nonce in second.output["response"]
