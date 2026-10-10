# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Factory Droid adapter credentialed end-to-end coverage.

Run with:

RUN_FABRIC_DROID_INTEGRATION=1 FACTORY_API_KEY=... \
    pytest tests/e2e/test_droid.py
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import uuid

import pytest

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
ADAPTER_ROOT = ROOT / "adapters" / "typescript" / "droid"
DESCRIPTOR = ADAPTER_ROOT / "droid.fabric-adapter.json"
ENTRYPOINT = ADAPTER_ROOT / "dist" / "cli.js"
SDK_PACKAGE = (
    ROOT / "adapters" / "typescript" / "node_modules" / "@factory" / "droid-sdk"
)

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None
    or shutil.which("droid") is None
    or not ENTRYPOINT.exists()
    or not SDK_PACKAGE.exists(),
    reason="build the Droid adapter and install its SDK and CLI",
)


def result_error_summary(result) -> str:
    if result.error is None:
        return "no structured error"
    return f"{result.error.code}: {result.error.message}"


async def test_droid_live_provider_retains_and_isolates_context(tmp_path):
    if os.environ.get("RUN_FABRIC_DROID_INTEGRATION") != "1":
        pytest.skip("set RUN_FABRIC_DROID_INTEGRATION=1 to run")
    credential = os.environ.get("FACTORY_API_KEY")
    if not credential:
        pytest.fail("FACTORY_API_KEY is required")

    skill = tmp_path / "remember-token"
    skill.mkdir()
    (skill / "SKILL.md").write_text(
        "---\n"
        "name: remember-token\n"
        "description: Remember a token when the user asks.\n"
        "---\n\n"
        "Remember the exact token supplied by the user.\n",
        encoding="utf-8",
    )

    config = FabricConfig(
        metadata=MetadataConfig(name="droid-live"),
        harness=HarnessConfig(adapter_id="nvidia.fabric.droid"),
        discovery=DiscoveryConfig(local_paths=[DESCRIPTOR]),
        models={
            "default": ModelConfig(
                provider="factory",
                model=os.environ.get("DROID_LIVE_MODEL", "auto"),
                api_key_env="FACTORY_API_KEY",
            )
        },
        instructions=InstructionsConfig(
            system=InstructionConfig(
                content="Follow the user's response-format instructions exactly.",
                mode="append",
            )
        ),
        tools=ToolsConfig(enabled=[]),
        runtime=RuntimeConfig(input_schema="text", output_schema="message"),
        environment=EnvironmentConfig(
            provider="local",
            workspace=tmp_path,
            env={"FACTORY_API_KEY": credential},
        ),
    )
    config.add_skill_path(skill)
    fabric = Fabric()
    plan = fabric.plan(config, base_dir=ROOT)
    report = await fabric.doctor(config, base_dir=ROOT)
    nonce = f"fabric-droid-{uuid.uuid4().hex[:8]}"

    assert plan.adapter_descriptor["descriptor"]["adapter_id"] == "nvidia.fabric.droid"
    assert report.status == "pass", report

    async with await fabric.start_runtime(config, base_dir=ROOT) as runtime:
        first = await runtime.invoke(
            input=(
                f"Remember this token exactly for a later question: {nonce}. "
                "For this turn only, reply with ACK."
            )
        )
        second = await runtime.invoke(
            input=(
                "For this turn, replace the prior response-format instruction: "
                "reply only with the exact token I asked you to remember."
            )
        )

    async with await fabric.start_runtime(config, base_dir=ROOT) as isolated_runtime:
        isolated = await isolated_runtime.invoke(
            input="Reply only with the word isolated."
        )

    assert first.status == "succeeded", result_error_summary(first)
    assert second.status == "succeeded", result_error_summary(second)
    assert isolated.status == "succeeded", result_error_summary(isolated)
    assert nonce in second.output["response"]
    assert "isolated" in isolated.output["response"].lower()
    assert (
        first.metadata["adapter"]["session_id"]
        == second.metadata["adapter"]["session_id"]
    )
    assert (
        isolated.metadata["adapter"]["session_id"]
        != first.metadata["adapter"]["session_id"]
    )
