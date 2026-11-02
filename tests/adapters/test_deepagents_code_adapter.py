# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Deep Agents Code adapter relies on its descriptor schema for settings."""

from __future__ import annotations

from pathlib import Path

import pytest

from nemo_fabric import (
    DiscoveryConfig,
    Fabric,
    FabricConfig,
    FabricConfigError,
    HarnessConfig,
    MetadataConfig,
    ModelConfig,
)

ROOT = Path(__file__).resolve().parents[2]
DESCRIPTOR = (
    ROOT / "adapters/typescript/deepagents-code/deepagents-code.fabric-adapter.json"
)


def config(settings: dict) -> FabricConfig:
    return FabricConfig(
        metadata=MetadataConfig(name="deepagents-code-test"),
        discovery=DiscoveryConfig(local_paths=[DESCRIPTOR]),
        harness=HarnessConfig(
            adapter_id="nvidia.fabric.langchain.deepagents-code", settings=settings
        ),
        models={"default": ModelConfig(provider="openai", model="test-model")},
    )


@pytest.mark.parametrize("approval_mode", ["auto", "yolo"])
def test_deepagents_code_plans_with_an_approval_mode(approval_mode):
    plan = Fabric().plan(config({"approval_mode": approval_mode}), base_dir=ROOT)
    assert plan.to_mapping()["agent_config"]["harness"]["settings"] == {
        "approval_mode": approval_mode
    }


@pytest.mark.parametrize(
    "settings",
    [{}, {"approval_mode": "manual"}, {"approval_mode": "auto", "unknown": True}],
)
def test_deepagents_code_rejects_invalid_settings(settings):
    with pytest.raises(FabricConfigError):
        Fabric().plan(config(settings), base_dir=ROOT)
