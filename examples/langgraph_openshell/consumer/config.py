# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Consumer-owned normalized Fabric environment configuration."""

from __future__ import annotations

from pathlib import Path

from nemo_fabric import DiscoveryConfig
from nemo_fabric import EnvironmentConfig
from nemo_fabric import FabricConfig
from nemo_fabric import HarnessConfig
from nemo_fabric import MetadataConfig
from nemo_fabric import RuntimeConfig

ADAPTER_ID = "nvidia.fabric.example.langgraph.portable-courier"
EXAMPLE_ROOT = Path(__file__).parents[1]


def courier_config(
    *, gateway: str, image: str, ownership: str, sandbox_name: str | None = None
) -> FabricConfig:
    """Build the complete OpenShell-backed demo configuration."""

    return FabricConfig(
        metadata=MetadataConfig(
            name="portable-courier",
            description="Stateful LangGraph agent running in an OpenShell sandbox.",
        ),
        discovery=DiscoveryConfig(local_paths=[EXAMPLE_ROOT / "adapter"]),
        harness=HarnessConfig(adapter_id=ADAPTER_ID, resolution="image_provided"),
        runtime=RuntimeConfig(
            input_schema="text",
            output_schema="object",
            artifacts="./artifacts",
            timeout_seconds=30,
        ),
        environment=EnvironmentConfig(
            provider="openshell",
            control_location="in_env_control",
            ownership=ownership,
            workspace="/sandbox",
            artifacts="/sandbox/artifacts",
            env={"PYTHONPATH": "/opt/nemo-fabric"},
            connection={"gateway": gateway},
            settings={
                "image": image,
                "command": ["fabric-runtime-server", "serve"],
                "policy_yaml": (EXAMPLE_ROOT / "policy.yaml").read_text(
                    encoding="utf-8"
                ),
                "ready_timeout_seconds": 90,
                "delete_timeout_seconds": 30,
                "exec_timeout_seconds": 35,
                **({"sandbox_name": sandbox_name} if sandbox_name else {}),
            },
        ),
    )
