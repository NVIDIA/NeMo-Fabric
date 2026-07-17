# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Complete, editable Fabric configuration for this example."""

from nemo_fabric import EnvironmentConfig
from nemo_fabric import FabricConfig
from nemo_fabric import HarnessConfig
from nemo_fabric import MetadataConfig
from nemo_fabric import RuntimeConfig


def build_config() -> FabricConfig:
    """Return a credential-free starting point for experimentation."""

    config = FabricConfig(
        metadata=MetadataConfig(
            name="code-review-agent",
            description="Editable SDK example for reviewing a small Python workspace.",
        ),
        harness=HarnessConfig(
            adapter_id="nvidia.fabric.scripted",
            resolution="preinstalled",
        ),
        runtime=RuntimeConfig(input_schema="text", output_schema="message"),
        environment=EnvironmentConfig(provider="local", workspace="repo"),
    )
    config.add_skill_path("skills/code-review.md")
    return config
