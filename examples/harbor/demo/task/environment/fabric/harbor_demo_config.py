# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed FabricConfig factories for the local Harbor calculator demo."""

from nemo_fabric import EnvironmentConfig
from nemo_fabric import FabricConfig
from nemo_fabric import HarnessConfig
from nemo_fabric import MetadataConfig
from nemo_fabric import ModelConfig
from nemo_fabric import RelayAtifConfig
from nemo_fabric import RelayAtofConfig
from nemo_fabric import RelayObservabilityConfig
from nemo_fabric import RelayOtlpConfig
from nemo_fabric import RuntimeConfig


def build_smoke() -> FabricConfig:
    """Return the deterministic, credential-free calculator config."""

    return FabricConfig(
        metadata=MetadataConfig(
            name="harbor-calculator-demo",
            description="Deterministic Harbor example pipeline check.",
        ),
        harness=HarnessConfig(
            adapter_id="demo.fabric.scripted",
            resolution="preinstalled",
        ),
        runtime=RuntimeConfig(
            input_schema="text",
            output_schema="message",
            artifacts="/logs/agent/fabric-artifacts/smoke",
        ),
        environment=EnvironmentConfig(
            provider="local",
            workspace="/app",
            artifacts="/logs/agent/fabric-artifacts/smoke",
        ),
    )


def build_hermes() -> FabricConfig:
    """Return the Hermes calculator config."""

    return FabricConfig(
        metadata=MetadataConfig(
            name="harbor-calculator-demo",
            description="Hermes code-repair example in a Harbor task environment.",
        ),
        harness=HarnessConfig(
            adapter_id="nvidia.fabric.hermes",
            resolution="preinstalled",
            settings={
                "cwd": "/app",
                "hermes_home": "/tmp/fabric-hermes",
                "base_url": "https://integrate.api.nvidia.com/v1",
                "max_iterations": 20,
            },
        ),
        models={
            "default": ModelConfig(
                provider="nvidia",
                model="nvidia/nemotron-3-nano-30b-a3b",
                temperature=0.0,
            )
        },
        runtime=RuntimeConfig(
            input_schema="text",
            output_schema="message",
            artifacts="/logs/agent/fabric-artifacts/hermes",
        ),
        environment=EnvironmentConfig(
            provider="local",
            workspace="/app",
            artifacts="/logs/agent/fabric-artifacts/hermes",
        ),
    )


def build_hermes_relay() -> FabricConfig:
    """Return a Hermes copy with Relay ATOF, ATIF, and Phoenix export."""

    config = build_hermes().model_copy(deep=True)
    config.metadata = MetadataConfig(
        name="harbor-calculator-demo",
        description="Hermes and Relay example in a Harbor task environment.",
    )
    config.harness.settings.update(max_iterations=4, terminal_timeout=120)
    config.runtime.artifacts = "/logs/agent/fabric-artifacts/hermes-relay"
    assert config.environment is not None
    config.environment.artifacts = "/logs/agent/fabric-artifacts/hermes-relay"
    config.enable_relay(
        output_dir="/logs/agent/fabric-artifacts/hermes-relay/relay",
        observability=RelayObservabilityConfig(
            atif=RelayAtifConfig(
                enabled=True,
                output_directory="/logs/agent/fabric-artifacts/hermes-relay/relay",
                filename_template="trajectory-{session_id}.atif.json",
                agent_name="harbor-calculator-demo",
            ),
            atof=RelayAtofConfig(
                enabled=True,
                output_directory="/logs/agent/fabric-artifacts/hermes-relay/relay",
                filename="events.atof.jsonl",
                mode="overwrite",
            ),
            openinference=RelayOtlpConfig(
                enabled=True,
                transport="http_binary",
                endpoint="http://host.docker.internal:6006/v1/traces",
            ),
        ),
    )
    return config


def build_claude() -> FabricConfig:
    """Return the Claude calculator config."""

    return FabricConfig(
        metadata=MetadataConfig(
            name="harbor-calculator-claude",
            description="Claude code-repair example in a Harbor task environment.",
        ),
        harness=HarnessConfig(
            adapter_id="nvidia.fabric.claude",
            resolution="preinstalled",
            settings={
                "permission_mode": "bypassPermissions",
                "max_turns": 20,
                "timeout_seconds": 600,
                "env": {"IS_SANDBOX": "1"},
            },
        ),
        models={
            "default": ModelConfig(
                provider="anthropic",
                model="claude-sonnet-4-5",
                api_key_env="ANTHROPIC_API_KEY",
            )
        },
        runtime=RuntimeConfig(
            input_schema="text",
            output_schema="message",
            artifacts="/logs/agent/fabric-artifacts/claude",
        ),
        environment=EnvironmentConfig(
            provider="local",
            workspace="/app",
            artifacts="/logs/agent/fabric-artifacts/claude",
        ),
    )
