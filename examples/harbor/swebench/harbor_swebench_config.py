# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed FabricConfig factories for one fixed Harbor SWE-Bench task."""

from nemo_fabric import EnvironmentConfig
from nemo_fabric import FabricConfig
from nemo_fabric import HarnessConfig
from nemo_fabric import MetadataConfig
from nemo_fabric import ModelConfig
from nemo_fabric import RelayAtifConfig
from nemo_fabric import RelayAtofConfig
from nemo_fabric import RelayObservabilityConfig
from nemo_fabric import RuntimeConfig
from nemo_fabric import ToolsConfig

ARTIFACT_ROOT = "/logs/agent/fabric-artifacts/swebench"


def _set_identity_and_artifacts(
    config: FabricConfig,
    *,
    name: str,
    description: str,
    artifact_name: str,
) -> FabricConfig:
    config.metadata = MetadataConfig(name=name, description=description)
    artifact_path = f"{ARTIFACT_ROOT}/{artifact_name}"
    config.runtime.artifacts = artifact_path
    assert config.environment is not None
    config.environment.artifacts = artifact_path
    return config


def build_hermes() -> FabricConfig:
    """Return the baseline Hermes SWE-Bench config."""

    return FabricConfig(
        metadata=MetadataConfig(
            name="harbor-swebench-hermes",
            description="Hermes baseline for one Harbor SWE-Bench task.",
        ),
        harness=HarnessConfig(
            adapter_id="nvidia.fabric.hermes",
            resolution="preinstalled",
            settings={
                "cwd": "/testbed",
                "hermes_home": "/tmp/fabric-hermes",
                "base_url": "https://integrate.api.nvidia.com/v1",
                "max_iterations": 50,
                "terminal_timeout": 300,
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
            artifacts=f"{ARTIFACT_ROOT}/hermes",
        ),
        environment=EnvironmentConfig(
            provider="local",
            workspace="/testbed",
            artifacts=f"{ARTIFACT_ROOT}/hermes",
        ),
    )


def build_claude() -> FabricConfig:
    """Return the Claude SWE-Bench config."""

    return FabricConfig(
        metadata=MetadataConfig(
            name="harbor-swebench-claude",
            description="Claude baseline for one Harbor SWE-Bench task.",
        ),
        harness=HarnessConfig(
            adapter_id="nvidia.fabric.claude",
            resolution="preinstalled",
            settings={
                "permission_mode": "bypassPermissions",
                "max_turns": 75,
                "timeout_seconds": 1800,
                "system_prompt": (
                    "Solve the requested repository task with the smallest relevant patch. "
                    "Do not create summary, demonstration, or scratch files. Run focused "
                    "tests, stop once the implementation is verified, and return a concise "
                    "result."
                ),
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
            artifacts=f"{ARTIFACT_ROOT}/claude",
        ),
        environment=EnvironmentConfig(
            provider="local",
            workspace="/testbed",
            artifacts=f"{ARTIFACT_ROOT}/claude",
        ),
    )


def build_hermes_skill() -> FabricConfig:
    """Return a Hermes copy with the SWE-Bench debugging skill."""

    config = build_hermes().model_copy(deep=True)
    _set_identity_and_artifacts(
        config,
        name="harbor-swebench-hermes-skill",
        description="Hermes with a config-owned SWE-Bench debugging skill.",
        artifact_name="hermes-skill",
    )
    config.add_skill_path("skills/swebench-debugging")
    return config


def build_hermes_mcp() -> FabricConfig:
    """Return a Hermes copy with the read-only repository-inspector MCP server."""

    config = build_hermes().model_copy(deep=True)
    _set_identity_and_artifacts(
        config,
        name="harbor-swebench-hermes-mcp",
        description="Hermes with a config-owned repository-inspector MCP server.",
        artifact_name="hermes-mcp",
    )
    config.add_mcp_server(
        "fabric-repo-inspector",
        transport="stdio",
        url="python3",
        exposure="harness_native",
        extra_fields={
            "args": ["/tmp/nemo-fabric-config/mcp/repo_inspector.py"],
        },
    )
    return config


def build_hermes_tools() -> FabricConfig:
    """Return a Hermes copy with Fabric's normalized blocked-tools policy."""

    config = build_hermes().model_copy(deep=True)
    _set_identity_and_artifacts(
        config,
        name="harbor-swebench-hermes-tools",
        description="Hermes with Fabric's normalized blocked-tools policy.",
        artifact_name="hermes-tools",
    )
    config.tools = ToolsConfig(blocked=["browser"])
    return config


def build_hermes_relay() -> FabricConfig:
    """Return a Hermes copy with Relay ATOF and ATIF run evidence."""

    config = build_hermes().model_copy(deep=True)
    _set_identity_and_artifacts(
        config,
        name="harbor-swebench-hermes-relay",
        description="Hermes with Relay ATOF and ATIF run evidence.",
        artifact_name="hermes-relay",
    )
    relay_output = f"{ARTIFACT_ROOT}/hermes-relay/relay"
    config.enable_relay(
        output_dir=relay_output,
        observability=RelayObservabilityConfig(
            atif=RelayAtifConfig(
                enabled=True,
                output_directory=relay_output,
                filename_template="trajectory-{session_id}.atif.json",
                agent_name="harbor-swebench-hermes",
            ),
            atof=RelayAtofConfig(
                enabled=True,
                output_directory=relay_output,
                filename="events.atof.jsonl",
                mode="overwrite",
            ),
        ),
    )
    return config
