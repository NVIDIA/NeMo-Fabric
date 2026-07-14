# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Complete Fabric configs and clone-based variants for the example agent."""

from __future__ import annotations

from pathlib import Path

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
from nemo_fabric import TelemetryConfig

BASE_DIR = Path(__file__).resolve().parent
WORKSPACE = "./repos/my-service"
SKILL_PATH = "./skills/code-review"


def base_config() -> FabricConfig:
    """Return a fresh common code-review config."""

    config = FabricConfig(
        metadata=MetadataConfig(
            name="code-review-agent",
            description="Reviews code changes and summarizes correctness risks.",
        ),
        harness=HarnessConfig(
            adapter_id="nvidia.fabric.hermes",
            resolution="preinstalled",
            settings={"workspace": WORKSPACE},
        ),
        models={
            "default": ModelConfig(
                provider="nvidia",
                model="nvidia/nemotron-3-nano-30b-a3b",
                temperature=0.0,
                api_key_env="NVIDIA_API_KEY",
            )
        },
        runtime=RuntimeConfig(
            input_schema="chat",
            output_schema="message",
            artifacts="./artifacts",
        ),
        environment=EnvironmentConfig(
            provider="local",
            workspace=WORKSPACE,
            artifacts="./artifacts/local",
        ),
        telemetry=TelemetryConfig(),
    )
    config.add_skill_path(SKILL_PATH)
    config.add_mcp_server(
        "github",
        transport="streamable-http",
        url="${GITHUB_MCP_URL}",
        exposure="harness_native",
    )
    return config


def hermes_config() -> FabricConfig:
    """Return the complete Hermes variant."""

    config = base_config().model_copy(deep=True)
    config.harness = HarnessConfig(
        adapter_id="nvidia.fabric.hermes",
        resolution="preinstalled",
        settings={
            "workspace": WORKSPACE,
            "hermes_home": "./artifacts/hermes-home",
            "base_url": "https://integrate.api.nvidia.com/v1",
            "max_iterations": 1,
            "max_tokens": 512,
            "temperature": 0.0,
            "reasoning_config": {"effort": "none"},
            "enabled_toolsets": [],
            "system_prompt": "You are a concise smoke test assistant.",
        },
    )
    config.runtime = RuntimeConfig(
        input_schema="chat",
        output_schema="message",
        artifacts="./artifacts/hermes",
    )
    config.environment = EnvironmentConfig(
        provider="local",
        workspace=WORKSPACE,
        artifacts="./artifacts/hermes",
    )
    return config


def codex_cli_config() -> FabricConfig:
    """Return the complete Codex CLI variant without inherited capabilities."""

    config = base_config().model_copy(deep=True)
    config.harness = HarnessConfig(
        adapter_id="nvidia.fabric.codex.cli",
        resolution="preinstalled",
        settings={
            "sandbox": "workspace-write",
            "skip_git_repo_check": True,
            "config_overrides": {"model_reasoning_effort": "high"},
        },
    )
    config.models = {"default": ModelConfig(provider="openai", model="openai/gpt-5.4")}
    config.runtime = RuntimeConfig(
        input_schema="text",
        output_schema="message",
        artifacts="./artifacts/codex-cli",
    )
    config.environment = EnvironmentConfig(
        provider="local",
        workspace=WORKSPACE,
        artifacts="./artifacts/codex-cli",
    )
    config.remove_mcp_server("github")
    config.remove_skill_path(SKILL_PATH)
    return config


def deepagents_config() -> FabricConfig:
    """Return the complete LangChain Deep Agents variant."""

    config = base_config().model_copy(deep=True)
    config.harness = HarnessConfig(
        adapter_id="nvidia.fabric.langchain.deepagents",
        resolution="preinstalled",
        settings={
            "workspace": WORKSPACE,
            "system_prompt": "You are a concise smoke test assistant.",
        },
    )
    config.runtime = RuntimeConfig(
        input_schema="chat",
        output_schema="message",
        artifacts="./artifacts/deepagents",
    )
    config.environment = EnvironmentConfig(
        provider="local",
        workspace=WORKSPACE,
        artifacts="./artifacts/deepagents",
    )
    config.remove_mcp_server("github")
    config.remove_skill_path(SKILL_PATH)
    return config


def with_opensandbox(base: FabricConfig) -> FabricConfig:
    """Return a copy configured for an externally controlled OpenSandbox."""

    config = base.model_copy(deep=True)
    config.environment = EnvironmentConfig(
        provider="opensandbox",
        control_location="external_control",
        workspace="/workspace",
        artifacts="/workspace/artifacts",
        metadata={
            "server_url": "http://127.0.0.1:8080",
            "image": "nvcr.io/nvidia/nemo/fabric-hermes:latest",
        },
    )
    return config


def with_fabric_managed_github_mcp(base: FabricConfig) -> FabricConfig:
    """Return a copy that routes the GitHub MCP server through Fabric."""

    config = base.model_copy(deep=True)
    config.add_mcp_server(
        "github",
        transport="streamable-http",
        url="${GITHUB_MCP_URL}",
        exposure="fabric_managed",
    )
    return config


def with_relay(base: FabricConfig) -> FabricConfig:
    """Return a copy with Relay ATOF and ATIF telemetry enabled."""

    config = base.model_copy(deep=True)
    config.enable_relay(
        output_dir="./artifacts/relay",
        observability=RelayObservabilityConfig(
            atif=RelayAtifConfig(
                enabled=True,
                output_directory="./artifacts/relay",
                filename_template="trajectory-{session_id}.atif.json",
                agent_name="code-review-agent",
                agent_version="fabric-sdk-example",
            ),
            atof=RelayAtofConfig(
                enabled=True,
                output_directory="./artifacts/relay",
                filename="events.atof.jsonl",
                mode="overwrite",
            ),
        ),
    )
    return config


def with_relay_otel(base: FabricConfig) -> FabricConfig:
    """Return a copy with Relay OpenTelemetry export enabled."""

    config = base.model_copy(deep=True)
    config.enable_relay(
        output_dir="./artifacts/relay-otel",
        observability=RelayObservabilityConfig(
            opentelemetry=RelayOtlpConfig(
                enabled=True,
                transport="http_binary",
                endpoint="http://localhost:4318/v1/traces",
                service_name="code-review-agent",
                service_namespace="fabric",
                service_version="fabric-sdk-example",
                instrumentation_scope="nemo-relay-otel",
                timeout_millis=3000,
                resource_attributes={"deployment.environment": "dev"},
            ),
        ),
    )
    return config


def with_relay_openinference(base: FabricConfig) -> FabricConfig:
    """Return a copy with Relay OpenInference export enabled."""

    config = with_relay(base)
    assert config.telemetry is not None
    assert config.relay is not None
    relay = config.relay
    assert not isinstance(relay, dict)
    assert relay.observability is not None
    observability = relay.observability
    assert not isinstance(observability, dict)

    relay.output_dir = "./artifacts/relay-openinference"
    observability.openinference = RelayOtlpConfig(
        enabled=True,
        transport="http_binary",
        endpoint="http://localhost:6006/v1/traces",
    )
    if isinstance(observability.atif, RelayAtifConfig):
        observability.atif.output_directory = "./artifacts/relay-openinference"
    if isinstance(observability.atof, RelayAtofConfig):
        observability.atof.output_directory = "./artifacts/relay-openinference"
    return config


def with_native_otel(base: FabricConfig) -> FabricConfig:
    """Return a copy with adapter-native OpenTelemetry enabled."""

    config = base.model_copy(deep=True)
    config.telemetry = TelemetryConfig(
        providers={
            "native": {
                "config": {
                    "version": 1,
                    "components": [
                        {
                            "kind": "observability",
                            "enabled": True,
                            "config": {
                                "version": 1,
                                "opentelemetry": {
                                    "enabled": True,
                                    "transport": "http_binary",
                                    "endpoint": "http://localhost:4318/v1/traces",
                                    "resource_attributes": {"deployment.environment": "dev"},
                                },
                            },
                        },
                    ],
                },
            }
        },
    )
    return config
