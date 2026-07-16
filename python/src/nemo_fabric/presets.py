# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Built-in, code-defined Fabric CLI presets and examples."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from nemo_fabric.models import EnvironmentConfig, FabricConfig, HarnessConfig, MetadataConfig, ModelConfig

ConfigFactory = Callable[[], FabricConfig]
BUNDLED_BASE_DIR = Path(__file__).resolve().parent / "_bundled"


@dataclass(frozen=True)
class Preset:
    """One maintained, complete Fabric configuration."""

    name: str
    description: str
    factory: ConfigFactory
    base_dir: Path = BUNDLED_BASE_DIR


@dataclass(frozen=True)
class Example:
    """One installed example with one or more complete variants."""

    name: str
    description: str
    variants: dict[str, ConfigFactory]
    default_variant: str
    base_dir: Path


def _preset(
    *,
    name: str,
    adapter_id: str,
    provider: str,
    model: str,
    settings: dict[str, object] | None = None,
) -> FabricConfig:
    return FabricConfig(
        metadata=MetadataConfig(name=f"{name}-agent", description=f"NeMo Fabric {name} CLI preset."),
        harness=HarnessConfig(adapter_id=adapter_id, resolution="preinstalled", settings=settings or {}),
        models={"default": ModelConfig(provider=provider, model=model)},
        environment=EnvironmentConfig(provider="local"),
    )


def hermes() -> FabricConfig:
    """Return a fresh Hermes preset."""

    return _preset(
        name="hermes",
        adapter_id="nvidia.fabric.hermes",
        provider="nvidia",
        model="nvidia/nemotron-3-nano-30b-a3b",
        settings={"base_url": "https://integrate.api.nvidia.com/v1"},
    )


def claude() -> FabricConfig:
    """Return a fresh Claude preset."""

    return _preset(
        name="claude",
        adapter_id="nvidia.fabric.claude",
        provider="anthropic",
        model="anthropic/claude-sonnet-4-5",
        settings={"permission_mode": "dontAsk"},
    )


def codex() -> FabricConfig:
    """Return a fresh Codex preset."""

    return _preset(
        name="codex",
        adapter_id="nvidia.fabric.codex",
        provider="openai",
        model="openai/gpt-5.4",
        settings={"sandbox": "workspace-write"},
    )


def deepagents() -> FabricConfig:
    """Return a fresh Deep Agents preset."""

    return _preset(
        name="deepagents",
        adapter_id="nvidia.fabric.langchain.deepagents",
        provider="nvidia",
        model="nvidia/nemotron-3-nano-30b-a3b",
    )


def _code_review(variant: str) -> FabricConfig:
    config = PRESETS[variant].factory()
    config.metadata = MetadataConfig(
        name="code-review-agent",
        description="Reviews a small Python workspace for correctness risks.",
    )
    config.environment = EnvironmentConfig(provider="local", workspace="examples/code_review_agent/repo")
    config.add_skill_path("examples/code_review_agent/skills/code-review.md")
    return config


PRESETS: dict[str, Preset] = {
    "hermes": Preset("hermes", "Hermes Agent with an NVIDIA-hosted model.", hermes),
    "claude": Preset("claude", "Claude Code with an Anthropic model.", claude),
    "codex": Preset("codex", "Codex with an OpenAI model.", codex),
    "deepagents": Preset("deepagents", "LangChain Deep Agents with an NVIDIA-hosted model.", deepagents),
}

EXAMPLES: dict[str, Example] = {
    "examples.code_review_agent": Example(
        name="examples.code_review_agent",
        description="Installed, editable-in-Python code review example.",
        variants={name: (lambda name=name: _code_review(name)) for name in PRESETS},
        default_variant="hermes",
        base_dir=BUNDLED_BASE_DIR,
    )
}
