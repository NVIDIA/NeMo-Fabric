# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Built-in, code-defined Fabric CLI presets and examples."""

from __future__ import annotations

import importlib.util
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from nemo_fabric.models import EnvironmentConfig
from nemo_fabric.models import FabricConfig
from nemo_fabric.models import HarnessConfig
from nemo_fabric.models import MetadataConfig
from nemo_fabric.models import ModelConfig
from nemo_fabric.models import RuntimeConfig

ConfigFactory = Callable[[], FabricConfig]
BUNDLED_BASE_DIR = Path(__file__).resolve().parent / "_bundled"


@dataclass(frozen=True)
class Preset:
    """One maintained, complete Fabric configuration."""

    name: str
    description: str
    factory: ConfigFactory
    base_dir: Path = BUNDLED_BASE_DIR
    install_extra: str | None = None
    required_env: tuple[str, ...] = ()
    probe_module: str | None = None

    @property
    def available(self) -> bool:
        """Return whether the preset's adapter implementation is importable."""

        if self.probe_module is None:
            return True
        try:
            return importlib.util.find_spec(self.probe_module) is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            return False

    def discovery(self) -> dict[str, object]:
        """Return user-facing preset requirements without serializing config."""

        install = "pip install nemo-fabric"
        if self.install_extra is not None:
            install = f"pip install 'nemo-fabric[{self.install_extra}]'"
        return {
            "name": self.name,
            "description": self.description,
            "available": self.available,
            "install": install,
            "required_env": list(self.required_env),
            "missing_env": [name for name in self.required_env if not os.environ.get(name)],
        }


@dataclass(frozen=True)
class Example:
    """One installed example with one or more complete variants."""

    name: str
    description: str
    variants: dict[str, ConfigFactory]
    default_variant: str
    base_dir: Path
    template_dir: Path | None = None


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


def scripted() -> FabricConfig:
    """Return a deterministic credential-free experimentation preset."""

    return FabricConfig(
        metadata=MetadataConfig(
            name="scripted-agent",
            description="Credential-free NeMo Fabric CLI smoke preset.",
        ),
        harness=HarnessConfig(
            adapter_id="nvidia.fabric.scripted",
            resolution="preinstalled",
        ),
        runtime=RuntimeConfig(input_schema="text", output_schema="message"),
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
    "scripted": Preset(
        "scripted",
        "Credential-free deterministic smoke preset.",
        scripted,
    ),
    "hermes": Preset(
        "hermes",
        "Hermes Agent with an NVIDIA-hosted model.",
        hermes,
        install_extra="hermes",
        required_env=("NVIDIA_API_KEY",),
        probe_module="nemo_fabric_adapters.hermes.adapter",
    ),
    "claude": Preset(
        "claude",
        "Claude Code with an Anthropic model.",
        claude,
        install_extra="claude",
        required_env=("ANTHROPIC_API_KEY",),
        probe_module="nemo_fabric_adapters.claude.adapter",
    ),
    "codex": Preset(
        "codex",
        "Codex with an OpenAI model.",
        codex,
        install_extra="codex",
        required_env=("OPENAI_API_KEY",),
        probe_module="nemo_fabric_adapters.codex.adapter",
    ),
    "deepagents": Preset(
        "deepagents",
        "LangChain Deep Agents with an NVIDIA-hosted model.",
        deepagents,
        install_extra="deepagents",
        required_env=("NVIDIA_API_KEY",),
        probe_module="nemo_fabric_adapters.deepagents.adapter",
    ),
}

EXAMPLE_VARIANTS = ("hermes", "claude", "codex", "deepagents")

EXAMPLES: dict[str, Example] = {
    "examples.code_review_agent": Example(
        name="examples.code_review_agent",
        description="Installed, editable-in-Python code review example.",
        variants={name: (lambda name=name: _code_review(name)) for name in EXAMPLE_VARIANTS},
        default_variant="hermes",
        base_dir=BUNDLED_BASE_DIR,
        template_dir=BUNDLED_BASE_DIR / "examples" / "code_review_agent",
    )
}
