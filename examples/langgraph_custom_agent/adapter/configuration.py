# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Translate typed AgentConfig into native LangChain dependencies."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from nemo_fabric_adapter_contract.models import AgentConfig
from nemo_fabric_adapters.common import instructions as common_instructions
from nemo_fabric_adapters.common import lifecycle

DEFAULT_SYSTEM_INSTRUCTION = (
    "Explain the fixed email-risk classification using only the supplied email "
    "and extracted signals."
)
OPENAI_COMPATIBLE_PROVIDERS = frozenset(
    {"nvidia", "openai", "openai-compatible"}
)
MODEL_REQUEST_TIMEOUT_SECONDS = 60


@dataclass(frozen=True, slots=True)
class AgentDependencies:
    """Native dependencies passed to the custom agent constructor."""

    model: BaseChatModel
    system_instruction: str


def _config_error(field: str, message: str) -> lifecycle.LifecycleError:
    return lifecycle.LifecycleError(
        "email_phishing_invalid_config",
        message,
        metadata={"field": field},
    )


def resolve_agent_dependencies(agent_config: AgentConfig) -> AgentDependencies:
    """Resolve the narrow configuration surface advertised by the descriptor."""

    if set(agent_config.models) != {"default"}:
        raise _config_error(
            "models",
            "The email-phishing adapter requires exactly one model named 'default'",
        )
    model_config = agent_config.models["default"]
    if model_config.settings:
        raise _config_error(
            "models.default.settings",
            "The email-phishing adapter does not accept model-specific settings",
        )
    if model_config.provider not in OPENAI_COMPATIBLE_PROVIDERS:
        raise _config_error(
            "models.default.provider",
            f"Unsupported OpenAI-compatible provider {model_config.provider!r}",
        )
    if model_config.provider != "openai" and model_config.base_url is None:
        raise _config_error(
            "models.default.base_url",
            f"Provider {model_config.provider!r} requires an explicit base URL",
        )

    api_key_env = model_config.api_key_env
    if api_key_env is None and model_config.provider == "openai":
        api_key_env = "OPENAI_API_KEY"
    if api_key_env is None:
        raise _config_error(
            "models.default.api_key_env",
            f"Provider {model_config.provider!r} requires a credential environment variable",
        )
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise _config_error(
            "models.default.api_key_env",
            f"Credential environment variable {api_key_env!r} is not set",
        )

    chat_model_options: dict[str, Any] = {
        "model": model_config.model,
        "api_key": api_key,
        "timeout": MODEL_REQUEST_TIMEOUT_SECONDS,
    }
    if model_config.base_url is not None:
        chat_model_options["base_url"] = model_config.base_url
    if model_config.temperature is not None:
        chat_model_options["temperature"] = model_config.temperature

    system_instruction = DEFAULT_SYSTEM_INSTRUCTION
    instruction = common_instructions.system_instruction(
        agent_config,
        adapter="email-phishing",
        supported_modes={"replace", "append"},
    )
    if instruction is not None:
        system_instruction = (
            instruction.content
            if instruction.mode == "replace"
            else f"{DEFAULT_SYSTEM_INSTRUCTION}\n\n{instruction.content}"
        )

    return AgentDependencies(
        model=ChatOpenAI(**chat_model_options),
        system_instruction=system_instruction,
    )
