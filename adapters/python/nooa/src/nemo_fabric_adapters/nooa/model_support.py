# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared OO Agents model construction and cleanup."""

from __future__ import annotations

import inspect
import os
from collections.abc import Mapping
from typing import Any

from nemo_fabric_adapter_contract.models import AgentConfig
from nemo_fabric_adapters.common import lifecycle


def _config_error(code: str, message: str, **metadata: Any) -> lifecycle.LifecycleError:
    return lifecycle.LifecycleError(code, message, metadata=metadata or None)


def _credential_env(provider: str, configured: str | None) -> str:
    if configured is not None:
        return configured
    defaults = {
        "anthropic": "ANTHROPIC_API_KEY",
        "nvidia": "NVIDIA_API_KEY",
        "openai": "OPENAI_API_KEY",
    }
    try:
        return defaults[provider]
    except KeyError as error:
        raise _config_error(
            "nooa_model_credential_required",
            f"OO Agents model provider {provider!r} requires api_key_env",
            field="models",
        ) from error


def _native_model_name(provider: str, model: str) -> str:
    if provider == "nvidia":
        if model.startswith("nvidia_nim/"):
            return model
        return f"nvidia_nim/{model}"
    if model.startswith(f"{provider}/"):
        return model
    return f"{provider}/{model}"


async def _await_if_needed(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


async def close_models(models: Mapping[str, Any]) -> None:
    """Close each distinct OO Agents model client exactly once."""

    closed: set[int] = set()
    for model in models.values():
        if id(model) in closed:
            continue
        closed.add(id(model))
        close = getattr(model, "aclose", None)
        if callable(close):
            await _await_if_needed(close())
            continue
        close = getattr(model, "close", None)
        if callable(close):
            await _await_if_needed(close())


def selected_model(models: Mapping[str, Any], *, target_name: str) -> Any:
    """Select the default or only model for an InteractiveAgent target."""

    if "default" in models:
        return models["default"]
    if len(models) == 1:
        return next(iter(models.values()))
    raise _config_error(
        "nooa_invalid_models",
        f"{target_name} requires a default model or exactly one model",
        field="models",
    )


async def build_models(config: AgentConfig) -> dict[str, Any]:
    """Translate normalized Fabric model roles into OO Agents clients."""

    if not config.models:
        return {}
    try:
        from nooa.unifiedllm import get_llm_client
    except Exception as error:
        raise _config_error(
            "nooa_dependency_missing",
            "OO Agents model support is not available in the adapter environment",
        ) from error

    result: dict[str, Any] = {}
    try:
        for role, model in config.models.items():
            credential_env = _credential_env(model.provider, model.api_key_env)
            api_key = os.environ.get(credential_env)
            if not api_key:
                raise _config_error(
                    "nooa_model_credential_missing",
                    f"OO Agents model credential environment variable {credential_env!r} is not set",
                    field=f"models.{role}.api_key_env",
                )
            settings = dict(model.settings)
            client_type = settings.pop("client_type", None)
            overrides: dict[str, Any] = {"api_key": api_key, **settings}
            if model.base_url is not None:
                overrides["api_base"] = model.base_url
            if model.temperature is not None:
                overrides["temperature"] = model.temperature
            result[role] = get_llm_client(
                _native_model_name(model.provider, model.model),
                client_type=client_type,
                **overrides,
            )
    except BaseException as error:
        try:
            await close_models(result)
        except BaseException as cleanup_error:
            error.add_note(
                f"OO Agents model cleanup also failed ({type(cleanup_error).__name__})"
            )
        raise
    return result
