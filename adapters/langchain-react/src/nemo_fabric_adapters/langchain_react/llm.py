# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenAI-compatible chat model binding for Fabric model config and trial overrides."""

from __future__ import annotations

import os
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI


def default_base_url(provider: str | None) -> str | None:
    if provider in {"nvidia", "openai"}:
        return "https://integrate.api.nvidia.com/v1"
    return None


def merge_mapping(base: dict[str, Any], overlay: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(base)
    if not overlay:
        return merged
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_mapping(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_model_config(
    models: dict[str, Any],
    llm_name: str,
    settings: dict[str, Any],
    overrides: Any | None,
) -> dict[str, Any]:
    model_config = models.get(llm_name) or models.get("default") or {}
    if not isinstance(model_config, dict):
        model_config = {}
    nested_settings = model_config.get("settings")
    if isinstance(nested_settings, dict):
        model_config = merge_mapping(model_config, nested_settings)

    resolved = {
        "provider": settings.get("provider") or model_config.get("provider"),
        "model": settings.get("model_name") or model_config.get("model") or model_config.get("model_name"),
        "temperature": model_config.get("temperature", 0.0),
        "top_p": model_config.get("top_p", 1.0),
        "max_tokens": model_config.get("max_tokens"),
        "base_url": settings.get("base_url") or model_config.get("base_url") or default_base_url(model_config.get("provider")),
        "api_key_env": model_config.get("api_key_env") or settings.get("api_key_env") or "OPENAI_API_KEY",
        "api_key": model_config.get("api_key"),
    }

    if isinstance(overrides, dict):
        models_override = overrides.get("models")
        if isinstance(models_override, dict):
            model_override = models_override.get(llm_name) or models_override.get("default")
            if isinstance(model_override, dict):
                resolved = merge_mapping(resolved, model_override)
        direct_keys = {
            key: overrides[key]
            for key in ("temperature", "top_p", "max_tokens", "model", "model_name", "base_url", "api_key")
            if key in overrides
        }
        if "model_name" in direct_keys and "model" not in direct_keys:
            direct_keys["model"] = direct_keys.pop("model_name")
        resolved = merge_mapping(resolved, direct_keys)
    return resolved


def build_chat_model(model_config: dict[str, Any]) -> BaseChatModel:
    api_key_env = str(model_config.get("api_key_env") or "OPENAI_API_KEY")
    api_key = model_config.get("api_key")
    if api_key is None:
        api_key = os.environ.get(api_key_env)
    if not api_key and model_config.get("allow_empty_api_key", False):
        api_key = "not-used"
    if not api_key:
        raise RuntimeError(
            f"{api_key_env} is required for langchain-react OpenAI-compatible mode "
            "(set models.<name>.api_key, allow_empty_api_key: true, or export the env var)"
        )

    model_name = model_config.get("model")
    if not model_name:
        raise RuntimeError("models.<llm_name>.model is required for langchain-react")

    kwargs: dict[str, Any] = {
        "model": model_name,
        "temperature": float(model_config.get("temperature", 0.0)),
        "top_p": float(model_config.get("top_p", 1.0)),
        "api_key": api_key,
    }
    if model_config.get("base_url"):
        kwargs["base_url"] = model_config["base_url"]
    if model_config.get("max_tokens") is not None:
        kwargs["max_tokens"] = int(model_config["max_tokens"])
    return ChatOpenAI(**kwargs)
