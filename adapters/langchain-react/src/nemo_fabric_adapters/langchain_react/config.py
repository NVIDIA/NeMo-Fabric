# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fabric payload helpers for langchain-react."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_core.messages import trim_messages

from nemo_fabric_adapters.langchain_react.react.graph import WorkflowSettings


def request_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return payload.get("request") or {}


def workflow_settings(payload: dict[str, Any]) -> WorkflowSettings:
    settings = (payload.get("effective_config") or {}).get("config", {}).get("harness", {}).get("settings") or {}
    if not isinstance(settings, dict):
        settings = {}
    workflow = settings.get("workflow") or {}
    if not isinstance(workflow, dict):
        workflow = {}
    return WorkflowSettings.model_validate(workflow)


def tools_config(payload: dict[str, Any]) -> dict[str, Any]:
    settings = (payload.get("effective_config") or {}).get("config", {}).get("harness", {}).get("settings") or {}
    if not isinstance(settings, dict):
        return {}
    tools = settings.get("tools") or {}
    return tools if isinstance(tools, dict) else {}


def harness_settings(payload: dict[str, Any]) -> dict[str, Any]:
    settings = (payload.get("effective_config") or {}).get("config", {}).get("harness", {}).get("settings") or {}
    return settings if isinstance(settings, dict) else {}


def models_payload(payload: dict[str, Any]) -> dict[str, Any]:
    config = (payload.get("effective_config") or {}).get("config") or {}
    models = config.get("models") or {}
    return models if isinstance(models, dict) else {}


def request_input_text(payload: dict[str, Any]) -> str:
    request = request_payload(payload)
    raw_input = request.get("input")
    if raw_input is None:
        return ""
    if isinstance(raw_input, str):
        return raw_input
    if isinstance(raw_input, dict):
        for key in ("question", "text", "input", "message"):
            value = raw_input.get(key)
            if isinstance(value, str):
                return value
    return str(raw_input)


def request_overrides(payload: dict[str, Any]) -> Any | None:
    request = request_payload(payload)
    return request.get("overrides")


def request_timezone(payload: dict[str, Any]) -> str | None:
    request = request_payload(payload)
    context = request.get("context") or {}
    if not isinstance(context, dict):
        return None
    for key in ("timezone", "x-timezone", "x_timezone"):
        value = context.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _message_from_mapping(item: dict[str, Any]) -> BaseMessage:
    role = str(item.get("role", "user")).lower()
    content = item.get("content", "")
    if role == "system":
        return SystemMessage(content=str(content))
    if role in {"assistant", "ai"}:
        return AIMessage(content=str(content))
    return HumanMessage(content=str(content))


def request_messages(payload: dict[str, Any], *, max_history: int) -> list[BaseMessage]:
    request = request_payload(payload)
    raw_input = request.get("input")
    messages: list[BaseMessage] = []

    if isinstance(raw_input, list):
        for item in raw_input:
            if isinstance(item, dict):
                messages.append(_message_from_mapping(item))
    elif isinstance(raw_input, dict):
        nested = raw_input.get("messages")
        if isinstance(nested, list):
            for item in nested:
                if isinstance(item, dict):
                    messages.append(_message_from_mapping(item))
        else:
            text = request_input_text(payload)
            if text:
                messages = [HumanMessage(content=text)]
    else:
        text = request_input_text(payload)
        if text:
            messages = [HumanMessage(content=text)]

    if not messages:
        return [HumanMessage(content="")]

    return list(
        trim_messages(
            messages=messages,
            max_tokens=max_history,
            strategy="last",
            token_counter=len,
            start_on="human",
            include_system=True,
        )
    )
