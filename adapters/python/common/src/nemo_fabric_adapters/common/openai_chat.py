# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for non-streaming OpenAI Chat Completions endpoints."""

from __future__ import annotations

from typing import Any, Protocol

from nemo_fabric_adapter_contract import models as contract


class AsyncJsonClient(Protocol):
    """The small HTTP client surface needed by Chat Completions."""

    async def post(self, url: str, *, json: dict[str, Any]) -> Any: ...


def endpoint(base_url: str) -> str:
    """Return the Chat Completions endpoint for an API root."""

    return f"{base_url.rstrip('/')}/chat/completions"


async def invoke(
    client: AsyncJsonClient,
    url: str,
    *,
    model: str,
    messages: list[dict[str, str]],
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
    metadata: dict[str, str] | None = None,
    user: str | None = None,
) -> tuple[str, contract.AgentUsage | None]:
    """Invoke a non-streaming Chat Completions endpoint and parse its result."""

    payload: dict[str, Any] = {"model": model, "messages": messages}
    if temperature is not None:
        payload["temperature"] = temperature
    if top_p is not None:
        payload["top_p"] = top_p
    if max_tokens is not None:
        payload["max_completion_tokens"] = max_tokens
    if metadata is not None:
        payload["metadata"] = metadata
    if user is not None:
        payload["user"] = user

    response = await client.post(url, json=payload)
    response.raise_for_status()
    value = response.json()
    usage = value.get("usage", {})
    return value["choices"][0]["message"]["content"], _usage(
        usage.get("prompt_tokens"),
        usage.get("completion_tokens"),
        usage.get("total_tokens"),
    )


def _usage(
    input_tokens: int | None,
    output_tokens: int | None,
    total_tokens: int | None,
) -> contract.AgentUsage | None:
    if input_tokens is None and output_tokens is None and total_tokens is None:
        return None
    return contract.AgentUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )
