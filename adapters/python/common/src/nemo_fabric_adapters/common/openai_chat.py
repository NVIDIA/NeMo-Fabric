# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for streaming OpenAI Chat Completions endpoints."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from types import TracebackType
from typing import Any, Protocol

from nemo_fabric_adapter_contract import models as contract


class AsyncStreamResponse(Protocol):
    """The response surface needed to consume server-sent events."""

    def raise_for_status(self) -> None: ...

    def aiter_lines(self) -> AsyncIterator[str]: ...


class AsyncStreamContext(Protocol):
    """The asynchronous context manager returned by an HTTP stream."""

    async def __aenter__(self) -> AsyncStreamResponse: ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None: ...


class AsyncChatClient(Protocol):
    """The small HTTP client surface needed by Chat Completions."""

    def stream(
        self,
        method: str,
        url: str,
        *,
        json: dict[str, Any],
    ) -> AsyncStreamContext: ...


def endpoint(base_url: str) -> str:
    """Return the Chat Completions endpoint for an API root."""

    return f"{base_url.rstrip('/')}/chat/completions"


async def invoke(
    client: AsyncChatClient,
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
    """Consume a Chat Completions SSE stream and aggregate its result."""

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    for key_name, value in (
        ("temperature", temperature),
        ("top_p", top_p),
        ("max_completion_tokens", max_tokens),
        ("metadata", metadata),
        ("user", user),
    ):
        if value is not None:
            payload[key_name] = value

    text: list[str] = []
    usage: dict[str, Any] = {}
    completed = False
    async with client.stream("POST", url, json=payload) as response:
        response.raise_for_status()
        async for data in _sse_data(response):
            if data == "[DONE]":
                completed = True
                break

            value = _json_object(data)
            if "error" in value:
                raise ValueError("Chat Completions stream returned an error")

            event_usage = value.get("usage")
            if event_usage is not None:
                if not isinstance(event_usage, dict):
                    raise TypeError("Chat Completions usage must be an object")
                usage = event_usage

            choices = value.get("choices", [])
            if not isinstance(choices, list):
                raise TypeError("Chat Completions choices must be an array")
            for choice in choices:
                if not isinstance(choice, dict):
                    raise TypeError("Chat Completions choice must be an object")
                if choice.get("index", 0) != 0:
                    continue
                delta = choice.get("delta", {})
                if not isinstance(delta, dict):
                    raise TypeError("Chat Completions delta must be an object")
                content = delta.get("content")
                if content is not None:
                    if not isinstance(content, str):
                        raise TypeError("Chat Completions content must be a string")
                    text.append(content)

    if not completed:
        raise ValueError("Chat Completions stream ended without [DONE]")

    return "".join(text), _usage(
        usage.get("prompt_tokens"),
        usage.get("completion_tokens"),
        usage.get("total_tokens"),
    )


async def _sse_data(response: AsyncStreamResponse) -> AsyncIterator[str]:
    data: list[str] = []
    async for line in response.aiter_lines():
        if not line:
            if data:
                yield "\n".join(data)
            data = []
        elif line.startswith("data:"):
            data.append(line.removeprefix("data:").lstrip())
    if data:
        yield "\n".join(data)


def _json_object(data: str) -> dict[str, Any]:
    value = json.loads(data)
    if not isinstance(value, dict):
        raise TypeError("Chat Completions event data must be an object")
    return value


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
