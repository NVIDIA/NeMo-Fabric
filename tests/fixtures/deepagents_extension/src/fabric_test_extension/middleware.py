# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Custom LangChain middleware referenced from harness.settings.deepagents.middleware."""

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import SystemMessage


def _with_text(request, text: str):
    base = request.system_message
    if base is None:
        blocks = []
    elif isinstance(base.content, str):
        blocks = [{"type": "text", "text": base.content}]
    else:
        blocks = list(base.content)
    return request.override(
        system_message=SystemMessage(content=[*blocks, {"type": "text", "text": text}])
    )


class SystemMarker(AgentMiddleware):
    """Append a marker to the system prompt before every model call."""

    def __init__(self, marker: str = "custom-middleware-marker") -> None:
        super().__init__()
        self.marker = marker

    def wrap_model_call(self, request, handler):
        return handler(_with_text(request, self.marker))

    async def awrap_model_call(self, request, handler):
        return await handler(_with_text(request, self.marker))


class SecondMarker(SystemMarker):
    """A second, distinctly named marker middleware."""


class ExtraState(AgentState):
    counter: int


class StatefulMiddleware(AgentMiddleware):
    """Middleware with a custom state schema, which the adapter rejects."""

    state_schema = ExtraState


class FilesystemMiddleware(AgentMiddleware):
    """Middleware that reuses a Deep Agents built-in middleware name."""


def not_middleware() -> None:
    """A plain function; the adapter must reject it without calling it."""
    raise AssertionError("the adapter called a function it should have rejected")
