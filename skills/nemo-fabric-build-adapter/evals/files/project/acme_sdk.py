# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic target SDK for the adapter exercises; no external service."""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any


@dataclass
class AcmeReply:
    text: str
    input_tokens: int
    output_tokens: int


class AcmeSession:
    """A live target session with cumulative usage and inspectable history."""

    DEFAULT_SYSTEM_PROMPT = "You are AcmeHarness."

    def __init__(self, options: dict[str, Any]):
        self.options = options
        self.system = options.get("system", self.DEFAULT_SYSTEM_PROMPT)
        self.history: list[str] = []
        self.input_tokens = 0
        self.output_tokens = 0
        self.closed = False
        self.last_reply: AcmeReply | None = None

    async def run(self, text: str) -> AcmeReply:
        if self.closed:
            raise RuntimeError("session_closed")
        self.history.append(text)
        self.input_tokens += len(text.split())
        self.output_tokens += 2
        self.last_reply = AcmeReply(
            text=" | ".join(self.history),
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
        )
        return self.last_reply

    async def stream(self, text: str) -> AsyncIterator[dict[str, Any]]:
        reply = await self.run(text)
        yield {
            "id": "acme-chunk",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "acme-model",
            "choices": [
                {"index": 0, "delta": {"content": reply.text}, "finish_reason": None}
            ],
        }

    def get_final_reply(self) -> AcmeReply:
        if self.last_reply is None:
            raise RuntimeError("no_completed_invocation")
        return self.last_reply

    async def close(self) -> None:
        self.closed = True
