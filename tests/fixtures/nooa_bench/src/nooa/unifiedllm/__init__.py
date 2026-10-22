# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic NOOA client seam with HTTPX model traffic kept in memory."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import httpx


class FixtureModel:
    def __init__(self, name: str, **settings: Any) -> None:
        self.name = name
        self.model = name.split("/", 1)[-1]
        self.settings = settings
        transport = httpx.MockTransport(self._respond)
        self._http = SimpleNamespace(
            httpx_sync=httpx.Client(transport=transport),
            httpx_async=httpx.AsyncClient(transport=transport),
        )

    def _respond(self, request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        return httpx.Response(
            200,
            request=request,
            json={
                "id": "fixture-completion",
                "model": payload["model"],
                "usage": {
                    "prompt_tokens": 12,
                    "prompt_tokens_details": {"cached_tokens": 0},
                    "completion_tokens": 4,
                    "total_tokens": 16,
                },
                "choices": [{"message": {"role": "assistant", "content": "done"}}],
            },
        )

    def call(self, messages: list[dict[str, str]], **kwargs: Any) -> dict[str, Any]:
        response = self._http.httpx_sync.post(
            "https://fixture.invalid/v1/chat/completions",
            json={"model": self.model, "messages": messages},
        )
        response.raise_for_status()
        return response.json()

    async def acall(
        self, messages: list[dict[str, str]], **kwargs: Any
    ) -> dict[str, Any]:
        response = await self._http.httpx_async.post(
            "https://fixture.invalid/v1/chat/completions",
            json={"model": self.model, "messages": messages},
        )
        response.raise_for_status()
        return response.json()

    async def aclose(self) -> None:
        self._http.httpx_sync.close()
        await self._http.httpx_async.aclose()


def get_llm_client(
    name: str,
    *,
    client_type: str | None = None,
    **overrides: Any,
) -> FixtureModel:
    return FixtureModel(name, client_type=client_type, **overrides)
