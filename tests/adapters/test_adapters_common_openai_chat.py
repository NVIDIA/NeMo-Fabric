# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Token usage parsing in the shared Chat Completions stream helper."""

from __future__ import annotations

import json

import httpx
import pytest
from nemo_fabric_adapter_contract.models import AgentUsage
from nemo_fabric_adapters.common import openai_chat

_URL = "https://gateway.example.test/v1/chat/completions"


def _stream(usage: object) -> str:
    """Return an OpenAI-style stream: content chunks carry ``usage: null``."""

    chunks = [
        {
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"content": "done"}}],
            "usage": None,
        },
        {"object": "chat.completion.chunk", "choices": [], "usage": usage},
    ]
    return "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + (
        "data: [DONE]\n\n"
    )


async def _invoke(usage: object) -> AgentUsage | None:
    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=_stream(usage))

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        text, reported = await openai_chat.invoke(
            client, _URL, model="openclaw/default", messages=[]
        )
    assert text == "done"
    return reported


@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        pytest.param(
            # OpenClaw 2026.9.4 sends prompt_tokens = input + cache reads, plus
            # both detail objects when it knows cache reads and reasoning.
            {
                "prompt_tokens": 12000,
                "completion_tokens": 300,
                "total_tokens": 12300,
                "prompt_tokens_details": {"cached_tokens": 9000},
                "completion_tokens_details": {"reasoning_tokens": 120},
            },
            AgentUsage(
                input_tokens=12000,
                cached_input_tokens=9000,
                input_tokens_include_cache=True,
                output_tokens=300,
                total_tokens=12300,
            ),
            id="cached-and-reasoning",
        ),
        pytest.param(
            # OpenClaw omits prompt_tokens_details when nothing was read from cache.
            {"prompt_tokens": 40, "completion_tokens": 5, "total_tokens": 45},
            AgentUsage(
                input_tokens=40,
                input_tokens_include_cache=True,
                output_tokens=5,
                total_tokens=45,
            ),
            id="cache-reads-not-reported",
        ),
        pytest.param(
            {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "prompt_tokens_details": {"cached_tokens": 0},
            },
            AgentUsage(
                input_tokens=0,
                cached_input_tokens=0,
                input_tokens_include_cache=True,
                output_tokens=0,
                total_tokens=0,
            ),
            id="explicit-zeros",
        ),
        pytest.param(
            {
                "prompt_tokens": 40,
                "completion_tokens": 5,
                "total_tokens": 45,
                "prompt_tokens_details": None,
                "completion_tokens_details": None,
            },
            AgentUsage(
                input_tokens=40,
                input_tokens_include_cache=True,
                output_tokens=5,
                total_tokens=45,
            ),
            id="null-details",
        ),
        pytest.param(
            {"completion_tokens": 5},
            AgentUsage(output_tokens=5),
            id="input-and-total-omitted",
        ),
        pytest.param(
            {"prompt_tokens_details": {"cached_tokens": 0}},
            AgentUsage(cached_input_tokens=0),
            id="only-cached-count",
        ),
        pytest.param(None, None, id="no-usage"),
    ],
)
async def test_invoke_usage_reports_cached_input_and_keeps_omitted_counts_unknown(
    usage: object, expected: AgentUsage | None
):
    assert await _invoke(usage) == expected


@pytest.mark.parametrize(
    "details",
    [
        pytest.param([], id="empty-list"),
        pytest.param(0, id="zero"),
        pytest.param(False, id="false"),
        pytest.param("", id="empty-string"),
    ],
)
async def test_invoke_usage_rejects_non_object_prompt_tokens_details(details: object):
    # Only a null details object means no cached count (see "null-details");
    # any other non-object value is an invalid report.
    usage = {
        "prompt_tokens": 40,
        "completion_tokens": 5,
        "total_tokens": 45,
        "prompt_tokens_details": details,
    }

    with pytest.raises(TypeError, match="prompt_tokens_details must be an object"):
        await _invoke(usage)


async def test_invoke_usage_keeps_every_count_known_across_a_cached_run():
    # An OpenAI run: the first call reads nothing from cache and reports an
    # explicit 0; later calls reuse the cached prompt prefix.
    reports = [
        (1200, 0, 80),
        (1350, 1024, 95),
        (1500, 1280, 60),
    ]

    for prompt, cached, completion in reports:
        usage = {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
            "prompt_tokens_details": {"cached_tokens": cached, "audio_tokens": 0},
            "completion_tokens_details": {"reasoning_tokens": 0},
        }

        assert await _invoke(usage) == AgentUsage(
            input_tokens=prompt,
            cached_input_tokens=cached,
            input_tokens_include_cache=True,
            output_tokens=completion,
            total_tokens=prompt + completion,
        )
