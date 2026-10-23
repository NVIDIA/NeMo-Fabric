# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Actual provider transformation and SDK telemetry; HTTP stays in memory."""

import json
from unittest.mock import MagicMock

import httpx
import pytest
from nemo_fabric_adapters.openhands.usage import CompletionReports

KEY = "nemo_fabric.usage_accounting.v1"


@pytest.mark.parametrize(
    "raw,expected_input,expected_cache,expected_writes,expected_output",
    [
        ({"input_tokens": 10, "output_tokens": 2}, None, None, None, 2),
        (
            {
                "input_tokens": 10,
                "output_tokens": 2,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            },
            None,
            None,
            None,
            2,
        ),
        (
            {
                "input_tokens": 10,
                "output_tokens": 2,
                "cache_read_input_tokens": 3,
                "cache_creation_input_tokens": 4,
            },
            17,
            3,
            4,
            2,
        ),
        (
            {
                "output_tokens": 2,
                "cache_read_input_tokens": 3,
                "cache_creation_input_tokens": 4,
            },
            None,
            3,
            4,
            2,
        ),
        (
            {
                "input_tokens": 10,
                "cache_read_input_tokens": 3,
                "cache_creation_input_tokens": 4,
            },
            17,
            3,
            4,
            None,
        ),
        (
            {
                "iterations": [
                    {
                        "input_tokens": 10,
                        "output_tokens": 2,
                        "cache_read_input_tokens": 3,
                        "cache_creation_input_tokens": 4,
                    },
                    {"input_tokens": 5, "output_tokens": 1},
                ]
            },
            None,
            None,
            None,
            None,
        ),
        (
            {"input_tokens": 10, "output_tokens": 20, "_thinking": True},
            None,
            None,
            None,
            20,
        ),
    ],
)
async def test_real_anthropic_transform_never_certifies_synthesized_presence(
    monkeypatch, raw, expected_input, expected_cache, expected_writes, expected_output
):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    litellm = pytest.importorskip("litellm")
    pytest.importorskip("openhands.sdk")
    from litellm.llms.custom_httpx.http_handler import HTTPHandler
    from openhands.sdk import LLM
    from openhands.sdk.llm import Message, TextContent
    from openhands.sdk.llm import llm as sdk_llm

    monkeypatch.setattr(
        "socket.getaddrinfo",
        MagicMock(side_effect=AssertionError("outbound DNS forbidden")),
    )
    monkeypatch.setattr(
        "socket.socket.connect",
        MagicMock(side_effect=AssertionError("outbound connection forbidden")),
    )
    raw = dict(raw)
    thinking = raw.pop("_thinking", False)
    body = {
        "id": "msg",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-5",
        "content": (
            [
                {
                    "type": "thinking",
                    "thinking": "Reason about the task",
                    "signature": "fake",
                }
            ]
            if thinking
            else []
        )
        + [{"type": "text", "text": "done"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": raw,
    }
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json=body, request=request)

    with httpx.Client(transport=httpx.MockTransport(respond)) as http:
        client = HTTPHandler(client=http)

        async def transport(**kwargs):
            return litellm.completion(
                model="anthropic/claude-sonnet-4-5",
                messages=[{"role": "user", "content": "scripted"}],
                api_key="fake",
                client=client,
            )

        monkeypatch.setattr(sdk_llm, "litellm_acompletion", transport)
        llm = LLM(model="anthropic/claude-sonnet-4-5", api_key="fake", num_retries=1)
        source = CompletionReports("claude-sonnet-4-5", "anthropic")
        captured = []

        def record(filename, data):
            captured.append(json.loads(data)["response"]["usage"])
            source.record(filename, data)

        llm.telemetry.set_log_completions_callback(record)
        llm.retry_listener = source.retry
        llm.telemetry.log_enabled = True
        baseline = llm.metrics.deep_copy()
        with source.invocation() as owner:
            await llm.agenerate(
                [Message(role="user", content=[TextContent(text="scripted")])]
            )
            usage = owner.usage(llm.metrics.diff(baseline))
    assert len(requests) == 1
    if thinking:
        assert captured[0]["completion_tokens_details"]["reasoning_tokens"] > 0
    assert usage.input_tokens == expected_input
    assert usage.cached_input_tokens == expected_cache
    assert usage.output_tokens == expected_output
    model = usage.extensions[KEY]["models"][0]
    assert model["reported"].get("cache_write_input_tokens") == expected_writes
    assert model["reported"].get("reasoning_tokens") is None
    assert model["model"] == "claude-sonnet-4-5"
    if raw.get("input_tokens") and expected_input is None:
        assert model["observed"]["input_tokens"] >= raw["input_tokens"]
    if raw.get("iterations"):
        assert captured[0]["iterations"] == raw["iterations"]
        assert model["reported"].get("input_tokens") is None
        assert model["reported"].get("output_tokens") is None


def test_custom_anthropic_protocol_keeps_presence_guard_without_native_provider():
    source = CompletionReports("custom-model", None, anthropic_usage=True)
    with source.invocation() as owner:
        source.record(
            "response.json",
            json.dumps(
                {
                    "response": {
                        "id": "msg",
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 2,
                            "prompt_tokens_details": {
                                "cached_tokens": 0,
                                "cache_creation_tokens": 0,
                                "text_tokens": 10,
                            },
                        },
                    }
                }
            ),
        )
        from types import SimpleNamespace

        usage = owner.usage(
            SimpleNamespace(
                token_usages=[], response_latencies=[SimpleNamespace(response_id="msg")]
            )
        )
    assert usage.input_tokens is None
    assert usage.output_tokens == 2
    assert usage.cached_input_tokens is None
    assert usage.extensions[KEY]["models"][0].get("provider") is None
