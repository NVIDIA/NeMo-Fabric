# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenHands source boundaries against scripted SDK transports and scoped logs."""

from contextvars import copy_context
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nemo_fabric_adapter_contract.models import AgentModelConfig
from nemo_fabric_adapters.openhands import adapter
from nemo_fabric_adapters.openhands.usage import CompletionReports

KEY = "nemo_fabric.usage_accounting.v1"


def _log(response_id="response", *, usage=None):
    return json.dumps(
        {
            "response": {
                "id": response_id,
                "usage": usage
                if usage is not None
                else {
                    "prompt_tokens": 100,
                    "completion_tokens": 5,
                    "prompt_tokens_details": {
                        "cached_tokens": 0,
                        "cache_creation_tokens": 0,
                    },
                },
            }
        }
    )


def _metrics(response_id="response", *, prompt=100, output=5):
    return SimpleNamespace(
        token_usages=[
            SimpleNamespace(
                response_id=response_id,
                prompt_tokens=prompt,
                completion_tokens=output,
                cache_read_tokens=0,
            )
        ]
        if prompt or output
        else [],
        response_latencies=[SimpleNamespace(response_id=response_id)],
    )


def test_late_callback_cannot_change_frozen_result_or_move_to_next_invocation():
    source = CompletionReports("configured-model", "openai")
    with source.invocation() as first:
        stale_context = copy_context()
        source.record("first.json", _log())
        first_usage = first.usage(_metrics())
    frozen = first_usage.to_mapping()
    with source.invocation() as second:
        stale_context.run(source.record, "late.json", _log("late"))
        source.record("second.json", _log("second"))
        second_usage = second.usage(_metrics("second"))
    assert first_usage.to_mapping() == frozen
    assert second_usage.input_tokens == 100
    assert second_usage.extensions[KEY]["models"][0]["reported"]["input_tokens"] == 100


@pytest.mark.parametrize("interrupted", [False, True])
def test_interruption_gaps_complete_totals_but_keeps_settled_observations(interrupted):
    pytest.importorskip("openhands.sdk")
    from openhands.sdk.event import InterruptEvent

    source = CompletionReports("configured-model", "openai")
    with source.invocation() as owner:
        source.record("response.json", _log())
        usage = owner.usage(
            _metrics(), events=[InterruptEvent()] if interrupted else []
        )
    assert usage.input_tokens == (None if interrupted else 100)
    assert usage.extensions[KEY]["observed"]["input_tokens"] == 100
    assert (
        any(
            gap["reason"] == "scope_unverified" for gap in usage.extensions[KEY]["gaps"]
        )
        is interrupted
    )

    frozen = usage.to_mapping()
    with source.invocation() as next_owner:
        source.record("next.json", _log("next"))
        next_usage = next_owner.usage(_metrics("next"))
    assert usage.to_mapping() == frozen
    assert next_usage.input_tokens == 100
    assert not any(
        gap["reason"] == "scope_unverified"
        for gap in next_usage.extensions[KEY]["gaps"]
    )


def test_ambiguous_main_zero_does_not_erase_literal_cache_zero():
    source = CompletionReports("configured-model", "openai")
    with source.invocation() as owner:
        source.record(
            "zero.json",
            _log(
                usage={
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "prompt_tokens_details": {"cached_tokens": 0},
                }
            ),
        )
        usage = owner.usage(_metrics(prompt=0, output=0))
    assert usage.input_tokens is None and usage.output_tokens is None
    assert usage.cached_input_tokens == 0


@pytest.mark.parametrize("missing", ["cache", "output"])
def test_omitted_dimension_stays_unknown_without_erasing_other_response_dimensions(
    missing,
):
    report = {
        "prompt_tokens": 100,
        "completion_tokens": 5,
        "prompt_tokens_details": {"cached_tokens": 0},
    }
    if missing == "cache":
        report.pop("prompt_tokens_details")
    else:
        report.pop("completion_tokens")
    source = CompletionReports("configured-model", "openai")
    with source.invocation() as owner:
        source.record("report.json", _log(usage=report))
        usage = owner.usage(_metrics(output=0 if missing == "output" else 5))
    assert usage.input_tokens == 100
    assert usage.output_tokens == (None if missing == "output" else 5)
    assert usage.cached_input_tokens == (None if missing == "cache" else 0)


@pytest.mark.parametrize("failure", [None, "server", "timeout", "empty"])
async def test_real_sdk_retry_reports_each_billed_response_and_retains_failure_gap(
    monkeypatch, failure
):
    pytest.importorskip("openhands.sdk")
    from litellm.exceptions import InternalServerError, Timeout
    from litellm.types.utils import ModelResponse
    from openhands.sdk import LLM
    from openhands.sdk.llm import Message, TextContent
    from openhands.sdk.llm import llm as openhands_llm

    def response(response_id, *, empty=False):
        return ModelResponse(
            id=response_id,
            model="test-model",
            choices=[]
            if empty
            else [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "done"},
                }
            ],
            usage={
                "prompt_tokens": 100,
                "completion_tokens": 5,
                "prompt_tokens_details": {"cached_tokens": 0},
            },
        )

    outcomes = []
    if failure == "server":
        outcomes.append(
            InternalServerError(
                message="scripted", model="test-model", llm_provider="openai"
            )
        )
    elif failure == "timeout":
        outcomes.append(
            Timeout(message="scripted", model="test-model", llm_provider="openai")
        )
    elif failure == "empty":
        outcomes.append(response("empty", empty=True))
    outcomes.append(response("success"))
    transport = AsyncMock(side_effect=outcomes)
    monkeypatch.setattr(openhands_llm, "litellm_acompletion", transport)
    llm = LLM(
        model="openai/test-model",
        api_key="not-a-key",
        num_retries=2,
        retry_min_wait=0,
        retry_max_wait=0,
        retry_multiplier=0,
    )
    source = CompletionReports("test-model", "openai")
    llm.telemetry.set_log_completions_callback(source.record)
    llm.retry_listener = source.retry
    llm.telemetry.log_enabled = True
    baseline = llm.metrics.deep_copy()
    with source.invocation() as owner:
        await llm.agenerate(
            [Message(role="user", content=[TextContent(text="scripted")])]
        )
        usage = owner.usage(llm.metrics.diff(baseline))
    assert transport.await_count == len(outcomes)
    expected = 200 if failure == "empty" else 100
    assert usage.input_tokens == (
        None if failure in {"server", "timeout"} else expected
    )
    assert usage.extensions[KEY]["observed"]["input_tokens"] == expected
    assert usage.cost_usd is None


@pytest.mark.parametrize(
    ("configured", "env", "expected"),
    [
        (None, None, "openai"),
        ("https://api.openai.com/v1", "https://custom.test/v1", "openai"),
        ("https://custom.test/v1", None, None),
        (None, "https://custom.test/v1", None),
    ],
)
def test_accounting_identity_uses_constructed_model_and_effective_route(
    monkeypatch, configured, env, expected
):
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    if env:
        monkeypatch.setenv("OPENAI_BASE_URL", env)
    model = AgentModelConfig(
        provider="configured-alias", model="openai/test-model", base_url=configured
    )
    assert adapter._accounting_identity(model) == (expected, "test-model")


@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        ({"prompt_tokens_details": {"cached_tokens": 0}}, 0),
        ({"prompt_tokens_details": {"cached_tokens": None}}, None),
        ({"prompt_tokens_details": {"audio_tokens": 0}}, None),
        (
            {"prompt_tokens_details": {}, "input_tokens_details": {"cached_tokens": 4}},
            4,
        ),
        (
            {
                "prompt_tokens_details": {"text_tokens": 10},
                "input_tokens_details": {"cached_tokens": 4},
            },
            4,
        ),
        ({"input_tokens_details": {"cached_tokens": None}}, None),
    ],
)
def test_cache_presence_handles_both_api_shapes_independently(usage, expected):
    source = CompletionReports("configured-model", "openai")
    with source.invocation() as owner:
        source.record("presence.json", _log(usage=usage))
        result = owner.usage(_metrics())
    assert result.cached_input_tokens == expected


@pytest.mark.parametrize("tool", ["terminal", "inspect_image_with_vision"])
def test_only_dispatched_unobserved_vision_client_gaps_scope(tool):
    source = CompletionReports("configured-model", "openai")
    with source.invocation() as owner:
        source.record("response.json", _log())
        usage = owner.usage(_metrics(), events=[SimpleNamespace(tool_name=tool)])
    assert usage.input_tokens == (None if tool == "inspect_image_with_vision" else 100)
    assert usage.extensions[KEY]["observed"]["input_tokens"] == 100


@pytest.mark.parametrize(
    ("wire", "configured", "env_key", "expected"),
    [
        ("anthropic/test-model", None, None, "anthropic"),
        ("anthropic/test-model", "https://custom.test", None, None),
        ("anthropic/test-model", None, "ANTHROPIC_BASE_URL", None),
        ("gemini/gemini-2.5-flash", None, None, "gemini"),
        ("gemini/gemini-2.5-flash", None, "GEMINI_API_BASE", None),
        ("vertex_ai/gemini-2.5-pro", "https://custom.test", None, None),
    ],
)
def test_custom_provider_routes_remain_unqualified(
    monkeypatch, wire, configured, env_key, expected
):
    import sys

    litellm = sys.modules.get("litellm")
    if litellm is not None:
        monkeypatch.setattr(litellm, "api_base", None)
    for key in ("ANTHROPIC_API_BASE", "ANTHROPIC_BASE_URL", "GEMINI_API_BASE"):
        monkeypatch.delenv(key, raising=False)
    if env_key:
        monkeypatch.setenv(env_key, "https://custom.test")
    model = AgentModelConfig(
        provider="configured-alias", model=wire, base_url=configured
    )
    assert adapter._accounting_identity(model) == (expected, wire.partition("/")[2])


def test_global_native_route_override_leaves_provider_unknown(monkeypatch):
    litellm = pytest.importorskip("litellm")
    monkeypatch.setattr(litellm, "api_base", "https://custom.test")
    model = AgentModelConfig(provider="openai", model="test-model")
    assert adapter._accounting_identity(model) == (None, "test-model")
