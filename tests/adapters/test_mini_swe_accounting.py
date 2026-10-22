# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qualified mini-SWE/LiteLLM boundaries, with in-memory HTTP only."""

import asyncio
from copy import deepcopy
import json
from threading import Event
from unittest.mock import MagicMock

import httpx
import litellm
import pytest
import pytest_asyncio
from minisweagent.exceptions import FormatError
from nemo_fabric_adapter_contract.models import ReportedUsage
from nemo_fabric_adapters.mini_swe_agent.usage import (
    MiniInvocationUsage,
    UsageRecordingLitellmModel,
    _reported,
)

KEY = "nemo_fabric.usage_accounting.v1"


def reply(usage, *, command="echo done", response_id="scripted"):
    return {
        "id": response_id,
        "object": "chat.completion",
        "created": 0,
        "model": "gpt-4.1-mini",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "done",
                    "tool_calls": [
                        {
                            "id": "tool",
                            "type": "function",
                            "function": {
                                "name": "bash",
                                "arguments": json.dumps({"command": command}),
                            },
                        }
                    ]
                    if command
                    else [],
                },
                "finish_reason": "tool_calls" if command else "stop",
            }
        ],
        **({"usage": usage} if usage is not None else {}),
    }


@pytest_asyncio.fixture(autouse=True, loop_scope="function")
async def deny_outbound_fixture(monkeypatch):
    # Use the test runner so Windows creates its loopback socketpair before
    # these guards apply, and restores them before that runner tears down.
    with monkeypatch.context() as patch:
        patch.setattr(
            "socket.getaddrinfo",
            MagicMock(side_effect=AssertionError("outbound DNS forbidden")),
        )
        patch.setattr(
            "socket.socket.connect",
            MagicMock(side_effect=AssertionError("outbound connection forbidden")),
        )
        patch.setenv("MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT", "1")
        yield


@pytest.fixture(name="make_model")
def make_model_fixture(monkeypatch):
    clients = []
    requests = []

    def make(
        responses,
        *,
        base_url="https://api.openai.com/v1",
        model_name="openai/gpt-4.1-mini",
    ):
        remaining = iter(responses)

        def respond(request):
            requests.append(request)
            response = next(remaining)
            if callable(response):
                response = response(request)
            if isinstance(response, BaseException):
                raise response
            status, body = response
            return httpx.Response(
                status, json=body, headers={"retry-after-ms": "1"}, request=request
            )

        def http_client():
            from nemo_fabric_adapters.mini_swe_agent.usage import _http_start, _http_end

            client = httpx.Client(
                transport=httpx.MockTransport(respond),
                event_hooks={"request": [_http_start], "response": [_http_end]},
            )
            clients.append(client)
            return client

        monkeypatch.setattr(
            "nemo_fabric_adapters.mini_swe_agent.usage._http_client", http_client
        )
        from litellm.llms.custom_httpx.http_handler import HTTPHandler

        # All qualified versions accept a supplied client; 1.89.3 does not
        # expose the newer create_client factory method.
        original_init = HTTPHandler.__init__

        def init_handler(handler, *args, **kwargs):
            client = httpx.Client(transport=httpx.MockTransport(respond))
            clients.append(client)
            return original_init(handler, *args, **{**kwargs, "client": client})

        monkeypatch.setattr(HTTPHandler, "__init__", init_handler)
        model = UsageRecordingLitellmModel(
            model_name=model_name,
            model_kwargs={
                "api_key": "local-only",
                "api_base": base_url,
            },
            cost_tracking="ignore_errors",
        )
        return model, requests

    yield make
    for client in clients:
        client.close()


@pytest.mark.parametrize(
    "raw",
    [
        {
            "prompt_tokens": 100,
            "completion_tokens": 5,
            "total_tokens": 105,
            "prompt_tokens_details": {"cached_tokens": 20},
        },
        {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        {"prompt_tokens": 100},
        {"total_tokens": 105},
        None,
    ],
)
def test_actual_model_raw_omission_zero_and_independent_fields(make_model, raw):
    model, requests = make_model([(200, reply(raw))])
    owner = MiniInvocationUsage()
    with owner.scope():
        result = model.query([{"role": "user", "content": "scripted"}])
    usage = owner.finish()
    assert result["extra"]["actions"][0]["command"] == "echo done"
    assert len(requests) == 1
    raw = raw or {}
    assert usage.input_tokens == raw.get("prompt_tokens")
    assert usage.output_tokens == raw.get("completion_tokens")
    assert usage.total_tokens == raw.get("total_tokens")
    assert usage.cached_input_tokens == raw.get("prompt_tokens_details", {}).get(
        "cached_tokens"
    )
    assert usage.cost_usd is None
    assert usage.extensions[KEY]["models"][0]["model"] == "gpt-4.1-mini"
    assert usage.extensions[KEY]["models"][0]["provider"] == "openai"
    assert "client" not in model.config.model_kwargs
    model.close_usage_client()


@pytest.mark.parametrize("status", [400, 408, 429, 503])
def test_sdk_retry_or_rejection_preserves_observations_without_status_zero(
    make_model, status
):
    responses = [(status, {"error": {"message": "scripted", "type": "server_error"}})]
    if status != 400:
        responses += [
            (
                200,
                reply(
                    {"prompt_tokens": 100, "completion_tokens": 5, "total_tokens": 105}
                ),
            )
        ]
    model, requests = make_model(responses)
    owner = MiniInvocationUsage()
    with owner.scope():
        if status == 400:
            with pytest.raises(litellm.BadRequestError):
                model._query([{"role": "user", "content": "scripted"}])
        else:
            model._query([{"role": "user", "content": "scripted"}])
    usage = owner.finish()
    assert len(requests) == (1 if status == 400 else 2)
    assert usage.input_tokens is None
    assert usage.output_tokens is None
    assert usage.extensions[KEY]["observed"].get("input_tokens") == (
        None if status == 400 else 100
    )
    assert {g["reason"] for g in usage.extensions[KEY]["gaps"]} == {"missing_usage"}
    model.close_usage_client()


def test_framework_retry_is_separate_attempt_without_double_count(
    make_model, monkeypatch
):
    # Exercise mini's actual query retry loop without its production backoff.
    from minisweagent.models.utils.retry import retry
    import logging
    from tenacity import wait_none

    monkeypatch.setenv("MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT", "2")
    retryer = retry(logger=logging.getLogger(__name__), abort_exceptions=[])
    retryer.wait = wait_none()
    monkeypatch.setattr("minisweagent.models.litellm_model.retry", lambda **_: retryer)
    model, requests = make_model(
        [
            (400, {"error": {"message": "scripted", "type": "bad_request"}}),
            (
                200,
                reply(
                    {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}
                ),
            ),
        ]
    )
    owner = MiniInvocationUsage()
    with owner.scope():
        model.query([{"role": "user", "content": "scripted"}])
    result = owner.finish()
    assert len(owner.calls) == 2
    assert len(requests) == 2
    assert result.input_tokens is None
    assert result.extensions[KEY]["observed"]["input_tokens"] == 10
    model.close_usage_client()


def test_format_error_and_failed_cost_lookup_keep_reported_usage(
    make_model, monkeypatch
):
    model, requests = make_model(
        [(200, reply({"prompt_tokens": 100, "completion_tokens": 5}, command=None))]
    )
    monkeypatch.setattr(
        litellm.cost_calculator,
        "completion_cost",
        MagicMock(side_effect=ValueError("not priced")),
    )
    owner = MiniInvocationUsage()
    with owner.scope():
        with pytest.raises(FormatError) as caught:
            model.query([{"role": "user", "content": "scripted"}])
    assert (
        caught.value.messages[0]["extra"]["response"]["usage"]["prompt_tokens"] == 100
    )
    assert owner.finish().input_tokens == 100
    assert owner.finish().output_tokens == 5
    assert len(requests) == 1
    model.close_usage_client()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            {
                "input_tokens": 100,
                "output_tokens": 5,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 20,
            },
            {
                "input_tokens": 100,
                "output_tokens": 5,
                "cached_input_tokens": 0,
                "cache_write_input_tokens": 20,
                "input_tokens_include_cache": False,
                "output_tokens_include_reasoning": True,
            },
        ),
        (
            {"input_tokens": 0, "output_tokens": 0},
            {
                "input_tokens": 0,
                "output_tokens": 0,
                "input_tokens_include_cache": False,
                "output_tokens_include_reasoning": True,
            },
        ),
        (
            {"inputTokens": 100, "outputTokens": 5, "totalTokens": 105},
            {
                "input_tokens": 100,
                "output_tokens": 5,
                "total_tokens": 105,
                "input_tokens_include_cache": False,
                "output_tokens_include_reasoning": True,
            },
        ),
        ({"cacheReadInputTokens": 5}, {"cached_input_tokens": 5}),
        (
            {"completion_tokens_details": {"reasoning_tokens": 7}},
            {"reasoning_tokens": 7},
        ),
        ({"prompt_tokens": -1, "completion_tokens": 1 << 64, "total_tokens": True}, {}),
    ],
)
def test_provider_mapping_preserves_raw_counts_and_conventions(raw, expected):
    assert _reported(raw).to_mapping() == expected


def test_repeated_raw_callback_deduplicates_but_different_call_does_not():
    owner = MiniInvocationUsage()
    body = {
        "id": "same-id",
        "usage": {
            "input_tokens": 10,
            "output_tokens": 2,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        },
    }
    details = {"log_event_type": "post_api_call", "original_response": json.dumps(body)}
    for _ in range(2):
        call = owner.start("claude-test", "anthropic")
        call.log(details)
        call.log(details)
        call.finished = True
    assert owner.finish().input_tokens == 20
    assert owner.finish().output_tokens == 4


def test_conflicting_duplicate_is_observed_with_gap():
    owner = MiniInvocationUsage()
    call = owner.start("claude-test", "anthropic")
    for count in [10, 11]:
        call.log(
            {
                "log_event_type": "post_api_call",
                "original_response": {
                    "id": "same",
                    "usage": {"input_tokens": count, "output_tokens": 2},
                },
            }
        )
    call.finished = True
    usage = owner.finish()
    assert usage.input_tokens is None
    assert usage.extensions[KEY]["observed"]["input_tokens"] == 10


def test_no_call_zero_requires_positive_no_call_proof():
    assert MiniInvocationUsage().finish().input_tokens is None
    usage = MiniInvocationUsage().finish(no_calls=True)
    assert usage.input_tokens == usage.output_tokens == usage.total_tokens == 0


async def test_late_worker_usage_is_frozen_and_never_transfers(make_model):
    entered = Event()
    release = Event()

    def blocked(request):
        entered.set()
        assert release.wait(10)
        return 200, reply({"prompt_tokens": 10, "completion_tokens": 2})

    model, requests = make_model([blocked])
    first = MiniInvocationUsage()
    with first.scope():
        task = asyncio.create_task(
            asyncio.to_thread(model.query, [{"role": "user", "content": "scripted"}])
        )
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        assert len(first.calls) == 1
        before = deepcopy(first.finish().to_mapping())
        assert "unfinished_work" in {
            gap["reason"] for gap in before["extensions"][KEY]["gaps"]
        }
        second = MiniInvocationUsage()
        with second.scope():
            release.set()
            await task
        assert first.finish().to_mapping() == before
        assert second.finish().extensions[KEY]["observed"] == {}
        assert len(requests) == 1
    finally:
        release.set()
        await task
        model.close_usage_client()


@pytest.mark.parametrize(
    "raw",
    [
        {
            "input_tokens": 100,
            "output_tokens": 5,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 20,
        },
        {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        },
        {"input_tokens": 100, "output_tokens": 5},
    ],
)
@pytest.mark.parametrize(
    "base_url,provider",
    [
        ("http://api.anthropic.com", None),
        ("https://api.anthropic.com:8443", None),
        ("https://model.example", None),
        ("https://api.anthropic.com", "anthropic"),
        ("https://api.anthropic.com:443", "anthropic"),
    ],
)
def test_actual_anthropic_logger_preserves_fields_before_litellm_zero_fills(
    raw, monkeypatch, base_url, provider
):
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    body = {
        "id": "msg_mini",
        "type": "message",
        "role": "assistant",
        "model": "claude-test",
        "content": [
            {
                "type": "tool_use",
                "id": "tool",
                "name": "bash",
                "input": {"command": "echo done"},
            }
        ],
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "usage": raw,
    }
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json=body, request=request)

    with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
        import importlib

        # Exercise the default handler lookup without configuring an opaque
        # external client on the model itself.
        handler = HTTPHandler(client=transport)
        monkeypatch.setattr(
            importlib.import_module("litellm.llms.anthropic.chat.handler"),
            "_get_httpx_client",
            lambda **_: handler,
        )
        model = UsageRecordingLitellmModel(
            model_name="anthropic/claude-sonnet-4-5",
            model_kwargs={"api_key": "local-only", "api_base": base_url},
            cost_tracking="ignore_errors",
        )
        owner = MiniInvocationUsage()
        with owner.scope():
            model.query([{"role": "user", "content": "scripted"}])
        usage = owner.finish()
    assert len(requests) == 1
    assert usage.input_tokens == raw["input_tokens"]
    assert usage.input_tokens_include_cache is False
    assert usage.output_tokens == raw["output_tokens"]
    assert usage.cached_input_tokens == raw.get("cache_read_input_tokens")
    assert usage.extensions[KEY]["observed"].get("cache_write_input_tokens") == raw.get(
        "cache_creation_input_tokens"
    )
    assert usage.extensions[KEY]["models"][0]["model"] == "claude-sonnet-4-5"

    assert usage.extensions[KEY]["models"][0].get("provider") == provider


def test_source_iteration_reports_are_retained_without_adding_top_level_again():
    owner = MiniInvocationUsage()
    call = owner.start("claude-test", "anthropic")
    call.log(
        {
            "log_event_type": "post_api_call",
            "original_response": {
                "id": "iterations",
                "usage": {
                    "input_tokens": 999,
                    "output_tokens": 999,
                    "iterations": [
                        {
                            "input_tokens": 10,
                            "output_tokens": 2,
                            "cache_read_input_tokens": 0,
                            "cache_creation_input_tokens": 0,
                        },
                        {
                            "input_tokens": 20,
                            "output_tokens": 3,
                            "cache_read_input_tokens": 0,
                            "cache_creation_input_tokens": 0,
                        },
                    ],
                },
            },
        }
    )
    call.finished = True
    usage = owner.finish()
    assert usage.input_tokens == 30
    assert usage.output_tokens == 5
    assert usage.extensions[KEY]["models"][0]["requests_exhaustive"] is False


async def test_cancelled_worker_invalidates_runtime_instead_of_reusing_agent(
    make_model,
):
    from nemo_fabric_adapter_contract.models import AgentRunRequest, RuntimeContext
    from nemo_fabric_adapters.mini_swe_agent.adapter import MiniSweAgentRuntime
    from nemo_fabric_adapters.common.lifecycle import LifecycleError

    entered = Event()
    release = Event()
    completed = Event()

    def blocked(request):
        entered.set()
        assert release.wait(10)
        return 200, reply({"prompt_tokens": 10, "completion_tokens": 2})

    model, requests = make_model([blocked])
    runtime = MiniSweAgentRuntime()
    runtime._model = model
    runtime._agent = MagicMock()
    runtime._agent.n_calls = 1

    def run(*args, **kwargs):
        try:
            model.query([{"role": "user", "content": "scripted"}])
            return {"exit_status": "Submitted", "submission": "done"}
        finally:
            completed.set()

    runtime._agent.run.side_effect = run
    request = AgentRunRequest(input="scripted")
    context = MagicMock(spec=RuntimeContext)
    task = asyncio.create_task(runtime.invoke(request, context))
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        with pytest.raises(LifecycleError) as caught:
            await runtime.invoke(request, context)
        assert caught.value.code == "mini_swe_agent_unusable"
    finally:
        release.set()
        assert await asyncio.to_thread(completed.wait, 10)
        await runtime.stop()
    assert len(requests) == 1


async def test_failed_task_with_settled_model_usage_keeps_complete_counts(make_model):
    from nemo_fabric_adapter_contract.models import AgentRunRequest, RuntimeContext
    from nemo_fabric_adapters.mini_swe_agent.adapter import MiniSweAgentRuntime

    model, requests = make_model(
        [
            (
                200,
                reply(
                    {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}
                ),
            )
        ]
    )
    runtime = MiniSweAgentRuntime()
    runtime._model = model
    runtime._agent = MagicMock()
    runtime._agent.n_calls = 1

    def run(*args, **kwargs):
        model.query([{"role": "user", "content": "scripted"}])
        return {"exit_status": "LimitsExceeded", "submission": ""}

    runtime._agent.run.side_effect = run
    try:
        result = await runtime.invoke(
            AgentRunRequest(input="scripted"), MagicMock(spec=RuntimeContext)
        )
        assert result.status == "failed"
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 2
        assert result.usage.total_tokens == 12
        assert len(requests) == 1
    finally:
        await runtime.stop()


def test_custom_endpoint_keeps_counts_without_claiming_native_billing(make_model):
    model, requests = make_model(
        [
            (
                200,
                reply(
                    {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}
                ),
            )
        ],
        base_url="https://fixture.invalid/v1",
    )
    owner = MiniInvocationUsage()
    with owner.scope():
        model.query([{"role": "user", "content": "scripted"}])
    usage = owner.finish()
    assert usage.input_tokens == 10
    assert usage.extensions[KEY]["models"][0].get("provider") is None
    assert requests[0].url.host == "fixture.invalid"
    model.close_usage_client()


def test_global_client_session_is_preserved_without_claiming_retry_coverage(
    monkeypatch,
):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(
            200,
            json=reply({"prompt_tokens": 10, "completion_tokens": 2}),
            request=request,
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr(litellm, "client_session", client)
        model = UsageRecordingLitellmModel(
            model_name="openai/gpt-4.1-mini",
            model_kwargs={
                "api_key": "local-only",
                "api_base": "https://api.openai.com/v1",
            },
            cost_tracking="ignore_errors",
        )
        owner = MiniInvocationUsage()
        with owner.scope():
            model.query([{"role": "user", "content": "scripted"}])
        usage = owner.finish()
        model.close_usage_client()
        assert not client.is_closed
    assert len(requests) == 1
    assert usage.input_tokens is None
    assert usage.extensions[KEY]["observed"]["input_tokens"] == 10
    assert "scope_unverified" in {
        gap["reason"] for gap in usage.extensions[KEY]["gaps"]
    }


def responses_reply(raw):
    return {
        "id": "resp_fixture",
        "object": "response",
        "created_at": 1,
        "model": "gpt-5.4",
        "status": "completed",
        "output": [
            {
                "id": "fc_fixture",
                "type": "function_call",
                "status": "completed",
                "call_id": "tool",
                "name": "bash",
                "arguments": '{"command":"echo done"}',
            }
        ],
        "usage": raw,
    }


@pytest.mark.parametrize(
    "raw",
    [
        {
            "input_tokens": 10,
            "output_tokens": 2,
            "total_tokens": 12,
            "input_tokens_details": {"cached_tokens": 0},
        },
        {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        {"input_tokens_details": {"cached_tokens": 5}},
        {"total_tokens": 12},
    ],
)
def test_actual_modern_model_responses_bridge_preserves_raw_presence_and_inclusion(
    make_model, raw
):
    model, requests = make_model(
        [(200, responses_reply(raw))], model_name="openai/gpt-5.4"
    )
    # Older qualified routers require explicit reasoning to select Responses.
    model.config.model_kwargs["reasoning_effort"] = "low"
    import importlib.metadata

    older_incomplete = importlib.metadata.version("litellm") != "1.100.1" and not {
        "input_tokens",
        "output_tokens",
        "total_tokens",
    }.issubset(raw)
    owner = MiniInvocationUsage()
    with owner.scope():
        if older_incomplete:
            # Older LiteLLM rejects partial Responses usage during conversion.
            # The collector must retain its independently usable raw fields.
            with pytest.raises(litellm.APIConnectionError):
                model.query([{"role": "user", "content": "scripted"}])
        else:
            message = model.query([{"role": "user", "content": "scripted"}])
            assert message["extra"]["actions"][0]["command"] == "echo done"
    result = owner.finish()
    assert len(requests) == 1
    assert requests[0].url.path == "/v1/responses"
    assert result.input_tokens == raw.get("input_tokens")
    assert result.output_tokens == raw.get("output_tokens")
    assert result.total_tokens == raw.get("total_tokens")
    assert result.cached_input_tokens == raw.get("input_tokens_details", {}).get(
        "cached_tokens"
    )
    assert result.input_tokens_include_cache == (
        True if "input_tokens" in raw else None
    )
    model.close_usage_client()


def test_global_responses_route_uses_the_same_owned_http_boundary(
    make_model, monkeypatch
):
    monkeypatch.setattr(litellm, "route_all_chat_openai_to_responses", True)
    model, requests = make_model(
        [
            (
                200,
                responses_reply(
                    {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}
                ),
            )
        ]
    )
    owner = MiniInvocationUsage()
    with owner.scope():
        model.query([{"role": "user", "content": "scripted"}])
    assert requests[0].url.path == "/v1/responses"
    assert owner.finish().input_tokens == 10
    assert owner.finish().input_tokens_include_cache is True
    model.close_usage_client()


def test_unqualified_package_version_preserves_client_and_raw_observations(monkeypatch):
    from openai import OpenAI
    import importlib.metadata

    real_version = importlib.metadata.version
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda name: "unqualified" if name == "litellm" else real_version(name),
    )
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(
            200,
            json=reply({"prompt_tokens": 10, "completion_tokens": 2}),
            request=request,
        )

    with OpenAI(
        api_key="local-only",
        base_url="https://fixture.invalid/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
    ) as client:
        model = UsageRecordingLitellmModel(
            model_name="openai/gpt-4.1-mini",
            model_kwargs={"client": client},
            cost_tracking="ignore_errors",
        )
        owner = MiniInvocationUsage()
        with owner.scope():
            model.query([{"role": "user", "content": "scripted"}])
        result = owner.finish()
        assert len(requests) == 1
        assert model.config.model_kwargs["client"] is client
        assert result.input_tokens is None
        assert result.extensions[KEY]["observed"]["input_tokens"] == 10
        assert "scope_unverified" in {
            gap["reason"] for gap in result.extensions[KEY]["gaps"]
        }


def test_missing_observer_seam_preserves_existing_model_execution(
    make_model, monkeypatch
):
    from litellm.llms.custom_httpx.http_handler import HTTPHandler
    from litellm.llms.openai.common_utils import BaseOpenAILLM

    # The framework itself retains its checker. Only the collector declines.
    declined = MagicMock(return_value=None)
    monkeypatch.setattr(
        "nemo_fabric_adapters.mini_swe_agent.usage._responses_mode", declined
    )
    model, requests = make_model(
        [(200, reply({"prompt_tokens": 10, "completion_tokens": 2}))]
    )
    model.config.model_kwargs["api_key"] = "declined-observer-local-only"
    handler = HTTPHandler()
    monkeypatch.setattr(
        BaseOpenAILLM, "_get_sync_http_client", staticmethod(lambda: handler.client)
    )
    owner = MiniInvocationUsage()
    try:
        with owner.scope():
            model.query([{"role": "user", "content": "scripted"}])
        result = owner.finish()
        assert len(requests) == 1
        declined.assert_called_once()
        assert result.input_tokens is None
        assert result.extensions[KEY]["observed"]["input_tokens"] == 10
    finally:
        handler.close()


def test_actual_default_route_matches_each_qualified_litellm_version(make_model):
    import importlib.metadata

    raw = {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}

    def respond(request):
        if request.url.path.endswith("/responses"):
            return 200, responses_reply(raw)
        return 200, reply(
            {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}
        )

    model, requests = make_model([respond], model_name="openai/gpt-5.4")
    owner = MiniInvocationUsage()
    with owner.scope():
        model.query([{"role": "user", "content": "scripted"}])
    expected = (
        "/v1/responses"
        if importlib.metadata.version("litellm") == "1.100.1"
        else "/v1/chat/completions"
    )
    assert requests[0].url.path == expected
    assert owner.finish().input_tokens == 10
    assert owner.finish().output_tokens == 2
    model.close_usage_client()


@pytest.mark.parametrize("version", ["1.89.3", "1.96.2", "1.100.1"])
def test_locked_litellm_version_is_qualified(monkeypatch, version):
    import importlib.metadata

    real_version = importlib.metadata.version
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda name: version if name == "litellm" else real_version(name),
    )
    model = UsageRecordingLitellmModel(
        model_name="openai/gpt-4.1-mini", cost_tracking="ignore_errors"
    )
    assert model._source_qualified is True


def test_qualified_old_bridge_signature_is_used_without_new_endpoint_argument(
    monkeypatch,
):
    import importlib
    from nemo_fabric_adapters.mini_swe_agent.usage import _responses_mode

    called = []

    def checker(
        model,
        custom_llm_provider,
        web_search_options=None,
        tools=None,
        reasoning_effort=None,
        reasoning_summary=None,
    ):
        called.append((model, custom_llm_provider, reasoning_effort))
        return {"mode": "responses"}, model

    monkeypatch.setattr(
        importlib.import_module("litellm.main"), "responses_api_bridge_check", checker
    )
    assert (
        _responses_mode(
            "gpt-5.4",
            "openai",
            {"reasoning_effort": "low", "api_base": "https://fixture.invalid/v1"},
        )
        is True
    )
    assert called == [("gpt-5.4", "openai", "low")]


def test_adapter_entry_module_imports_without_optional_harness_packages():
    import subprocess
    import sys

    probe = "import sys; sys.modules.update({name: None for name in ('minisweagent', 'litellm', 'httpx')}); from nemo_fabric_adapters.mini_swe_agent.adapter import MiniSweAgentRuntime"
    completed = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr


def test_custom_anthropic_client_can_hide_retry_without_certifying_complete_usage():
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    statuses = iter([503, 200])
    requests = []

    def respond(request):
        status = next(statuses)
        requests.append(status)
        body = (
            {"error": {"type": "api_error", "message": "fixture"}}
            if status == 503
            else {
                "id": "msg_fixture",
                "type": "message",
                "role": "assistant",
                "model": "claude-test",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tool",
                        "name": "bash",
                        "input": {"command": "echo done"},
                    }
                ],
                "stop_reason": "tool_use",
                "stop_sequence": None,
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 2,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                },
            }
        )
        return httpx.Response(status, request=request, json=body)

    class RetryingHandler(HTTPHandler):
        def post(self, *args, **kwargs):
            try:
                return super().post(*args, **kwargs)
            except httpx.HTTPStatusError:
                return super().post(*args, **kwargs)

    with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
        handler = RetryingHandler(client=transport)
        model = UsageRecordingLitellmModel(
            model_name="anthropic/claude-sonnet-4-5",
            model_kwargs={"api_key": "local-only", "client": handler},
            cost_tracking="ignore_errors",
        )
        owner = MiniInvocationUsage()
        with owner.scope():
            result = model.query([{"role": "user", "content": "scripted"}])
        usage = owner.finish()
        assert result["extra"]["actions"][0]["command"] == "echo done"
        assert model.config.model_kwargs["client"] is handler
        assert requests == [503, 200]
        assert usage.extensions[KEY]["observed"]["input_tokens"] == 10
        assert usage.input_tokens is None
        assert usage.output_tokens is None
        assert "scope_unverified" in {
            gap["reason"] for gap in usage.extensions[KEY]["gaps"]
        }
        assert "provider" not in usage.extensions[KEY]["models"][0]


@pytest.mark.parametrize(
    "base_url,provider",
    [
        ("http://api.openai.com/v1", None),
        ("https://api.openai.com:8443/v1", None),
        ("https://model.example/v1", None),
        ("https://api.openai.com/v1", "openai"),
        ("https://api.openai.com:443/v1", "openai"),
    ],
)
def test_actual_openai_route_requires_native_https_for_pricing(
    make_model, base_url, provider
):
    model, requests = make_model(
        [
            (
                200,
                reply(
                    {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}
                ),
            )
        ],
        base_url=base_url,
    )
    owner = MiniInvocationUsage()
    with owner.scope():
        model.query([{"role": "user", "content": "scripted"}])
    usage = owner.finish()
    assert len(requests) == 1
    assert (usage.input_tokens, usage.output_tokens) == (10, 2)
    assert usage.extensions[KEY]["models"][0]["model"] == "gpt-4.1-mini"
    assert usage.extensions[KEY]["models"][0].get("provider") == provider
    model.close_usage_client()


def test_observed_openai_redirect_cannot_restore_native_provider_after_custom_attempt():
    from nemo_fabric_adapters.mini_swe_agent import usage as collector

    def respond(request):
        if request.url.host == "api.openai.com":
            return httpx.Response(
                307, headers={"location": "https://model.example/v1/chat/completions"}
            )
        return httpx.Response(
            200,
            json={
                "id": "redirected",
                "usage": {"prompt_tokens": 10, "completion_tokens": 2},
            },
        )

    owner = MiniInvocationUsage()
    call = owner.start("fixture-model", "openai")
    call.http_expected = True
    token = collector._ACTIVE.set(call)
    try:
        with httpx.Client(
            transport=httpx.MockTransport(respond),
            follow_redirects=True,
            event_hooks={
                "request": [collector._http_start],
                "response": [collector._http_end],
            },
        ) as client:
            client.post(
                "https://api.openai.com/v1/chat/completions",
                json={"model": "fixture-model"},
            )
    finally:
        call.finished = True
        collector._ACTIVE.reset(token)
    usage = owner.finish()
    assert usage.extensions[KEY]["observed"]["input_tokens"] == 10
    assert usage.extensions[KEY]["models"][0].get("provider") is None
