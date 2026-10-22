# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned NOOA clients with in-memory HTTP; no external model requests."""

import asyncio
from copy import deepcopy
import json
import os
import sys
from unittest.mock import MagicMock

import httpx
import pytest

if not ((3, 12) <= sys.version_info[:2] < (3, 14)):
    pytest.skip("NOOA supports Python 3.12 and 3.13", allow_module_level=True)

pytest.importorskip("nooa.unifiedllm", reason="NOOA harness extra is required")
from litellm.exceptions import InternalServerError
from nooa.runtime.channels import QueueManager
from nooa.unifiedllm import CompletionClient, ResponsesClient, RetryConfig
from nemo_fabric_adapters.nooa.usage import (
    InvocationUsage,
    _report,
    install_model_observer,
    owned_work_pending,
    running_jobs,
)

KEY = "nemo_fabric.usage_accounting.v1"
MESSAGES = [{"role": "user", "content": "hi"}]


def _body(responses, usage, identity):
    if responses:
        body = {
            "id": identity,
            "object": "response",
            "created_at": 1,
            "status": "completed",
            "model": "fixture-model",
            "output": [
                {
                    "id": "msg1",
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": "ok", "annotations": []}
                    ],
                }
            ],
        }
    else:
        body = {
            "id": identity,
            "object": "chat.completion",
            "created": 1,
            "model": "fixture-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
        }
    if usage is not None:
        body["usage"] = usage
    return body


def _counts(responses, input_tokens=40, output_tokens=8):
    if responses:
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_tokens_details": {"cached_tokens": 6},
        }
    return {
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "prompt_tokens_details": {"cached_tokens": 6},
    }


@pytest.fixture(name="make_client")
async def make_client_fixture():
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    clients = []

    def make(
        *,
        responses=False,
        usages=None,
        statuses=None,
        handler=None,
        native=False,
        provider="openai",
    ):
        client = (ResponsesClient if responses else CompletionClient)(
            model=f"{provider}/fixture-model",
            api_key="local-fixture",
            api_base=(
                "https://api.anthropic.com"
                if provider == "anthropic"
                else "https://api.openai.com/v1"
            )
            if native
            else "https://model.example.test/v1",
            retry_config=RetryConfig(max_retries=0, rate_limit_extra_retries=0),
        )
        for api in (client._http.sync_client, client._http.async_client):
            if hasattr(api, "max_retries"):
                api.max_retries = 0
        requests = []
        usages = iter(usages if usages is not None else [_counts(responses)])
        statuses = iter(statuses if statuses is not None else [200])

        def respond(request):
            requests.append(json.loads(request.content))
            if handler is not None:
                return handler(request, len(requests))
            status = next(statuses)
            return httpx.Response(
                status,
                json=_body(responses, next(usages), f"reply-{len(requests)}"),
                headers={"retry-after-ms": "1"},
            )

        transport = httpx.MockTransport(respond)
        for http in (client._http.httpx_sync, client._http.httpx_async):
            http._transport = transport
            http._mounts = {}
        install_model_observer(client)
        clients.append(client)
        return client, requests

    yield make
    for client in clients:
        await client.aclose()


@pytest.mark.parametrize("responses", [False, True])
@pytest.mark.parametrize("synchronous", [False, True])
async def test_real_client_raw_counts_are_known_before_litellm_fills_fields(
    make_client, responses, synchronous
):
    client, requests = make_client(responses=responses, native=True)
    owner = InvocationUsage()
    with owner.scope():
        client.call(MESSAGES) if synchronous else await client.acall(MESSAGES)
    usage = owner.finish()
    assert (
        usage.input_tokens,
        usage.cached_input_tokens,
        usage.output_tokens,
        usage.total_tokens,
    ) == (40, 6, 8, 48)
    model = usage.extensions[KEY]["models"][0]
    assert (model["model"], model["provider"]) == ("fixture-model", "openai")
    assert len(requests) == 1


@pytest.mark.parametrize("responses", [False, True])
@pytest.mark.parametrize("value", [None, 0])
async def test_real_client_omitted_input_is_unknown_and_reported_zero_survives(
    make_client, responses, value
):
    raw = _counts(responses, input_tokens=0)
    if value is None:
        raw.pop("input_tokens" if responses else "prompt_tokens")
    client, _ = make_client(responses=responses, usages=[raw])
    owner = InvocationUsage()
    with owner.scope():
        if responses and value is None:
            # The pinned Responses parser rejects the malformed primary field,
            # after the raw usage reached the observer. Independent fields survive.
            with pytest.raises(AttributeError, match="type"):
                await client.acall(MESSAGES)
        else:
            await client.acall(MESSAGES)
    usage = owner.finish()
    assert usage.input_tokens == value
    assert usage.cached_input_tokens == 6
    assert usage.output_tokens == 8
    assert usage.total_tokens == 8


@pytest.mark.parametrize("responses", [False, True])
async def test_custom_route_does_not_inherit_native_provider_identity(
    make_client, responses
):
    client, _ = make_client(responses=responses)
    owner = InvocationUsage()
    with owner.scope():
        await client.acall(MESSAGES, model="openai/actual-dispatch")
    usage = owner.finish()
    model = usage.extensions[KEY]["models"][0]
    assert model["model"] == "actual-dispatch"
    assert model.get("provider") is None
    assert usage.input_tokens == 40


@pytest.mark.parametrize("responses", [False, True])
async def test_recovered_response_without_usage_keeps_observations_with_a_gap(
    make_client, responses
):
    client, requests = make_client(
        responses=responses, usages=[None, _counts(responses)], statuses=[503, 200]
    )
    client.retry_config = RetryConfig(
        max_retries=1, rate_limit_extra_retries=0, base_delay=0, jitter_factor=0
    )
    # The OpenAI SDK has its own retry loop in the CompletionClient path.
    if hasattr(client._http.async_client, "max_retries"):
        client._http.async_client.max_retries = 0
    owner = InvocationUsage()
    with owner.scope():
        await client.acall(MESSAGES)
    usage = owner.finish()
    assert len(requests) == 2
    assert usage.input_tokens is None
    assert usage.output_tokens is None
    assert usage.extensions[KEY]["observed"]["input_tokens"] == 40
    assert any(
        gap["reason"] == "missing_usage" for gap in usage.extensions[KEY]["gaps"]
    )


async def test_raw_billed_response_survives_a_later_parser_failure(make_client):
    def respond(request, index):
        body = _body(False, _counts(False), "billed")
        body["choices"] = []
        return httpx.Response(200, json=body)

    client, _ = make_client(handler=respond)
    owner = InvocationUsage()
    with owner.scope(), pytest.raises(InternalServerError, match="no .choices."):
        await client.acall(MESSAGES)
    usage = owner.finish()
    assert (usage.input_tokens, usage.output_tokens) == (40, 8)


async def test_in_flight_return_freezes_owner_and_never_moves_to_next_invocation(
    make_client,
):
    started, release = asyncio.Event(), asyncio.Event()

    async def respond(request, index):
        if index == 1:
            started.set()
            await release.wait()
        return httpx.Response(
            200,
            json=_body(
                False,
                _counts(False, input_tokens=40 if index == 1 else 20),
                f"reply-{index}",
            ),
        )

    client, _ = make_client(handler=respond)
    first = InvocationUsage()
    with first.scope():
        task = asyncio.create_task(client.acall(MESSAGES))
    await started.wait()
    frozen = first.finish()
    before = deepcopy(frozen.to_mapping())
    assert frozen.input_tokens is None
    second = InvocationUsage()
    with second.scope():
        await client.acall(MESSAGES)
    assert second.finish().input_tokens == 20
    release.set()
    await task
    assert first.finish().to_mapping() == before
    assert second.finish().input_tokens == 20


async def test_pending_queue_job_is_owned_before_its_first_model_call():
    release = asyncio.Event()
    queue = QueueManager()
    queue.queue("jobs")
    agent = MagicMock(_summarizers=[], queue_manager=queue)
    owner = InvocationUsage()
    before = running_jobs(agent)
    with owner.scope():
        handle = queue.spawn(release.wait(), channel="jobs")
    assert owned_work_pending(agent, owner, before)
    unrelated = InvocationUsage()
    assert not owned_work_pending(agent, unrelated, [])
    usage = owner.finish(unfinished=True)
    assert usage.input_tokens is None
    release.set()
    await handle._task
    await queue.shutdown()


async def test_pending_summary_task_is_owned_before_its_first_call():
    release = asyncio.Event()
    owner = InvocationUsage()
    with owner.scope():
        task = asyncio.create_task(release.wait())
    agent = MagicMock(_summarizers=[MagicMock(_pending_task=task)])
    assert owned_work_pending(agent, owner, [])
    assert not owned_work_pending(agent, InvocationUsage(), [])
    release.set()
    await task


@pytest.mark.parametrize("value", [-1, True, 1 << 64, 1.5])
def test_invalid_provider_counts_stay_unknown(value):
    report = _report(
        {"prompt_tokens": value, "completion_tokens": 3, "total_tokens": 10}
    )
    assert report.input_tokens is None
    assert report.output_tokens == 3


def test_messages_cache_convention_does_not_depend_on_cache_fields_being_present():
    report = _report({"input_tokens": 40, "output_tokens": 8}, messages_api=True)
    assert report.input_tokens == 40
    assert report.input_tokens_include_cache is False
    assert report.cached_input_tokens is None


def test_no_call_requires_source_proof():
    assert InvocationUsage().finish().input_tokens is None
    assert InvocationUsage().finish(no_calls_verified=True).input_tokens == 0


@pytest.mark.parametrize("interactive", [False])
@pytest.mark.parametrize("responses", [False, True])
@pytest.mark.parametrize("outcome", ["succeeded", "failed", "raised", "malformed"])
async def test_runtime_result_keeps_model_usage_independently_of_execution_status(
    make_client, monkeypatch, tmp_path, interactive, responses, outcome
):
    from unittest.mock import AsyncMock
    from nemo_fabric_adapter_contract.models import AgentRunRequest, RuntimeContext
    from nemo_fabric_adapters.nooa import adapter, bench_adapter
    from nemo_fabric_adapters.nooa.telemetry import RelayInvocation

    failed = outcome != "succeeded"
    client, _ = make_client(responses=responses)
    agent = MagicMock(_summarizers=[], queue_manager=MagicMock())
    agent.queue_manager.running_handles.return_value = []
    agent.shell.close = AsyncMock()

    async def run(*args):
        await client.acall(MESSAGES)
        if outcome == "raised":
            raise RuntimeError("scripted target failure after a billed call")
        if outcome == "malformed":
            return None
        if interactive:
            if failed:
                raise RuntimeError("scripted target failure after a billed call")
            return "DONE", "done"
        return {"success": not failed, "response": "done", "result": None}

    async def telemetry(*, call, **kwargs):
        return RelayInvocation(called=True, result=await call(), report=None)

    if interactive:
        runtime = adapter.NooaRuntime()
        runtime._target = adapter.InteractiveAgentTarget(agent=agent)
        monkeypatch.setattr(adapter, "dispatch", run)
    else:
        runtime = bench_adapter.BenchRuntime()
        runtime._model_role = "default"
        runtime._workspace = tmp_path
        runtime._bench_agent_factory = MagicMock(return_value=agent)
        agent._run_evaluation = AsyncMock(side_effect=run)
    runtime._models = {"default": client}
    runtime._runtime_id = "runtime"
    runtime._telemetry = MagicMock(invoke=AsyncMock(side_effect=telemetry))
    context = RuntimeContext.from_mapping(
        {
            "runtime_id": "runtime",
            "invocation_id": "invocation",
            "request_id": "request",
            "environment": {
                "environment_id": "environment",
                "provider": "test",
                "control_location": "in_env_control",
                "ownership": "caller_owned",
            },
            "artifacts": {},
        }
    )
    result = await runtime.invoke(AgentRunRequest(input="task"), context)
    assert result.status.value == ("failed" if failed else "succeeded")
    assert (result.usage.input_tokens, result.usage.output_tokens) == (40, 8)


@pytest.mark.parametrize("responses", [False, True])
@pytest.mark.parametrize("status", [400, 429])
async def test_http_rejection_without_usage_does_not_fabricate_zero(
    make_client, responses, status
):
    from litellm.exceptions import BadRequestError, RateLimitError

    client, _ = make_client(
        responses=responses,
        handler=lambda request, index: httpx.Response(
            status,
            json={"error": {"message": "scripted", "type": "rate_limit_error"}},
            headers={"retry-after-ms": "1"},
        ),
    )
    owner = InvocationUsage()
    with owner.scope(), pytest.raises((BadRequestError, RateLimitError)):
        await client.acall(MESSAGES)
    usage = owner.finish()
    assert usage.input_tokens is None
    assert usage.output_tokens is None
    assert usage.total_tokens is None


async def test_client_callback_and_existing_http_hooks_are_preserved(make_client):
    calls = []
    client, _ = make_client()

    async def before(request):
        calls.append(request.url.path)

    client._http.httpx_async.event_hooks["request"].insert(0, before)
    callback = MagicMock()
    owner = InvocationUsage()
    with owner.scope():
        await client.acall(MESSAGES, logger_fn=callback)
    assert calls == ["/v1/chat/completions"]
    assert callback.called
    assert owner.finish().input_tokens == 40


async def test_internal_empty_content_retry_counts_both_paid_responses(make_client):
    def respond(request, index):
        body = _body(False, _counts(False), f"reply-{index}")
        if index == 1:
            body["choices"][0]["message"] = {
                "role": "assistant",
                "content": "",
                "reasoning_content": "reason",
            }
        return httpx.Response(200, json=body)

    client, requests = make_client(handler=respond)
    client.retry_config = RetryConfig(
        max_retries=1,
        rate_limit_extra_retries=0,
        base_delay=0,
        jitter_factor=0,
        retry_on_empty_content=True,
    )
    owner = InvocationUsage()
    with owner.scope():
        await client.acall(MESSAGES)
    usage = owner.finish()
    assert len(requests) == 2
    assert (usage.input_tokens, usage.cached_input_tokens, usage.output_tokens) == (
        80,
        12,
        16,
    )


async def test_stream_does_not_promote_litellm_token_estimates(
    make_client,
):
    def respond(request, index):
        chunks = [
            {
                "id": "stream",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "fixture-model",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "ok"},
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "stream",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "fixture-model",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": _counts(False),
            },
        ]
        text = (
            "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
            + "data: [DONE]\n\n"
        )
        return httpx.Response(
            200, content=text, headers={"content-type": "text/event-stream"}
        )

    client, _ = make_client(handler=respond)
    owner = InvocationUsage()
    with owner.scope():
        response = await client.acall(MESSAGES, stream=True)
    assert response.content == "ok"
    usage = owner.finish()
    assert usage.input_tokens is None
    assert "input_tokens" not in usage.extensions[KEY]["observed"]
    assert response.usage["prompt_tokens"] != 40
    assert any(
        gap["reason"] == "scope_unverified" for gap in usage.extensions[KEY]["gaps"]
    )


@pytest.mark.parametrize("synchronous", [False, True])
async def test_real_anthropic_client_preserves_native_cache_convention(
    make_client, synchronous
):
    def respond(request, index):
        assert request.url.path == "/v1/messages"
        return httpx.Response(
            200,
            json={
                "id": "msg-fixture",
                "type": "message",
                "role": "assistant",
                "model": "fixture-model",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {
                    "input_tokens": 40,
                    "cache_read_input_tokens": 6,
                    "cache_creation_input_tokens": 2,
                    "output_tokens": 8,
                },
            },
        )

    client, _ = make_client(handler=respond, provider="anthropic", native=True)
    owner = InvocationUsage()
    with owner.scope():
        client.call(MESSAGES) if synchronous else await client.acall(MESSAGES)
    usage = owner.finish()
    assert (usage.input_tokens, usage.cached_input_tokens, usage.output_tokens) == (
        40,
        6,
        8,
    )
    assert usage.input_tokens_include_cache is False
    raw = usage.extensions[KEY]["models"][0]
    assert raw["provider"] == "anthropic"
    assert raw["reported"]["cache_write_input_tokens"] == 2


async def test_parallel_managed_calls_keep_actual_per_request_models(make_client):
    client, _ = make_client(
        usages=[_counts(False), _counts(False)], statuses=[200, 200], native=True
    )
    owner = InvocationUsage()
    with owner.scope():
        await asyncio.gather(
            client.acall(MESSAGES, model="openai/first-model"),
            client.acall(MESSAGES, model="openai/second-model"),
        )
    usage = owner.finish()
    assert (usage.input_tokens, usage.output_tokens) == (80, 16)
    models = usage.extensions[KEY]["models"]
    assert [model["model"] for model in models] == ["first-model", "second-model"]
    assert [model["reported"]["input_tokens"] for model in models] == [40, 40]


async def test_responses_cache_and_total_survive_when_both_primary_counts_are_omitted(
    make_client,
):
    client, _ = make_client(
        responses=True,
        usages=[
            {
                "input_tokens_details": {"cached_tokens": 6},
                "total_tokens": 7,
            }
        ],
    )
    owner = InvocationUsage()
    with owner.scope(), pytest.raises(AttributeError, match="type"):
        await client.acall(MESSAGES)
    usage = owner.finish()
    assert usage.input_tokens is None
    assert usage.output_tokens is None
    assert usage.cached_input_tokens == 6
    assert usage.total_tokens == 7


@pytest.mark.parametrize("conflicting", [False, True])
async def test_repeated_response_id_deduplicates_only_identical_reports(
    make_client, conflicting
):
    def respond(request, index):
        body = _body(
            False,
            _counts(False, input_tokens=60 if conflicting and index == 2 else 40),
            "same-response",
        )
        if index == 1:
            body["choices"][0]["message"] = {
                "role": "assistant",
                "content": "",
                "reasoning_content": "reason",
            }
        return httpx.Response(200, json=body)

    client, requests = make_client(handler=respond)
    client.retry_config = RetryConfig(
        max_retries=1,
        rate_limit_extra_retries=0,
        base_delay=0,
        jitter_factor=0,
        retry_on_empty_content=True,
    )
    owner = InvocationUsage()
    with owner.scope():
        await client.acall(MESSAGES)
    usage = owner.finish()
    assert len(requests) == 2
    if conflicting:
        assert usage.input_tokens is None
        assert usage.output_tokens is None
        assert usage.extensions[KEY]["observed"]["input_tokens"] == 40
        assert any(
            gap["reason"] == "scope_unverified" for gap in usage.extensions[KEY]["gaps"]
        )
    else:
        assert usage.input_tokens == 40
        assert usage.output_tokens == 8


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize(
    "route", ["http", "nonstandard_port", "custom", "native", "explicit443"]
)
def test_actual_http_route_qualifies_native_pricing_without_erasing_counts(
    provider, route
):
    from nemo_fabric_adapters.nooa import usage as collector

    host = "api.openai.com" if provider == "openai" else "api.anthropic.com"
    base = {
        "http": f"http://{host}",
        "nonstandard_port": f"https://{host}:8443",
        "custom": "https://model.example",
        "native": f"https://{host}",
        "explicit443": f"https://{host}:443",
    }[route]
    raw = (
        {
            "input_tokens": 10,
            "output_tokens": 2,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        }
        if provider == "anthropic"
        else {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}
    )
    owner = InvocationUsage()
    call = owner.start("fixture-model")
    token = collector._CALL.set(call)
    try:
        with httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200, json={"id": "response", "usage": raw}
                )
            ),
            event_hooks={
                "request": [collector._request],
                "response": [collector._response],
            },
        ) as client:
            client.post(
                base
                + (
                    "/v1/messages"
                    if provider == "anthropic"
                    else "/v1/chat/completions"
                ),
                json={"model": "fixture-model"},
            )
    finally:
        call.finished = True
        collector._CALL.reset(token)
    usage = owner.finish()
    assert usage.input_tokens == 10
    assert usage.output_tokens == 2
    model = usage.extensions[KEY]["models"][0]
    assert model["model"] == "fixture-model"
    assert model.get("provider") == (
        provider if route in {"native", "explicit443"} else None
    )


def test_redirected_actual_request_does_not_inherit_initial_native_pricing_identity():
    from nemo_fabric_adapters.nooa import usage as collector

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

    owner = InvocationUsage()
    call = owner.start("fixture-model")
    token = collector._CALL.set(call)
    try:
        with httpx.Client(
            transport=httpx.MockTransport(respond),
            follow_redirects=True,
            event_hooks={
                "request": [collector._request],
                "response": [collector._response],
            },
        ) as client:
            client.post(
                "https://api.openai.com/v1/chat/completions",
                json={"model": "fixture-model"},
            )
    finally:
        call.finished = True
        collector._CALL.reset(token)
    usage = owner.finish()
    assert usage.extensions[KEY]["observed"]["input_tokens"] == 10
    with_counts = [
        model
        for model in usage.extensions[KEY]["models"]
        if model["observed"].get("input_tokens") is not None
    ]
    assert len(with_counts) == 1
    assert with_counts[0].get("provider") is None
