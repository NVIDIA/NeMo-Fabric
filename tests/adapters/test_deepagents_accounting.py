# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real pinned ChatOpenAI boundaries against scripted, in-memory HTTP transports."""

import asyncio
from copy import deepcopy
import json
import os
from uuid import uuid4

import httpx
import pytest
from langchain_core.caches import InMemoryCache
from langchain_core.globals import get_llm_cache, set_llm_cache
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_openai import ChatOpenAI

from nemo_fabric_adapters.deepagents.usage import (
    AccountingCache,
    ModelCallUsage,
    async_request_hook,
    async_response_hook,
    model_metadata,
    request_hook,
    response_hook,
)
from nemo_fabric_adapters.deepagents import adapter
from nemo_fabric_adapter_contract.models import (
    AgentConfig,
    AgentRunRequest,
    RuntimeContext,
)

KEY = "nemo_fabric.usage_accounting.v1"


def _reply(*, model="model", usage=None):
    return {
        "id": "scripted",
        "object": "chat.completion",
        "created": 1,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "done"},
                "finish_reason": "stop",
            }
        ],
        "usage": usage
        if usage is not None
        else {"prompt_tokens": 100, "completion_tokens": 2, "total_tokens": 102},
    }


@pytest.fixture(name="make_model")
async def make_model_fixture():
    clients = []
    old_cache = get_llm_cache()
    set_llm_cache(None)

    def make(
        *,
        statuses=(200,),
        usage=None,
        hooks=True,
        transport_error=False,
        observer="observer",
        served_model="model",
        model_field="reported",
        finish_reason="stop",
        base_url="https://example.invalid/v1",
    ):
        requests = []
        remaining = iter(statuses)

        def respond(request):
            requests.append(json.loads(request.content))
            if transport_error:
                raise httpx.ConnectError("scripted connection failure", request=request)
            status = next(remaining)
            body = _reply(model=served_model, usage=usage)
            body["choices"][0]["finish_reason"] = finish_reason
            if model_field == "omitted":
                body.pop("model")
            elif model_field == "null":
                body["model"] = None
            return httpx.Response(
                status,
                json=body
                if status == 200
                else {"error": {"message": "scripted", "type": "api_error"}},
                headers={"retry-after-ms": "1"},
            )

        transport = httpx.MockTransport(respond)
        sync = httpx.Client(
            transport=transport,
            event_hooks={"request": [request_hook], "response": [response_hook]}
            if hooks
            else {},
        )
        async_client = httpx.AsyncClient(
            transport=transport,
            event_hooks={
                "request": [async_request_hook],
                "response": [async_response_hook],
            }
            if hooks
            else {},
        )
        clients.append((sync, async_client))
        model = ChatOpenAI(
            model="model",
            api_key="scripted-unused",
            base_url=base_url,
            http_client=sync,
            http_async_client=async_client,
            metadata=model_metadata(observer),
            cache=AccountingCache(),
        )
        return model, requests

    yield make
    set_llm_cache(old_cache)
    for sync, async_client in clients:
        sync.close()
        await async_client.aclose()


@pytest.mark.parametrize("sync", [False, True])
@pytest.mark.parametrize("model_field", ["reported", "omitted", "null"])
async def test_model_identity_requires_a_reported_name(make_model, sync, model_field):
    model, requests = make_model(
        served_model="literal-served-model", model_field=model_field
    )
    recorder = ModelCallUsage("observer")
    with recorder.scope():
        if sync:
            model.invoke("task", config={"callbacks": [recorder]})
        else:
            await model.ainvoke("task", config={"callbacks": [recorder]})
    usage = recorder.finish()
    assert usage.input_tokens == 100
    assert usage.output_tokens == 2
    assert usage.total_tokens == 102
    models = usage.extensions[KEY].get("models", [])
    assert [entry["model"] for entry in models] == (
        ["literal-served-model"] if model_field == "reported" else []
    )
    assert len(requests) == 1


@pytest.mark.parametrize("sync", [False, True])
async def test_sync_and_async_keep_raw_presence_and_reported_zero(make_model, sync):
    model, requests = make_model(usage={"prompt_tokens": 0})
    recorder = ModelCallUsage("observer")
    with recorder.scope():
        if sync:
            model.invoke("task", config={"callbacks": [recorder]})
        else:
            await model.ainvoke("task", config={"callbacks": [recorder]})
    usage = recorder.finish()
    assert usage.input_tokens == 0
    assert usage.output_tokens is None
    assert usage.total_tokens is None
    assert len(requests) == 1


@pytest.mark.parametrize(
    "operation", ["unused", "start_async_task", "update_async_task", "check_async_task"]
)
@pytest.mark.parametrize("foreground_call", [False, True])
async def test_remote_tool_callback_dispatch_marks_new_work_only_unit(
    make_model, operation, foreground_call
):
    from deepagents.middleware.async_subagents import (
        StartAsyncTaskSchema,
        UpdateAsyncTaskSchema,
        CheckAsyncTaskSchema,
    )
    from langchain_core.tools import StructuredTool

    model, _ = make_model()
    recorder = ModelCallUsage("observer", remote_names=frozenset({"researcher"}))
    schemas = {
        "start_async_task": StartAsyncTaskSchema,
        "update_async_task": UpdateAsyncTaskSchema,
        "check_async_task": CheckAsyncTaskSchema,
    }
    inputs = {
        "start_async_task": {"description": "research", "subagent_type": "researcher"},
        "update_async_task": {"task_id": "prior-task", "message": "continue"},
        "check_async_task": {"task_id": "prior-task"},
    }
    with recorder.scope():
        if foreground_call:
            await model.ainvoke("task", config={"callbacks": [recorder]})
        if operation != "unused":
            tool = StructuredTool.from_function(
                func=lambda **values: "fixture result",
                name=operation,
                description="Callback-dispatch unit fixture",
                args_schema=schemas[operation],
            )
            tool.invoke(inputs[operation], config={"callbacks": [recorder]})
    result = recorder.finish()
    dispatched = operation in {"start_async_task", "update_async_task"}
    assert result.input_tokens == (100 if foreground_call and not dispatched else None)
    reasons = {gap["reason"] for gap in result.extensions[KEY]["gaps"]}
    assert ("scope_unverified" in reasons) is dispatched
    if foreground_call:
        assert result.extensions[KEY]["observed"]["input_tokens"] == 100
    elif dispatched:
        assert "input_tokens" not in (result.extensions[KEY].get("observed") or {})


@pytest.mark.parametrize("status", [429, 503])
async def test_sdk_internal_retry_retains_observation_without_certifying_total(
    make_model, status
):
    model, requests = make_model(statuses=(status, 200))
    recorder = ModelCallUsage("observer")
    with recorder.scope():
        await model.ainvoke("task", config={"callbacks": [recorder]})
    usage = recorder.finish()
    assert len(requests) == 2
    assert usage.input_tokens is None
    assert usage.extensions[KEY]["observed"]["input_tokens"] == 100
    assert any(
        gap["reason"] == "missing_usage" for gap in usage.extensions[KEY]["gaps"]
    )


async def test_instrumentation_preserves_request_bodies_counts_and_response(make_model):
    observed, observed_requests = make_model(statuses=(503, 200))
    baseline, baseline_requests = make_model(statuses=(503, 200), hooks=False)
    baseline_result = await baseline.ainvoke("task")
    recorder = ModelCallUsage("observer")
    with recorder.scope():
        observed_result = await observed.ainvoke(
            "task", config={"callbacks": [recorder]}
        )
    assert baseline_requests == observed_requests
    assert baseline_result.content == observed_result.content
    assert baseline_result.usage_metadata == observed_result.usage_metadata


async def test_owned_direct_sdk_call_without_a_callback_is_not_silently_omitted(
    make_model,
):
    model, requests = make_model(statuses=(200, 200))
    recorder = ModelCallUsage("observer")
    with recorder.scope():
        await model.ainvoke("visible", config={"callbacks": [recorder]})
        await model.root_async_client.chat.completions.create(
            model="model", messages=[{"role": "user", "content": "auxiliary"}]
        )
    result = recorder.finish()
    assert len(requests) == 2
    assert result.input_tokens is None
    assert result.extensions[KEY]["observed"]["input_tokens"] == 100


async def test_worker_without_transport_owner_retains_observation_with_a_scope_gap(
    make_model,
):
    model, _ = make_model()
    recorder = ModelCallUsage("observer")
    with recorder.scope():
        # Plain run_in_executor does not copy ContextVars; LangChain's own
        # context-preserving executor and asyncio.to_thread do.
        await asyncio.get_running_loop().run_in_executor(
            None, lambda: model.invoke("task", config={"callbacks": [recorder]})
        )
    result = recorder.finish()
    assert result.input_tokens is None
    assert result.extensions[KEY]["observed"]["input_tokens"] == 100


async def test_local_cache_replay_does_not_recharge_historical_tokens(make_model):
    cache = InMemoryCache()
    set_llm_cache(cache)
    model, requests = make_model()
    totals = []
    for _ in range(2):
        recorder = ModelCallUsage("observer")
        with recorder.scope():
            await model.ainvoke("identical task", config={"callbacks": [recorder]})
        totals.append(recorder.finish().input_tokens)
    assert len(requests) == 1
    assert totals == [100, 0]


async def test_observer_identity_does_not_change_cross_model_cache_reuse(make_model):
    set_llm_cache(InMemoryCache())
    first_model, first_requests = make_model(observer="first")
    second_model, second_requests = make_model(observer="second")
    amounts = []
    for model, name in [(first_model, "first"), (second_model, "second")]:
        owner = ModelCallUsage(name)
        with owner.scope():
            await model.ainvoke("identical", config={"callbacks": [owner]})
        amounts.append(owner.finish().input_tokens)
    assert len(first_requests) + len(second_requests) == 1
    assert amounts == [100, 0]


async def test_concurrent_invocation_owners_and_late_callbacks_stay_isolated(
    make_model,
):
    model, requests = make_model(statuses=(200, 200))
    first, second = ModelCallUsage("observer"), ModelCallUsage("observer")
    barrier = asyncio.Event()

    async def invoke(recorder, prompt):
        with recorder.scope():
            await barrier.wait()
            await model.ainvoke(prompt, config={"callbacks": [recorder]})
        return recorder.finish()

    a = asyncio.create_task(invoke(first, "first"))
    b = asyncio.create_task(invoke(second, "second"))
    barrier.set()
    results = await asyncio.gather(a, b)
    assert [result.input_tokens for result in results] == [100, 100]
    frozen = deepcopy(results[0].to_mapping())
    late = uuid4()
    first.on_chat_model_start({}, [], run_id=late)
    first.on_llm_end(
        LLMResult(generations=[[ChatGeneration(message=AIMessage(content="late"))]]),
        run_id=late,
    )
    assert first.finish().to_mapping() == frozen
    assert second.finish().input_tokens == 100
    assert len(requests) == 2


async def test_terminal_failure_is_not_a_zero_and_does_not_poison_the_next_owner(
    make_model,
):
    model, requests = make_model(statuses=(400, 200))
    failed = ModelCallUsage("observer")
    with failed.scope(), pytest.raises(Exception, match="scripted"):
        await model.ainvoke("first", config={"callbacks": [failed]})
    assert failed.finish().input_tokens is None
    next_owner = ModelCallUsage("observer")
    with next_owner.scope():
        await model.ainvoke("next", config={"callbacks": [next_owner]})
    assert next_owner.finish().input_tokens == 100
    assert len(requests) == 2


async def test_non_model_failure_keeps_completed_call_usage(make_model):
    model, _ = make_model()
    recorder = ModelCallUsage("observer")
    with recorder.scope(), pytest.raises(ValueError, match="tool failed"):
        await model.ainvoke("task", config={"callbacks": [recorder]})
        raise ValueError("tool failed")
    assert recorder.finish().input_tokens == 100


@pytest.fixture(name="make_runtime")
async def make_runtime_fixture(tmp_path, monkeypatch):
    runtimes = []
    monkeypatch.setenv("SCRIPTED_MODEL_KEY", "unused-scripted")
    monkeypatch.delenv("OPENAI_PROXY", raising=False)

    async def make(replies, *, identity="runtime", profile=None, barrier=None):
        ledger = []
        remaining = iter(replies)

        async def respond(request):
            ledger.append(json.loads(request.content))
            if barrier is not None:
                await barrier()
            reply = next(remaining)
            status, body = reply if isinstance(reply, tuple) else (200, reply)
            return httpx.Response(status, json=body, headers={"retry-after-ms": "1"})

        def clients():
            transport = httpx.MockTransport(respond)
            return (
                httpx.Client(
                    transport=transport,
                    timeout=None,
                    event_hooks={
                        "request": [request_hook],
                        "response": [response_hook],
                    },
                ),
                httpx.AsyncClient(
                    transport=transport,
                    timeout=None,
                    event_hooks={
                        "request": [async_request_hook],
                        "response": [async_response_hook],
                    },
                ),
            )

        monkeypatch.setattr(adapter, "observed_http_clients", clients)
        build = adapter.build_chat_model
        if profile is not None:

            def profiled_model(config, **kwargs):
                model, name, url = build(config, **kwargs)
                model.profile = profile
                return model, name, url

            monkeypatch.setattr(adapter, "build_chat_model", profiled_model)
        workspace = tmp_path / identity
        workspace.mkdir()
        context = RuntimeContext.from_mapping(
            {
                "runtime_id": identity,
                "invocation_id": "invocation",
                "request_id": "request",
                "environment": {
                    "environment_id": "environment",
                    "provider": "local",
                    "control_location": "in_env_control",
                    "workspace": str(workspace),
                    "ownership": "caller_owned",
                },
                "artifacts": {},
            }
        )
        config = AgentConfig.from_mapping(
            {
                "harness": {"settings": {}},
                "models": {
                    "default": {
                        "provider": "openai",
                        "model": "model",
                        "api_key_env": "SCRIPTED_MODEL_KEY",
                        "base_url": "https://example.invalid/v1",
                    }
                },
            }
        )
        runtime = adapter.DeepAgentsRuntime()
        runtimes.append(runtime)
        await runtime.start(
            {
                "config": config,
                "runtime_context": context.to_mapping(),
                "base_dir": str(workspace),
            }
        )
        return runtime, context, ledger

    yield make
    for runtime in runtimes:
        await runtime.stop()


async def test_two_actual_runtimes_preserve_concurrent_graph_ownership(make_runtime):
    arrived = 0
    both = asyncio.Event()

    async def barrier():
        nonlocal arrived
        arrived += 1
        if arrived == 2:
            both.set()
        await both.wait()

    a, context_a, ledger_a = await make_runtime(
        [
            _reply(
                usage={
                    "prompt_tokens": 100,
                    "completion_tokens": 2,
                    "total_tokens": 102,
                }
            )
        ],
        identity="a",
        barrier=barrier,
    )
    b, context_b, ledger_b = await make_runtime(
        [
            _reply(
                usage={
                    "prompt_tokens": 200,
                    "completion_tokens": 3,
                    "total_tokens": 203,
                }
            )
        ],
        identity="b",
        barrier=barrier,
    )
    results = await asyncio.wait_for(
        asyncio.gather(
            a.invoke(AgentRunRequest(input="first"), context_a),
            b.invoke(AgentRunRequest(input="second"), context_b),
        ),
        timeout=20,
    )
    assert [result.usage.input_tokens for result in results] == [100, 200]
    assert [len(ledger_a), len(ledger_b)] == [1, 1]


async def test_actual_summarization_is_counted_outside_graph_messages(make_runtime):
    replies = [
        _reply(
            usage={"prompt_tokens": i, "completion_tokens": o, "total_tokens": i + o}
        )
        for i, o in [(1500, 10), (1400, 60), (300, 5)]
    ]
    runtime, context, ledger = await make_runtime(
        replies, profile={"max_input_tokens": 1000}
    )
    first = await runtime.invoke(AgentRunRequest(input="word " * 6000), context)
    second = await runtime.invoke(AgentRunRequest(input="continue"), context)
    assert first.usage.input_tokens == 1500
    assert second.usage.input_tokens == 1700
    assert second.usage.output_tokens == 65
    assert len(ledger) == 3
    assert "tools" not in ledger[1]
    assert "tools" in ledger[2]


async def test_actual_parallel_task_subagents_count_each_call_once(make_runtime):
    first = _reply(
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
    )
    first["choices"][0]["message"]["tool_calls"] = [
        {
            "id": f"task-{index}",
            "type": "function",
            "function": {
                "name": "task",
                "arguments": json.dumps(
                    {
                        "description": f"Echo {index}.",
                        "subagent_type": "general-purpose",
                    }
                ),
            },
        }
        for index in (1, 2)
    ]
    first["choices"][0]["finish_reason"] = "tool_calls"
    replies = [
        first,
        *[
            _reply(
                usage={
                    "prompt_tokens": count,
                    "completion_tokens": count,
                    "total_tokens": count * 2,
                }
            )
            for count in (10, 100, 1000)
        ],
    ]
    runtime, context, ledger = await make_runtime(replies)
    result = await runtime.invoke(
        AgentRunRequest(input="Delegate two tasks, then answer."), context
    )
    assert result.usage.input_tokens == 1111
    assert result.usage.output_tokens == 1111
    assert len(ledger) == 4


@pytest.mark.parametrize("sync", [False, True])
@pytest.mark.parametrize("completion", [2, 0])
@pytest.mark.parametrize("extra_frame", [False, True])
@pytest.mark.parametrize("model_field", ["reported", "omitted", "null"])
async def test_real_streaming_usage_is_preserved_when_the_source_reports_it(
    sync, completion, extra_frame, model_field
):
    chunks = [
        {
            "id": "stream",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "model",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "done"},
                    "finish_reason": "stop",
                }
            ],
        },
        {
            "id": "stream",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "model",
            "choices": [],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": completion,
                "total_tokens": 100 + completion,
            },
        },
    ]
    for chunk in chunks:
        if model_field == "omitted":
            chunk.pop("model")
        elif model_field == "null":
            chunk["model"] = None
    if extra_frame:
        earlier = deepcopy(chunks[-1])
        earlier["usage"] = {
            "prompt_tokens": 90,
            "completion_tokens": 1,
            "total_tokens": 91,
        }
        chunks.insert(1, earlier)
    sse = (
        "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
        + "data: [DONE]\n\n"
    )
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200, text=sse, headers={"content-type": "text/event-stream"}
        )
    )
    with httpx.Client(
        transport=transport,
        event_hooks={"request": [request_hook], "response": [response_hook]},
    ) as client:
        async with httpx.AsyncClient(
            transport=transport,
            event_hooks={
                "request": [async_request_hook],
                "response": [async_response_hook],
            },
        ) as async_client:
            model = ChatOpenAI(
                model="configured-fallback",
                api_key="scripted-unused",
                base_url="https://example.invalid/v1",
                http_client=client,
                http_async_client=async_client,
                metadata=model_metadata("observer"),
                stream_usage=True,
            )
            recorder = ModelCallUsage("observer")
            with recorder.scope():
                if sync:
                    list(model.stream("task", config={"callbacks": [recorder]}))
                else:
                    async for _ in model.astream(
                        "task", config={"callbacks": [recorder]}
                    ):
                        pass
            result = recorder.finish()
            if extra_frame:
                assert result.input_tokens is None
                assert result.extensions[KEY]["observed"]["input_tokens"] == 100
            else:
                assert result.input_tokens == 100
                assert result.output_tokens == (completion if completion > 0 else None)
            assert result.total_tokens is None
            models = result.extensions[KEY].get("models", [])
            assert [entry["model"] for entry in models] == (
                ["model"] if model_field == "reported" else []
            )


async def test_late_http_completion_never_moves_to_next_invocation():
    started, release = asyncio.Event(), asyncio.Event()

    async def respond(request):
        prompt = json.loads(request.content)["messages"][-1]["content"]
        if prompt == "late":
            started.set()
            await release.wait()
        return httpx.Response(200, json=_reply())

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(respond),
        event_hooks={
            "request": [async_request_hook],
            "response": [async_response_hook],
        },
    ) as client:
        model = ChatOpenAI(
            model="model",
            api_key="scripted-unused",
            base_url="https://example.invalid/v1",
            http_async_client=client,
            metadata=model_metadata("observer"),
        )
        old_owner = ModelCallUsage("observer")
        with old_owner.scope():
            late = asyncio.create_task(
                model.ainvoke("late", config={"callbacks": [old_owner]})
            )
        await asyncio.wait_for(started.wait(), timeout=10)
        frozen = old_owner.finish().to_mapping()
        next_owner = ModelCallUsage("observer")
        try:
            with next_owner.scope():
                await model.ainvoke("next", config={"callbacks": [next_owner]})
            assert next_owner.finish().input_tokens == 100
        finally:
            release.set()
            await asyncio.wait_for(late, timeout=10)
        assert old_owner.finish().to_mapping() == frozen
        assert frozen.get("input_tokens") is None


def test_failed_call_partial_response_is_retained_as_an_observation_unit():
    recorder = ModelCallUsage("observer")
    identity = uuid4()
    recorder.on_chat_model_start(
        {"id": ["ChatOpenAI"]}, [], run_id=identity, metadata=model_metadata("observer")
    )
    response = LLMResult(
        generations=[
            [
                ChatGeneration(
                    message=AIMessage(
                        content="partial",
                        response_metadata={
                            "model_name": "model",
                            "token_usage": {
                                "prompt_tokens": 100,
                                "completion_tokens": 2,
                                "total_tokens": 102,
                            },
                        },
                    )
                )
            ]
        ]
    )
    recorder.on_llm_error(
        RuntimeError("stream interrupted"), run_id=identity, response=response
    )
    result = recorder.finish()
    assert result.input_tokens is None
    assert result.extensions[KEY]["observed"]["input_tokens"] == 100


async def test_one_models_retry_does_not_erase_another_models_complete_evidence(
    make_model,
):
    clean, _ = make_model(served_model="clean")
    retried, _ = make_model(served_model="retried", statuses=(503, 200))
    recorder = ModelCallUsage("observer")
    with recorder.scope():
        await clean.ainvoke("task", config={"callbacks": [recorder]})
        await retried.ainvoke("task", config={"callbacks": [recorder]})
    usage = recorder.finish()
    models = {model["model"]: model for model in usage.extensions[KEY]["models"]}
    assert usage.input_tokens is None
    assert models["clean"]["reported"]["input_tokens"] == 100
    assert "input_tokens" not in models["retried"]["reported"]
    assert models["retried"]["observed"]["input_tokens"] == 100


@pytest.mark.parametrize("sync", [False, True])
@pytest.mark.parametrize("partial", [False, True])
async def test_structured_output_length_error_keeps_reported_usage(
    make_model, sync, partial
):
    from openai import LengthFinishReasonError
    from pydantic import BaseModel

    class Answer(BaseModel):
        answer: str

    model, requests = make_model(
        finish_reason="length", usage={"prompt_tokens": 100} if partial else None
    )
    structured = model.with_structured_output(Answer, method="json_schema")
    recorder = ModelCallUsage("observer")
    with recorder.scope(), pytest.raises(LengthFinishReasonError):
        if sync:
            structured.invoke("task", config={"callbacks": [recorder]})
        else:
            await structured.ainvoke("task", config={"callbacks": [recorder]})
    usage = recorder.finish()
    assert usage.input_tokens == 100
    assert usage.output_tokens == (None if partial else 2)
    assert usage.total_tokens == (None if partial else 102)
    assert len(requests) == 1


def test_unknown_input_convention_does_not_invent_an_inclusive_peak():
    recorder = ModelCallUsage(None)
    identity = uuid4()
    recorder.on_chat_model_start({}, [], run_id=identity)
    recorder.on_llm_end(
        LLMResult(
            generations=[
                [
                    ChatGeneration(
                        message=AIMessage(
                            content="done",
                            response_metadata={"token_usage": {"prompt_tokens": 100}},
                        )
                    )
                ]
            ]
        ),
        run_id=identity,
    )
    observed = recorder.finish().extensions[KEY]["observed"]
    assert observed["input_tokens"] == 100
    assert "peak_request_input_tokens" not in observed


@pytest.mark.parametrize("sync", [False, True])
@pytest.mark.parametrize(
    ("base_url", "environment", "provider"),
    [
        (None, {}, "openai"),
        ("https://api.openai.com/v1", {}, "openai"),
        ("https://example.invalid/v1", {}, None),
        (None, {"OPENAI_BASE_URL": "https://example.invalid/v1"}, None),
        (None, {"OPENAI_BASE_URL": "https://api.openai.com/v1"}, "openai"),
        (None, {"OPENAI_API_BASE": "https://example.invalid/v1"}, None),
        (
            "https://api.openai.com/v1",
            {"OPENAI_BASE_URL": "https://example.invalid/v1"},
            "openai",
        ),
        (
            "https://example.invalid/v1",
            {"OPENAI_BASE_URL": "https://api.openai.com/v1"},
            None,
        ),
        ("https://api.openai.com:8443/v1", {}, None),
        ("http://api.openai.com/v1", {}, None),
    ],
)
async def test_actual_request_endpoint_controls_accounting_provider(
    make_model, monkeypatch, sync, base_url, environment, provider
):
    for name in ("OPENAI_BASE_URL", "OPENAI_API_BASE"):
        monkeypatch.delenv(name, raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    model, requests = make_model(base_url=base_url, served_model="literal-served-model")
    recorder = ModelCallUsage("observer")
    with recorder.scope():
        if sync:
            model.invoke("task", config={"callbacks": [recorder]})
        else:
            await model.ainvoke("task", config={"callbacks": [recorder]})
    usage = recorder.finish()
    assert len(requests) == 1
    assert usage.input_tokens == 100
    assert usage.output_tokens == 2
    assert usage.total_tokens == 102
    accounted = usage.extensions[KEY]["models"]
    assert len(accounted) == 1
    assert accounted[0]["model"] == "literal-served-model"
    assert accounted[0].get("provider") == provider


async def test_parallel_calls_keep_independent_native_and_custom_routes(make_model):
    native, _ = make_model(base_url="https://api.openai.com/v1", served_model="native")
    custom, _ = make_model(base_url="https://example.invalid/v1", served_model="custom")
    recorder = ModelCallUsage("observer")
    with recorder.scope():
        await asyncio.gather(
            native.ainvoke("one", config={"callbacks": [recorder]}),
            custom.ainvoke("two", config={"callbacks": [recorder]}),
        )
    result = recorder.finish()
    models = {model["model"]: model for model in result.extensions[KEY]["models"]}
    assert result.input_tokens == 200
    assert result.output_tokens == 4
    assert models["native"]["provider"] == "openai"
    assert "provider" not in models["custom"]
