# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract checks for owned, bounded and frozen Hermes accounting evidence."""

from __future__ import annotations

import builtins
from contextlib import contextmanager
import json
import socket
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
from nemo_fabric_adapter_contract.models import GapReason
from nemo_fabric_adapters.hermes import usage


@pytest.fixture(autouse=True)
def deny_network(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("network access is forbidden in accounting tests")

    monkeypatch.setattr(socket, "getaddrinfo", denied)
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)


def _body(*, input_tokens=10, cached=0, output_tokens=2):
    counters = {"completion_tokens": output_tokens}
    if input_tokens is not None:
        counters["prompt_tokens"] = input_tokens
        counters["total_tokens"] = input_tokens + output_tokens
    if cached is not None:
        counters["prompt_tokens_details"] = {"cached_tokens": cached}
    return {
        "id": "response",
        "model": "model-a",
        "created": 0,
        "object": "chat.completion",
        "usage": counters,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "done"},
                "finish_reason": "stop",
            }
        ],
    }


def _owner():
    return usage.InvocationUsage(object())


def _client(handler):
    return httpx.Client(
        transport=httpx.MockTransport(handler),
        event_hooks={
            "request": [usage.observe_request],
            "response": [usage.observe_response],
        },
    )


def _accounting(result):
    return result.extensions["nemo_fabric.usage_accounting.v1"]


def test_reported_zero_and_omission_are_independent():
    for count in (0, None):
        owner = _owner()
        with (
            owner.activate(),
            _client(
                lambda request: httpx.Response(200, json=_body(input_tokens=count))
            ) as client,
        ):
            client.post(
                "https://example.invalid/v1/chat/completions", json={"model": "model-a"}
            )
        result = owner.finish()
        assert result.input_tokens == count
        assert result.output_tokens == 2
        assert result.cached_input_tokens == 0


def test_stream_preserves_lazy_bytes_and_counts_terminal_usage_once():
    owner = _owner()
    chunks = [
        b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n',
        b"data: " + json.dumps(_body()).encode() + b"\n\n",
        b"data: [DONE]\n\n",
    ]
    stream = MagicMock(spec=httpx.SyncByteStream)
    stream.__iter__.side_effect = lambda: iter(chunks)
    with (
        owner.activate(),
        _client(lambda request: httpx.Response(200, stream=stream)) as client,
    ):
        with client.stream(
            "POST",
            "https://example.invalid/v1/chat/completions",
            json={"model": "model-a", "stream": True},
        ) as response:
            stream.__iter__.assert_not_called()
            assert list(response.iter_bytes()) == chunks
    result = owner.finish()
    assert (result.input_tokens, result.cached_input_tokens, result.output_tokens) == (
        10,
        0,
        2,
    )
    assert len(_accounting(result)["models"][0]["requests"]) == 1


@pytest.mark.parametrize("status", [400, 408, 429, 503])
def test_hidden_http_retry_retains_observation_with_incomplete_total(status):
    owner = _owner()
    replies = iter(
        [
            httpx.Response(status, json={"error": "failed"}),
            httpx.Response(200, json=_body()),
        ]
    )
    with owner.activate(), _client(lambda request: next(replies)) as client:
        for _ in range(2):
            client.post(
                "https://example.invalid/v1/chat/completions", json={"model": "model-a"}
            )
    result = owner.finish()
    assert result.input_tokens is None
    assert _accounting(result)["observed"]["input_tokens"] == 10
    assert len(_accounting(result)["models"][0]["requests"]) == 2


def test_dropped_stream_retains_usage_but_not_complete_total():
    owner = _owner()

    def chunks():
        yield b"data: " + json.dumps(_body()).encode() + b"\n\n"
        raise httpx.ReadError("dropped")

    stream = MagicMock(spec=httpx.SyncByteStream)
    stream.__iter__.side_effect = chunks
    with (
        owner.activate(),
        _client(lambda request: httpx.Response(200, stream=stream)) as client,
    ):
        with pytest.raises(httpx.ReadError):
            client.post(
                "https://example.invalid/v1/chat/completions",
                json={"model": "model-a", "stream": True},
            )
    result = owner.finish()
    assert result.input_tokens is None
    assert _accounting(result)["observed"]["input_tokens"] == 10


def test_unread_response_marks_owned_work_missing():
    owner = _owner()
    with (
        owner.activate(),
        _client(
            lambda request: httpx.Response(
                200, stream=httpx.ByteStream(json.dumps(_body()).encode())
            )
        ) as client,
    ):
        with client.stream(
            "POST",
            "https://example.invalid/v1/chat/completions",
            json={"model": "model-a"},
        ):
            pass
    assert owner.finish().input_tokens is None


def test_late_usage_cannot_mutate_snapshot_or_transfer_to_next_owner():
    first, second = _owner(), _owner()
    with (
        first.activate(),
        _client(
            lambda request: httpx.Response(
                200, stream=httpx.ByteStream(json.dumps(_body()).encode())
            )
        ) as client,
    ):
        request = client.build_request(
            "POST",
            "https://example.invalid/v1/chat/completions",
            json={"model": "model-a"},
        )
        response = client.send(request, stream=True)
        frozen = first.finish().to_mapping()
        with second.activate():
            response.read()
            response.close()
    assert first.finish().to_mapping() == frozen
    assert not second.calls
    assert first.finish().input_tokens is None


def test_namespace_streaming_response_keeps_literal_field_presence():
    raw, model = usage.raw_usage(
        SimpleNamespace(
            model="actual", usage=SimpleNamespace(prompt_tokens=0, completion_tokens=2)
        ),
        "chat_completions",
    )
    assert (raw.input_tokens, raw.cached_input_tokens, raw.output_tokens, model) == (
        0,
        None,
        2,
        "actual",
    )


@contextmanager
def _observers():
    tag = object()
    registrations = usage.install_observer(tag)
    try:
        yield tag
    finally:
        for registration in registrations:
            registration.dispose()


def test_public_middleware_does_not_double_count_wire_evidence():
    pytest.importorskip("hermes_cli.middleware")
    from hermes_cli.middleware import run_llm_execution_middleware

    with _observers() as tag:
        owner = usage.InvocationUsage(tag)
        with (
            owner.activate(),
            _client(lambda request: httpx.Response(200, json=_body())) as client,
        ):
            result = run_llm_execution_middleware(
                {"model": "model-a"},
                lambda request: client.post(
                    "https://example.invalid/v1/chat/completions", json=request
                ).json(),
                api_mode="chat_completions",
                model="model-a",
                provider="openai",
            )
        assert result["usage"]["prompt_tokens"] == 10
        assert owner.finish().input_tokens == 10
        assert len(owner.calls) == 1


def test_auxiliary_keeps_positive_counters_and_cannot_invent_zero_presence():
    pytest.importorskip("hermes_cli.lifecycle")
    from hermes_cli.lifecycle import invoke_hook

    with _observers() as tag:
        owner = usage.InvocationUsage(tag)
        with owner.activate():
            invoke_hook(
                "pre_auxiliary_call",
                api_request_id="aux",
                api_call_count=1,
                model="aux-model",
                provider="openai",
            )
        # End can run in a different thread/context. It retains its initiating owner.
        invoke_hook(
            "post_auxiliary_call",
            api_request_id="aux",
            api_call_count=1,
            response_model="aux-model",
            usage={"input_tokens": 10, "cache_read_tokens": 0, "output_tokens": 2},
        )
        result = owner.finish()
        assert result.input_tokens == 10
        assert result.output_tokens == 2
        assert result.cached_input_tokens is None
        assert _accounting(result)["observed"]["input_tokens"] == 10
        assert "cached_input_tokens" not in _accounting(result)["observed"]


def test_actual_child_dispatch_is_unfinished_until_stop_and_not_a_capability_gap():
    pytest.importorskip("hermes_cli.lifecycle")
    from hermes_cli.lifecycle import invoke_hook

    with _observers() as tag:
        owner = usage.InvocationUsage(tag)
        index = owner.start("main")
        owner.settle(index, usage.raw_usage(_body(), "chat_completions")[0])
        with owner.activate():
            invoke_hook("subagent_start", child_session_id="child")
        assert owner.finish().input_tokens is None
        invoke_hook("subagent_stop", child_session_id="child")
        assert owner.finish().input_tokens is None
        assert any(
            g["reason"] == GapReason.UNFINISHED_WORK.value
            for g in _accounting(owner.finish())["gaps"]
        )


def test_completed_child_owner_is_collectible_and_late_usage_cannot_transfer():
    import gc
    import weakref

    pytest.importorskip("hermes_cli.middleware")
    from hermes_cli.lifecycle import invoke_hook
    from hermes_cli.middleware import run_llm_execution_middleware

    with _observers() as tag:
        first = usage.InvocationUsage(tag)
        with first.activate():
            invoke_hook("subagent_start", child_session_id="completed-child")
            run_llm_execution_middleware(
                {},
                lambda request: _body(),
                session_id="completed-child",
                api_mode="chat_completions",
                model="model-a",
            )
            invoke_hook("subagent_stop", child_session_id="completed-child")
        first_result = first.finish()
        frozen = first_result.to_mapping()
        first_reference = weakref.ref(first)
        del first
        gc.collect()
        assert first_reference() is None

        second = usage.InvocationUsage(tag)
        with second.activate():
            run_llm_execution_middleware(
                {},
                lambda request: _body(),
                session_id="completed-child",
                api_mode="chat_completions",
                model="model-a",
            )
            invoke_hook(
                "pre_auxiliary_call",
                session_id="completed-child",
                api_request_id="late-child-call",
                api_call_count=1,
                model="model-a",
            )
            invoke_hook(
                "post_auxiliary_call",
                api_request_id="late-child-call",
                api_call_count=1,
                usage={"input_tokens": 10, "output_tokens": 2},
            )
            run_llm_execution_middleware(
                {},
                lambda request: _body(),
                session_id="parent",
                api_mode="chat_completions",
                model="model-a",
            )
        assert _accounting(second.finish())["observed"]["input_tokens"] == 10
        assert first_result.to_mapping() == frozen


def test_source_qualification_accepts_exact_pinned_seam_and_rejects_unknown_signature():
    pytest.importorskip("run_agent")
    from run_agent import AIAgent

    assert usage.transport_qualified(AIAgent)
    unknown = MagicMock()
    unknown._build_keepalive_http_client = lambda unexpected: None
    assert not usage.transport_qualified(unknown)


@pytest.mark.parametrize(
    "scenario",
    [
        "stream",
        "retry",
        "summary",
        "summary-malformed",
        "summary-error",
        "zero",
        "silent-eof",
        "custom-alias",
    ],
)
def test_pinned_hermes_real_client_paths_use_owned_passive_transport(
    monkeypatch, tmp_path, scenario
):
    pytest.importorskip("run_agent")
    from run_agent import AIAgent
    from agent import process_bootstrap
    from agent import chat_completion_helpers
    from agent.turn_api_call import perform_api_call

    observed_type = usage.observed_agent_type(AIAgent)
    assert observed_type is not AIAgent
    model_name = "gpt-4.1-mini" if scenario == "custom-alias" else "model-a"
    requests = []
    failures_left = int(scenario == "retry")

    def transport(request):
        nonlocal failures_left
        body = json.loads(request.content)
        requests.append(body)
        if failures_left or scenario == "summary-error":
            failures_left -= 1
            return httpx.Response(
                503, json={"error": {"message": "temporary", "type": "server_error"}}
            )
        response = _body(input_tokens=0 if scenario == "zero" else 10)
        response["model"] = model_name
        if scenario == "summary-malformed":
            response["choices"] = []
        if body.get("stream"):
            content = {
                "id": "response",
                "model": model_name,
                "created": 0,
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "done"},
                        "finish_reason": "stop",
                    }
                ],
            }
            response["object"] = "chat.completion.chunk"
            response["choices"] = []
            payload = (
                b"data: "
                + json.dumps(content).encode()
                + b"\n\ndata: "
                + json.dumps(response).encode()
                + b"\n\ndata: [DONE]\n\n"
            )
            if scenario == "silent-eof":
                payload = payload.replace(b"data: [DONE]\n\n", b"")
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=httpx.ByteStream(payload),
            )
        return httpx.Response(200, json=response)

    monkeypatch.setattr(
        process_bootstrap,
        "build_keepalive_http_client",
        lambda *args, **kwargs: httpx.Client(transport=httpx.MockTransport(transport)),
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_STREAM_RETRY_DELAY", "0")
    # Skip retry backoff in this controlled source probe, without changing count.
    monkeypatch.setattr(
        chat_completion_helpers, "_wait_stream_retry_backoff", lambda agent, delay: None
    )
    agent = observed_type(
        model=model_name,
        provider="openai",
        base_url="https://example.invalid/v1",
        api_key="fixture",
        enabled_toolsets=[],
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        skip_background_review=True,
        session_id="source-probe",
        reasoning_config={"effort": "none"},
    )
    try:
        with _observers() as tag:
            owner = usage.InvocationUsage(tag)
            with owner.activate():
                if scenario.startswith("summary"):
                    result = chat_completion_helpers.handle_max_iterations(
                        agent,
                        [
                            {"role": "system", "content": "fixture"},
                            {"role": "user", "content": "finish"},
                        ],
                        1,
                    )
                    if scenario == "summary":
                        assert result == "done"
                else:
                    result = perform_api_call(
                        agent,
                        api_kwargs={
                            "model": model_name,
                            "messages": [{"role": "user", "content": "fixture"}],
                            "stream_options": {"include_usage": True},
                        },
                        _original_api_kwargs={},
                        _llm_middleware_trace=[],
                        _moa_prepared_request=None,
                        _retry=False,
                        thinking_spinner=None,
                        retry_count=0,
                        api_call_count=1,
                        api_request_id="probe",
                        effective_task_id="probe",
                        turn_id="probe",
                        interrupted=False,
                    )
                    assert result.response is not None
                    assert result.response.choices[0].message.content == "done"
            counts = owner.finish()
            if scenario == "custom-alias":
                assert all(
                    call.provider is None and call.model == model_name
                    for call in owner.calls
                )
            (tmp_path / "usage.json").write_text(
                json.dumps(counts.to_mapping(), indent=2)
            )
            if scenario == "summary-error":
                assert counts.input_tokens is None
                assert len(owner.calls) == len(requests) == 1
            elif scenario in {"retry", "silent-eof"}:
                assert counts.input_tokens is None
                assert _accounting(counts)["observed"]["input_tokens"] == 10
                assert len(owner.calls) == len(requests)
                assert len(owner.calls) == (1 if scenario == "silent-eof" else 2)
            else:
                assert counts.input_tokens == (0 if scenario == "zero" else 10)
                assert counts.cached_input_tokens == 0
                assert counts.output_tokens == 2
                assert len(owner.calls) == len(requests) == 1
    finally:
        agent.close()


def test_plugin_compression_observes_actual_owned_call_and_disposes(monkeypatch):
    pytest.importorskip("agent.context_engine")
    from agent.context_engine import ContextEngine
    from hermes_cli import plugins

    engine = MagicMock(spec=ContextEngine)
    engine.compress = MagicMock(
        return_value=([{"role": "user", "content": "kept"}], "system")
    )
    original = engine.compress
    monkeypatch.setattr(plugins, "get_plugin_context_engine", lambda: object())
    agent = SimpleNamespace(context_compressor=engine)
    owner = _owner()
    lease = usage.observe_context_engine(agent, owner.registration_tag)
    assert lease is not None
    assert not owner.calls
    with owner.activate():
        result = engine.compress([], force=True)
    assert result == ([{"role": "user", "content": "kept"}], "system")
    assert len(owner.calls) == 1
    assert owner.calls[0].gap is GapReason.SCOPE_UNVERIFIED
    lease.dispose()
    assert engine.compress is original


def test_shared_plugin_engine_keeps_execution_unmodified_and_declines_scope(
    monkeypatch,
):
    pytest.importorskip("agent.context_engine")
    from agent.context_engine import ContextEngine
    from hermes_cli import plugins

    engine = MagicMock(spec=ContextEngine)
    original = engine.compress
    monkeypatch.setattr(plugins, "get_plugin_context_engine", lambda: engine)
    lease = usage.observe_context_engine(
        SimpleNamespace(context_compressor=engine), object()
    )
    assert lease is not None and lease.scope_unverified
    assert engine.compress is original
    lease.dispose()


def test_missing_context_compressor_module_keeps_execution_unmodified(monkeypatch):
    engine = MagicMock()
    original = engine.compress
    real_import = builtins.__import__

    def import_without_compressor(name, *args, **kwargs):
        if name == "agent.context_compressor":
            raise ImportError("context compressor unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_compressor)
    lease = usage.observe_context_engine(
        SimpleNamespace(context_compressor=engine), object()
    )

    assert lease is not None and lease.scope_unverified
    assert engine.compress is original
    lease.dispose()


def test_source_auxiliary_attempt_hook_retains_real_normalized_positive_fields():
    pytest.importorskip("agent.auxiliary_hooks")
    from agent.auxiliary_hooks import run_with_aux_hooks
    from openai.types.chat import ChatCompletion

    response = ChatCompletion.model_validate(_body())
    with _observers() as tag:
        owner = usage.InvocationUsage(tag)
        with owner.activate():
            actual = run_with_aux_hooks(
                lambda: response,
                aux_task="compression",
                metadata={"api_request_id": "compression"},
                client=SimpleNamespace(base_url="https://example.invalid/v1"),
                kwargs={"model": "model-a"},
                provider="openai",
                model="model-a",
                api_mode="chat_completions",
            )
        assert actual is response
        result = owner.finish()
        assert result.input_tokens == 10
        assert result.input_tokens_include_cache is True
        assert result.output_tokens == 2
        assert result.cached_input_tokens is None
        assert len(owner.calls) == 1


def test_changed_source_keeps_original_agent_and_scoped_middleware_observations(
    monkeypatch,
):
    pytest.importorskip("run_agent")
    from run_agent import AIAgent
    from hermes_cli.middleware import run_llm_execution_middleware

    monkeypatch.setattr(usage, "_SOURCE_HASHES", {"client_lifecycle.py": "unqualified"})
    assert usage.observed_agent_type(AIAgent) is AIAgent
    with _observers() as tag:
        owner = usage.InvocationUsage(tag)
        with owner.activate():
            response = run_llm_execution_middleware(
                {"model": "model-a"},
                lambda request: _body(),
                api_mode="chat_completions",
                model="model-a",
                provider="openai",
            )
        assert response["choices"][0]["message"]["content"] == "done"
        result = owner.finish()
        assert result.input_tokens is None
        assert _accounting(result)["observed"]["input_tokens"] == 10


def test_two_runtime_registrations_count_only_their_own_invocation():
    pytest.importorskip("hermes_cli.middleware")
    from hermes_cli.middleware import run_llm_execution_middleware

    with _observers() as first_tag, _observers() as second_tag:
        first, second = (
            usage.InvocationUsage(first_tag),
            usage.InvocationUsage(second_tag),
        )
        for owner in (first, second):
            with owner.activate():
                run_llm_execution_middleware(
                    {},
                    lambda request: _body(),
                    api_mode="chat_completions",
                    model="model-a",
                    provider="openai",
                )
        assert len(first.calls) == len(second.calls) == 1


def test_descriptor_copies_declare_identical_accounting_payload():
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    canonical = (
        root / "adapters/python/hermes/hermes.fabric-adapter.json"
    ).read_bytes()
    cli = (
        root / "crates/fabric-cli/assets/adapters/hermes/hermes.fabric-adapter.json"
    ).read_bytes()
    assert canonical == cli
    declaration = json.loads(canonical)["extension_schemas"]["usage"]
    assert "nemo_fabric.usage_accounting.v1" in declaration["properties"]


def test_auxiliary_cached_only_usage_cannot_fabricate_omitted_prompt():
    pytest.importorskip("agent.auxiliary_hooks")
    from agent.auxiliary_hooks import run_with_aux_hooks

    response = SimpleNamespace(
        model="model-a",
        usage=SimpleNamespace(
            completion_tokens=2, prompt_tokens_details=SimpleNamespace(cached_tokens=10)
        ),
        choices=[],
    )
    with _observers() as tag:
        owner = usage.InvocationUsage(tag)
        with owner.activate():
            run_with_aux_hooks(
                lambda: response,
                aux_task="compression",
                metadata={"api_request_id": "missing-prompt"},
                client=SimpleNamespace(base_url="https://example.invalid/v1"),
                kwargs={"model": "model-a"},
                provider="openai",
                model="model-a",
                api_mode="chat_completions",
            )
        assert owner.finish().input_tokens is None
        assert owner.finish().cached_input_tokens == 10
        assert owner.finish().output_tokens == 2


@pytest.mark.parametrize("damage", ["malformed", "oversized"])
def test_stream_parsing_loss_cannot_certify_an_earlier_usage_snapshot(damage):
    owner = _owner()
    good = b"data: " + json.dumps(_body()).encode() + b"\n\n"
    final = (
        b"data: {invalid json}\n\n"
        if damage == "malformed"
        else b"data: " + b"x" * (usage._MAX_EVENT_BYTES + 1) + b"\n\n"
    )
    payload = good + final + b"data: [DONE]\n\n"
    with (
        owner.activate(),
        _client(
            lambda request: httpx.Response(200, stream=httpx.ByteStream(payload))
        ) as client,
    ):
        response = client.post(
            "https://example.invalid/v1/chat/completions",
            json={"model": "model-a", "stream": True},
        )
        assert response.content == payload
    result = owner.finish()
    assert result.input_tokens is None
    assert _accounting(result)["observed"]["input_tokens"] == 10
    assert any(g["reason"] == "scope_unverified" for g in _accounting(result)["gaps"])


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://api.openai.com/v1/chat/completions", "openai"),
        ("https://api.anthropic.com/v1/messages", "anthropic"),
        ("https://api.openai.com:444/v1/chat/completions", None),
        ("http://api.openai.com/v1/chat/completions", None),
        ("http://localhost:1234/v1/chat/completions", None),
        ("https://custom.invalid/v1/chat/completions", None),
    ],
)
def test_actual_http_route_qualifies_provider_pricing_identity(url, expected):
    owner = _owner()
    with (
        owner.activate(),
        _client(lambda request: httpx.Response(200, json=_body())) as client,
    ):
        client.post(url, json={"model": "gpt-5.4"})
    assert owner.calls[0].provider == expected
    assert owner.calls[0].model == "model-a"
