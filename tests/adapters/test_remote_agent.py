# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end protocol tests for the remote-agent adapter."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import httpx
import pytest
from nemo_fabric_adapter_contract.models import AgentConfig
from nemo_fabric_adapter_contract.models import AgentRunRequest
from nemo_fabric_adapter_contract.models import AgentRunResult
from nemo_fabric_adapter_contract.models import AgentUsage
from nemo_fabric_adapter_contract.models import RuntimeContext
from nemo_fabric_adapters.remote_agent import adapter


def _context(*, relay: bool = False) -> RuntimeContext:
    payload = {
        "runtime_id": "remote-agent-runtime",
        "invocation_id": "remote-agent-invocation",
        "request_id": "remote-agent-request",
        "environment": {
            "environment_id": "remote-agent-environment",
            "provider": "local",
            "control_location": "in_env_control",
            "workspace": ".",
            "env": {},
            "ownership": "caller_owned",
        },
        "artifacts": {},
    }
    if relay:
        payload["telemetry"] = {"relay_enabled": True}
    return RuntimeContext.from_mapping(payload)


async def test_sse_events_flushes_unterminated_final_event():
    async def lines():
        yield "event: response.completed"
        yield 'data: {"response": {"id": "response-id"}}'

    mock_response = MagicMock()
    mock_response.aiter_lines.return_value = lines()

    events = [event async for event in adapter._sse_events(mock_response)]

    assert events == [("response.completed", {"response": {"id": "response-id"}})]


@pytest.mark.parametrize(
    ("api_type", "path"),
    [
        pytest.param("openai-responses", "/responses", id="openai-responses"),
        pytest.param(
            "openai-completions", "/chat/completions", id="openai-completions"
        ),
        pytest.param("anthropic-messages", "/messages", id="anthropic-messages"),
    ],
)
async def test_remote_agent_invokes_supported_protocol(
    api_server: str, api_type: str, path: str, repo_root: Path
):
    config = AgentConfig.from_mapping(
        {
            "harness": {
                "settings": {"base_url": f"{api_server}/v1", "api_type": api_type}
            },
            "instructions": {"system": {"content": "Be concise."}},
            "models": {
                "default": {
                    "provider": "test",
                    "model": "fabric-echo",
                    "temperature": 0.2,
                    "top_p": 0.8,
                    "max_tokens": 64,
                }
            },
        }
    )
    context = _context()
    runtime = adapter.RemoteAgentRuntime()
    await runtime.start(
        {
            "config": config,
            "runtime_context": context.to_mapping(),
            "base_dir": str(repo_root),
        }
    )

    try:
        assert runtime._endpoint == f"{api_server}/v1{path}"

        result = await runtime.invoke(AgentRunRequest(input="Hello."), context)
    finally:
        await runtime.stop()

    async with httpx.AsyncClient() as control_client:
        captured = (await control_client.get(f"{api_server}/_requests")).json()
    assert result.status == "succeeded"
    expected_text = (
        "echo user_count=1 latest=Hello."
        if api_type == "openai-completions"
        else "echo response"
    )
    assert result.output == {"response": expected_text}
    assert result.usage.input_tokens == 0
    assert captured[-1]["model"] == "fabric-echo"
    assert captured[-1]["temperature"] == 0.2
    assert captured[-1]["top_p"] == 0.8
    max_tokens_field = {
        "openai-responses": "max_output_tokens",
        "openai-completions": "max_completion_tokens",
        "anthropic-messages": "max_tokens",
    }[api_type]
    assert captured[-1][max_tokens_field] == 64
    assert captured[-1]["stream"] is True
    if api_type == "openai-completions":
        assert captured[-1]["stream_options"] == {"include_usage": True}
    assert captured[-1]["messages" if api_type != "openai-responses" else "input"]


@pytest.mark.parametrize(
    ("normalized_max_tokens", "expected_max_tokens"),
    [
        pytest.param(None, 32, id="legacy-settings-fallback"),
        pytest.param(64, 64, id="normalized-field-precedence"),
    ],
)
async def test_anthropic_max_tokens_fallback_and_precedence(
    api_server: str,
    repo_root: Path,
    normalized_max_tokens: int | None,
    expected_max_tokens: int,
):
    model: dict[str, object] = {
        "provider": "test",
        "model": "fabric-echo",
        "settings": {"max_tokens": 32},
    }
    if normalized_max_tokens is not None:
        model["max_tokens"] = normalized_max_tokens
    config = AgentConfig.from_mapping(
        {
            "harness": {
                "settings": {
                    "base_url": f"{api_server}/v1",
                    "api_type": "anthropic-messages",
                }
            },
            "models": {"default": model},
        }
    )
    context = _context()
    runtime = adapter.RemoteAgentRuntime()
    await runtime.start(
        {
            "config": config,
            "runtime_context": context.to_mapping(),
            "base_dir": str(repo_root),
        }
    )

    try:
        result = await runtime.invoke(AgentRunRequest(input="Hello."), context)
    finally:
        await runtime.stop()

    async with httpx.AsyncClient() as control_client:
        captured = (await control_client.get(f"{api_server}/_requests")).json()
    assert result.status == "succeeded"
    assert captured[-1]["max_tokens"] == expected_max_tokens


async def test_remote_agent_retains_transcript_and_reports_http_failure(
    api_server: str,
    repo_root: Path,
):
    config = AgentConfig.from_mapping(
        {
            "harness": {"settings": {"base_url": f"{api_server}/v1"}},
            "models": {"default": {"provider": "test", "model": "fabric-echo"}},
        }
    )
    context = _context()
    runtime = adapter.RemoteAgentRuntime()
    await runtime.start(
        {
            "config": config,
            "runtime_context": context.to_mapping(),
            "base_dir": str(repo_root),
        }
    )

    try:
        await runtime.invoke(AgentRunRequest(input="First."), context)
        await runtime.invoke(AgentRunRequest(input="Second."), context)

        async with httpx.AsyncClient() as control_client:
            captured = (await control_client.get(f"{api_server}/_requests")).json()
        assert [message["role"] for message in captured[-1]["input"]] == [
            "user",
            "assistant",
            "user",
        ]

        async with httpx.AsyncClient() as control_client:
            await control_client.post(
                f"{api_server}/_scenario", json={"status_code": 503}
            )
        result = await runtime.invoke(AgentRunRequest(input="Retry."), context)
    finally:
        await runtime.stop()

    assert result.status == "failed"
    assert result.error.code == "remote_agent_http_error"
    assert result.error.retryable is True


@pytest.mark.parametrize(
    "api_type",
    ["openai-responses", "openai-completions"],
)
async def test_remote_agent_maps_relay_request_id_metadata(
    api_server: str,
    api_type: str,
    repo_root: Path,
):
    config = AgentConfig.from_mapping(
        {
            "harness": {
                "settings": {
                    "base_url": f"{api_server}/v1",
                    "api_type": api_type,
                    "relay_streaming": True,
                }
            },
            "models": {"default": {"provider": "test", "model": "fabric-echo"}},
        }
    )
    context = _context(relay=True)
    runtime = adapter.RemoteAgentRuntime()
    await runtime.start(
        {
            "config": config,
            "runtime_context": context.to_mapping(),
            "base_dir": str(repo_root),
        }
    )

    try:
        result = await runtime.invoke(AgentRunRequest(input="Hello."), context)
    finally:
        await runtime.stop()

    async with httpx.AsyncClient() as control_client:
        captured = (await control_client.get(f"{api_server}/_requests")).json()
    assert result.status == "succeeded"
    assert captured[-1]["metadata"] == {
        adapter.FABRIC_REQUEST_ID_METADATA: context.request_id
    }


async def test_remote_agent_does_not_add_metadata_without_relay(
    api_server: str,
    repo_root: Path,
):
    config = AgentConfig.from_mapping(
        {
            "harness": {"settings": {"base_url": f"{api_server}/v1"}},
            "models": {"default": {"provider": "test", "model": "fabric-echo"}},
        }
    )
    context = _context()
    runtime = adapter.RemoteAgentRuntime()
    await runtime.start(
        {
            "config": config,
            "runtime_context": context.to_mapping(),
            "base_dir": str(repo_root),
        }
    )

    try:
        result = await runtime.invoke(AgentRunRequest(input="Hello."), context)
    finally:
        await runtime.stop()

    async with httpx.AsyncClient() as control_client:
        captured = (await control_client.get(f"{api_server}/_requests")).json()
    assert result.status == "succeeded"
    assert "metadata" not in captured[-1]


@pytest.mark.parametrize(
    ("relay", "relay_streaming"),
    [
        pytest.param(True, False, id="relay-without-opt-in"),
        pytest.param(False, True, id="opt-in-without-relay"),
    ],
)
async def test_remote_agent_rejects_mismatched_relay_streaming_configuration(
    repo_root: Path,
    relay: bool,
    relay_streaming: bool,
):
    config = AgentConfig.from_mapping(
        {
            "harness": {
                "settings": {
                    "base_url": "https://agents.example.test/v1",
                    "relay_streaming": relay_streaming,
                }
            },
            "models": {"default": {"provider": "test", "model": "fabric-echo"}},
        }
    )
    runtime = adapter.RemoteAgentRuntime()

    with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
        await runtime.start(
            {
                "config": config,
                "runtime_context": _context(relay=relay).to_mapping(),
                "base_dir": str(repo_root),
            }
        )

    assert caught.value.code == "remote_agent_invalid_relay_configuration"


async def test_remote_agent_rejects_anthropic_messages_relay_streaming(
    repo_root: Path,
):
    config = AgentConfig.from_mapping(
        {
            "harness": {
                "settings": {
                    "base_url": "https://agents.example.test/v1",
                    "api_type": "anthropic-messages",
                    "relay_streaming": True,
                }
            },
            "models": {"default": {"provider": "test", "model": "fabric-echo"}},
        }
    )
    runtime = adapter.RemoteAgentRuntime()

    with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
        await runtime.start(
            {
                "config": config,
                "runtime_context": _context(relay=True).to_mapping(),
                "base_dir": str(repo_root),
            }
        )

    assert caught.value.code == "remote_agent_invalid_relay_configuration"
    assert caught.value.metadata == {"field": "harness.settings.api_type"}


async def test_remote_agent_configures_http_client(
    monkeypatch: pytest.MonkeyPatch, repo_root: Path
):
    mock_client = MagicMock()
    mock_client.aclose = AsyncMock()
    mock_async_client = MagicMock(return_value=mock_client)
    monkeypatch.setattr(adapter.httpx, "AsyncClient", mock_async_client)

    config = AgentConfig.from_mapping(
        {
            "harness": {
                "settings": {
                    "base_url": "https://agents.example.test/v1",
                    "connect_timeout_seconds": 2.5,
                    "read_timeout_seconds": 30,
                }
            },
            "models": {"default": {"provider": "test", "model": "fabric-echo"}},
        }
    )
    context = _context()
    runtime = adapter.RemoteAgentRuntime()
    await runtime.start(
        {
            "config": config,
            "runtime_context": context.to_mapping(),
            "base_dir": str(repo_root),
        }
    )
    await runtime.stop()
    mock_client.aclose.assert_awaited_once()

    assert mock_async_client.call_args.kwargs["http2"] is True
    timeout = mock_async_client.call_args.kwargs["timeout"]
    assert timeout.connect == 2.5
    assert timeout.read == 30
    assert timeout.write == 30
    assert timeout.pool == 2.5


async def test_remote_agent_maps_timeout_and_closes_client(
    monkeypatch: pytest.MonkeyPatch, repo_root: Path
):
    request = httpx.Request("POST", "https://agents.example.test/v1/chat/completions")
    mock_client = MagicMock()
    mock_client.stream = MagicMock(
        side_effect=httpx.ReadTimeout("remote agent timed out", request=request)
    )
    mock_client.aclose = AsyncMock()
    monkeypatch.setattr(
        adapter.httpx, "AsyncClient", MagicMock(return_value=mock_client)
    )

    config = AgentConfig.from_mapping(
        {
            "harness": {
                "settings": {
                    "base_url": "https://agents.example.test/v1",
                    "api_type": "openai-completions",
                }
            },
            "models": {"default": {"provider": "test", "model": "fabric-echo"}},
        }
    )
    context = _context()
    runtime = adapter.RemoteAgentRuntime()
    await runtime.start(
        {
            "config": config,
            "runtime_context": context.to_mapping(),
            "base_dir": str(repo_root),
        }
    )

    try:
        with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
            await runtime.invoke(AgentRunRequest(input="Hello."), context)
    finally:
        await runtime.stop()

    assert caught.value.code == "remote_agent_transport_failed"
    assert caught.value.retryable is True
    mock_client.aclose.assert_awaited_once()


async def test_remote_agent_rejects_append_system_instruction(repo_root: Path):
    config = AgentConfig.from_mapping(
        {
            "harness": {"settings": {"base_url": "https://agents.example.test/v1"}},
            "instructions": {
                "system": {"content": "Additional guidance.", "mode": "append"}
            },
            "models": {"default": {"provider": "test", "model": "fabric-echo"}},
        }
    )
    context = _context()
    runtime = adapter.RemoteAgentRuntime()

    with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
        await runtime.start(
            {
                "config": config,
                "runtime_context": context.to_mapping(),
                "base_dir": str(repo_root),
            }
        )

    assert caught.value.code == "unsupported_system_instruction_mode"
    assert caught.value.metadata["field"] == "instructions.system.mode"


def _sse(events: list[tuple[str, dict[str, object]]]) -> str:
    return "".join(
        f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events
    )


def _responses_stream(usage: object) -> str:
    return _sse(
        [
            (
                "response.completed",
                {
                    "type": "response.completed",
                    "response": {
                        "id": "resp_fabric_test",
                        "object": "response",
                        "status": "completed",
                        "output": [
                            {
                                "type": "message",
                                "role": "assistant",
                                "content": [{"type": "output_text", "text": "done"}],
                            }
                        ],
                        "usage": usage,
                    },
                },
            )
        ]
    )


_OMITTED = object()


def _messages_stream(start_usage: object, delta_usage: object) -> str:
    delta: dict[str, object] = {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
    }
    if delta_usage is not _OMITTED:
        delta["usage"] = delta_usage
    return _sse(
        [
            (
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_fabric_test",
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": "fabric-echo",
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": start_usage,
                    },
                },
            ),
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "done"},
                },
            ),
            ("message_delta", delta),
            ("message_stop", {"type": "message_stop"}),
        ]
    )


async def _invoke_scripted_stream(
    monkeypatch: pytest.MonkeyPatch, repo_root: Path, api_type: str, body: str
) -> AgentRunResult:
    [result] = await _invoke_scripted_streams(monkeypatch, repo_root, api_type, [body])
    return result


async def _invoke_scripted_streams(
    monkeypatch: pytest.MonkeyPatch,
    repo_root: Path,
    api_type: str,
    bodies: list[str],
) -> list[AgentRunResult]:
    """Invoke one runtime once per body; each call receives the next body."""

    async_client = httpx.AsyncClient
    pending = iter(bodies)

    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, text=next(pending), headers={"content-type": "text/event-stream"}
        )

    monkeypatch.setattr(
        adapter.httpx,
        "AsyncClient",
        lambda **_: async_client(transport=httpx.MockTransport(respond)),
    )
    config = AgentConfig.from_mapping(
        {
            "harness": {
                "settings": {
                    "base_url": "https://agents.example.test/v1",
                    "api_type": api_type,
                }
            },
            "models": {"default": {"provider": "test", "model": "fabric-echo"}},
        }
    )
    context = _context()
    runtime = adapter.RemoteAgentRuntime()
    await runtime.start(
        {
            "config": config,
            "runtime_context": context.to_mapping(),
            "base_dir": str(repo_root),
        }
    )
    try:
        return [
            await runtime.invoke(AgentRunRequest(input="Hello."), context)
            for _ in bodies
        ]
    finally:
        await runtime.stop()


@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        pytest.param(
            {
                "input_tokens": 12000,
                "input_tokens_details": {"cached_tokens": 9000},
                "output_tokens": 300,
                "output_tokens_details": {"reasoning_tokens": 120},
                "total_tokens": 12300,
            },
            AgentUsage(
                input_tokens=12000,
                cached_input_tokens=9000,
                input_tokens_include_cache=True,
                output_tokens=300,
                total_tokens=12300,
            ),
            id="cached-input",
        ),
        pytest.param(
            {"output_tokens": 5},
            AgentUsage(output_tokens=5),
            id="input-and-total-omitted",
        ),
        pytest.param(
            {"input_tokens_details": {"cached_tokens": 9}},
            AgentUsage(cached_input_tokens=9),
            id="only-cached-count",
        ),
        pytest.param(None, None, id="null-usage"),
    ],
)
async def test_responses_usage_reports_cached_input_and_tolerates_null_usage(
    monkeypatch: pytest.MonkeyPatch,
    repo_root: Path,
    usage: object,
    expected: AgentUsage | None,
):
    result = await _invoke_scripted_stream(
        monkeypatch, repo_root, "openai-responses", _responses_stream(usage)
    )

    assert result.status == "succeeded"
    assert result.output == {"response": "done"}
    assert result.usage == expected


@pytest.mark.parametrize(
    ("start_usage", "delta_usage", "expected"),
    [
        pytest.param(
            {
                "input_tokens": 50,
                "cache_creation_input_tokens": 2000,
                "cache_read_input_tokens": 48000,
                "output_tokens": 1,
            },
            {"output_tokens": 200},
            AgentUsage(
                input_tokens=50050,
                cached_input_tokens=48000,
                input_tokens_include_cache=True,
                output_tokens=200,
            ),
            id="cache-reads-and-writes",
        ),
        pytest.param(
            # Anthropic's server-tool example: message_delta carries cumulative
            # input and cache counts that replace the message_start values.
            {
                "input_tokens": 2679,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "output_tokens": 3,
            },
            {
                "input_tokens": 10682,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "output_tokens": 510,
                "server_tool_use": {"web_search_requests": 1},
            },
            AgentUsage(
                input_tokens=10682,
                cached_input_tokens=0,
                input_tokens_include_cache=True,
                output_tokens=510,
            ),
            id="cumulative-message-delta",
        ),
        pytest.param(
            # vLLM's default: no cache fields, and input_tokens is the full prompt.
            {"input_tokens": 1200, "output_tokens": 1},
            {"output_tokens": 200},
            AgentUsage(
                input_tokens=1200,
                input_tokens_include_cache=True,
                output_tokens=200,
            ),
            id="cache-fields-omitted",
        ),
        pytest.param(
            # vLLM with prompt token details: cache reads only.
            {"input_tokens": 200, "cache_read_input_tokens": 1000, "output_tokens": 1},
            {"output_tokens": 200},
            AgentUsage(
                input_tokens=1200,
                cached_input_tokens=1000,
                input_tokens_include_cache=True,
                output_tokens=200,
            ),
            id="cache-reads-only",
        ),
        pytest.param(
            {
                "input_tokens": 50,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "output_tokens": 1,
            },
            _OMITTED,
            AgentUsage(
                input_tokens=50,
                cached_input_tokens=0,
                input_tokens_include_cache=True,
            ),
            id="message-delta-without-usage",
        ),
        pytest.param(
            {
                "input_tokens": 0,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "output_tokens": 0,
            },
            {"output_tokens": 0},
            AgentUsage(
                input_tokens=0,
                cached_input_tokens=0,
                input_tokens_include_cache=True,
                output_tokens=0,
            ),
            id="explicit-zeros",
        ),
        pytest.param(
            # Without input_tokens the inclusive input is unknown, but the
            # reported cache reads and output still count.
            {"cache_read_input_tokens": 9},
            {"output_tokens": 3},
            AgentUsage(cached_input_tokens=9, output_tokens=3),
            id="input-omitted",
        ),
        pytest.param(None, None, None, id="null-usage"),
    ],
)
async def test_messages_usage_adds_reported_cache_counts_to_input(
    monkeypatch: pytest.MonkeyPatch,
    repo_root: Path,
    start_usage: object,
    delta_usage: object,
    expected: AgentUsage | None,
):
    result = await _invoke_scripted_stream(
        monkeypatch,
        repo_root,
        "anthropic-messages",
        _messages_stream(start_usage, delta_usage),
    )

    assert result.status == "succeeded"
    assert result.output == {"response": "done"}
    assert result.usage == expected


async def test_responses_usage_keeps_every_count_known_across_a_cached_run(
    monkeypatch: pytest.MonkeyPatch, repo_root: Path
):
    # An OpenAI run: the first call reads nothing from cache and reports an
    # explicit 0; later calls reuse the cached prompt prefix.
    reports = [(1200, 0, 80), (1350, 1024, 95), (1500, 1280, 60)]
    bodies = [
        _responses_stream(
            {
                "input_tokens": prompt,
                "input_tokens_details": {"cached_tokens": cached},
                "output_tokens": output,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": prompt + output,
            }
        )
        for prompt, cached, output in reports
    ]

    results = await _invoke_scripted_streams(
        monkeypatch, repo_root, "openai-responses", bodies
    )

    assert [result.usage for result in results] == [
        AgentUsage(
            input_tokens=prompt,
            cached_input_tokens=cached,
            input_tokens_include_cache=True,
            output_tokens=output,
            total_tokens=prompt + output,
        )
        for prompt, cached, output in reports
    ]


async def test_messages_usage_keeps_every_count_known_across_a_cached_run(
    monkeypatch: pytest.MonkeyPatch, repo_root: Path
):
    # An Anthropic run: the first call writes the cache and reads 0 from it;
    # later calls read the cached prefix and send only new tokens as input.
    reports = [
        # (input_tokens, cache writes, cache reads, output_tokens)
        (12, 4000, 0, 150),
        (40, 0, 4000, 90),
        (35, 300, 4000, 120),
    ]
    bodies = [
        _messages_stream(
            {
                "input_tokens": uncached,
                "cache_creation_input_tokens": writes,
                "cache_read_input_tokens": reads,
                "output_tokens": 1,
            },
            {"output_tokens": output},
        )
        for uncached, writes, reads, output in reports
    ]

    results = await _invoke_scripted_streams(
        monkeypatch, repo_root, "anthropic-messages", bodies
    )

    assert [result.usage for result in results] == [
        AgentUsage(
            input_tokens=uncached + writes + reads,
            cached_input_tokens=reads,
            input_tokens_include_cache=True,
            output_tokens=output,
        )
        for uncached, writes, reads, output in reports
    ]


@pytest.mark.parametrize(
    ("api_type", "body"),
    [
        pytest.param(
            "openai-responses",
            _sse([("response.completed", {"response": None})]),
            id="null-response",
        ),
        pytest.param(
            "anthropic-messages",
            _sse([("message_start", {"type": "message_start", "message": None})]),
            id="null-message",
        ),
    ],
)
async def test_null_event_object_is_an_invalid_response(
    monkeypatch: pytest.MonkeyPatch, repo_root: Path, api_type: str, body: str
):
    with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
        await _invoke_scripted_stream(monkeypatch, repo_root, api_type, body)

    assert caught.value.code == "remote_agent_invalid_response"


@pytest.mark.parametrize(
    ("key", "value"),
    [
        pytest.param("input_tokens", -1, id="negative-input"),
        pytest.param("input_tokens", True, id="bool-input"),
        pytest.param("cache_creation_input_tokens", -1, id="negative-cache-writes"),
        pytest.param("cache_creation_input_tokens", True, id="bool-cache-writes"),
        pytest.param("cache_read_input_tokens", "9", id="string-cache-reads"),
    ],
)
async def test_messages_malformed_input_count_is_an_invalid_response(
    monkeypatch: pytest.MonkeyPatch, repo_root: Path, key: str, value: object
):
    # The adapter sums these counts, so it must reject a bad one before the sum
    # turns it into a valid-looking input count.
    start_usage = {
        "input_tokens": 50,
        "cache_creation_input_tokens": 20,
        "cache_read_input_tokens": 1000,
        "output_tokens": 1,
    }
    start_usage[key] = value

    with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
        await _invoke_scripted_stream(
            monkeypatch,
            repo_root,
            "anthropic-messages",
            _messages_stream(start_usage, {"output_tokens": 7}),
        )

    assert caught.value.code == "remote_agent_invalid_response"


def test_remote_agent_descriptor_and_module_entrypoint(repo_root: Path):
    descriptor = json.loads(
        (
            repo_root / "adapters/python/remote-agent/remote-agent.fabric-adapter.json"
        ).read_text(encoding="utf-8")
    )
    result = subprocess.run(
        [sys.executable, "-m", "nemo_fabric_adapters.remote_agent.adapter"],
        input="",
        text=True,
        capture_output=True,
        check=False,
    )

    assert descriptor["adapter_id"] == "nvidia.fabric.remote-agent"
    assert descriptor["settings_schema"]["properties"]["api_type"]["default"] == (
        "openai-responses"
    )
    assert descriptor["config"]["system_instruction_modes"] == ["replace"]
    assert descriptor["capabilities"]["streaming"] is True
    assert descriptor["telemetry"]["providers"]["relay"] == {
        "outputs": ["atof"],
        "integration_modes": ["remote_service"],
    }
    assert result.returncode == 0, result.stderr
