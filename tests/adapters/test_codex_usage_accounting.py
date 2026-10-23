# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scripted notifications through the supported SDK's real router and turn handle.

No app-server process, network request, provider, or timing-based timeout is used.
"""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from jsonschema import Draft202012Validator
from nemo_fabric_adapter_contract.models import (
    AgentConfig,
    AgentRunRequest,
    RuntimeContext,
)
from nemo_fabric_adapters.codex import adapter
from nemo_fabric_adapter_contract.usage import UsageObservation
from openai_codex import AsyncCodex, AsyncThread, AsyncTurnHandle, TransportClosedError
from openai_codex._message_router import MessageRouter
from openai_codex.generated.v2_all import (
    ErrorNotification,
    ItemCompletedNotification,
    ItemStartedNotification,
    ThreadTokenUsageUpdatedNotification,
    TurnCompletedNotification,
)
from openai_codex.models import Notification, UnknownNotification

ACCOUNTING_KEY = "nemo_fabric.usage_accounting.v1"


@pytest.fixture(name="scripted_turn")
def scripted_turn_fixture(tmp_path):
    router = MessageRouter()
    codex = MagicMock(spec=AsyncCodex)
    codex._ensure_initialized = AsyncMock()
    codex._client = MagicMock()
    codex._client.turn_interrupt = AsyncMock()
    handle = AsyncTurnHandle(
        codex, "thread-1", "turn-1", _subscription=router.subscribe_turn("turn-1")
    )
    thread = MagicMock(spec=AsyncThread)
    thread.id = "thread-1"
    thread.turn = AsyncMock(return_value=handle)
    config = AgentConfig.from_mapping(
        {
            "harness": {"settings": {}},
            "models": {"default": {"provider": "openai", "model": "gpt-5.4"}},
        }
    )
    context = RuntimeContext.from_mapping(
        {
            "runtime_id": "runtime-1",
            "invocation_id": "invocation-1",
            "request_id": "request-1",
            "environment": {
                "environment_id": "environment-1",
                "provider": "local",
                "control_location": "in_env_control",
                "ownership": "caller_owned",
                "workspace": str(tmp_path),
            },
            "artifacts": {"root": str(tmp_path / "artifacts")},
        }
    )
    return router, handle, thread, config, context


def report(router, input_tokens, output_tokens, *, cached=0, turn_id="turn-1"):
    count = {
        "inputTokens": input_tokens,
        "cachedInputTokens": cached,
        "outputTokens": output_tokens,
        "reasoningOutputTokens": 0,
        "totalTokens": input_tokens + output_tokens,
    }
    router.route_notification(
        Notification(
            method="thread/tokenUsage/updated",
            payload=ThreadTokenUsageUpdatedNotification.model_validate(
                {
                    "threadId": "thread-1",
                    "turnId": turn_id,
                    "tokenUsage": {"total": count, "last": count},
                }
            ),
        )
    )


def respond(router, raw, *, response_id="response-1", turn_id="turn-1"):
    router.route_notification(
        Notification(
            method="rawResponse/completed",
            payload=UnknownNotification(
                {
                    "threadId": "thread-1",
                    "turnId": turn_id,
                    "responseId": response_id,
                    "usageMetadata": None if raw is None else {"metadata": raw},
                }
            ),
        )
    )


def source_result(raw):
    return adapter._agent_run_result(
        {
            "usage_observations": [
                UsageObservation(
                    reported=adapter._response_usage(raw), is_model_call=True
                )
            ]
        }
    )


def complete(router, *, failed=False):
    if not failed:
        router.route_notification(
            Notification(
                method="item/completed",
                payload=ItemCompletedNotification.model_validate(
                    {
                        "completedAtMs": 101000,
                        "threadId": "thread-1",
                        "turnId": "turn-1",
                        "item": {
                            "id": "answer-1",
                            "type": "agentMessage",
                            "phase": "final_answer",
                            "text": "done",
                        },
                    }
                ),
            )
        )
    turn = {"id": "turn-1", "items": [], "status": "failed" if failed else "completed"}
    if failed:
        turn["error"] = {"message": "turn failed after accounted work"}
    router.route_notification(
        Notification(
            method="turn/completed",
            payload=TurnCompletedNotification.model_validate(
                {"threadId": "thread-1", "turn": turn}
            ),
        )
    )


async def invoke_script(scripted_turn):
    _, _, thread, config, context = scripted_turn
    output, usable = await adapter._invoke_thread(
        config,
        context,
        str(Path(context.environment.workspace)),
        AgentRunRequest(input="Inspect the change."),
        thread,
        asyncio.get_running_loop().time() + 60,
    )
    return adapter._agent_run_result(output), usable


@pytest.mark.parametrize("failed", [False, True])
async def test_response_records_ignore_thread_snapshots_and_keep_terminal_failure_usage(
    scripted_turn, failed
):
    router = scripted_turn[0]
    report(router, 10, 3)
    report(router, 40, 8, cached=6)
    report(router, 999, 999, turn_id="another-turn")
    respond(
        router,
        {
            "input_tokens": 40,
            "input_tokens_details": {"cached_tokens": 6},
            "output_tokens": 8,
            "total_tokens": 48,
        },
    )
    respond(router, {"input_tokens": 999}, turn_id="another-turn")
    complete(router, failed=failed)

    result, usable = await invoke_script(scripted_turn)

    assert result.usage.input_tokens == 40
    assert result.usage.output_tokens == 8
    assert result.usage.total_tokens == 48
    assert result.usage.cached_input_tokens == 6
    accounting = result.usage.extensions[ACCOUNTING_KEY]
    assert accounting["observed"]["cached_input_tokens"] == 6
    assert accounting["models_exhaustive"] is False
    assert "models" not in accounting
    assert usable is not failed


@pytest.mark.parametrize(
    "error",
    [TimeoutError("request timed out"), TransportClosedError("connection closed")],
)
async def test_sdk_transport_failure_keeps_only_frozen_observations(
    scripted_turn, error
):
    router = scripted_turn[0]
    report(router, 40, 8)
    respond(router, {"input_tokens": 40, "output_tokens": 8, "total_tokens": 48})
    router.fail_all(error)

    result, usable = await invoke_script(scripted_turn)
    before = result.to_mapping()
    report(router, 100, 20)

    assert usable is False
    assert result.usage.input_tokens is None
    assert result.usage.output_tokens is None
    accounting = result.usage.extensions[ACCOUNTING_KEY]
    assert accounting["observed"]["input_tokens"] == 40
    assert any(gap["reason"] == "unfinished_work" for gap in accounting["gaps"])
    assert result.to_mapping() == before


@pytest.mark.parametrize(
    ("will_retry", "info", "complete_input"),
    [
        (True, None, False),
        (False, "sandboxError", True),
        (False, {"responseStreamDisconnected": {"httpStatusCode": 503}}, False),
    ],
)
async def test_error_scope_keeps_accounted_tool_failures_distinct_from_model_gaps(
    scripted_turn, will_retry, info, complete_input
):
    router = scripted_turn[0]
    report(router, 40, 8)
    respond(router, {"input_tokens": 40, "output_tokens": 8, "total_tokens": 48})
    error = {"message": "scripted failure", "codexErrorInfo": info}
    router.route_notification(
        Notification(
            method="error",
            payload=ErrorNotification.model_validate(
                {
                    "threadId": "thread-1",
                    "turnId": "turn-1",
                    "willRetry": will_retry,
                    "error": error,
                }
            ),
        )
    )
    complete(router)

    result, _ = await invoke_script(scripted_turn)

    assert (result.usage.input_tokens == 40) is complete_input
    assert result.usage.extensions[ACCOUNTING_KEY]["observed"]["input_tokens"] == 40


@pytest.mark.parametrize("value", [None, 0, 5])
def test_original_cache_presence_is_kept_independently_from_input_and_output(value):
    raw = {"input_tokens": 40, "output_tokens": 8, "total_tokens": 48}
    if value is not None:
        raw["input_tokens_details"] = {"cached_tokens": value}
    result = source_result(raw)
    assert result.usage.input_tokens == 40
    assert result.usage.output_tokens == 8
    assert result.usage.cached_input_tokens == value


def test_literal_primary_zero_is_preserved_and_omitted_primary_counts_stay_unknown():
    result = source_result({"input_tokens": 0})
    assert result.usage.input_tokens == 0
    assert result.usage.output_tokens is None
    assert result.usage.total_tokens is None
    assert result.usage.input_tokens_include_cache is True


def test_usage_descriptor_matches_the_shared_schema_and_accepts_real_output():
    root = Path(__file__).resolve().parents[2]
    canonical = root / "adapters/python/codex/codex.fabric-adapter.json"
    packaged = (
        root / "crates/fabric-cli/assets/adapters/codex/codex.fabric-adapter.json"
    )
    assert canonical.read_bytes() == packaged.read_bytes()
    declared = json.loads(canonical.read_text())["extension_schemas"]["usage"]
    raw = json.loads(
        (root / "schemas/extensions/usage-accounting.schema.json").read_text()
    )
    definitions = raw.pop("$defs")
    raw.pop("$schema")
    assert declared["properties"][ACCOUNTING_KEY] == raw
    assert declared["$defs"] == definitions
    Draft202012Validator.check_schema(declared)
    result = adapter._agent_run_result(
        {"usage": {"inputTokens": 40, "outputTokens": 8}, "usage_complete": False}
    )
    assert ACCOUNTING_KEY in result.usage.extensions
    Draft202012Validator(declared).validate(result.usage.extensions)


@pytest.mark.parametrize(
    ("tool", "receivers", "is_dispatch"),
    [
        ("spawnAgent", [], False),
        ("spawnAgent", ["child-1"], True),
        ("wait", ["child-1"], False),
    ],
)
async def test_real_sdk_only_dispatched_child_work_creates_a_scope_gap(
    scripted_turn, tool, receivers, is_dispatch
):
    router = scripted_turn[0]
    router.route_notification(
        Notification(
            method="item/started",
            payload=ItemStartedNotification.model_validate(
                {
                    "startedAtMs": 100000,
                    "threadId": "thread-1",
                    "turnId": "turn-1",
                    "item": {
                        "type": "collabAgentToolCall",
                        "id": "child-dispatch",
                        "tool": tool,
                        "status": "inProgress",
                        "agentsStates": {},
                        "receiverThreadIds": receivers,
                        "senderThreadId": "thread-1",
                    },
                }
            ),
        )
    )
    report(router, 40, 8)
    respond(router, {"input_tokens": 40, "output_tokens": 8, "total_tokens": 48})
    complete(router)

    result, _ = await invoke_script(scripted_turn)

    assert (result.usage.input_tokens is None) is is_dispatch
    assert result.usage.extensions[ACCOUNTING_KEY]["observed"]["input_tokens"] == 40


@pytest.mark.parametrize("missing_usage", [False, True])
async def test_real_sdk_raw_completed_response_exposes_an_omitted_report(
    scripted_turn, missing_usage
):
    router = scripted_turn[0]
    respond(router, {"input_tokens": 40, "output_tokens": 8, "total_tokens": 48})
    if missing_usage:
        respond(router, None, response_id="response-2")
    report(router, 40, 8)
    complete(router)
    result, _ = await invoke_script(scripted_turn)
    assert (result.usage.input_tokens is None) is missing_usage
    assert result.usage.extensions[ACCOUNTING_KEY]["observed"]["input_tokens"] == 40


def test_original_cache_write_and_reasoning_presence_are_preserved():
    result = source_result(
        {
            "input_tokens": 40,
            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 5},
            "output_tokens": 8,
            "output_tokens_details": {"reasoning_tokens": 3},
            "total_tokens": 48,
        }
    )
    assert result.usage.input_tokens == 40
    detail = result.usage.extensions["nemo_fabric.detailed_token_usage.v1"]
    assert detail["cache_write_input_tokens"] == 5
    assert detail["reasoning_tokens"] == 3


async def test_raw_response_duplicates_are_counted_once_and_conflicts_create_a_gap(
    scripted_turn,
):
    router = scripted_turn[0]
    raw = {"input_tokens": 40, "output_tokens": 8, "total_tokens": 48}
    respond(router, raw)
    respond(router, raw)
    respond(router, {**raw, "input_tokens": 41})
    complete(router)
    result, _ = await invoke_script(scripted_turn)
    assert result.usage.input_tokens is None
    assert result.usage.extensions[ACCOUNTING_KEY]["observed"]["input_tokens"] == 40


@pytest.mark.parametrize("approval_mode", ["auto_review", "deny_all"])
@pytest.mark.parametrize(
    "sandbox_mode", ["read-only", "workspace-write", "danger-full-access"]
)
async def test_raw_event_thread_request_preserves_every_pinned_sdk_parameter(
    scripted_turn, approval_mode, sandbox_mode
):
    from types import SimpleNamespace

    _, _, _, _, context = scripted_turn
    config = AgentConfig.from_mapping(
        {
            "harness": {
                "settings": {
                    "approval_mode": approval_mode,
                    "sandbox": sandbox_mode,
                    "personality": "pragmatic",
                    "service_tier": "flex",
                    "developer_instructions": "Inspect before editing.",
                    "config_overrides": {"features.web_search": False},
                }
            },
            "instructions": {
                "system": {"content": "Review carefully.", "mode": "replace"}
            },
            "models": {"default": {"provider": "openai", "model": "gpt-5.4"}},
        }
    )
    base_dir = str(context.environment.workspace)
    codex = MagicMock(spec=AsyncCodex)
    codex._ensure_initialized = AsyncMock()
    response = SimpleNamespace(thread=SimpleNamespace(id="thread-1"))
    codex._client = SimpleNamespace(
        thread_start=AsyncMock(return_value=response),
        request=AsyncMock(return_value=response),
        next_notification=AsyncMock(),
    )

    # Let the pinned SDK itself establish the expected mapping, including
    # approval and full-access sandbox conversions.
    await AsyncCodex.thread_start(
        codex, **adapter._thread_options(config, context, base_dir, None)
    )
    expected = codex._client.thread_start.await_args.args[0].model_dump(
        mode="json", by_alias=True, exclude_none=True
    )
    thread = await adapter._open_thread(codex, config, context, base_dir, relay=None)

    assert thread.id == "thread-1"
    method, actual = codex._client.request.await_args.args
    assert method == "thread/start"
    assert actual == {**expected, "experimentalRawEvents": True}
    assert (
        codex._client.request.await_args.kwargs["response_model"]
        is adapter.ThreadStartResponse
    )


async def test_custom_provider_without_login_initializes_real_sdk_before_generic_request(
    scripted_turn, monkeypatch
):
    import os
    from types import SimpleNamespace
    from openai_codex.models import InitializeResponse

    _, _, _, _, context = scripted_turn
    os.environ["FABRIC_SOURCE_INIT_KEY"] = "local-test-only"
    config = AgentConfig.from_mapping(
        {
            "harness": {"settings": {}},
            "models": {
                "default": {
                    "provider": "local",
                    "model": "gpt-5.4",
                    "base_url": "http://127.0.0.1:1/v1",
                    "api_key_env": "FABRIC_SOURCE_INIT_KEY",
                }
            },
        }
    )
    sdk = AsyncCodex()
    protocol = sdk._client
    monkeypatch.setattr(protocol, "start", AsyncMock())
    monkeypatch.setattr(
        protocol,
        "initialize",
        AsyncMock(return_value=InitializeResponse(userAgent="codex/0.154.0")),
    )
    monkeypatch.setattr(protocol, "close", AsyncMock())

    async def request(method, params, *, response_model):
        assert sdk.metadata.serverInfo.version == "0.154.0"
        assert method == "thread/start"
        assert params["experimentalRawEvents"] is True
        return SimpleNamespace(thread=SimpleNamespace(id="thread-1"))

    monkeypatch.setattr(protocol, "request", AsyncMock(side_effect=request))
    monkeypatch.setattr(adapter, "AsyncCodex", MagicMock(return_value=sdk))
    runtime = adapter.CodexRuntime()
    await runtime.start(
        {
            "config": config,
            "runtime_context": context,
            "base_dir": str(context.environment.workspace),
        }
    )
    try:
        protocol.start.assert_awaited_once()
        protocol.initialize.assert_awaited_once()
        protocol.request.assert_awaited_once()
    finally:
        await runtime.stop()
    protocol.close.assert_awaited_once()
