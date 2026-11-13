# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native Relay roots and per-turn ATIF session identity for NOOA."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from nemo_fabric_adapter_contract.models import AgentConfig
from nemo_fabric_adapter_contract.models import RuntimeContext

if not ((3, 12) <= sys.version_info[:2] < (3, 14)):
    pytest.skip("NOOA supports Python 3.12 and 3.13", allow_module_level=True)

pytest.importorskip("nooa")

from nooa.runtime.event_manager import EventManager
from nooa.runtime.middleware import AgentCallContext
from nemo_fabric_adapters.nooa.telemetry import RelayTelemetry
from nooa.nemo_relay_middleware import nemo_relay_agent_call_middleware


@pytest.fixture(name="relay_session")
def relay_session_fixture(tmp_path, nemo_relay):
    config_path = tmp_path / "relay.json"
    config_path.write_text(
        json.dumps(
            {
                "relay": {
                    "config": {
                        "version": 1,
                        "components": [
                            {
                                "kind": "observability",
                                "enabled": True,
                                "config": {
                                    "version": 4,
                                    "atof": {
                                        "enabled": True,
                                        "sinks": [
                                            {
                                                "type": "file",
                                                "output_directory": str(
                                                    tmp_path / "atof"
                                                ),
                                                "filename": "events.jsonl",
                                                "mode": "overwrite",
                                            }
                                        ],
                                    },
                                    "atif": {
                                        "enabled": True,
                                        "output_directory": str(tmp_path / "atif"),
                                        "filename_template": "trajectory-{session_id}.json",
                                    },
                                },
                            }
                        ],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    os.environ["FABRIC_RELAY_CONFIG_PATH"] = str(config_path)
    context = RuntimeContext.from_mapping(
        {
            "runtime_id": "33333333-3333-4333-8333-333333333333",
            "invocation_id": "default-invocation",
            "request_id": str(uuid.uuid4()),
            "environment": {
                "environment_id": "default-environment",
                "provider": "local",
                "control_location": "in_env_control",
                "ownership": "caller_owned",
            },
            "artifacts": {},
            "telemetry": {
                "relay_enabled": True,
                "config_path": str(config_path),
                "metadata": {"telemetry_providers": ["relay"]},
            },
        }
    )
    telemetry = RelayTelemetry(
        agent_name="default",
        base_dir=tmp_path,
        config=AgentConfig.from_mapping(
            {
                "workflow": {
                    "entrypoint": {
                        "kind": "interactive_agent_factory",
                        "ref": "nemo_fabric_adapters.nooa.targets.coding_agent:create_agent",
                    },
                    "settings": {},
                }
            }
        ),
    )
    return nemo_relay, telemetry, context, SimpleNamespace(event_manager=EventManager())


async def _nested_work(relay):
    first_started = asyncio.Event()
    second_started = asyncio.Event()
    first_finished = asyncio.Event()

    async def nested(context):
        async def model(_request):
            return {"message": {"role": "assistant", "content": "42"}}

        async def tool(_arguments):
            return relay.ToolExecutionResult({"answer": 42})

        await relay.llm.execute(
            "default", relay.LLMRequest({}, {"messages": []}), model
        )
        await relay.tools.execute("execute_python", {}, tool)
        return context

    async def first(context):
        first_started.set()
        await second_started.wait()
        try:
            await nemo_relay_agent_call_middleware(
                AgentCallContext(method_name="nested"), nested
            )
        finally:
            first_finished.set()
        return context

    async def second(context):
        await first_started.wait()
        second_started.set()
        await first_finished.wait()
        await nemo_relay_agent_call_middleware(
            AgentCallContext(method_name="nested"), nested
        )
        return context

    await asyncio.gather(
        nemo_relay_agent_call_middleware(AgentCallContext(method_name="first"), first),
        nemo_relay_agent_call_middleware(
            AgentCallContext(method_name="second"), second
        ),
    )
    return 42


def _documents(result):
    artifacts = {item["kind"]: Path(item["path"]) for item in result.report.artifacts}
    atif = json.loads(artifacts["atif"].read_text(encoding="utf-8"))
    events = [
        json.loads(line)
        for line in artifacts["atof"].read_text(encoding="utf-8").splitlines()
    ]
    return artifacts, atif, events


@pytest.mark.parametrize(
    "roots",
    [
        ("11111111-1111-4111-8111-111111111111",) * 2,
        (
            "11111111-1111-4111-8111-111111111111",
            "22222222-2222-4222-8222-222222222222",
        ),
        (None, None),
        ("not-a-uuid", "not-a-uuid"),
        ("00000000-0000-0000-0000-000000000000",) * 2,
        ("11111111-1111-4111-0000-000000000000",) * 2,
    ],
)
async def test_atif_session_grouping_keeps_distinct_turns(relay_session, roots):
    relay, telemetry, context, agent = relay_session
    turns = []
    for root in roots:
        context.invocation_id = str(uuid.uuid4())
        context.request_id = str(uuid.uuid4())
        result = await telemetry.invoke(
            agent=agent,
            runtime_context=context,
            relay_session_root=root,
            call=lambda: _nested_work(relay),
        )
        assert result.result == 42
        assert result.report.error is None
        artifacts, atif, events = _documents(result)
        expected_root = (
            root
            if root
            in {
                "11111111-1111-4111-8111-111111111111",
                "22222222-2222-4222-8222-222222222222",
            }
            else context.request_id
        )
        assert atif["session_id"] == expected_root
        scopes = [
            event
            for event in events
            if event["kind"] == "scope" and event["scope_category"] == "start"
        ]
        (request,) = [event for event in scopes if event["category"] == "agent"]
        assert request["parent_uuid"] == context.request_id
        assert request["metadata"]["nemo_fabric_runtime_id"] == context.runtime_id
        assert request["metadata"]["nemo_fabric_invocation_id"] == context.invocation_id
        assert {event["propagation_root_uuid"] for event in events} == {expected_root}
        assert {event["category"] for event in scopes} >= {
            "agent",
            "function",
            "llm",
            "tool",
        }
        functions = [event for event in scopes if event["category"] == "function"]
        siblings = [event for event in functions if event["name"] != "NoneType.nested"]
        assert {event["name"] for event in siblings} == {
            "NoneType.first",
            "NoneType.second",
        }
        assert {event["parent_uuid"] for event in siblings} == {request["uuid"]}
        assert {
            event["parent_uuid"]
            for event in functions
            if event["name"] == "NoneType.nested"
        } == {event["uuid"] for event in siblings}
        turns.append((atif["trajectory_id"], artifacts["atif"]))
    assert turns[0][0] != turns[1][0]
    assert turns[0][1] != turns[1][1]


async def test_non_uuid_requests_keep_native_per_request_sessions(relay_session):
    relay, telemetry, context, agent = relay_session
    sessions = []
    for request_id in ("default-first", "default-second"):
        context.request_id = request_id
        context.invocation_id = str(uuid.uuid4())
        result = await telemetry.invoke(
            agent=agent, runtime_context=context, call=lambda: _nested_work(relay)
        )
        assert result.report.error is None
        artifacts, atif, events = _documents(result)
        (request,) = [
            event
            for event in events
            if event["category"] == "agent" and event["scope_category"] == "start"
        ]
        assert request["metadata"]["nemo_fabric_request_id"] == request_id
        assert atif["session_id"] == request["uuid"]
        sessions.append((atif["session_id"], artifacts["atif"]))
    assert sessions[0][0] != sessions[1][0]
    assert sessions[0][1] != sessions[1][1]


@pytest.mark.parametrize("ending", ["success", "failure", "cancel"])
async def test_invocation_restores_outer_session_context(relay_session, ending):
    relay, telemetry, context, agent = relay_session

    async def method(context):
        if ending == "failure":
            raise RuntimeError("default failure")
        if ending == "cancel":
            raise asyncio.CancelledError()
        context.result = 42
        return context

    async def work():
        result = await nemo_relay_agent_call_middleware(
            AgentCallContext(method_name="cleanup"), method
        )
        return result.result

    with relay.scope.scope("default-outer", relay.ScopeType.Agent) as outer:
        baseline = relay.capture_propagation_context()
        if ending == "cancel":
            with pytest.raises(asyncio.CancelledError):
                await telemetry.invoke(agent=agent, runtime_context=context, call=work)
        else:
            result = await telemetry.invoke(
                agent=agent, runtime_context=context, call=work
            )
            assert result.called
            assert (result.report.error is None) == (ending == "success")
        assert relay.scope.get_handle().uuid == outer.uuid
        restored = relay.capture_propagation_context()
        assert restored.parent_uuid == baseline.parent_uuid
        assert restored.root_uuid == baseline.root_uuid
        result = await telemetry.invoke(
            agent=agent, runtime_context=context, call=lambda: _nested_work(relay)
        )
        assert result.result == 42
        assert result.report.error is None
        assert relay.scope.get_handle().uuid == outer.uuid


async def test_leaked_request_scope_is_quarantined_before_context_restore(
    relay_session,
):
    relay, telemetry, context, agent = relay_session

    async def leak():
        relay.scope.push("default-leaked", relay.ScopeType.Function)
        return 42

    async def later():
        return 43

    with relay.scope.scope("default-outer", relay.ScopeType.Agent) as outer:
        first = await telemetry.invoke(agent=agent, runtime_context=context, call=leak)
        assert first.result == 42
        assert first.report.error is not None
        assert relay.scope.get_handle().uuid == outer.uuid
        second = await telemetry.invoke(
            agent=agent, runtime_context=context, call=later
        )
        assert second.result == 43
        assert "telemetry is disabled for later turns" in second.report.error
        assert second.report.quarantine_cause is not None
        assert relay.scope.get_handle().uuid == outer.uuid


async def test_genai_export_preserves_native_and_mapped_session_identity(relay_session):
    relay, telemetry, context, agent = relay_session
    root = "11111111-1111-4111-8111-111111111111"
    config_path = Path(context.telemetry.config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["relay"]["config"]["components"][0]["config"]["opentelemetry"] = {
        "enabled": True,
        "file_sinks": [
            {
                "type": "gen_ai",
                "format": "json_lines",
                "output_directory": str(config_path.parent / "traces"),
                "filename": "default.jsonl",
                "attribute_mappings": [
                    {"key": "gen_ai.conversation.id", "alias": "session_id"}
                ],
            }
        ],
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")

    result = await telemetry.invoke(
        agent=agent,
        runtime_context=context,
        relay_session_root=root,
        call=lambda: _nested_work(relay),
    )
    assert result.result == 42
    assert result.report.error is None
    artifacts, _atif, events = _documents(result)
    spans = [
        span
        for line in artifacts["otlp"].read_text(encoding="utf-8").splitlines()
        for resource in json.loads(line)["resourceSpans"]
        for scope in resource["scopeSpans"]
        for span in scope["spans"]
    ]
    assert spans
    scope_categories = {
        uuid.UUID(event["uuid"]).hex[-16:]: event["category"]
        for event in events
        if event["kind"] == "scope" and event["scope_category"] == "start"
    }
    assert set(scope_categories.values()) >= {"agent", "function", "llm", "tool"}
    assert set(scope_categories) <= {span["spanId"] for span in spans}
    for span in spans:
        for key in ("gen_ai.conversation.id", "session_id"):
            assert [
                attribute["value"]
                for attribute in span["attributes"]
                if attribute["key"] == key
            ] == [{"stringValue": root}]
