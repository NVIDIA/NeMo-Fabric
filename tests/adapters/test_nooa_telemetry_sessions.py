# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native Relay session attribution and configuration isolation for NOOA."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from nemo_fabric_adapter_contract.models import AgentConfig
from nemo_fabric_adapter_contract.models import RuntimeContext

if not ((3, 12) <= sys.version_info[:2] < (3, 14)):
    pytest.skip("NOOA supports Python 3.12 and 3.13", allow_module_level=True)

pytest.importorskip("nooa")

from nooa.runtime.event_manager import EventManager
from nooa.runtime.middleware import AgentCallContext
from nemo_fabric_adapters.nooa.telemetry import RelayTelemetry
from nemo_fabric_adapters.nooa.relay_compat import _agent_call_middleware


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
                                    "opentelemetry": {
                                        "enabled": True,
                                        "file_sinks": [
                                            {
                                                "type": "gen_ai",
                                                "format": "json_lines",
                                                "output_directory": str(
                                                    tmp_path / "traces"
                                                ),
                                                "filename": "default.jsonl",
                                                "promote_metadata_prefixes": ["label."],
                                            }
                                        ],
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
            "runtime_id": "default-runtime",
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
    async def nested(context):
        await asyncio.sleep(0)

        async def model(_request):
            return {"message": {"role": "assistant", "content": "42"}}

        async def tool(_arguments):
            return relay.ToolExecutionResult({"answer": 42})

        await relay.llm.execute(
            "default", relay.LLMRequest({}, {"messages": []}), model
        )
        await relay.tools.execute("execute_python", {}, tool)
        return context

    async def child(context):
        await _agent_call_middleware(AgentCallContext(method_name="nested"), nested)
        return context

    await asyncio.gather(
        *(
            _agent_call_middleware(AgentCallContext(method_name=name), child)
            for name in ("first", "second")
        )
    )
    return 42


def _spans(path):
    return [
        span
        for line in path.read_text(encoding="utf-8").splitlines()
        for resource in json.loads(line)["resourceSpans"]
        for scope in resource["scopeSpans"]
        for span in scope["spans"]
    ]


@pytest.mark.parametrize(
    "roots",
    [
        ("11111111-1111-4111-8111-111111111111",) * 2,
        (
            "11111111-1111-4111-8111-111111111111",
            "22222222-2222-4222-8222-222222222222",
        ),
        (None, "not-a-uuid"),
    ],
)
async def test_every_span_preserves_session_across_nested_nooa_stacks(
    relay_session, roots
):
    relay, telemetry, context, agent = relay_session
    previous = 0
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
        artifact = next(
            item for item in result.report.artifacts if item["kind"] == "otlp"
        )
        spans = _spans(Path(artifact["path"]))
        current = spans[previous:]
        previous = len(spans)
        assert len(current) == 9
        attributes = [
            {
                item["key"]: item["value"]["stringValue"]
                for item in span["attributes"]
                if item["key"] == "session_id"
            }
            for span in current
        ]
        expected = (
            root
            if root
            in (
                "11111111-1111-4111-8111-111111111111",
                "22222222-2222-4222-8222-222222222222",
            )
            else context.request_id
        )
        assert [item["session_id"] for item in attributes] == [expected] * len(current)


async def test_ambient_configuration_is_ignored_and_preserved(relay_session, tmp_path):
    relay, telemetry, context, agent = relay_session
    user_config = Path(os.environ["XDG_CONFIG_HOME"]) / "nemo-relay" / "plugins.toml"
    user_config.parent.mkdir(parents=True)
    user_config.write_text("malformed user TOML [", encoding="utf-8")
    project_config = tmp_path / ".nemo-relay" / "plugins.toml"
    project_config.parent.mkdir()
    project_config.write_text("malformed project TOML [", encoding="utf-8")
    previous = Path.cwd()
    try:
        os.chdir(tmp_path)
        result = await telemetry.invoke(
            agent=agent, runtime_context=context, call=lambda: _nested_work(relay)
        )
    finally:
        os.chdir(previous)
    assert result.result == 42
    assert result.report.error is None
    assert user_config.read_text(encoding="utf-8") == "malformed user TOML ["
    assert project_config.read_text(encoding="utf-8") == "malformed project TOML ["


@pytest.mark.parametrize("ending", ["success", "failure", "cancel"])
async def test_session_injector_is_removed_after_invocation(relay_session, ending):
    relay, telemetry, context, agent = relay_session

    async def method(context):
        if ending == "failure":
            raise RuntimeError("default failure")
        if ending == "cancel":
            raise asyncio.CancelledError()
        context.result = 42
        return context

    async def work():
        result = await _agent_call_middleware(
            AgentCallContext(method_name="cleanup"), method
        )
        return result.result

    if ending == "cancel":
        with pytest.raises(asyncio.CancelledError):
            await telemetry.invoke(agent=agent, runtime_context=context, call=work)
    else:
        result = await telemetry.invoke(agent=agent, runtime_context=context, call=work)
        assert result.called
        assert (result.report.error is None) == (ending == "success")

    received = threading.Event()
    metadata = []

    def observe(event):
        metadata.append(event.to_dict()["metadata"] or {})
        received.set()

    relay.subscribers.register("default-after-invocation", observe)
    try:
        with relay.scope.scope("outside", relay.ScopeType.Agent):
            pass
        assert await asyncio.to_thread(received.wait, 5)
        assert all("session_id" not in item for item in metadata)
    finally:
        relay.subscribers.deregister("default-after-invocation")


async def test_export_preserves_existing_metadata_prefixes(relay_session):
    relay, telemetry, context, agent = relay_session
    relay.event_metadata.register_injector(
        "default-label", 10, lambda _event: {"label.source": "default"}
    )
    try:
        result = await telemetry.invoke(
            agent=agent, runtime_context=context, call=lambda: _nested_work(relay)
        )
    finally:
        relay.event_metadata.deregister_injector("default-label")
    assert result.report.error is None
    artifact = next(item for item in result.report.artifacts if item["kind"] == "otlp")
    spans = _spans(Path(artifact["path"]))
    assert spans
    for span in spans:
        attributes = {item["key"]: item["value"] for item in span["attributes"]}
        assert attributes["label.source"] == {"stringValue": "default"}
        assert attributes["session_id"] == {"stringValue": context.request_id}


async def test_disabled_trace_export_remains_disabled(relay_session):
    relay, telemetry, context, agent = relay_session
    config_path = Path(context.telemetry.config_path)
    document = json.loads(config_path.read_text(encoding="utf-8"))
    document["relay"]["config"]["components"][0]["config"]["opentelemetry"][
        "enabled"
    ] = False
    config_path.write_text(json.dumps(document), encoding="utf-8")
    result = await telemetry.invoke(
        agent=agent, runtime_context=context, call=lambda: _nested_work(relay)
    )
    assert result.result == 42
    assert result.report.error is None
    assert result.report.artifacts == ()
