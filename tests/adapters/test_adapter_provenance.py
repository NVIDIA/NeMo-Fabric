# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Returned execution/accounting survives independent provenance collection."""

from __future__ import annotations

import asyncio
import importlib
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from claude_agent_sdk import AssistantMessage, ResultMessage, SystemMessage, TextBlock
from nemo_fabric_adapter_contract.models import (
    AgentConfig,
    AgentRunRequest,
    AgentRunResult,
    AgentRunStatus,
    AgentUsage,
)
from nemo_fabric_adapters.common import provenance


_NOOA_PYTHON = pytest.mark.skipif(
    not ((3, 12) <= sys.version_info[:2] < (3, 14)),
    reason="NOOA requires Python 3.12 or 3.13",
)


@pytest.mark.parametrize(
    ("module", "runtime_type"),
    [
        ("claude", "ClaudeRuntime"),
        ("codex", "CodexRuntime"),
        ("deepagents", "DeepAgentsRuntime"),
        ("hermes", "HermesRuntime"),
        ("mini_swe_agent", "MiniSweAgentRuntime"),
        pytest.param("nooa", "NooaRuntime", marks=_NOOA_PYTHON),
        pytest.param("nooa.bench", "BenchRuntime", marks=_NOOA_PYTHON),
        ("openhands", "OpenHandsRuntime"),
        ("remote_agent", "RemoteAgentRuntime"),
    ],
)
async def test_returned_failure_and_usage_survive_version_reporting(
    module, runtime_type, monkeypatch
):
    module_name = (
        "nemo_fabric_adapters.nooa.bench_adapter"
        if module == "nooa.bench"
        else f"nemo_fabric_adapters.{module}.adapter"
    )
    adapter = importlib.import_module(module_name)
    runtime = getattr(adapter, runtime_type)()
    config = AgentConfig.from_mapping(
        {"models": {"default": {"provider": "openai", "model": "configured-model"}}}
    )
    for attribute in ("_agent_config", "_config"):
        if hasattr(runtime, attribute):
            setattr(runtime, attribute, config)
    if hasattr(runtime, "_start_payload"):
        runtime._start_payload = {"config": config}
    if hasattr(runtime, "_model_roles"):
        runtime._model_roles = ["default"]
    usage = AgentUsage(input_tokens=0, output_tokens=7)
    returned = AgentRunResult(
        status=AgentRunStatus.FAILED,
        output={"response": "kept"},
        error={"code": "target_failed", "message": "kept"},
        usage=usage,
        extensions={"other": {"value": "kept"}},
    )
    monkeypatch.setattr(runtime, "_invoke", AsyncMock(return_value=returned))
    result = await runtime.invoke(AgentRunRequest(input="hello"), SimpleNamespace())
    assert result is returned
    assert result.usage is usage
    assert result.output == {"response": "kept"}
    assert result.status is AgentRunStatus.FAILED
    assert result.error.code == "target_failed"
    assert result.extensions["other"] == {"value": "kept"}
    report = result.extensions[provenance.VERSIONS_REPORT_KEY]
    assert report["adapter"]["package_version"]["source"] == "reported"
    assert report["adapter"]["package_version"]["collected_by"] == "adapter"
    assert report["models"]["default"]["served"]["reason"] in {
        "no_source",
        "not_collected",
    }


def assistant(model, parent=None):
    return AssistantMessage(
        content=[TextBlock(text="done")], model=model, parent_tool_use_id=parent
    )


def test_claude_served_model_excludes_children_and_synthetic_messages():
    from nemo_fabric_adapters.claude import adapter

    messages = adapter.InvocationMessages(
        items=[
            assistant("child", "tool-1"),
            assistant("<synthetic>"),
            assistant("actual"),
        ]
    )
    assert adapter.served_model_fact(messages)["value"] == "actual"
    messages.items.append(assistant("other"))
    assert adapter.served_model_fact(messages)["reason"] == "not_attributed"
    messages.failure = "timeout"
    assert adapter.served_model_fact(messages)["state"] == "collection_failed"


@pytest.mark.parametrize("version", ["1.2.3", None, ""])
async def test_codex_version_uses_thread_read_response(version):
    from nemo_fabric_adapters.codex import adapter

    thread = SimpleNamespace(
        read=AsyncMock(
            return_value=SimpleNamespace(thread=SimpleNamespace(cli_version=version))
        )
    )
    fact = await adapter.read_cli_version(thread)
    thread.read.assert_awaited_once_with()
    assert (
        fact.get("value") == version if version else fact["reason"] == "empty_response"
    )


async def test_codex_version_read_failure_does_not_fail_execution():
    from nemo_fabric_adapters.codex import adapter

    fact = await adapter.read_cli_version(
        SimpleNamespace(read=AsyncMock(side_effect=TypeError("SDK changed")))
    )
    assert fact["state"] == "collection_failed"
    assert fact["reason"] == "no_result"
    with pytest.raises(asyncio.CancelledError):
        await adapter.read_cli_version(
            SimpleNamespace(read=AsyncMock(side_effect=asyncio.CancelledError()))
        )


async def test_hermes_version_uses_derived_version_and_degrades_probe_errors(
    monkeypatch,
):
    from nemo_fabric_adapters.hermes import adapter
    import sys

    fake = SimpleNamespace(
        get_version_info=lambda: SimpleNamespace(derived_version="0.12.0+abc.dirty")
    )
    monkeypatch.setitem(sys.modules, "hermes_cli.version_info", fake)
    assert adapter.hermes_version() == "0.12.0+abc.dirty"
    fact = await adapter.probe_hermes_version(
        provenance.BlockingProbe(), adapter.hermes_version
    )
    assert fact["value"] == "0.12.0+abc.dirty"

    def fail():
        raise OSError("no metadata")

    fact = await adapter.probe_hermes_version(provenance.BlockingProbe(), fail)
    assert fact["state"] == "collection_failed"


async def test_openclaw_http_failure_keeps_startup_version(monkeypatch):
    import httpx
    from nemo_fabric_adapters.openclaw import adapter

    runtime = adapter.OpenClawRuntime()
    runtime._config = AgentConfig.from_mapping(
        {"models": {"default": {"provider": "openai", "model": "configured"}}}
    )
    runtime._context = SimpleNamespace(runtime_id="runtime")
    runtime._client = object()
    runtime._gateway_url = "http://gateway.example"
    runtime._harness_version = provenance.known_fact("2026.4.1", "gateway_api")
    response = httpx.Response(
        503, request=httpx.Request("POST", "http://gateway.example")
    )
    monkeypatch.setattr(
        runtime,
        "_invoke_monitored_gateway",
        AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "failed", request=response.request, response=response
            )
        ),
    )
    result = await runtime.invoke(AgentRunRequest(input="hello"), runtime._context)
    assert result.status is AgentRunStatus.FAILED
    assert result.error.code == "openclaw_http_error"
    assert result.usage is None
    assert (
        result.extensions[provenance.VERSIONS_REPORT_KEY]["harness"]["version"]["value"]
        == "2026.4.1"
    )
