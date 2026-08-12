# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the minimum custom-agent lifecycle."""

from __future__ import annotations

import asyncio
import io
import json

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.tools import tool
from nemo_fabric_adapters.common import lifecycle
from nemo_fabric_adapter_contract.models import AgentConfig

from examples.langgraph_custom_agent.adapter.configuration import AgentDependencies
from examples.langgraph_custom_agent.adapter import runtime as runtime_module


def test_lifecycle_host_starts_once_invokes_repeatedly_and_stops(
    monkeypatch,
    runtime_context_factory,
    agent_config_mapping,
    lifecycle_request_factory,
):
    model = FakeListChatModel(responses=["first explanation", "second explanation"])
    monkeypatch.setattr(
        runtime_module,
        "resolve_agent_dependencies",
        lambda _config: AgentDependencies(model, "Explain the assessment."),
    )
    runtime_id = "runtime-1"
    requests = [
        lifecycle_request_factory(
            "start",
            {
                "config": agent_config_mapping,
                "runtime_context": runtime_context_factory(
                    runtime_id, "runtime-start"
                ),
            },
        ),
        lifecycle_request_factory(
            "invoke",
            {
                "runtime_context": runtime_context_factory(
                    runtime_id, "invocation-1"
                ),
                "request": {
                    "input": "Urgent: verify your password at https://one.invalid."
                },
            },
        ),
        lifecycle_request_factory(
            "invoke",
            {
                "runtime_context": runtime_context_factory(
                    runtime_id, "invocation-2"
                ),
                "request": {"input": "Team lunch is at noon."},
            },
        ),
        lifecycle_request_factory("stop", {"runtime_id": runtime_id}),
    ]
    input_stream = io.StringIO(
        "".join(f"{json.dumps(request)}\n" for request in requests)
    )
    output_stream = io.StringIO()

    lifecycle.serve(
        runtime_module.EmailPhishingRuntime,
        config_loader=AgentConfig.from_mapping,
        input_stream=input_stream,
        output_stream=output_stream,
    )

    responses = [json.loads(line) for line in output_stream.getvalue().splitlines()]
    assert [response["outcome"]["status"] for response in responses] == [
        "succeeded",
        "succeeded",
        "succeeded",
        "succeeded",
    ]
    assert responses[1]["outcome"]["output"] == {
        "response": "first explanation",
        "classification": "phishing",
        "signals": ["urgency", "credential_request", "external_link"],
    }
    assert responses[2]["outcome"]["output"] == {
        "response": "second explanation",
        "classification": "benign",
        "signals": [],
    }


def test_invocation_failure_does_not_invalidate_runtime(
    monkeypatch,
    runtime_context_factory,
    agent_config_mapping,
    lifecycle_request_factory,
):
    class FailThenSucceedGraph:
        def __init__(self):
            self.calls = 0

        async def ainvoke(self, _input, *, config):
            self.calls += 1
            if self.calls == 1:
                raise TimeoutError("provider details must not escape")
            return {
                "explanation": "second invocation succeeded",
                "classification": "benign",
                "signals": [],
            }

    graph = FailThenSucceedGraph()
    monkeypatch.setattr(
        runtime_module,
        "resolve_agent_dependencies",
        lambda _config: AgentDependencies(
            FakeListChatModel(responses=["unused"]),
            "Explain the assessment.",
        ),
    )
    monkeypatch.setattr(
        runtime_module,
        "build_email_phishing_graph",
        lambda *_args: graph,
    )
    runtime_id = "runtime-1"
    requests = [
        lifecycle_request_factory(
            "start",
            {
                "config": agent_config_mapping,
                "runtime_context": runtime_context_factory(
                    runtime_id, "runtime-start"
                ),
            },
        ),
        lifecycle_request_factory(
            "invoke",
            {
                "runtime_context": runtime_context_factory(
                    runtime_id, "invocation-1"
                ),
                "request": {"input": "first"},
            },
        ),
        lifecycle_request_factory(
            "invoke",
            {
                "runtime_context": runtime_context_factory(
                    runtime_id, "invocation-2"
                ),
                "request": {"input": "second"},
            },
        ),
        lifecycle_request_factory("stop", {"runtime_id": runtime_id}),
    ]
    input_stream = io.StringIO(
        "".join(f"{json.dumps(request)}\n" for request in requests)
    )
    output_stream = io.StringIO()

    lifecycle.serve(
        runtime_module.EmailPhishingRuntime,
        config_loader=AgentConfig.from_mapping,
        input_stream=input_stream,
        output_stream=output_stream,
    )

    responses = [json.loads(line) for line in output_stream.getvalue().splitlines()]
    assert responses[1]["outcome"] == {
        "status": "succeeded",
        "output": {
            "response": None,
            "completed": False,
            "failed": True,
            "error": {
                "code": "email_phishing_invoke_failed",
                "message": "The email-phishing agent invocation failed",
                "retryable": False,
            },
        },
    }
    assert responses[2]["outcome"]["output"] == {
        "response": "second invocation succeeded",
        "classification": "benign",
        "signals": [],
    }
    assert graph.calls == 2


def test_runtime_rejects_invoke_before_start(runtime_context_factory):
    runtime = runtime_module.EmailPhishingRuntime()

    with pytest.raises(lifecycle.LifecycleError) as error:
        asyncio.run(
            runtime.invoke(
                {
                    "runtime_context": runtime_context_factory(
                        "runtime-1", "invocation-1"
                    ),
                    "request": {"input": "hello"},
                }
            )
        )

    assert error.value.code == "email_phishing_runtime_not_started"


def test_runtime_rejects_runtime_mismatch(
    monkeypatch,
    runtime_context_factory,
    agent_config_mapping,
):
    monkeypatch.setattr(
        runtime_module,
        "resolve_agent_dependencies",
        lambda _config: AgentDependencies(
            FakeListChatModel(responses=["unused"]),
            "Explain the assessment.",
        ),
    )
    runtime = runtime_module.EmailPhishingRuntime()
    asyncio.run(
        runtime.start(
            {
                "config": AgentConfig.from_mapping(agent_config_mapping),
                "runtime_context": runtime_context_factory(
                    "runtime-1", "runtime-start"
                ),
            }
        )
    )

    with pytest.raises(lifecycle.LifecycleError) as error:
        asyncio.run(
            runtime.invoke(
                {
                    "runtime_context": runtime_context_factory(
                        "runtime-2", "invocation-1"
                    ),
                    "request": {"input": "hello"},
                }
            )
        )

    assert error.value.code == "email_phishing_runtime_mismatch"
    asyncio.run(runtime.stop())


def test_stop_is_safe_after_partial_start(
    monkeypatch,
    runtime_context_factory,
    agent_config_mapping,
):
    def fail_resolution(_config):
        raise lifecycle.LifecycleError("test_start_failure", "start failed")

    monkeypatch.setattr(
        runtime_module,
        "resolve_agent_dependencies",
        fail_resolution,
    )
    runtime = runtime_module.EmailPhishingRuntime()
    start_payload = {
        "config": AgentConfig.from_mapping(agent_config_mapping),
        "runtime_context": runtime_context_factory("runtime-1", "runtime-start"),
    }

    with pytest.raises(lifecycle.LifecycleError, match="start failed"):
        asyncio.run(runtime.start(start_payload))

    asyncio.run(runtime.stop())
    with pytest.raises(lifecycle.LifecycleError) as error:
        asyncio.run(
            runtime.invoke(
                {
                    "runtime_context": runtime_context_factory(
                        "runtime-1", "invocation-after-stop"
                    ),
                    "request": {"input": "hello"},
                }
            )
        )

    assert error.value.code == "email_phishing_runtime_not_started"


def test_runtime_returns_optional_mcp_link_inspections(
    monkeypatch,
    runtime_context_factory,
    agent_config_mapping,
):
    @tool
    async def inspect_url(url: str) -> str:
        """Inspect one URL."""

        return json.dumps(
            {
                "hostname": "example.invalid",
                "indicators": ["reserved_test_domain"],
            }
        )

    async def resolve(_config):
        return inspect_url

    monkeypatch.setattr(
        runtime_module,
        "resolve_agent_dependencies",
        lambda _config: AgentDependencies(
            FakeListChatModel(responses=["The link is suspicious."]),
            "Explain the assessment.",
        ),
    )
    monkeypatch.setattr(runtime_module, "resolve_url_inspector", resolve)
    runtime = runtime_module.EmailPhishingRuntime()
    asyncio.run(
        runtime.start(
            {
                "config": AgentConfig.from_mapping(agent_config_mapping),
                "runtime_context": runtime_context_factory(
                    "runtime-1", "runtime-start"
                ),
            }
        )
    )

    output = asyncio.run(
        runtime.invoke(
            {
                "runtime_context": runtime_context_factory(
                    "runtime-1", "invocation-1"
                ),
                "request": {"input": "Review https://example.invalid/login."},
            }
        )
    )

    assert output["signals"] == [
        "credential_request",
        "external_link",
        "suspicious_link",
    ]
    assert output["link_inspections"] == [
        {
            "url": "https://example.invalid/login",
            "hostname": "example.invalid",
            "indicators": ["reserved_test_domain"],
        }
    ]
    asyncio.run(runtime.stop())
