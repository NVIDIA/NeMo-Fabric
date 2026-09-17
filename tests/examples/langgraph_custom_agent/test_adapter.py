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
from nemo_fabric_adapter_contract.models import AgentRunRequest
from nemo_fabric_adapter_contract.models import AgentRunStatus
from nemo_fabric_adapter_contract.models import RuntimeContext

from examples.langgraph_custom_agent.adapter.configuration import AgentDependencies
from examples.langgraph_custom_agent.adapter import runtime as runtime_module


def invocation(
    runtime_context: dict[str, object],
    input_value: object,
) -> tuple[AgentRunRequest, RuntimeContext]:
    return (
        AgentRunRequest(input=input_value),
        RuntimeContext.from_mapping(runtime_context),
    )


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
                "runtime_context": runtime_context_factory(runtime_id, "runtime-start"),
            },
        ),
        lifecycle_request_factory(
            "invoke",
            {
                "runtime_context": runtime_context_factory(runtime_id, "invocation-1"),
                "request": {
                    "input": "Urgent: verify your password at https://one.invalid."
                },
            },
        ),
        lifecycle_request_factory(
            "invoke",
            {
                "runtime_context": runtime_context_factory(runtime_id, "invocation-2"),
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
    assert responses[1]["outcome"]["output"]["output"] == {
        "response": "first explanation",
        "classification": "phishing",
        "signals": ["urgency", "credential_request", "external_link"],
    }
    assert responses[2]["outcome"]["output"]["output"] == {
        "response": "second explanation",
        "classification": "benign",
        "signals": [],
        "previous_classification": "phishing",
    }


def test_runtime_continuation_state_is_isolated_between_runtimes(
    monkeypatch,
    runtime_context_factory,
    agent_config_mapping,
):
    model = FakeListChatModel(responses=["first explanation", "isolated explanation"])
    monkeypatch.setattr(
        runtime_module,
        "resolve_agent_dependencies",
        lambda _config: AgentDependencies(model, "Explain the assessment."),
    )
    config = AgentConfig.from_mapping(agent_config_mapping)
    first_runtime = runtime_module.EmailPhishingRuntime()
    second_runtime = runtime_module.EmailPhishingRuntime()
    asyncio.run(
        first_runtime.start(
            {
                "config": config,
                "runtime_context": runtime_context_factory(
                    "runtime-1", "runtime-start-1"
                ),
            }
        )
    )
    asyncio.run(
        second_runtime.start(
            {
                "config": config,
                "runtime_context": runtime_context_factory(
                    "runtime-2", "runtime-start-2"
                ),
            }
        )
    )

    asyncio.run(
        first_runtime.invoke(
            *invocation(
                runtime_context_factory("runtime-1", "invocation-1"),
                "Urgent: verify your password immediately.",
            )
        )
    )
    isolated = asyncio.run(
        second_runtime.invoke(
            *invocation(
                runtime_context_factory("runtime-2", "invocation-2"),
                "Team lunch is at noon.",
            )
        )
    )

    assert "previous_classification" not in isolated.output
    asyncio.run(first_runtime.stop())
    asyncio.run(second_runtime.stop())


def test_stop_discards_runtime_continuation_state(
    monkeypatch,
    runtime_context_factory,
    agent_config_mapping,
):
    model = FakeListChatModel(responses=["first explanation", "fresh explanation"])
    monkeypatch.setattr(
        runtime_module,
        "resolve_agent_dependencies",
        lambda _config: AgentDependencies(model, "Explain the assessment."),
    )
    config = AgentConfig.from_mapping(agent_config_mapping)
    runtime = runtime_module.EmailPhishingRuntime()
    start_payload = {
        "config": config,
        "runtime_context": runtime_context_factory("runtime-1", "runtime-start"),
    }

    asyncio.run(runtime.start(start_payload))
    asyncio.run(
        runtime.invoke(
            *invocation(
                runtime_context_factory("runtime-1", "invocation-1"),
                "Urgent: verify your password immediately.",
            )
        )
    )
    asyncio.run(runtime.stop())

    asyncio.run(runtime.start(start_payload))
    fresh = asyncio.run(
        runtime.invoke(
            *invocation(
                runtime_context_factory("runtime-1", "invocation-2"),
                "Team lunch is at noon.",
            )
        )
    )

    assert "previous_classification" not in fresh.output
    asyncio.run(runtime.stop())


def test_runtime_bounds_continuation_history(
    monkeypatch,
    runtime_context_factory,
    agent_config_mapping,
):
    model = FakeListChatModel(
        responses=["first explanation", "second explanation", "third explanation"]
    )
    monkeypatch.setattr(
        runtime_module,
        "resolve_agent_dependencies",
        lambda _config: AgentDependencies(
            model,
            "Explain the assessment.",
            max_history_entries=2,
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

    for number, email in enumerate(
        (
            "Urgent: verify your password immediately.",
            "Team lunch is at noon.",
            "Act now and sign in to avoid suspension.",
        ),
        start=1,
    ):
        asyncio.run(
            runtime.invoke(
                *invocation(
                    runtime_context_factory(
                        "runtime-1", f"invocation-{number}"
                    ),
                    email,
                )
            )
        )

    assert [
        assessment["classification"]
        for assessment in runtime._assessment_history
    ] == ["benign", "phishing"]
    asyncio.run(runtime.stop())


def test_invocation_failure_propagates_to_lifecycle_host(
    monkeypatch,
    runtime_context_factory,
    agent_config_mapping,
):
    class FailingGraph:
        async def ainvoke(self, _input, *, config):
            raise TimeoutError("provider request timed out")

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
        lambda *_args, **_kwargs: FailingGraph(),
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

    with pytest.raises(TimeoutError, match="provider request timed out"):
        asyncio.run(
            runtime.invoke(
                *invocation(
                    runtime_context_factory("runtime-1", "invocation-1"),
                    "first",
                )
            )
        )

    asyncio.run(runtime.stop())


def test_runtime_rejects_invoke_before_start(runtime_context_factory):
    runtime = runtime_module.EmailPhishingRuntime()

    with pytest.raises(lifecycle.LifecycleError) as error:
        asyncio.run(
            runtime.invoke(
                *invocation(
                    runtime_context_factory("runtime-1", "invocation-1"),
                    "hello",
                )
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
                *invocation(
                    runtime_context_factory("runtime-2", "invocation-1"),
                    "hello",
                )
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
                *invocation(
                    runtime_context_factory("runtime-1", "invocation-after-stop"),
                    "hello",
                )
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
            *invocation(
                runtime_context_factory("runtime-1", "invocation-1"),
                "Review https://example.invalid/login.",
            )
        )
    )

    assert output.status is AgentRunStatus.SUCCEEDED
    assert output.output["signals"] == [
        "credential_request",
        "external_link",
        "suspicious_link",
    ]
    assert output.output["link_inspections"] == [
        {
            "url": "https://example.invalid/login",
            "hostname": "example.invalid",
            "indicators": ["reserved_test_domain"],
        }
    ]
    asyncio.run(runtime.stop())
