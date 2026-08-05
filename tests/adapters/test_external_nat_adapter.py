# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for the source-only NeMo Agent Toolkit reference adapter."""

from __future__ import annotations

import json
import os
import sys
import types
from copy import deepcopy
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import call

import pytest

ROOT = Path(__file__).parents[2]
NAT_ADAPTER_SOURCE = ROOT / "external" / "nat" / "src"
sys.path.insert(0, str(NAT_ADAPTER_SOURCE))

from nemo_fabric_adapters.nat import adapter  # noqa: E402


@pytest.fixture(name="make_payload")
def make_payload_fixture(tmp_path: Path):
    """Return a factory for canonical Fabric lifecycle payloads."""

    def make(
        *,
        workflow: dict[str, Any] | None = None,
        functions: dict[str, Any] | None = None,
        function_groups: dict[str, Any] | None = None,
        models: dict[str, Any] | None = None,
        instruction: str | None = None,
        tools: dict[str, Any] | None = None,
        mcp_servers: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        config: dict[str, Any] = {
            "harness": {
                "settings": {
                    "workflow": deepcopy(
                        workflow
                        or {
                            "_type": "react_agent",
                            "llm_name": "default",
                            "tool_names": [],
                        }
                    ),
                    "functions": deepcopy(functions or {}),
                    "function_groups": deepcopy(function_groups or {}),
                }
            },
            "models": deepcopy(models or {}),
        }
        if instruction is not None:
            config["instructions"] = {
                "system": {"content": instruction, "mode": "replace"}
            }
        if tools is not None:
            config["tools"] = deepcopy(tools)

        return {
            "base_dir": str(tmp_path),
            "config": config,
            "runtime_context": {
                "runtime_id": "runtime-1",
                "environment": {"workspace": str(tmp_path)},
            },
            "capability_plan": {
                "native": {"mcp_servers": deepcopy(mcp_servers or {})}
            },
        }

    return make


@pytest.fixture(name="mock_nat")
def mock_nat_fixture(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Install mocked NAT modules and return lifecycle call recorders."""

    mock_typed_config = MagicMock(name="typed_nat_config")
    mock_config_type = MagicMock(name="Config")
    mock_config_type.model_validate = MagicMock(return_value=mock_typed_config)
    mock_discover = MagicMock(name="discover_and_register_plugins")
    mock_plugin_types = MagicMock(name="PluginTypes")
    mock_plugin_types.CONFIG_OBJECT = "config-object"

    mock_builder = MagicMock(name="builder")
    mock_builder_context = MagicMock(name="builder_context")
    mock_builder_context.__aenter__ = AsyncMock(return_value=mock_builder)
    mock_builder_context.__aexit__ = AsyncMock(return_value=False)
    mock_workflow_builder = MagicMock(name="WorkflowBuilder")
    mock_workflow_builder.from_config = MagicMock(return_value=mock_builder_context)

    mock_runner = MagicMock(name="runner")
    mock_runner.result = AsyncMock(side_effect=[{"answer": 42}, {"answer": 84}])
    mock_run_context = MagicMock(name="run_context")
    mock_run_context.__aenter__ = AsyncMock(return_value=mock_runner)
    mock_run_context.__aexit__ = AsyncMock(return_value=False)
    mock_session = MagicMock(name="session")
    mock_session.run = MagicMock(return_value=mock_run_context)
    mock_session_context = MagicMock(name="session_context")
    mock_session_context.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session_context.__aexit__ = AsyncMock(return_value=False)
    mock_sessions = MagicMock(name="sessions")
    mock_sessions.session = MagicMock(return_value=mock_session_context)
    mock_sessions.shutdown = AsyncMock()
    mock_session_manager = MagicMock(name="SessionManager")
    mock_session_manager.create = AsyncMock(return_value=mock_sessions)

    mock_runtime_type = MagicMock(name="RuntimeTypeEnum")
    mock_runtime_type.RUN_OR_SERVE = "run-or-serve"
    mock_to_jsonable = MagicMock(
        name="to_jsonable_python",
        side_effect=lambda value, **_kwargs: value,
    )

    modules: dict[str, types.ModuleType] = {}
    for name in (
        "nat",
        "nat.builder",
        "nat.builder.workflow_builder",
        "nat.runtime",
        "nat.runtime.loader",
        "nat.runtime.session",
        "nat.data_models",
        "nat.data_models.config",
        "nat.data_models.runtime_enum",
        "pydantic_core",
    ):
        module = types.ModuleType(name)
        if name in {"nat", "nat.builder", "nat.runtime", "nat.data_models"}:
            module.__path__ = []  # type: ignore[attr-defined]
        modules[name] = module
        monkeypatch.setitem(sys.modules, name, module)

    modules["nat.runtime.loader"].PluginTypes = mock_plugin_types
    modules["nat.runtime.loader"].discover_and_register_plugins = mock_discover
    modules["nat.data_models.config"].Config = mock_config_type
    modules["nat.builder.workflow_builder"].WorkflowBuilder = mock_workflow_builder
    modules["nat.runtime.session"].SessionManager = mock_session_manager
    modules["nat.data_models.runtime_enum"].RuntimeTypeEnum = mock_runtime_type
    modules["pydantic_core"].to_jsonable_python = mock_to_jsonable

    return {
        "typed_config": mock_typed_config,
        "config_type": mock_config_type,
        "discover": mock_discover,
        "plugin_types": mock_plugin_types,
        "workflow_builder": mock_workflow_builder,
        "builder_context": mock_builder_context,
        "builder": mock_builder,
        "session_manager": mock_session_manager,
        "sessions": mock_sessions,
        "session": mock_session,
        "runner": mock_runner,
        "run_context": mock_run_context,
        "session_context": mock_session_context,
        "runtime_type": mock_runtime_type,
        "to_jsonable": mock_to_jsonable,
    }


def invocation_payload(
    *,
    input_value: Any = "hello",
    request_id: str = "request-1",
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    request: dict[str, Any] = {"input": input_value, "request_id": request_id}
    if context is not None:
        request["context"] = context
    return {
        "runtime_context": {"runtime_id": "runtime-1"},
        "request": request,
    }


def test_descriptor_declares_exact_source_reference_contract():
    descriptor = json.loads(
        (ROOT / "external" / "nat" / "fabric-adapter.json").read_text(
            encoding="utf-8"
        )
    )

    assert descriptor["adapter_id"] == "nvidia.fabric.nat"
    assert descriptor["harness"] == "nat"
    assert descriptor["adapter_kind"] == "python"
    assert descriptor["runner"] == {
        "module": "nemo_fabric_adapters.nat.adapter"
    }
    assert descriptor["requirements"] == {}
    assert descriptor["config"]["accepts"] == [
        "models",
        "models.base_url",
        "models.temperature",
        "instructions.system",
        "tools.enabled",
        "tools.blocked",
        "mcp",
        "mcp.tool_filters",
    ]
    assert descriptor["settings_schema"]["required"] == ["workflow"]
    assert descriptor["settings_schema"]["additionalProperties"] is False
    assert descriptor["capabilities"] == {
        "cancellation": False,
        "service": False,
        "streaming": False,
        "updates": False,
    }


def test_build_mapping_translates_components_models_and_instruction(
    make_payload,
):
    os.environ["NVIDIA_API_KEY"] = "test-key"
    workflow = {
        "_type": "react_agent",
        "llm_name": "default",
        "tool_names": ["clock", "calculator"],
    }
    functions = {"clock": {"_type": "current_datetime"}}
    function_groups = {
        "calculator": {"_type": "calculator", "include": ["add", "subtract"]}
    }
    payload = make_payload(
        workflow=workflow,
        functions=functions,
        function_groups=function_groups,
        models={
            "default": {
                "provider": "nvidia",
                "model": "nvidia/test-model",
                "api_key_env": "NVIDIA_API_KEY",
                "base_url": "https://integrate.api.nvidia.com/v1",
                "temperature": 0.2,
                "settings": {"max_tokens": 512},
            },
            "reviewer": {
                "provider": "openai",
                "model": "gpt-test",
            },
        },
        instruction="Use portable instructions.",
    )

    result = adapter.build_nat_config_mapping(payload)

    assert result == {
        "workflow": {
            "_type": "react_agent",
            "llm_name": "default",
            "tool_names": ["clock", "calculator"],
            "additional_instructions": "Use portable instructions.",
        },
        "functions": functions,
        "function_groups": function_groups,
        "llms": {
            "default": {
                "_type": "nim",
                "model_name": "nvidia/test-model",
                "api_key": "test-key",
                "base_url": "https://integrate.api.nvidia.com/v1",
                "temperature": 0.2,
                "max_tokens": 512,
            },
            "reviewer": {
                "_type": "openai",
                "model_name": "gpt-test",
            },
        },
    }
    assert payload["config"]["harness"]["settings"]["workflow"] == workflow


def test_system_instruction_rejects_duplicate_nat_instruction_source(make_payload):
    payload = make_payload(
        workflow={
            "_type": "react_agent",
            "llm_name": "default",
            "additional_instructions": "adapter-local value",
        },
        instruction="Portable instruction",
    )

    with pytest.raises(adapter.lifecycle.LifecycleError) as error:
        adapter.build_nat_config_mapping(payload)

    assert error.value.code == "nat_system_instruction_conflict"


def test_react_agent_without_tool_names_defaults_to_empty_list(make_payload):
    payload = make_payload(
        workflow={"_type": "react_agent", "llm_name": "default"}
    )

    result = adapter.build_nat_config_mapping(payload)

    assert result["workflow"]["tool_names"] == []


def test_build_typed_config_discovers_components_before_validation(
    make_payload,
    mock_nat,
):
    events: list[str] = []
    mock_nat["discover"].side_effect = lambda _plugin_type: events.append("discover")
    mock_nat["config_type"].model_validate.side_effect = (
        lambda _mapping: events.append("validate") or mock_nat["typed_config"]
    )

    result = adapter.build_nat_config(make_payload())

    assert result is mock_nat["typed_config"]
    assert events == ["discover", "validate"]
    mock_nat["discover"].assert_called_once_with(
        mock_nat["plugin_types"].CONFIG_OBJECT
    )


@pytest.mark.parametrize(
    ("server_policy", "expected_group"),
    [
        (
            {},
            {
                "_type": "mcp_client",
                "server": {"transport": "sse", "url": "https://mcp.test/sse"},
            },
        ),
        (
            {"blocked_tools": ["delete"]},
            {
                "_type": "mcp_client",
                "server": {"transport": "sse", "url": "https://mcp.test/sse"},
                "exclude": ["delete"],
            },
        ),
        (
            {"allowed_tools": ["read", "list"], "blocked_tools": ["delete"]},
            {
                "_type": "mcp_client",
                "server": {"transport": "sse", "url": "https://mcp.test/sse"},
                "include": ["read", "list"],
            },
        ),
    ],
)
def test_mcp_filter_states_generate_one_nat_group_policy(
    make_payload,
    server_policy: dict[str, Any],
    expected_group: dict[str, Any],
):
    server = {
        "transport": "sse",
        "url": "https://mcp.test/sse",
        **server_policy,
    }
    payload = make_payload(mcp_servers={"docs": server})

    result = adapter.build_nat_config_mapping(payload)

    assert result["function_groups"]["docs"] == expected_group
    assert result["workflow"]["tool_names"] == ["docs"]


def test_mcp_policy_deduplicates_valid_fabric_names(make_payload):
    payload = make_payload(
        mcp_servers={
            "docs": {
                "transport": "sse",
                "url": "https://mcp.test/sse",
                "allowed_tools": ["read", "read"],
                "blocked_tools": ["delete", "delete"],
            }
        },
    )

    result = adapter.build_nat_config_mapping(payload)

    assert result["function_groups"] == {
        "docs": {
            "_type": "mcp_client",
            "server": {"transport": "sse", "url": "https://mcp.test/sse"},
            "include": ["read"],
        },
    }
    assert result["workflow"]["tool_names"] == ["docs"]


def test_root_tool_policy_deduplicates_valid_fabric_names(make_payload):
    payload = make_payload(
        function_groups={"calculator": {"_type": "calculator"}},
        tools={"enabled": ["calculator", "calculator"]},
    )

    result = adapter.build_nat_config_mapping(payload)

    assert result["function_groups"] == {
        "calculator": {"_type": "calculator"},
    }
    assert result["workflow"]["tool_names"] == ["calculator"]


def test_empty_mcp_allowlist_suppresses_server_and_existing_workflow_ref(
    make_payload,
):
    payload = make_payload(
        workflow={
            "_type": "react_agent",
            "llm_name": "default",
            "tool_names": ["clock", "docs"],
        },
        functions={"clock": {"_type": "current_datetime"}},
        mcp_servers={
            "docs": {
                "transport": "streamable-http",
                "url": "https://mcp.test/mcp",
                "allowed_tools": [],
            }
        },
    )

    result = adapter.build_nat_config_mapping(payload)

    assert result["workflow"]["tool_names"] == ["clock"]
    assert "docs" not in result["function_groups"]


@pytest.mark.parametrize("component_field", ["functions", "function_groups"])
def test_empty_mcp_allowlist_rejects_same_name_nat_component(
    make_payload,
    component_field: str,
):
    components = {"docs": {"_type": "native_docs"}}
    payload = make_payload(
        **{component_field: components},
        mcp_servers={
            "docs": {
                "transport": "sse",
                "url": "https://mcp.test/sse",
                "allowed_tools": [],
            }
        },
    )

    with pytest.raises(adapter.lifecycle.LifecycleError) as error:
        adapter.build_nat_config_mapping(payload)

    assert error.value.code == "nat_mcp_name_conflict"


def test_all_suppressed_mcp_does_not_require_custom_workflow_tool_names(
    make_payload,
):
    payload = make_payload(
        workflow={"_type": "custom_workflow"},
        mcp_servers={
            "docs": {
                "transport": "sse",
                "url": "https://mcp.test/sse",
                "allowed_tools": [],
            }
        },
    )

    result = adapter.build_nat_config_mapping(payload)

    assert result["workflow"] == {"_type": "custom_workflow"}
    assert result["function_groups"] == {}


def test_root_tool_policy_selects_and_blocks_exact_group_members(make_payload):
    payload = make_payload(
        workflow={
            "_type": "react_agent",
            "llm_name": "default",
            "tool_names": ["clock", "unused", "calculator", "search"],
        },
        functions={
            "clock": {"_type": "current_datetime"},
            "unused": {"_type": "unused_function"},
        },
        function_groups={
            "calculator": {
                "_type": "calculator",
                "include": ["add", "subtract", "multiply"],
            },
            "search": {"_type": "search", "exclude": ["delete"]},
        },
        tools={
            "enabled": [
                "clock",
                "calculator__add",
                "calculator__subtract",
                "search__find",
            ],
            "blocked": ["calculator__subtract", "search__secret"],
        },
    )

    result = adapter.build_nat_config_mapping(payload)

    assert result["functions"] == {"clock": {"_type": "current_datetime"}}
    assert result["function_groups"] == {
        "calculator": {"_type": "calculator", "include": ["add"]},
        "search": {"_type": "search", "include": ["find"]},
    }
    assert result["workflow"]["tool_names"] == ["clock", "calculator", "search"]


@pytest.mark.parametrize(
    "tools",
    [
        {"enabled": ["missing"]},
        {"blocked": ["missing"]},
        {"enabled": ["calculator__missing"]},
    ],
)
def test_root_tool_policy_rejects_unknown_exact_selectors(make_payload, tools):
    payload = make_payload(
        workflow={
            "_type": "react_agent",
            "llm_name": "default",
            "tool_names": ["calculator"],
        },
        function_groups={
            "calculator": {"_type": "calculator", "include": ["add"]}
        },
        tools=tools,
    )

    with pytest.raises(adapter.lifecycle.LifecycleError) as error:
        adapter.build_nat_config_mapping(payload)

    assert error.value.code == "nat_unknown_tool_selector"


async def test_runtime_reuses_one_builder_across_invocations_and_cleans_up(
    make_payload,
    mock_nat,
):
    runtime = adapter.NatRuntime()
    await runtime.start(make_payload())

    first = await runtime.invoke(
        invocation_payload(
            request_id="message-1",
            context={"user_id": "user-1", "conversation_id": "conversation-1"},
        )
    )
    second = await runtime.invoke(
        invocation_payload(
            input_value="again",
            request_id="message-2",
            context={"user_id": "user-1", "conversation_id": "conversation-1"},
        )
    )
    await runtime.stop()
    await runtime.stop()

    assert first == {
        "harness": "nat",
        "adapter": "python",
        "mode": "nat_workflow",
        "response": {"answer": 42},
        "completed": True,
        "failed": False,
        "error": None,
    }
    assert second["response"] == {"answer": 84}
    mock_nat["workflow_builder"].from_config.assert_called_once_with(
        config=mock_nat["typed_config"]
    )
    mock_nat["session_manager"].create.assert_awaited_once_with(
        config=mock_nat["typed_config"],
        shared_builder=mock_nat["builder"],
    )
    assert mock_nat["sessions"].session.call_args_list == [
        call(
            user_id="user-1",
            conversation_id="conversation-1",
            user_message_id="message-1",
        ),
        call(
            user_id="user-1",
            conversation_id="conversation-1",
            user_message_id="message-2",
        ),
    ]
    assert mock_nat["session"].run.call_args_list == [
        call("hello", runtime_type="run-or-serve"),
        call("again", runtime_type="run-or-serve"),
    ]
    assert mock_nat["runner"].result.await_count == 2
    mock_nat["sessions"].shutdown.assert_awaited_once_with()
    assert mock_nat["builder_context"].__aexit__.await_count == 1


async def test_start_failure_cleans_builder_and_redacts_cause(
    make_payload,
    mock_nat,
    caplog,
):
    mock_nat["session_manager"].create.side_effect = RuntimeError(
        "api-key=super-secret"
    )
    runtime = adapter.NatRuntime()

    with pytest.raises(adapter.lifecycle.LifecycleError) as error:
        await runtime.start(make_payload())

    assert error.value.code == "nat_workflow_start_failed"
    assert "super-secret" not in error.value.message
    assert "super-secret" not in caplog.text
    assert mock_nat["builder_context"].__aexit__.await_count == 1
    await runtime.stop()
    assert mock_nat["builder_context"].__aexit__.await_count == 1


async def test_invoke_failure_is_normalized_and_redacts_cause(
    make_payload,
    mock_nat,
    caplog,
):
    mock_nat["runner"].result = AsyncMock(
        side_effect=RuntimeError("api-key=super-secret")
    )
    runtime = adapter.NatRuntime()
    await runtime.start(make_payload())
    try:
        result = await runtime.invoke(invocation_payload())
    finally:
        await runtime.stop()

    assert result["failed"] is True
    assert result["completed"] is False
    assert result["response"] is None
    assert result["error"] == {
        "code": "nat_workflow_invoke_failed",
        "message": "NAT workflow invocation failed; inspect adapter stderr for details",
        "retryable": False,
    }
    assert "super-secret" not in json.dumps(result)
    assert "super-secret" not in caplog.text


async def test_non_json_result_is_normalized_without_value_leak(
    make_payload,
    mock_nat,
    caplog,
):
    mock_nat["runner"].result = AsyncMock(return_value=object())
    mock_nat["to_jsonable"].side_effect = TypeError("secret-object-repr")
    runtime = adapter.NatRuntime()
    await runtime.start(make_payload())
    try:
        result = await runtime.invoke(invocation_payload())
    finally:
        await runtime.stop()

    assert result["error"] == {
        "code": "nat_result_not_json_serializable",
        "message": "NAT workflow returned a result that cannot be represented as JSON",
        "retryable": False,
    }
    assert "secret-object-repr" not in json.dumps(result)
    assert "secret-object-repr" not in caplog.text


def test_system_instruction_rejects_unsupported_workflow(make_payload):
    payload = make_payload(
        workflow={"_type": "custom_workflow"},
        instruction="Portable instruction",
    )

    with pytest.raises(adapter.lifecycle.LifecycleError) as error:
        adapter.build_nat_config_mapping(payload)

    assert error.value.code == "nat_system_instruction_unsupported"


@pytest.mark.parametrize("transport", ["http", "streamable_http", "streamablehttp"])
def test_mcp_server_normalizes_streamable_http_aliases(transport: str):
    result = adapter.nat_mcp_server_config(
        "docs",
        {"transport": transport, "url": "https://mcp.test"},
    )

    assert result == {
        "transport": "streamable-http",
        "url": "https://mcp.test",
    }


@pytest.mark.parametrize("transport", ["websocket", ""])
def test_mcp_server_rejects_unsupported_transport(transport: str):
    with pytest.raises(adapter.lifecycle.LifecycleError) as error:
        adapter.nat_mcp_server_config(
            "docs",
            {"transport": transport, "url": "https://mcp.test"},
        )

    assert error.value.code == "nat_unsupported_mcp_transport"
