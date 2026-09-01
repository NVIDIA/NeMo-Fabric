# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for the source-only NeMo Agent Toolkit reference adapter."""

from __future__ import annotations

import asyncio
import json
import runpy
import sys
import types
from copy import deepcopy
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import call

import pytest
from nemo_fabric import Fabric
from nemo_fabric_adapter_contract.codec import ContractValidationError
from nemo_fabric_adapter_contract.models import AgentConfig
from nemo_fabric_adapter_contract.models import AgentMcpServerConfig
from nemo_fabric_adapter_contract.models import AgentRunRequest
from nemo_fabric_adapter_contract.models import AgentRunStatus
from nemo_fabric_adapter_contract.models import RuntimeContext

ROOT = Path(__file__).parents[2]
NAT_ADAPTER_SOURCE = ROOT / "external" / "nat" / "src"
sys.path.insert(0, str(NAT_ADAPTER_SOURCE))

from nemo_fabric_adapters.nat import adapter  # noqa: E402


class _AsyncChunkStream:
    """Small controllable async stream used to verify NeMo Agent Toolkit stream ownership."""

    def __init__(
        self,
        items: list[Any],
        *,
        close_error: BaseException | None = None,
    ) -> None:
        self._items = iter(items)
        self._close_error = close_error
        self.close_count = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            item = next(self._items)
        except StopIteration:
            raise StopAsyncIteration from None
        if isinstance(item, BaseException):
            raise item
        return item

    async def aclose(self) -> None:
        self.close_count += 1
        if self._close_error is not None:
            raise self._close_error


def _fabric_workflow(
    ref: str = "fabric.agent.react",
    *,
    kind: str = "factory",
    **settings: Any,
) -> dict[str, Any]:
    return {
        "entrypoint": {"kind": kind, "ref": ref},
        "settings": settings,
    }


def _mcp_server(**values: Any) -> AgentMcpServerConfig:
    return AgentMcpServerConfig.from_mapping(values)


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
        definitions: dict[str, Any] = {}
        for name, component in (functions or {}).items():
            component = deepcopy(component)
            definitions[name] = {
                "kind": "function",
                "ref": component.pop("_type"),
                "settings": component,
            }
        for name, component in (function_groups or {}).items():
            component = deepcopy(component)
            definitions[name] = {
                "kind": "function_group",
                "ref": component.pop("_type"),
                "settings": component,
            }

        config: dict[str, Any] = {
            "harness": {},
            "workflow": deepcopy(
                _fabric_workflow(llm_name="default") if workflow is None else workflow
            ),
            "models": deepcopy(models or {}),
        }
        if instruction is not None:
            config["instructions"] = {
                "system": {"content": instruction, "mode": "append"}
            }
        if tools is not None or definitions:
            config["tools"] = {**deepcopy(tools or {}), "definitions": definitions}
        if mcp_servers:
            config["mcp"] = {"servers": deepcopy(mcp_servers)}

        return {
            "base_dir": str(tmp_path),
            "config": AgentConfig.from_mapping(config),
            "runtime_context": {
                "runtime_id": "runtime-1",
                "environment": {"workspace": str(tmp_path)},
            },
        }

    return make


@pytest.fixture(name="mock_nat")
def mock_nat_fixture(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Install mocked NeMo Agent Toolkit modules and return lifecycle call recorders."""

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
    mock_runner.result_stream = MagicMock(name="result_stream")
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
        "nat.data_models.api_server",
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
    mock_chat_response_chunk = MagicMock(name="ChatResponseChunk")
    modules["nat.data_models.api_server"].ChatResponseChunk = mock_chat_response_chunk
    mock_sessions.get_workflow_streaming_output_schema.return_value = (
        mock_chat_response_chunk
    )
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
        "chat_response_chunk": mock_chat_response_chunk,
    }


@pytest.fixture(name="make_invocation_payload")
def make_invocation_payload_fixture():
    """Return a factory for typed adapter invocation arguments."""

    def make(
        *,
        input_value: Any = "hello",
        request_id: str = "request-1",
        context: Any = None,
        raw_request: Any = None,
        runtime_id: str = "runtime-1",
    ) -> tuple[AgentRunRequest, RuntimeContext]:
        request = raw_request
        if request is None:
            request = {"input": input_value}
            if context is not None:
                request["context"] = context
        return (
            AgentRunRequest.from_mapping(request),
            RuntimeContext.from_mapping(
                {
                    "runtime_id": runtime_id,
                    "invocation_id": f"invocation-{request_id}",
                    "request_id": request_id,
                    "environment": {
                        "environment_id": "environment-1",
                        "provider": "test",
                        "control_location": "in_env_control",
                        "ownership": "caller_owned",
                    },
                    "artifacts": {},
                }
            ),
        )

    return make


def test_descriptor_declares_exact_source_reference_contract():
    descriptor = json.loads(
        (ROOT / "external" / "nat" / "nat.fabric-adapter.json").read_text(
            encoding="utf-8"
        )
    )

    assert descriptor["adapter_id"] == "nvidia.fabric.nat"
    assert descriptor["adapter_kind"] == "python"
    assert descriptor["target_types"] == ["workflow"]
    assert descriptor["runner"] == {"module": "nemo_fabric_adapters.nat.adapter"}
    assert descriptor["requirements"] == {}
    assert descriptor["config"]["accepts"] == [
        "models",
        "models.base_url",
        "models.temperature",
        "instructions.system",
        "tools.definitions",
        "tools.enabled",
        "tools.blocked",
        "mcp",
        "mcp.tool_filters",
    ]
    assert descriptor["config"]["system_instruction_modes"] == ["append"]
    settings_schema = descriptor["settings_schema"]
    assert settings_schema["properties"] == {}
    assert "required" not in settings_schema
    assert settings_schema["additionalProperties"] is False
    target = json.loads(
        (
            ROOT
            / "external"
            / "nat"
            / "targets"
            / "email-phishing-analyzer.fabric-target.json"
        ).read_text(encoding="utf-8")
    )
    assert target["adapter_id"] == descriptor["adapter_id"]
    assert target["spec"]["entrypoint"] == {
        "kind": "factory",
        "ref": "fabric.agent.react",
    }
    assert target["spec"]["settings_schema"]["additionalProperties"] is False
    definition_schema = descriptor["tool_definition_schema"]
    assert definition_schema["properties"]["kind"]["enum"] == [
        "function",
        "function_group",
    ]
    assert definition_schema["properties"]["settings"]["properties"]["_type"] is False
    assert descriptor["capabilities"] == {
        "cancellation": False,
        "service": False,
        "streaming": True,
        "updates": False,
    }


def test_installed_nat_chat_response_chunk_serializes_to_openai_mapping():
    """Keep the adapter serializer aligned with the NeMo Agent Toolkit public chunk model."""

    api_server = pytest.importorskip("nat.data_models.api_server")
    pydantic_core = pytest.importorskip("pydantic_core")

    chunk = api_server.ChatResponseChunk.from_string(
        "hello",
        id_="chunk-1",
        model="test-model",
        finish_reason="stop",
    )
    serialized = pydantic_core.to_jsonable_python(
        chunk,
        serialize_unknown=False,
    )

    assert serialized["id"] == "chunk-1"
    assert serialized["object"] == "chat.completion.chunk"
    assert serialized["model"] == "test-model"
    assert isinstance(serialized["created"], int)
    assert serialized["choices"] == [
        {
            "finish_reason": "stop",
            "index": 0,
            "delta": {
                "content": "hello",
                "role": "assistant",
                "tool_calls": None,
            },
        }
    ]


def test_main_opts_the_nat_host_into_typed_agent_config(
    monkeypatch: pytest.MonkeyPatch,
):
    serve = MagicMock()
    monkeypatch.setattr(adapter.lifecycle, "serve", serve)

    adapter.main()

    serve.assert_called_once_with(
        adapter.NatRuntime,
        config_loader=AgentConfig.from_mapping,
    )


def test_build_mapping_translates_components_models_and_instruction(
    make_payload,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    workflow_settings = {
        "llm_name": "default",
        "tool_names": ["clock", "calculator"],
    }
    workflow = _fabric_workflow(**workflow_settings)
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

    result = adapter.build_nat_config_mapping(payload["config"])

    assert result == {
        "workflow": {
            "_type": "react_agent",
            **workflow_settings,
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
    assert payload["config"].workflow.to_mapping() == workflow


def test_system_instruction_rejects_duplicate_nat_instruction_source(make_payload):
    payload = make_payload(
        workflow=_fabric_workflow(
            llm_name="default",
            additional_instructions="adapter-local value",
        ),
        instruction="Portable instruction",
    )

    with pytest.raises(adapter.lifecycle.LifecycleError) as error:
        adapter.build_nat_config_mapping(payload["config"])

    assert error.value.code == "nat_system_instruction_conflict"


def test_system_instruction_rejects_replace_mode(make_payload):
    payload = make_payload(instruction="Portable instruction")
    payload["config"].instructions.system.mode = "replace"

    with pytest.raises(adapter.lifecycle.LifecycleError) as error:
        adapter.build_nat_config_mapping(payload["config"])

    assert error.value.code == "unsupported_system_instruction_mode"
    assert error.value.metadata["field"] == "instructions.system.mode"


def test_react_agent_without_tool_names_defaults_to_empty_list(make_payload):
    payload = make_payload(workflow=_fabric_workflow(llm_name="default"))

    result = adapter.build_nat_config_mapping(payload["config"])

    assert result["workflow"]["tool_names"] == []


def test_shared_nat_adapter_rejects_an_unknown_fabric_factory(make_payload):
    payload = make_payload(workflow=_fabric_workflow("fabric.agent.custom"))

    with pytest.raises(adapter.lifecycle.LifecycleError) as error:
        adapter.build_nat_config_mapping(payload["config"])

    assert error.value.code == "nat_invalid_workflow"
    assert error.value.metadata["field"] == "workflow.entrypoint.ref"


def test_missing_root_workflow_is_rejected(make_payload):
    payload = make_payload()
    payload["config"].workflow = None

    with pytest.raises(adapter.lifecycle.LifecycleError) as error:
        adapter.build_nat_config_mapping(payload["config"])

    assert error.value.code == "nat_invalid_workflow"
    assert error.value.metadata["field"] == "workflow"


@pytest.mark.parametrize(
    ("workflow", "field"),
    [
        (_fabric_workflow(kind="python_callable"), "workflow.entrypoint.kind"),
        (_fabric_workflow("fabric.agent.unknown"), "workflow.entrypoint.ref"),
        (_fabric_workflow(_type="react_agent"), "workflow.settings._type"),
    ],
)
def test_invalid_root_workflow_is_rejected(make_payload, workflow, field):
    payload = make_payload(workflow=workflow)

    with pytest.raises(adapter.lifecycle.LifecycleError) as error:
        adapter.build_nat_config_mapping(payload["config"])

    assert error.value.code == "nat_invalid_workflow"
    assert error.value.metadata["field"] == field


@pytest.mark.parametrize(
    ("example", "target_id"),
    [
        ("calculator.py", "nvidia.examples.nat.calculator"),
        (
            "email_phishing.py",
            "nvidia.examples.nat.email-phishing-analyzer",
        ),
    ],
)
def test_typed_examples_project_and_translate_through_one_nat_adapter(
    tmp_path: Path,
    example: str,
    target_id: str,
    monkeypatch: pytest.MonkeyPatch,
):
    namespace = runpy.run_path(str(ROOT / "external" / "nat" / "examples" / example))
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")

    plan = Fabric().plan(namespace["build_config"](), base_dir=tmp_path)

    assert plan.config.workflow.target_id == target_id
    assert plan["adapter_target_descriptor"]["descriptor"]["id"] == target_id
    assert plan.config.harness is None
    southbound = plan.to_mapping()["agent_config"]
    assert southbound["workflow"]["entrypoint"] == {
        "kind": "factory",
        "ref": "fabric.agent.react",
    }
    assert "schema_version" not in southbound
    nat_config = adapter.build_nat_config_mapping(AgentConfig.from_mapping(southbound))
    assert nat_config["workflow"]["_type"] == "react_agent"
    if example == "calculator.py":
        assert nat_config["function_groups"]["calculator"]["_type"] == "mcp_client"
    else:
        assert nat_config["functions"]["email_phishing_analyzer"] == {
            "_type": "email_phishing_analyzer",
            "llm": "default",
        }


def test_calculator_example_uses_the_source_stdio_server():
    namespace = runpy.run_path(
        str(ROOT / "external" / "nat" / "examples" / "calculator.py")
    )

    config = namespace["build_config"]()
    calculator = config.mcp.servers["calculator"]

    assert calculator.transport == "stdio"
    assert calculator.url == sys.executable
    assert calculator.args == [
        str(ROOT / "external" / "nat" / "examples" / "calculator_mcp.py")
    ]
    assert calculator.blocked_tools == ["divide"]


def test_email_example_uses_a_normalized_function_definition():
    namespace = runpy.run_path(
        str(ROOT / "external" / "nat" / "examples" / "email_phishing.py")
    )

    config = namespace["build_config"]()
    definition = config.tools.definitions["email_phishing_analyzer"]

    assert config.harness is None
    assert definition.kind == "function"
    assert definition.ref == "email_phishing_analyzer"
    assert definition.settings == {"llm": "default"}


def test_build_typed_config_discovers_components_before_validation(
    make_payload,
    mock_nat,
):
    events: list[str] = []
    mock_nat["discover"].side_effect = lambda _plugin_type: events.append("discover")
    mock_nat["config_type"].model_validate.side_effect = lambda _mapping: (
        events.append("validate") or mock_nat["typed_config"]
    )

    result = adapter.build_nat_config(make_payload()["config"])

    assert result is mock_nat["typed_config"]
    assert events == ["discover", "validate"]
    mock_nat["discover"].assert_called_once_with(mock_nat["plugin_types"].CONFIG_OBJECT)


def test_build_typed_config_contract_with_installed_nat(
    make_payload,
    monkeypatch: pytest.MonkeyPatch,
):
    config_module = pytest.importorskip(
        "nat.data_models.config",
        reason="NeMo Agent Toolkit is not installed in the base Fabric test environment",
    )
    pytest.importorskip(
        "nat.plugins.langchain.agent.react_agent",
        reason="The NeMo Agent Toolkit LangChain extra is not installed",
    )
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    payload = make_payload(
        workflow=_fabric_workflow(llm_name="default"),
        models={
            "default": {
                "provider": "nvidia",
                "model": "nvidia/test-model",
                "api_key_env": "NVIDIA_API_KEY",
            }
        },
    )

    result = adapter.build_nat_config(payload["config"])

    assert isinstance(result, config_module.Config)
    assert result.workflow.type == "react_agent"
    assert result.llms["default"].model_name == "nvidia/test-model"


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

    result = adapter.build_nat_config_mapping(payload["config"])

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

    result = adapter.build_nat_config_mapping(payload["config"])

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

    result = adapter.build_nat_config_mapping(payload["config"])

    assert result["function_groups"] == {
        "calculator": {"_type": "calculator"},
    }
    assert result["workflow"]["tool_names"] == ["calculator"]


def test_empty_mcp_allowlist_suppresses_server_and_existing_workflow_ref(
    make_payload,
):
    payload = make_payload(
        workflow=_fabric_workflow(
            llm_name="default",
            tool_names=["clock", "docs"],
        ),
        functions={"clock": {"_type": "current_datetime"}},
        mcp_servers={
            "docs": {
                "transport": "streamable-http",
                "url": "https://mcp.test/mcp",
                "allowed_tools": [],
            }
        },
    )

    result = adapter.build_nat_config_mapping(payload["config"])

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
        adapter.build_nat_config_mapping(payload["config"])

    assert error.value.code == "nat_mcp_name_conflict"


def test_all_suppressed_mcp_leaves_factory_workflow_with_no_tools(make_payload):
    payload = make_payload(
        mcp_servers={
            "docs": {
                "transport": "sse",
                "url": "https://mcp.test/sse",
                "allowed_tools": [],
            }
        },
    )

    result = adapter.build_nat_config_mapping(payload["config"])

    assert result["workflow"] == {
        "_type": "react_agent",
        "llm_name": "default",
        "tool_names": [],
    }
    assert result["function_groups"] == {}


def test_root_tool_policy_selects_and_blocks_exact_group_members(make_payload):
    payload = make_payload(
        workflow=_fabric_workflow(
            llm_name="default",
            tool_names=["clock", "unused", "calculator", "search"],
        ),
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
                "search__find",
            ],
            "blocked": ["calculator__subtract", "search__secret"],
        },
    )

    result = adapter.build_nat_config_mapping(payload["config"])

    assert result["functions"] == {"clock": {"_type": "current_datetime"}}
    assert result["function_groups"] == {
        "calculator": {"_type": "calculator", "include": ["add"]},
        "search": {"_type": "search", "include": ["find"]},
    }
    assert result["workflow"]["tool_names"] == ["clock", "calculator", "search"]


def test_blocking_last_group_member_and_function_removes_both_tool_refs(
    make_payload,
):
    payload = make_payload(
        workflow=_fabric_workflow(
            llm_name="default",
            tool_names=["calculator", "clock"],
        ),
        functions={"clock": {"_type": "current_datetime"}},
        function_groups={"calculator": {"_type": "calculator", "include": ["add"]}},
        tools={"blocked": ["calculator__add", "clock"]},
    )

    result = adapter.build_nat_config_mapping(payload["config"])

    assert result["functions"] == {}
    assert result["function_groups"] == {}
    assert result["workflow"]["tool_names"] == []


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
        workflow=_fabric_workflow(
            llm_name="default",
            tool_names=["calculator"],
        ),
        function_groups={"calculator": {"_type": "calculator", "include": ["add"]}},
        tools=tools,
    )

    with pytest.raises(adapter.lifecycle.LifecycleError) as error:
        adapter.build_nat_config_mapping(payload["config"])

    assert error.value.code == "nat_unknown_tool_selector"


async def test_runtime_reuses_one_builder_across_invocations_and_cleans_up(
    make_payload,
    make_invocation_payload,
    mock_nat,
):
    runtime = adapter.NatRuntime()
    await runtime.start(make_payload())

    first = await runtime.invoke(
        *make_invocation_payload(
            request_id="message-1",
            context={"user_id": "user-1", "conversation_id": "conversation-1"},
        )
    )
    second = await runtime.invoke(
        *make_invocation_payload(
            input_value="again",
            request_id="message-2",
            context={"user_id": "user-1", "conversation_id": "conversation-1"},
        )
    )
    await runtime.stop()
    await runtime.stop()

    assert first.status is AgentRunStatus.SUCCEEDED
    assert first.output == {
        "harness": "nat",
        "adapter": "python",
        "mode": "agent_config",
        "response": {"answer": 42},
        "completed": True,
    }
    assert second.output["response"] == {"answer": 84}
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


async def test_openai_stream_forwards_ordered_chunks_and_returns_content(
    make_payload,
    make_invocation_payload,
    mock_nat,
):
    chunks = [
        {
            "id": "chunk-1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "hel"},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chunk-2",
            "object": "chat.completion.chunk",
            "created": 2,
            "model": "test-model",
            "choices": [
                {
                    "index": 1,
                    "delta": {"content": "ignored"},
                    "finish_reason": None,
                },
                {
                    "index": 0,
                    "delta": {"content": "lo"},
                    "finish_reason": "stop",
                },
            ],
        },
    ]
    stream = _AsyncChunkStream(chunks)
    mock_nat["runner"].result_stream.return_value = stream
    emitted: list[dict[str, Any]] = []

    async def emit(chunk: dict[str, Any]) -> None:
        emitted.append(chunk)

    input_value = {"message": "hello"}
    runtime = adapter.NatRuntime()
    await runtime.start(make_payload())
    try:
        result = await runtime.invoke_openai_stream(
            *make_invocation_payload(
                input_value=input_value,
                context={"conversation_id": "conversation-1"},
            ),
            emit,
        )
    finally:
        await runtime.stop()

    assert emitted == chunks
    assert result.status is AgentRunStatus.SUCCEEDED
    assert result.output == {
        "harness": "nat",
        "adapter": "python",
        "mode": "agent_config",
        "response": "hello",
        "completed": True,
    }
    mock_nat["sessions"].session.assert_called_once_with(
        conversation_id="conversation-1",
        user_message_id="request-1",
    )
    mock_nat["session"].run.assert_called_once_with(
        input_value,
        runtime_type="run-or-serve",
    )
    mock_nat["runner"].result_stream.assert_called_once_with(
        to_type=mock_nat["chat_response_chunk"]
    )
    mock_nat["runner"].result.assert_not_awaited()
    assert mock_nat["to_jsonable"].call_args_list == [
        call(chunk, serialize_unknown=False) for chunk in chunks
    ]
    assert stream.close_count == 1


async def test_openai_stream_reuses_invocation_state_validation(
    make_payload,
    make_invocation_payload,
    mock_nat,
):
    runtime = adapter.NatRuntime()
    emit = AsyncMock()

    with pytest.raises(adapter.lifecycle.LifecycleError) as not_started:
        await runtime.invoke_openai_stream(*make_invocation_payload(), emit)
    assert not_started.value.code == "nat_runtime_not_started"

    await runtime.start(make_payload())
    try:
        with pytest.raises(adapter.lifecycle.LifecycleError) as mismatch:
            await runtime.invoke_openai_stream(
                *make_invocation_payload(runtime_id="runtime-2"), emit
            )
    finally:
        await runtime.stop()

    assert mismatch.value.code == "nat_runtime_mismatch"
    mock_nat["sessions"].session.assert_not_called()
    emit.assert_not_awaited()


async def test_openai_stream_accepts_empty_and_usage_only_streams(
    make_payload,
    make_invocation_payload,
    mock_nat,
):
    usage_chunk = {
        "id": "usage-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "test-model",
        "choices": [],
        "usage": {"prompt_tokens": 1, "completion_tokens": 0, "total_tokens": 1},
    }
    streams = [_AsyncChunkStream([]), _AsyncChunkStream([usage_chunk])]
    mock_nat["runner"].result_stream.side_effect = streams
    emitted: list[dict[str, Any]] = []

    async def emit(chunk: dict[str, Any]) -> None:
        emitted.append(chunk)

    runtime = adapter.NatRuntime()
    await runtime.start(make_payload())
    try:
        empty_result = await runtime.invoke_openai_stream(
            *make_invocation_payload(request_id="empty"), emit
        )
        usage_result = await runtime.invoke_openai_stream(
            *make_invocation_payload(request_id="usage"), emit
        )
    finally:
        await runtime.stop()

    assert empty_result.output["response"] == ""
    assert usage_result.output["response"] == ""
    assert emitted == [usage_chunk]
    assert mock_nat["runner"].result_stream.call_count == 2
    mock_nat["runner"].result.assert_not_awaited()
    assert [stream.close_count for stream in streams] == [1, 1]


async def test_openai_stream_rejects_non_openai_schema_before_session(
    make_payload,
    make_invocation_payload,
    mock_nat,
):
    mock_nat["sessions"].get_workflow_streaming_output_schema.return_value = None
    runtime = adapter.NatRuntime()
    await runtime.start(make_payload())
    try:
        result = await runtime.invoke_openai_stream(
            *make_invocation_payload(),
            AsyncMock(),
        )
    finally:
        await runtime.stop()

    assert result.error.code == "nat_openai_stream_unsupported_schema"
    assert result.error.message == (
        "Native NeMo Agent Toolkit OpenAI streaming requires a ChatResponseChunk output schema"
    )
    mock_nat["sessions"].get_workflow_streaming_output_schema.assert_called_once_with()
    mock_nat["sessions"].session.assert_not_called()
    mock_nat["runner"].result_stream.assert_not_called()
    mock_nat["runner"].result.assert_not_awaited()


async def test_openai_stream_normalizes_partial_nat_failure_without_leaking_cause(
    make_payload,
    make_invocation_payload,
    mock_nat,
    caplog,
):
    chunk = {
        "id": "chunk-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "test-model",
        "choices": [{"index": 0, "delta": {"content": "partial"}}],
    }
    stream = _AsyncChunkStream([chunk, RuntimeError("api-key=super-secret")])
    mock_nat["runner"].result_stream.return_value = stream
    emit = AsyncMock()
    runtime = adapter.NatRuntime()
    await runtime.start(make_payload())
    try:
        result = await runtime.invoke_openai_stream(*make_invocation_payload(), emit)
    finally:
        await runtime.stop()

    assert result.error.code == "nat_workflow_stream_failed"
    assert result.error.message == (
        "NeMo Agent Toolkit workflow streaming failed; inspect adapter stderr for details"
    )
    emit.assert_awaited_once_with(chunk)
    assert stream.close_count == 1
    mock_nat["runner"].result.assert_not_awaited()
    assert "super-secret" not in json.dumps(result.to_mapping())
    assert "super-secret" not in caplog.text


@pytest.mark.parametrize(
    "serialization_failure",
    [TypeError("secret-object-repr"), ["not", "a", "mapping"]],
)
async def test_openai_stream_normalizes_chunk_serialization_failure(
    make_payload,
    make_invocation_payload,
    mock_nat,
    caplog,
    serialization_failure: Any,
):
    chunk = {
        "id": "chunk-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "test-model",
        "choices": [],
    }
    stream = _AsyncChunkStream([chunk])
    mock_nat["runner"].result_stream.return_value = stream
    if isinstance(serialization_failure, BaseException):
        mock_nat["to_jsonable"].side_effect = serialization_failure
    else:
        mock_nat["to_jsonable"].side_effect = None
        mock_nat["to_jsonable"].return_value = serialization_failure
    emit = AsyncMock()
    runtime = adapter.NatRuntime()
    await runtime.start(make_payload())
    try:
        result = await runtime.invoke_openai_stream(*make_invocation_payload(), emit)
    finally:
        await runtime.stop()

    assert result.error.code == "nat_stream_chunk_not_json_serializable"
    assert result.error.message == (
        "NeMo Agent Toolkit workflow returned a stream chunk that cannot be represented as JSON"
    )
    mock_nat["to_jsonable"].assert_called_once_with(
        chunk,
        serialize_unknown=False,
    )
    emit.assert_not_awaited()
    assert stream.close_count == 1
    assert "secret-object-repr" not in json.dumps(result.to_mapping())
    assert "secret-object-repr" not in caplog.text


@pytest.mark.parametrize(
    "emitter_error",
    [
        adapter.lifecycle.LifecycleError(
            "lifecycle_stream_transport_failed",
            "stream listener disconnected",
        ),
        asyncio.CancelledError(),
    ],
)
async def test_openai_stream_propagates_emitter_lifecycle_and_cancellation(
    make_payload,
    make_invocation_payload,
    mock_nat,
    caplog,
    emitter_error: BaseException,
):
    chunk = {
        "id": "chunk-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "test-model",
        "choices": [{"index": 0, "delta": {"content": "hello"}}],
    }
    stream = _AsyncChunkStream(
        [chunk],
        close_error=RuntimeError("cleanup-secret"),
    )
    mock_nat["runner"].result_stream.return_value = stream
    emit = AsyncMock(side_effect=emitter_error)
    runtime = adapter.NatRuntime()
    await runtime.start(make_payload())
    try:
        with pytest.raises(type(emitter_error)) as raised:
            await runtime.invoke_openai_stream(*make_invocation_payload(), emit)
        assert raised.value is emitter_error
    finally:
        await runtime.stop()

    assert stream.close_count == 1
    assert mock_nat["run_context"].__aexit__.await_count == 1
    assert mock_nat["session_context"].__aexit__.await_count == 1
    mock_nat["runner"].result.assert_not_awaited()
    assert "cleanup-secret" not in caplog.text
    assert "error_type=RuntimeError" in caplog.text


async def test_start_rejects_an_already_started_runtime(make_payload, mock_nat):
    runtime = adapter.NatRuntime()
    await runtime.start(make_payload())
    try:
        with pytest.raises(adapter.lifecycle.LifecycleError) as error:
            await runtime.start(make_payload())
    finally:
        await runtime.stop()

    assert error.value.code == "nat_runtime_already_started"
    assert error.value.message == "NeMo Agent Toolkit runtime is already started"
    mock_nat["workflow_builder"].from_config.assert_called_once()


async def test_start_rejects_a_legacy_fabric_config_mapping(tmp_path: Path):
    runtime = adapter.NatRuntime()

    with pytest.raises(adapter.lifecycle.LifecycleError) as error:
        await runtime.start(
            {
                "config": {"schema_version": "fabric.agent/v1alpha1"},
                "runtime_context": {"runtime_id": "runtime-1"},
                "base_dir": str(tmp_path),
            }
        )

    assert error.value.code == "nat_invalid_agent_config"


async def test_invoke_rejects_a_runtime_that_has_not_started(
    make_invocation_payload,
):
    runtime = adapter.NatRuntime()

    with pytest.raises(adapter.lifecycle.LifecycleError) as error:
        await runtime.invoke(*make_invocation_payload())

    assert error.value.code == "nat_runtime_not_started"
    assert error.value.message == "NeMo Agent Toolkit runtime is not started"


async def test_invoke_rejects_a_different_runtime_id(
    make_payload,
    make_invocation_payload,
    mock_nat,
):
    runtime = adapter.NatRuntime()
    await runtime.start(make_payload())
    try:
        with pytest.raises(adapter.lifecycle.LifecycleError) as error:
            await runtime.invoke(*make_invocation_payload(runtime_id="runtime-2"))
    finally:
        await runtime.stop()

    assert error.value.code == "nat_runtime_mismatch"
    assert (
        error.value.message
        == "NeMo Agent Toolkit invocation does not match the active runtime"
    )
    mock_nat["sessions"].session.assert_not_called()


def test_typed_invocation_rejects_a_non_mapping_request(make_invocation_payload):
    with pytest.raises(ContractValidationError):
        make_invocation_payload(raw_request=["not", "a", "mapping"])


def test_typed_invocation_rejects_a_non_mapping_request_context(
    make_invocation_payload,
):
    with pytest.raises(ContractValidationError):
        make_invocation_payload(context=["not", "a", "mapping"])


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


def test_config_translation_failure_is_normalized_and_redacts_cause(
    make_payload,
    mock_nat,
):
    mock_nat["config_type"].model_validate.side_effect = RuntimeError(
        "api-key=super-secret"
    )

    with pytest.raises(adapter.lifecycle.LifecycleError) as error:
        adapter.build_nat_config(make_payload()["config"])

    assert error.value.code == "nat_config_translation_failed"
    assert error.value.message == (
        "Fabric config could not be translated into a valid NeMo Agent Toolkit config"
    )
    assert "super-secret" not in str(error.value)


async def test_stop_failure_clears_runtime_state_and_redacts_cause(
    make_payload,
    mock_nat,
):
    mock_nat["sessions"].shutdown.side_effect = RuntimeError("api-key=super-secret")
    runtime = adapter.NatRuntime()
    await runtime.start(make_payload())

    with pytest.raises(adapter.lifecycle.LifecycleError) as error:
        await runtime.stop()

    assert error.value.code == "nat_runtime_stop_failed"
    assert error.value.message == "NeMo Agent Toolkit runtime failed to stop cleanly"
    assert "super-secret" not in str(error.value)
    mock_nat["sessions"].shutdown.assert_awaited_once_with()
    assert mock_nat["builder_context"].__aexit__.await_count == 1

    await runtime.stop()
    mock_nat["sessions"].shutdown.assert_awaited_once_with()


async def test_invoke_failure_is_normalized_and_redacts_cause(
    make_payload,
    make_invocation_payload,
    mock_nat,
    caplog,
):
    mock_nat["runner"].result = AsyncMock(
        side_effect=RuntimeError("api-key=super-secret")
    )
    runtime = adapter.NatRuntime()
    await runtime.start(make_payload())
    try:
        result = await runtime.invoke(*make_invocation_payload())
    finally:
        await runtime.stop()

    assert result.status is AgentRunStatus.FAILED
    assert result.output["completed"] is False
    assert result.output["response"] is None
    assert result.error.code == "nat_workflow_invoke_failed"
    assert result.error.message == (
        "NeMo Agent Toolkit workflow invocation failed; inspect adapter stderr for details"
    )
    assert "super-secret" not in json.dumps(result.to_mapping())
    assert "super-secret" not in caplog.text


async def test_non_json_result_is_normalized_without_value_leak(
    make_payload,
    make_invocation_payload,
    mock_nat,
    caplog,
):
    mock_nat["runner"].result = AsyncMock(return_value=object())
    mock_nat["to_jsonable"].side_effect = TypeError("secret-object-repr")
    runtime = adapter.NatRuntime()
    await runtime.start(make_payload())
    try:
        result = await runtime.invoke(*make_invocation_payload())
    finally:
        await runtime.stop()

    assert result.error.code == "nat_result_not_json_serializable"
    assert result.error.message == (
        "NeMo Agent Toolkit workflow returned a result that cannot be represented as JSON"
    )
    assert "secret-object-repr" not in json.dumps(result.to_mapping())
    assert "secret-object-repr" not in caplog.text


@pytest.mark.parametrize("transport", ["http", "streamable_http", "streamablehttp"])
def test_mcp_server_normalizes_streamable_http_aliases(transport: str):
    result = adapter.nat_mcp_server_config(
        "docs",
        _mcp_server(transport=transport, url="https://mcp.test"),
    )

    assert result == {
        "transport": "streamable-http",
        "url": "https://mcp.test",
    }


def test_mcp_stdio_expands_command_and_maps_structured_args_and_env(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("NAT_TEST_MCP_COMMAND", "/opt/nat/bin/mcp-server")

    result = adapter.nat_mcp_server_config(
        "calculator",
        _mcp_server(
            transport="stdio",
            url="$NAT_TEST_MCP_COMMAND",
            args=[
                "--label",
                "safe mode",
                "--port",
                "9000",
                "--trace",
                "--request-timeout=10",
            ],
            env={"NAT_MCP_TOKEN": "test-token"},
        ),
    )

    assert result == {
        "transport": "stdio",
        "command": "/opt/nat/bin/mcp-server",
        "args": [
            "--label",
            "safe mode",
            "--port",
            "9000",
            "--trace",
            "--request-timeout=10",
        ],
        "env": {"NAT_MCP_TOKEN": "test-token"},
    }


def test_mcp_stdio_preserves_a_command_with_spaces_without_shell_parsing():
    result = adapter.nat_mcp_server_config(
        "calculator",
        _mcp_server(transport="stdio", url="/opt/MCP Servers/calculator"),
    )

    assert result == {
        "transport": "stdio",
        "command": "/opt/MCP Servers/calculator",
    }


def test_mcp_stdio_rejects_a_whitespace_only_command():
    server = _mcp_server(transport="stdio", url="placeholder")
    object.__setattr__(server, "url", " \t\n ")

    with pytest.raises(adapter.lifecycle.LifecycleError) as error:
        adapter.nat_mcp_server_config("calculator", server)

    assert error.value.code == "nat_invalid_mcp_server"
    assert error.value.message == (
        "NeMo Agent Toolkit MCP server 'calculator' requires a non-empty url"
    )


@pytest.mark.parametrize("transport", ["websocket", ""])
def test_mcp_server_rejects_unsupported_transport(transport: str):
    server = _mcp_server(transport=transport or "placeholder", url="https://mcp.test")
    object.__setattr__(server, "transport", transport)
    with pytest.raises(adapter.lifecycle.LifecycleError) as error:
        adapter.nat_mcp_server_config(
            "docs",
            server,
        )

    assert error.value.code == "nat_unsupported_mcp_transport"
