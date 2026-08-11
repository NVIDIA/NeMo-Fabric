# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for the source-only NeMo Agent Toolkit reference adapter."""

from __future__ import annotations

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
from nemo_fabric_adapter_contract.models import AgentConfig
from nemo_fabric_adapter_contract.models import AgentMcpServerConfig

ROOT = Path(__file__).parents[2]
NAT_ADAPTER_SOURCE = ROOT / "external" / "nat" / "src"
sys.path.insert(0, str(NAT_ADAPTER_SOURCE))

from nemo_fabric_adapters.nat import adapter  # noqa: E402


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
                "system": {"content": instruction, "mode": "replace"}
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
    mock_sessions.is_workflow_per_user = False
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


@pytest.fixture(name="make_invocation_payload")
def make_invocation_payload_fixture():
    """Return a factory for canonical Fabric invocation payloads."""

    def make(
        *,
        input_value: Any = "hello",
        request_id: str = "request-1",
        context: Any = None,
        raw_request: Any = None,
        runtime_id: str = "runtime-1",
    ) -> dict[str, Any]:
        request = raw_request
        if request is None:
            request = {"input": input_value, "request_id": request_id}
            if context is not None:
                request["context"] = context
        return {
            "runtime_context": {"runtime_id": runtime_id},
            "request": request,
        }

    return make


def test_descriptor_declares_exact_source_reference_contract():
    descriptor = json.loads(
        (ROOT / "external" / "nat" / "fabric-adapter.json").read_text(encoding="utf-8")
    )

    assert descriptor["adapter_id"] == "nvidia.fabric.nat"
    assert descriptor["harness"] == "nat"
    assert descriptor["adapter_kind"] == "python"
    assert descriptor["runner"] == {"module": "nemo_fabric_adapters.nat.adapter"}
    assert descriptor["requirements"] == {}
    assert descriptor["config"]["input"] == "agent_config"
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
    settings_schema = descriptor["settings_schema"]
    assert settings_schema["properties"] == {}
    assert "required" not in settings_schema
    assert settings_schema["additionalProperties"] is False
    workflow_schema = descriptor["workflow_schema"]
    assert workflow_schema["required"] == ["entrypoint"]
    assert workflow_schema["additionalProperties"] is False
    entrypoint_schema = workflow_schema["properties"]["entrypoint"]
    assert entrypoint_schema["properties"]["kind"]["const"] == "factory"
    ref_schema = entrypoint_schema["properties"]["ref"]
    assert ref_schema["type"] == "string"
    assert ref_schema["minLength"] == 1
    assert ref_schema["pattern"] == r"^\S+$"
    assert entrypoint_schema["required"] == ["kind", "ref"]
    assert entrypoint_schema["additionalProperties"] is False
    assert workflow_schema["properties"]["settings"]["properties"]["_type"] is False
    definition_schema = descriptor["tool_definition_schema"]
    assert definition_schema["properties"]["kind"]["enum"] == [
        "function",
        "function_group",
    ]
    assert definition_schema["properties"]["settings"]["properties"]["_type"] is False
    assert descriptor["capabilities"] == {
        "cancellation": False,
        "service": False,
        "streaming": False,
        "updates": False,
    }


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


@pytest.mark.parametrize(
    ("ref", "expected_type", "uses_react_config_shape"),
    [
        ("fabric.agent.react", "react_agent", True),
        ("react_agent", "react_agent", True),
        (
            "nat.plugins.langchain.agent.react_agent/react_agent",
            "nat.plugins.langchain.agent.react_agent/react_agent",
            True,
        ),
        ("per_user_react_agent", "per_user_react_agent", True),
        (
            "nat.plugins.langchain.agent.react_agent/per_user_react_agent",
            "nat.plugins.langchain.agent.react_agent/per_user_react_agent",
            True,
        ),
        ("tool_calling_agent", "tool_calling_agent", False),
        (
            "nat.plugins.langchain.agent.tool_calling_agent/tool_calling_agent",
            "nat.plugins.langchain.agent.tool_calling_agent/tool_calling_agent",
            False,
        ),
        ("third.party/react_agent", "third.party/react_agent", False),
        ("third.party/per_user_react_agent", "third.party/per_user_react_agent", False),
    ],
)
def test_workflow_refs_pass_through_with_react_tool_defaults(
    make_payload,
    ref: str,
    expected_type: str,
    uses_react_config_shape: bool,
):
    payload = make_payload(workflow=_fabric_workflow(ref, llm_name="default"))

    result = adapter.build_nat_config_mapping(payload["config"])

    assert result["workflow"]["_type"] == expected_type
    if uses_react_config_shape:
        assert result["workflow"]["tool_names"] == []
    else:
        assert "tool_names" not in result["workflow"]


@pytest.mark.parametrize(
    "ref",
    ["third.party/react_agent", "third.party/per_user_react_agent"],
)
def test_arbitrary_registry_ref_does_not_inherit_react_field_translation(
    make_payload,
    ref: str,
):
    payload = make_payload(
        workflow=_fabric_workflow(ref, llm_name="default"),
        instruction="Portable instruction",
    )

    with pytest.raises(adapter.lifecycle.LifecycleError) as error:
        adapter.build_nat_config_mapping(payload["config"])

    assert error.value.code == "nat_system_instruction_unsupported"


@pytest.mark.parametrize(
    "ref",
    [
        "per_user_react_agent",
        "nat.plugins.langchain.agent.react_agent/per_user_react_agent",
    ],
)
def test_per_user_react_maps_system_instruction_and_tool_policy(
    make_payload,
    ref: str,
):
    payload = make_payload(
        workflow=_fabric_workflow(
            ref,
            llm_name="default",
            tool_names=["clock", "unused"],
        ),
        functions={
            "clock": {"_type": "current_datetime"},
            "unused": {"_type": "unused_function"},
        },
        instruction="Use the clock for time questions.",
        tools={"enabled": ["clock"]},
    )

    result = adapter.build_nat_config_mapping(payload["config"])

    assert result["workflow"]["additional_instructions"] == (
        "Use the clock for time questions."
    )
    assert result["workflow"]["tool_names"] == ["clock"]
    assert result["functions"] == {"clock": {"_type": "current_datetime"}}


def test_nat_registry_ref_passes_through_without_adapter_catalog(make_payload):
    payload = make_payload(workflow=_fabric_workflow("installed.custom/workflow"))

    result = adapter.build_nat_config_mapping(payload["config"])

    assert result["workflow"] == {"_type": "installed.custom/workflow"}


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
        (_fabric_workflow(_type="react_agent"), "workflow.settings._type"),
    ],
)
def test_invalid_root_workflow_is_rejected(make_payload, workflow, field):
    payload = make_payload(workflow=workflow)

    with pytest.raises(adapter.lifecycle.LifecycleError) as error:
        adapter.build_nat_config_mapping(payload["config"])

    assert error.value.code == "nat_invalid_workflow"
    assert error.value.metadata["field"] == field


@pytest.mark.parametrize("example", ["calculator.py", "email_phishing.py"])
def test_typed_examples_project_and_translate_through_one_nat_adapter(
    tmp_path: Path,
    example: str,
    monkeypatch: pytest.MonkeyPatch,
):
    descriptor = ROOT / "external" / "nat" / "fabric-adapter.json"
    staged_descriptor = tmp_path / "adapters" / "nat" / "fabric-adapter.json"
    staged_descriptor.parent.mkdir(parents=True)
    staged_descriptor.write_text(
        descriptor.read_text(encoding="utf-8"), encoding="utf-8"
    )
    namespace = runpy.run_path(str(ROOT / "external" / "nat" / "examples" / example))
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")

    plan = Fabric().plan(namespace["build_config"](), base_dir=tmp_path)

    assert plan.config.workflow.entrypoint.kind == "factory"
    assert plan.config.workflow.entrypoint.ref == "fabric.agent.react"
    assert "workflow" not in plan.config.harness.settings
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


@pytest.mark.parametrize(
    ("ref", "expected_type", "uses_react_config_shape"),
    [
        ("per_user_react_agent", "per_user_react_agent", True),
        (
            "nat.plugins.langchain.agent.react_agent/per_user_react_agent",
            "nat.plugins.langchain.agent.react_agent/per_user_react_agent",
            True,
        ),
        ("tool_calling_agent", "tool_calling_agent", False),
        (
            "nat.plugins.langchain.agent.tool_calling_agent/tool_calling_agent",
            "nat.plugins.langchain.agent.tool_calling_agent/tool_calling_agent",
            False,
        ),
    ],
)
def test_nat_workflow_refs_plan_through_generic_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ref: str,
    expected_type: str,
    uses_react_config_shape: bool,
):
    descriptor = ROOT / "external" / "nat" / "fabric-adapter.json"
    staged_descriptor = tmp_path / "adapters" / "nat" / "fabric-adapter.json"
    staged_descriptor.parent.mkdir(parents=True)
    staged_descriptor.write_text(
        descriptor.read_text(encoding="utf-8"), encoding="utf-8"
    )
    namespace = runpy.run_path(
        str(ROOT / "external" / "nat" / "examples" / "calculator.py")
    )
    config = namespace["build_config"]()
    config.workflow.entrypoint.ref = ref
    if not uses_react_config_shape:
        config.instructions = None
        config.mcp = None
        config.tools = None
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")

    plan = Fabric().plan(config, base_dir=tmp_path)

    assert plan.config.workflow.entrypoint.ref == ref
    southbound = AgentConfig.from_mapping(plan.to_mapping()["agent_config"])
    nat_config = adapter.build_nat_config_mapping(southbound)
    assert nat_config["workflow"]["_type"] == expected_type


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

    assert config.harness.settings == {}
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


@pytest.mark.parametrize(
    ("ref", "expected_type"),
    [
        ("fabric.agent.react", "react_agent"),
        ("per_user_react_agent", "per_user_react_agent"),
        (
            "nat.plugins.langchain.agent.react_agent/per_user_react_agent",
            "per_user_react_agent",
        ),
        ("tool_calling_agent", "tool_calling_agent"),
        (
            "nat.plugins.langchain.agent.tool_calling_agent/tool_calling_agent",
            "tool_calling_agent",
        ),
    ],
)
def test_build_typed_config_contract_with_installed_nat(
    make_payload,
    monkeypatch: pytest.MonkeyPatch,
    ref: str,
    expected_type: str,
):
    config_module = pytest.importorskip(
        "nat.data_models.config",
        reason="NAT is not installed in the base Fabric test environment",
    )
    pytest.importorskip(
        "nat.plugins.langchain.agent.react_agent",
        reason="The NAT LangChain extra is not installed",
    )
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    payload = make_payload(
        workflow=_fabric_workflow(ref, llm_name="default"),
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
    assert result.workflow.type == expected_type
    assert result.llms["default"].model_name == "nvidia/test-model"


async def test_installed_nat_reuses_isolates_and_cleans_per_user_builders(
    make_payload,
    monkeypatch: pytest.MonkeyPatch,
):
    session_module = pytest.importorskip(
        "nat.runtime.session",
        reason="NAT is not installed in the base Fabric test environment",
    )
    per_user_builder_module = pytest.importorskip(
        "nat.builder.per_user_workflow_builder",
        reason="The installed NAT version does not support per-user workflows",
    )
    pytest.importorskip(
        "nat.plugins.langchain.agent.react_agent",
        reason="The NAT LangChain extra is not installed",
    )
    config = adapter.build_nat_config(
        make_payload(
            workflow=_fabric_workflow(
                "per_user_react_agent",
                llm_name="default",
            )
        )["config"]
    )
    shared_builder = MagicMock(name="shared-builder")
    workflows = [MagicMock(name="workflow-a"), MagicMock(name="workflow-b")]
    builders: list[MagicMock] = []
    for index, workflow in enumerate(workflows):
        builder = MagicMock(name=f"per-user-builder-{index}")
        builder.__aenter__ = AsyncMock(return_value=builder)
        builder.__aexit__ = AsyncMock(return_value=False)
        builder.populate_builder = AsyncMock()
        builder.build = AsyncMock(return_value=workflow)
        builders.append(builder)
    mock_per_user_builder = MagicMock(
        name="PerUserWorkflowBuilder",
        side_effect=builders,
    )
    monkeypatch.setattr(
        per_user_builder_module,
        "PerUserWorkflowBuilder",
        mock_per_user_builder,
    )

    manager = await session_module.SessionManager.create(
        config=config,
        shared_builder=shared_builder,
    )
    assert manager.is_workflow_per_user is True
    try:
        async with manager.session(user_id="user-a") as first_a:
            first_a_workflow = first_a.workflow
        async with manager.session(user_id="user-b") as first_b:
            first_b_workflow = first_b.workflow
        async with manager.session(user_id="user-a") as second_a:
            second_a_workflow = second_a.workflow
    finally:
        await manager.shutdown()

    assert first_a_workflow is workflows[0]
    assert first_b_workflow is workflows[1]
    assert second_a_workflow is workflows[0]
    assert mock_per_user_builder.call_args_list == [
        call(user_id="user-a", shared_builder=shared_builder),
        call(user_id="user-b", shared_builder=shared_builder),
    ]
    for builder in builders:
        builder.populate_builder.assert_awaited_once_with(config)
        builder.build.assert_awaited_once_with(entry_function=None)
        builder.__aexit__.assert_awaited_once_with(None, None, None)


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
        make_invocation_payload(
            request_id="message-1",
            context={"user_id": "user-1", "conversation_id": "conversation-1"},
        )
    )
    second = await runtime.invoke(
        make_invocation_payload(
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
        "mode": "agent_config",
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


async def test_per_user_runtime_forwards_identity_and_cleans_nat_before_builder(
    make_payload,
    make_invocation_payload,
    mock_nat,
):
    mock_nat["sessions"].is_workflow_per_user = True
    events: list[str] = []
    mock_nat["runner"].result = AsyncMock(
        side_effect=[{"answer": "a1"}, {"answer": "b1"}, {"answer": "a2"}]
    )
    mock_nat["sessions"].shutdown.side_effect = lambda: events.append("sessions")
    mock_nat["builder_context"].__aexit__.side_effect = lambda *_args: (
        events.append("builder") or False
    )
    runtime = adapter.NatRuntime()
    await runtime.start(
        make_payload(
            workflow=_fabric_workflow(
                "nat.plugins.langchain.agent.react_agent/per_user_react_agent"
            )
        )
    )

    results = [
        await runtime.invoke(
            make_invocation_payload(
                input_value=message,
                request_id=f"message-{index}",
                context={"user_id": user},
            )
        )
        for index, (user, message) in enumerate(
            [("user-a", "first"), ("user-b", "first"), ("user-a", "second")],
            start=1,
        )
    ]
    await runtime.stop()

    assert [result["response"] for result in results] == [
        {"answer": "a1"},
        {"answer": "b1"},
        {"answer": "a2"},
    ]
    assert mock_nat["sessions"].session.call_args_list == [
        call(user_id="user-a", user_message_id="message-1"),
        call(user_id="user-b", user_message_id="message-2"),
        call(user_id="user-a", user_message_id="message-3"),
    ]
    mock_nat["workflow_builder"].from_config.assert_called_once_with(
        config=mock_nat["typed_config"]
    )
    mock_nat["session_manager"].create.assert_awaited_once_with(
        config=mock_nat["typed_config"],
        shared_builder=mock_nat["builder"],
    )
    assert events == ["sessions", "builder"]


@pytest.mark.parametrize(
    "context",
    [
        None,
        {},
        {"user_id": ""},
        {"user_id": " \t "},
        {"user_id": 42},
    ],
)
async def test_per_user_runtime_rejects_missing_or_invalid_user_before_session(
    make_payload,
    make_invocation_payload,
    mock_nat,
    context: dict[str, Any] | None,
):
    mock_nat["sessions"].is_workflow_per_user = True
    runtime = adapter.NatRuntime()
    await runtime.start(make_payload(workflow=_fabric_workflow("tool_calling_agent")))
    try:
        result = await runtime.invoke(make_invocation_payload(context=context))
    finally:
        await runtime.stop()

    assert result["error"] == {
        "code": "nat_invalid_request",
        "message": (
            "NAT per-user workflow requires request.context.user_id "
            "as a non-empty string"
        ),
        "retryable": False,
    }
    mock_nat["sessions"].session.assert_not_called()


async def test_runtime_defaults_missing_per_user_metadata_to_shared(
    make_payload,
    make_invocation_payload,
    mock_nat,
):
    del mock_nat["sessions"].is_workflow_per_user
    runtime = adapter.NatRuntime()
    await runtime.start(make_payload(workflow=_fabric_workflow("per_user_react_agent")))
    try:
        result = await runtime.invoke(make_invocation_payload())
    finally:
        await runtime.stop()

    assert result["response"] == {"answer": 42}
    mock_nat["sessions"].session.assert_called_once_with(
        user_message_id="request-1"
    )


async def test_independent_fabric_runtimes_create_separate_nat_managers(
    make_payload,
    mock_nat,
):
    builders: list[MagicMock] = []
    builder_contexts: list[MagicMock] = []
    managers: list[MagicMock] = []
    for index in range(2):
        builder = MagicMock(name=f"builder-{index}")
        builder_context = MagicMock(name=f"builder-context-{index}")
        builder_context.__aenter__ = AsyncMock(return_value=builder)
        builder_context.__aexit__ = AsyncMock(return_value=False)
        manager = MagicMock(name=f"manager-{index}")
        manager.shutdown = AsyncMock()
        builders.append(builder)
        builder_contexts.append(builder_context)
        managers.append(manager)

    mock_nat["workflow_builder"].from_config.side_effect = builder_contexts
    mock_nat["session_manager"].create.side_effect = managers
    first_payload = make_payload(workflow=_fabric_workflow("per_user_react_agent"))
    second_payload = make_payload(workflow=_fabric_workflow("per_user_react_agent"))
    second_payload["runtime_context"]["runtime_id"] = "runtime-2"
    first_runtime = adapter.NatRuntime()
    second_runtime = adapter.NatRuntime()
    await first_runtime.start(first_payload)
    await second_runtime.start(second_payload)
    await first_runtime.stop()
    await second_runtime.stop()

    assert mock_nat["session_manager"].create.await_args_list == [
        call(config=mock_nat["typed_config"], shared_builder=builders[0]),
        call(config=mock_nat["typed_config"], shared_builder=builders[1]),
    ]
    managers[0].shutdown.assert_awaited_once_with()
    managers[1].shutdown.assert_awaited_once_with()


async def test_start_rejects_an_already_started_runtime(make_payload, mock_nat):
    runtime = adapter.NatRuntime()
    await runtime.start(make_payload())
    try:
        with pytest.raises(adapter.lifecycle.LifecycleError) as error:
            await runtime.start(make_payload())
    finally:
        await runtime.stop()

    assert error.value.code == "nat_runtime_already_started"
    assert error.value.message == "NAT runtime is already started"
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
        await runtime.invoke(make_invocation_payload())

    assert error.value.code == "nat_runtime_not_started"
    assert error.value.message == "NAT runtime is not started"


async def test_invoke_rejects_a_different_runtime_id(
    make_payload,
    make_invocation_payload,
    mock_nat,
):
    runtime = adapter.NatRuntime()
    await runtime.start(make_payload())
    try:
        with pytest.raises(adapter.lifecycle.LifecycleError) as error:
            await runtime.invoke(make_invocation_payload(runtime_id="runtime-2"))
    finally:
        await runtime.stop()

    assert error.value.code == "nat_runtime_mismatch"
    assert error.value.message == "NAT invocation does not match the active runtime"
    mock_nat["sessions"].session.assert_not_called()


async def test_invoke_normalizes_a_non_mapping_request(
    make_payload,
    make_invocation_payload,
    mock_nat,
):
    runtime = adapter.NatRuntime()
    await runtime.start(make_payload())
    try:
        result = await runtime.invoke(
            make_invocation_payload(raw_request=["not", "a", "mapping"])
        )
    finally:
        await runtime.stop()

    assert result["error"] == {
        "code": "nat_invalid_request",
        "message": "NAT invocation request must be a mapping",
        "retryable": False,
    }
    mock_nat["sessions"].session.assert_not_called()


async def test_invoke_normalizes_a_non_mapping_request_context(
    make_payload,
    make_invocation_payload,
    mock_nat,
):
    runtime = adapter.NatRuntime()
    await runtime.start(make_payload())
    try:
        result = await runtime.invoke(
            make_invocation_payload(context=["not", "a", "mapping"])
        )
    finally:
        await runtime.stop()

    assert result["error"] == {
        "code": "nat_invalid_request",
        "message": "NAT invocation request.context must be a mapping",
        "retryable": False,
    }
    mock_nat["sessions"].session.assert_not_called()


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
        "Fabric config could not be translated into a valid NAT config"
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
    assert error.value.message == "NAT runtime failed to stop cleanly"
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
        result = await runtime.invoke(make_invocation_payload())
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
    make_invocation_payload,
    mock_nat,
    caplog,
):
    mock_nat["runner"].result = AsyncMock(return_value=object())
    mock_nat["to_jsonable"].side_effect = TypeError("secret-object-repr")
    runtime = adapter.NatRuntime()
    await runtime.start(make_payload())
    try:
        result = await runtime.invoke(make_invocation_payload())
    finally:
        await runtime.stop()

    assert result["error"] == {
        "code": "nat_result_not_json_serializable",
        "message": "NAT workflow returned a result that cannot be represented as JSON",
        "retryable": False,
    }
    assert "secret-object-repr" not in json.dumps(result)
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
        "NAT MCP server 'calculator' requires a non-empty url"
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
