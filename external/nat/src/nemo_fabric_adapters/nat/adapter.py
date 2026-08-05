#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NeMo Agent Toolkit adapter for NeMo Fabric.

The adapter builds one in-memory NAT configuration from Fabric's normalized
configuration and adapter-owned NAT component settings. One persistent adapter
host owns the resulting workflow for the complete Fabric runtime lifecycle.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import os
import shlex
from contextlib import AsyncExitStack
from typing import Any

from nemo_fabric_adapters.common import lifecycle
import nemo_fabric_adapters.common.utils as common_utils

HARNESS = "nat"
MODE = "nat_workflow"
FUNCTION_GROUP_SEPARATOR = "__"
NAT_SETTINGS_FIELDS = frozenset({"workflow", "functions", "function_groups"})
RESERVED_MODEL_SETTINGS = frozenset(
    {
        "_type",
        "type",
        "provider",
        "model",
        "model_name",
        "api_key",
        "api_key_env",
        "base_url",
        "temperature",
    }
)

LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Serve the persistent local-host lifecycle protocol."""

    lifecycle.serve(NatRuntime)


def _config_error(code: str, message: str, **metadata: Any) -> lifecycle.LifecycleError:
    return lifecycle.LifecycleError(code, message, metadata=metadata or None)


def _runtime_id(payload: dict[str, Any]) -> str:
    try:
        return common_utils.runtime_id(payload)
    except ValueError as error:
        raise _config_error(
            "nat_invalid_runtime_context",
            "NAT lifecycle payload is missing a runtime ID",
        ) from error


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _config_error(
            "nat_invalid_harness_settings",
            f"{field} must be a mapping",
            field=field,
        )
    return copy.deepcopy(value)


def _nat_component_settings(payload: dict[str, Any]) -> dict[str, Any]:
    settings = common_utils.settings_payload(payload)
    if not isinstance(settings, dict):
        raise _config_error(
            "nat_invalid_harness_settings",
            "harness.settings must be a mapping",
            field="harness.settings",
        )

    unknown = sorted(set(settings).difference(NAT_SETTINGS_FIELDS))
    if unknown:
        raise _config_error(
            "nat_invalid_harness_settings",
            "NAT harness settings contain unsupported top-level fields",
            fields=unknown,
        )

    workflow = _mapping(settings.get("workflow"), "harness.settings.workflow")
    workflow_type = workflow.get("_type")
    if not isinstance(workflow_type, str) or not workflow_type.strip():
        raise _config_error(
            "nat_invalid_harness_settings",
            "harness.settings.workflow._type must be a non-empty string",
            field="harness.settings.workflow._type",
        )
    if _workflow_type(workflow) == "react_agent":
        workflow.setdefault("tool_names", [])

    functions = settings.get("functions", {})
    function_groups = settings.get("function_groups", {})
    return {
        "workflow": workflow,
        "functions": _mapping(functions, "harness.settings.functions"),
        "function_groups": _mapping(
            function_groups,
            "harness.settings.function_groups",
        ),
    }


def _nat_llm_type(provider: str) -> str:
    # NAT calls NVIDIA's OpenAI-compatible provider ``nim``. Other provider
    # identifiers are already NAT component type names and are validated after
    # installed NAT plugins register their config objects.
    return "nim" if provider == "nvidia" else provider


def _nat_llms(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    models = common_utils.models_payload(payload)
    if not isinstance(models, dict):
        raise _config_error("nat_invalid_models", "Fabric models must be a mapping")

    llms: dict[str, dict[str, Any]] = {}
    for role, raw_model in models.items():
        if not isinstance(role, str) or not role:
            raise _config_error(
                "nat_invalid_models",
                "Fabric model role names must be non-empty strings",
            )
        if not isinstance(raw_model, dict):
            raise _config_error(
                "nat_invalid_models",
                f"Fabric model {role!r} must be a mapping",
                role=role,
            )

        provider = raw_model.get("provider")
        model_name = raw_model.get("model")
        if not isinstance(provider, str) or not provider:
            raise _config_error(
                "nat_invalid_models",
                f"Fabric model {role!r} requires a non-empty provider",
                role=role,
            )
        if not isinstance(model_name, str) or not model_name:
            raise _config_error(
                "nat_invalid_models",
                f"Fabric model {role!r} requires a non-empty model",
                role=role,
            )

        settings = raw_model.get("settings") or {}
        if not isinstance(settings, dict):
            raise _config_error(
                "nat_invalid_models",
                f"Fabric model {role!r} settings must be a mapping",
                role=role,
            )
        reserved = sorted(RESERVED_MODEL_SETTINGS.intersection(settings))
        if reserved:
            raise _config_error(
                "nat_model_settings_reserved",
                f"Fabric model {role!r} settings cannot replace normalized model fields",
                role=role,
                fields=reserved,
            )

        llm = copy.deepcopy(settings)
        llm.update(
            {
                "_type": _nat_llm_type(provider),
                "model_name": model_name,
            }
        )
        if raw_model.get("base_url") is not None:
            llm["base_url"] = raw_model["base_url"]
        if raw_model.get("temperature") is not None:
            llm["temperature"] = raw_model["temperature"]

        api_key_env = raw_model.get("api_key_env")
        if api_key_env is not None:
            if not isinstance(api_key_env, str) or not api_key_env:
                raise _config_error(
                    "nat_invalid_models",
                    f"Fabric model {role!r} api_key_env must be a non-empty string",
                    role=role,
                )
            api_key = os.environ.get(api_key_env)
            if not api_key:
                raise _config_error(
                    "nat_model_api_key_missing",
                    f"Fabric model {role!r} requires environment variable {api_key_env!r}",
                    role=role,
                    api_key_env=api_key_env,
                )
            llm["api_key"] = api_key

        llms[role] = llm
    return llms


def _workflow_type(workflow: dict[str, Any]) -> str:
    value = workflow.get("_type")
    return value.rsplit("/", 1)[-1] if isinstance(value, str) else ""


def _apply_system_instruction(config: dict[str, Any], payload: dict[str, Any]) -> None:
    instruction = common_utils.system_instruction(payload)
    if instruction is None:
        return

    workflow = config["workflow"]
    if _workflow_type(workflow) != "react_agent":
        raise _config_error(
            "nat_system_instruction_unsupported",
            "instructions.system is supported only for a NAT react_agent workflow",
            workflow_type=workflow.get("_type"),
        )
    if "additional_instructions" in workflow:
        raise _config_error(
            "nat_system_instruction_conflict",
            "instructions.system conflicts with harness.settings.workflow.additional_instructions",
            fields=[
                "instructions.system",
                "harness.settings.workflow.additional_instructions",
            ],
        )
    workflow["additional_instructions"] = instruction


def _string_list(value: Any, field: str, *, optional: bool = False) -> list[str] | None:
    if value is None and optional:
        return None
    if value is None:
        return []
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() or item != item.strip()
        for item in value
    ):
        raise _config_error(
            "nat_invalid_tool_policy",
            f"{field} must be a list of non-empty strings",
            field=field,
        )
    return list(dict.fromkeys(value))


def nat_mcp_server_config(name: str, server: Any) -> dict[str, Any]:
    """Translate one Fabric MCP server plan into a NAT server mapping."""

    if not isinstance(server, dict):
        raise _config_error(
            "nat_invalid_mcp_server",
            f"NAT MCP server {name!r} must be a mapping",
            server=name,
        )

    transport = str(server.get("transport") or "").strip().lower().replace("_", "-")
    target = os.path.expandvars(str(server.get("url") or "")).strip()
    if not target:
        raise _config_error(
            "nat_invalid_mcp_server",
            f"NAT MCP server {name!r} requires a non-empty url",
            server=name,
        )

    if transport == "stdio":
        try:
            command = shlex.split(target)
        except ValueError as error:
            raise _config_error(
                "nat_invalid_mcp_server",
                f"NAT MCP server {name!r} has an invalid stdio command",
                server=name,
            ) from error
        if not command:
            raise _config_error(
                "nat_invalid_mcp_server",
                f"NAT MCP server {name!r} has an empty stdio command",
                server=name,
            )
        result: dict[str, Any] = {
            "transport": "stdio",
            "command": command[0],
        }
        if command[1:]:
            result["args"] = command[1:]
        return result

    if transport in {"http", "streamablehttp"}:
        transport = "streamable-http"
    if transport not in {"sse", "streamable-http"}:
        raise _config_error(
            "nat_unsupported_mcp_transport",
            f"NAT MCP server {name!r} has unsupported transport {transport!r}",
            server=name,
            transport=transport,
        )
    return {"transport": transport, "url": target}


def _workflow_tool_names(config: dict[str, Any], reason: str) -> list[str]:
    tool_names = config["workflow"].get("tool_names")
    if not isinstance(tool_names, list) or any(
        not isinstance(name, str) or not name.strip() or name != name.strip()
        for name in tool_names
    ):
        raise _config_error(
            "nat_workflow_tools_unsupported",
            f"{reason} requires a NAT workflow with a string-list tool_names field",
            field="harness.settings.workflow.tool_names",
        )
    return tool_names


def _native_mcp_servers(payload: dict[str, Any]) -> dict[str, Any]:
    plan = common_utils.capability_plan(payload)
    if not isinstance(plan, dict):
        raise _config_error(
            "nat_invalid_capability_plan",
            "NAT capability plan must be a mapping",
        )
    native = plan.get("native", {})
    if not isinstance(native, dict):
        raise _config_error(
            "nat_invalid_capability_plan",
            "NAT native capability plan must be a mapping",
        )
    servers = native.get("mcp_servers", {})
    if not isinstance(servers, dict):
        raise _config_error(
            "nat_invalid_mcp_config",
            "NAT native MCP capability plan must be a mapping",
        )
    return servers


def _mcp_group(name: str, server: dict[str, Any]) -> dict[str, Any] | None:
    allowed = _string_list(
        server.get("allowed_tools"),
        f"capability_plan.native.mcp_servers.{name}.allowed_tools",
        optional=True,
    )
    blocked = _string_list(
        server.get("blocked_tools"),
        f"capability_plan.native.mcp_servers.{name}.blocked_tools",
    )
    assert blocked is not None

    if allowed is not None:
        overlap = sorted(set(allowed).intersection(blocked))
        if overlap:
            raise _config_error(
                "nat_invalid_mcp_tool_policy",
                f"NAT MCP server {name!r} cannot both allow and block a tool",
                server=name,
                tool=overlap[0],
            )
        if not allowed:
            return None

    group: dict[str, Any] = {
        "_type": "mcp_client",
        "server": nat_mcp_server_config(name, server),
    }
    # NAT forbids include and exclude together. A non-empty Fabric allowlist is
    # already the effective exposed set after the disjoint blocklist is applied.
    if allowed is not None:
        group["include"] = allowed
    elif blocked:
        group["exclude"] = blocked
    return group


def _apply_mcp_servers(config: dict[str, Any], payload: dict[str, Any]) -> set[str]:
    servers = _native_mcp_servers(payload)
    if not servers:
        return set()

    functions = config["functions"]
    function_groups = config["function_groups"]
    suppressed: set[str] = set()
    tool_names: list[str] | None = None

    for name, raw_server in sorted(servers.items()):
        if not isinstance(name, str) or not name.strip() or name != name.strip():
            raise _config_error(
                "nat_invalid_mcp_server",
                "NAT MCP server names must be non-empty strings",
            )
        if not isinstance(raw_server, dict):
            raise _config_error(
                "nat_invalid_mcp_server",
                f"NAT MCP server {name!r} must be a mapping",
                server=name,
            )

        if name in functions or name in function_groups:
            raise _config_error(
                "nat_mcp_name_conflict",
                f"NAT MCP server {name!r} conflicts with an existing function or function group",
                server=name,
            )

        group = _mcp_group(name, raw_server)
        if group is None:
            # An explicit empty allowlist means the server exposes no tools. It
            # must not remain reachable through a pre-existing workflow ref.
            existing_names = config["workflow"].get("tool_names")
            if isinstance(existing_names, list):
                existing_names[:] = [
                    tool_name for tool_name in existing_names if tool_name != name
                ]
            suppressed.add(name)
            continue

        if tool_names is None:
            tool_names = _workflow_tool_names(config, "Normalized MCP configuration")
        function_groups[name] = group
        if name not in tool_names:
            tool_names.append(name)

    return suppressed


def _group_member_identity(name: str) -> tuple[str, str] | None:
    if FUNCTION_GROUP_SEPARATOR not in name:
        return None
    group, member = name.split(FUNCTION_GROUP_SEPARATOR, 1)
    if not group or not member:
        return None
    return group, member


def _group_mapping(groups: dict[str, Any], name: str) -> dict[str, Any] | None:
    value = groups.get(name)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise _config_error(
            "nat_invalid_harness_settings",
            f"NAT function group {name!r} must be a mapping",
            group=name,
        )
    return value


def _unknown_tool_selector(selector: str) -> lifecycle.LifecycleError:
    return _config_error(
        "nat_unknown_tool_selector",
        f"Fabric tool selector {selector!r} does not match a configured NAT function or function group",
        selector=selector,
    )


def _validate_tool_selectors(
    config: dict[str, Any],
    selectors: list[str],
    suppressed: set[str],
    *,
    enabling: bool,
) -> None:
    functions = config["functions"]
    groups = config["function_groups"]
    for selector in selectors:
        if selector in suppressed:
            raise _unknown_tool_selector(selector)
        if selector in functions or selector in groups:
            continue
        member = _group_member_identity(selector)
        if member is None:
            raise _unknown_tool_selector(selector)
        group = _group_mapping(groups, member[0])
        if group is None:
            raise _unknown_tool_selector(selector)
        if not enabling:
            continue

        include = _string_list(group.get("include"), "function group include") or []
        exclude = _string_list(group.get("exclude"), "function group exclude") or []
        if (include and member[1] not in include) or member[1] in exclude:
            raise _unknown_tool_selector(selector)


def _select_group_members(requested: list[str]) -> list[str]:
    # Selector compatibility with existing include/exclude policy is validated
    # before mutation. Preserve caller order while removing duplicates.
    return list(dict.fromkeys(requested))


def _apply_enabled_tools(
    config: dict[str, Any], enabled: list[str], suppressed: set[str]
) -> None:
    functions = config["functions"]
    groups = config["function_groups"]
    selected_functions: dict[str, Any] = {}
    selected_groups: dict[str, Any] = {}
    selected_refs: list[str] = []

    requested_members: dict[str, list[str]] = {}
    for identity in enabled:
        if identity in functions or identity in groups:
            continue
        member = _group_member_identity(identity)
        if member is not None:
            requested_members.setdefault(member[0], []).append(member[1])

    for identity in enabled:
        if identity in suppressed:
            continue
        if identity in functions:
            selected_functions[identity] = functions[identity]
            if identity not in selected_refs:
                selected_refs.append(identity)
            continue

        if identity in groups:
            if identity in suppressed:
                continue
            selected_groups[identity] = groups[identity]
            if identity not in selected_refs:
                selected_refs.append(identity)
            continue

        member = _group_member_identity(identity)
        if member is None:
            continue
        group_name, _ = member
        if group_name in suppressed or group_name in selected_groups:
            continue
        group = _group_mapping(groups, group_name)
        if group is None:
            continue
        selected = _select_group_members(requested_members[group_name])
        if not selected:
            continue
        selected_group = copy.deepcopy(group)
        selected_group["include"] = selected
        selected_group.pop("exclude", None)
        selected_groups[group_name] = selected_group
        if group_name not in selected_refs:
            selected_refs.append(group_name)

    functions.clear()
    functions.update(selected_functions)
    groups.clear()
    groups.update(selected_groups)
    config["workflow"]["tool_names"] = selected_refs


def _block_group_member(
    config: dict[str, Any], group_name: str, member_name: str
) -> None:
    groups = config["function_groups"]
    group = _group_mapping(groups, group_name)
    if group is None:
        return

    include = _string_list(group.get("include"), "function group include") or []
    if include:
        remaining = [name for name in include if name != member_name]
        if remaining:
            group["include"] = remaining
            return
        groups.pop(group_name, None)
        config["workflow"]["tool_names"] = [
            name for name in config["workflow"]["tool_names"] if name != group_name
        ]
        return

    exclude = _string_list(group.get("exclude"), "function group exclude") or []
    if member_name not in exclude:
        group["exclude"] = [*exclude, member_name]


def _apply_blocked_tools(config: dict[str, Any], blocked: list[str]) -> None:
    functions = config["functions"]
    groups = config["function_groups"]
    tool_names = config["workflow"]["tool_names"]

    for identity in blocked:
        if identity in functions:
            functions.pop(identity, None)
            tool_names[:] = [name for name in tool_names if name != identity]
            continue
        if identity in groups:
            groups.pop(identity, None)
            tool_names[:] = [name for name in tool_names if name != identity]
            continue
        member = _group_member_identity(identity)
        if member is not None:
            _block_group_member(config, *member)


def _apply_tool_policy(
    config: dict[str, Any], payload: dict[str, Any], suppressed: set[str]
) -> None:
    tools = common_utils.tools_config(payload)
    enabled = _string_list(tools.get("enabled"), "config.tools.enabled", optional=True)
    blocked = _string_list(tools.get("blocked"), "config.tools.blocked")
    assert blocked is not None
    if enabled is None and not blocked:
        return

    _workflow_tool_names(config, "Normalized tool policy")
    if enabled is not None:
        _validate_tool_selectors(
            config,
            enabled,
            suppressed,
            enabling=True,
        )
    _validate_tool_selectors(
        config,
        blocked,
        suppressed,
        enabling=False,
    )
    if enabled is not None:
        _apply_enabled_tools(config, enabled, suppressed)
    else:
        config["workflow"]["tool_names"] = [
            name for name in config["workflow"]["tool_names"] if name not in suppressed
        ]
    if blocked:
        _apply_blocked_tools(config, blocked)


def apply_nat_capabilities(config: dict[str, Any], payload: dict[str, Any]) -> None:
    """Compile Fabric MCP routing and tool policy into a raw NAT config."""

    suppressed = _apply_mcp_servers(config, payload)
    _apply_tool_policy(config, payload, suppressed)


def build_nat_config_mapping(payload: dict[str, Any]) -> dict[str, Any]:
    """Build the raw in-memory NAT mapping from one Fabric start payload."""

    config = _nat_component_settings(payload)
    llms = _nat_llms(payload)
    if llms:
        config["llms"] = llms
    _apply_system_instruction(config, payload)
    apply_nat_capabilities(config, payload)
    return config


def build_nat_config(payload: dict[str, Any]) -> Any:
    """Build and validate a typed NAT Config without a workflow YAML file."""

    raw_config = build_nat_config_mapping(payload)

    try:
        from nat.runtime.loader import PluginTypes
        from nat.runtime.loader import discover_and_register_plugins

        discover_and_register_plugins(PluginTypes.CONFIG_OBJECT)

        from nat.data_models.config import Config

        return Config.model_validate(raw_config)
    except lifecycle.LifecycleError:
        raise
    except Exception as error:
        raise _config_error(
            "nat_config_translation_failed",
            "Fabric config could not be translated into a valid NAT config",
        ) from error


def _session_kwargs(request: dict[str, Any]) -> dict[str, str]:
    context = request.get("context") or {}
    if not isinstance(context, dict):
        raise ValueError("request.context must be a mapping")

    values = {
        "user_id": context.get("user_id"),
        "conversation_id": context.get("conversation_id"),
        "user_message_id": context.get("user_message_id") or request.get("request_id"),
    }
    result: dict[str, str] = {}
    for name, value in values.items():
        if value is None:
            continue
        if not isinstance(value, str) or not value:
            raise ValueError(f"request context {name} must be a non-empty string")
        result[name] = value
    return result


def _success_output(response: Any) -> dict[str, Any]:
    return {
        "harness": HARNESS,
        "adapter": "python",
        "mode": MODE,
        "response": response,
        "completed": True,
        "failed": False,
        "error": None,
    }


def _failure_output(code: str, message: str) -> dict[str, Any]:
    return {
        "harness": HARNESS,
        "adapter": "python",
        "mode": MODE,
        "response": None,
        "completed": False,
        "failed": True,
        "error": {
            "code": code,
            "message": message,
            "retryable": False,
        },
    }


async def _close_after_failed_start(stack: AsyncExitStack) -> None:
    try:
        await stack.aclose()
    except asyncio.CancelledError:
        raise
    except Exception as error:
        LOGGER.error(
            "NAT workflow cleanup failed after start error (error_type=%s)",
            type(error).__name__,
        )


class NatRuntime:
    """One NAT WorkflowBuilder and SessionManager owned by a Fabric runtime."""

    def __init__(self) -> None:
        self._runtime_id: str | None = None
        self._sessions: Any = None
        self._exit_stack: AsyncExitStack | None = None

    async def start(self, payload: dict[str, Any]) -> None:
        if self._exit_stack is not None:
            raise lifecycle.LifecycleError(
                "nat_runtime_already_started",
                "NAT runtime is already started",
            )

        runtime_id = _runtime_id(payload)
        stack = AsyncExitStack()
        try:
            from nat.builder.workflow_builder import WorkflowBuilder
            from nat.runtime.session import SessionManager

            config = build_nat_config(payload)
            builder = await stack.enter_async_context(
                WorkflowBuilder.from_config(config=config)
            )
            sessions = await SessionManager.create(
                config=config,
                shared_builder=builder,
            )
            stack.push_async_callback(sessions.shutdown)
        except asyncio.CancelledError:
            await _close_after_failed_start(stack)
            raise
        except lifecycle.LifecycleError:
            await _close_after_failed_start(stack)
            raise
        except Exception as error:
            await _close_after_failed_start(stack)
            raise lifecycle.LifecycleError(
                "nat_workflow_start_failed",
                "NAT workflow failed to load; inspect adapter stderr for details",
            ) from error

        self._runtime_id = runtime_id
        self._sessions = sessions
        self._exit_stack = stack

    async def invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._sessions is None or self._runtime_id is None:
            raise lifecycle.LifecycleError(
                "nat_runtime_not_started",
                "NAT runtime is not started",
            )
        if _runtime_id(payload) != self._runtime_id:
            raise lifecycle.LifecycleError(
                "nat_runtime_mismatch",
                "NAT invocation does not match the active runtime",
            )

        request = common_utils.request_payload(payload)
        if not isinstance(request, dict):
            return _failure_output(
                "nat_invalid_request",
                "NAT invocation request must be a mapping",
            )

        try:
            from nat.data_models.runtime_enum import RuntimeTypeEnum

            async with self._sessions.session(**_session_kwargs(request)) as session:
                async with session.run(
                    request.get("input", ""),
                    runtime_type=RuntimeTypeEnum.RUN_OR_SERVE,
                ) as runner:
                    result = await runner.result()
        except asyncio.CancelledError:
            raise
        except Exception as error:
            LOGGER.error(
                "NAT workflow invocation failed (error_type=%s)",
                type(error).__name__,
            )
            return _failure_output(
                "nat_workflow_invoke_failed",
                "NAT workflow invocation failed; inspect adapter stderr for details",
            )

        try:
            from pydantic_core import to_jsonable_python

            response = to_jsonable_python(result, serialize_unknown=False)
        except (TypeError, ValueError) as error:
            LOGGER.error(
                "NAT workflow returned a non-JSON result (error_type=%s)",
                type(error).__name__,
            )
            return _failure_output(
                "nat_result_not_json_serializable",
                "NAT workflow returned a result that cannot be represented as JSON",
            )
        return _success_output(response)

    async def stop(self) -> None:
        stack = self._exit_stack
        self._runtime_id = None
        self._sessions = None
        self._exit_stack = None

        if stack is None:
            return
        try:
            await stack.aclose()
        except asyncio.CancelledError:
            raise
        except Exception as error:
            raise lifecycle.LifecycleError(
                "nat_runtime_stop_failed",
                "NAT runtime failed to stop cleanly",
            ) from error


if __name__ == "__main__":
    main()
