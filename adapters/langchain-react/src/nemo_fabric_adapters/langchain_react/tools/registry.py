# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NAT langchain tool parity registry for Fabric ReAct agents."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import StructuredTool

from nemo_fabric_adapters.langchain_react.tools.calculator import build_calculator_tools
from nemo_fabric_adapters.langchain_react.tools.code_generation import build_code_generation_tool_from_spec
from nemo_fabric_adapters.langchain_react.tools.datetime_tools import build_current_datetime_tool
from nemo_fabric_adapters.langchain_react.tools.datetime_tools import build_current_timezone_tool
from nemo_fabric_adapters.langchain_react.tools.exa import build_exa_internet_search_tool
from nemo_fabric_adapters.langchain_react.tools.wiki import build_wiki_search_tool

TAVILY_MIGRATION_MESSAGE = (
    "`tavily_internet_search` was removed from NAT `nvidia-nat[langchain]` in NeMo Agent Toolkit 1.8. "
    "Use a Tavily function group or configure `kind: exa_internet_search` / `kind: wiki_search` instead."
)


@dataclass(frozen=True)
class ToolResolutionContext:
    timezone: str | None = None
    build_llm: Callable[[str], BaseChatModel] | None = None


def _tool_from_spec(name: str, spec: dict[str, Any], context: ToolResolutionContext) -> list[StructuredTool]:
    kind = str(spec.get("kind") or spec.get("_type") or name)
    timezone_name = str(spec.get("timezone", "Etc/UTC"))

    if kind in {"wiki_search", "wikipedia_search", "wiki"}:
        return [build_wiki_search_tool(max_results=int(spec.get("max_results", 2)), name=name)]

    if kind in {"current_datetime", "clock", "datetime"}:
        return [
            build_current_datetime_tool(
                timezone_name=timezone_name,
                context_timezone=context.timezone,
                name=name,
            )
        ]

    if kind in {"current_timezone", "timezone"}:
        return [
            build_current_timezone_tool(
                timezone_name=timezone_name,
                context_timezone=context.timezone,
                name=name,
            )
        ]

    if kind in {"function_group", "calculator"}:
        include = spec.get("include")
        if isinstance(include, list):
            return build_calculator_tools(include=[str(item) for item in include])
        return build_calculator_tools()

    if kind in {"exa_internet_search", "exa"}:
        return [
            build_exa_internet_search_tool(
                name=name,
                max_results=int(spec.get("max_results", 5)),
                api_key=spec.get("api_key"),
                api_key_env=str(spec.get("api_key_env", "EXA_API_KEY")),
                max_retries=int(spec.get("max_retries", 3)),
                search_type=spec.get("search_type", "auto"),
                livecrawl=spec.get("livecrawl", "fallback"),
                max_query_length=int(spec.get("max_query_length", 2000)),
                highlights=bool(spec.get("highlights", True)),
                max_content_length=spec.get("max_content_length", 10000),
            )
        ]

    if kind in {"code_generation", "code_gen"}:
        if context.build_llm is None:
            raise ValueError(f"Tool '{name}' requires a configured workflow LLM for code generation.")
        return [build_code_generation_tool_from_spec(spec, name=name, build_llm=context.build_llm)]

    if kind in {"tavily_internet_search", "tavily"}:
        raise RuntimeError(TAVILY_MIGRATION_MESSAGE)

    raise ValueError(f"Unsupported tool kind '{kind}' for tool '{name}'")


def resolve_tools(
    tool_names: list[str],
    tools_config: dict[str, Any],
    *,
    context: ToolResolutionContext | None = None,
) -> list[StructuredTool]:
    ctx = context or ToolResolutionContext()
    resolved: list[StructuredTool] = []
    for tool_name in tool_names:
        spec = tools_config.get(tool_name) or {}
        if not isinstance(spec, dict):
            raise ValueError(f"Tool '{tool_name}' must be configured as a mapping.")
        if not spec:
            spec = {"kind": tool_name}
        kind = str(spec.get("kind") or spec.get("_type") or tool_name)
        if kind in {"function_group", "calculator"}:
            resolved.extend(_tool_from_spec(tool_name, spec, ctx))
            continue
        resolved.extend(_tool_from_spec(tool_name, {**spec, "kind": kind}, ctx))
    if not resolved:
        raise ValueError("No tools resolved for the ReAct workflow.")
    return resolved
