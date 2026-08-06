# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration shared by the native LangGraph examples.

This is application configuration for the examples, not a NeMo Fabric adapter
descriptor or a supported Fabric configuration format.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from typing import Literal

import yaml
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from pydantic import Field
from pydantic import HttpUrl
from pydantic import model_validator

NIM_OPENAI_BASE_URL = "https://integrate.api.nvidia.com/v1"


class NimModelConfig(BaseModel):
    """One NVIDIA NIM model binding used by a LangGraph example."""

    provider: Literal["nim"]
    model: str = Field(min_length=1)
    api_key_env: str = "NVIDIA_API_KEY"
    base_url: HttpUrl = NIM_OPENAI_BASE_URL
    temperature: float = 0.0
    max_tokens: int = Field(default=1024, gt=0)


class McpServerConfig(BaseModel):
    """The streamable HTTP MCP server exposed to the calculator workflow."""

    transport: Literal["streamable-http"]
    url: HttpUrl
    include: list[str] = Field(default_factory=list)


class McpConfig(BaseModel):
    """MCP servers available to a LangGraph example."""

    servers: dict[str, McpServerConfig] = Field(default_factory=dict)


class LocalToolConfig(BaseModel):
    """A local tool supplied by the workflow rather than an MCP server."""

    kind: Literal["local"]
    description: str = Field(min_length=1)


class WorkflowConfig(BaseModel):
    """Select a LangGraph graph and its workflow-owned settings."""

    entrypoint: Literal[
        "langgraph:per_user_react_agent", "langgraph:email_phishing_analyzer"
    ]
    settings: dict[str, Any] = Field(default_factory=dict)


class LangGraphExampleConfig(BaseModel):
    """Validated source configuration for the two native LangGraph examples."""

    models: dict[str, NimModelConfig] = Field(min_length=1)
    mcp: McpConfig | None = None
    tools: dict[str, LocalToolConfig] = Field(default_factory=dict)
    workflow: WorkflowConfig

    @model_validator(mode="after")
    def _validate_workflow_references(self) -> "LangGraphExampleConfig":
        llm_name = self.workflow.settings.get("llm_name")
        if not isinstance(llm_name, str) or llm_name not in self.models:
            raise ValueError("workflow.settings.llm_name must name a configured model")

        tool_names = self.workflow.settings.get("tool_names", [])
        if not isinstance(tool_names, list) or not all(
            isinstance(name, str) for name in tool_names
        ):
            raise ValueError("workflow.settings.tool_names must be a list of strings")

        available = set(self.tools)
        if self.mcp is not None:
            available.update(self.mcp.servers)
        unknown = set(tool_names) - available
        if unknown:
            raise ValueError(
                "workflow.settings.tool_names names unknown tool source(s): "
                + ", ".join(sorted(unknown))
            )
        return self

    def selected_model(self) -> NimModelConfig:
        """Return the NIM model selected by the workflow."""

        return self.models[self.workflow.settings["llm_name"]]


def load_config(path: str | Path) -> LangGraphExampleConfig:
    """Load and validate one LangGraph example YAML configuration file."""

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("LangGraph example configuration must be a mapping")
    return LangGraphExampleConfig.model_validate(raw)


def build_nim_chat_model(model: NimModelConfig) -> ChatOpenAI:
    """Build the OpenAI-compatible NIM chat-model binding for a graph."""

    api_key = os.environ.get(model.api_key_env)
    if not api_key:
        raise RuntimeError(
            f"Set {model.api_key_env} before running a workflow that uses {model.model}."
        )
    return ChatOpenAI(
        model=model.model,
        api_key=api_key,
        base_url=str(model.base_url),
        temperature=model.temperature,
        max_tokens=model.max_tokens,
    )
