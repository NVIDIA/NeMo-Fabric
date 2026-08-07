# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pydantic models for the southbound NeMo Fabric adapter contract."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any
from typing import Literal
from typing import Self

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import JsonValue
from pydantic import TypeAdapter
from pydantic import field_validator
from pydantic import model_validator


_EXTENSIONS_ADAPTER = TypeAdapter(dict[str, JsonValue])


def extension_schema(model: type[BaseModel]) -> dict[str, JsonValue]:
    """Return a JSON-safe schema for one descriptor extension point."""

    return _EXTENSIONS_ADAPTER.validate_python(model.model_json_schema(mode="validation"))


class AgentConfigBlock(BaseModel):
    """Base for explicitly extensible adapter-facing config blocks."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        allow_inf_nan=False,
    )

    extensions: dict[str, JsonValue] = Field(
        default_factory=dict,
        exclude_if=lambda value: not value,
        description="Adapter-owned fields validated by the selected adapter descriptor.",
    )

    def set_extensions(self, value: BaseModel | Mapping[str, Any]) -> Self:
        """Replace this block's adapter-owned extensions and return the block."""

        raw = (
            value.model_dump(mode="json", exclude_none=True)
            if isinstance(value, BaseModel)
            else value
        )
        self.extensions = _EXTENSIONS_ADAPTER.validate_python(raw)
        return self

    def to_mapping(self) -> dict[str, Any]:
        """Return a detached JSON-compatible adapter wire mapping."""

        return self.model_dump(mode="json", exclude_none=True)


class AgentHarnessConfig(AgentConfigBlock):
    """Adapter-owned target settings projected from the selected harness."""

    settings: dict[str, JsonValue] = Field(
        default_factory=dict,
        exclude_if=lambda value: not value,
    )


class AgentModelConfig(AgentConfigBlock):
    """Configuration for one named model role."""

    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    api_key_env: str | None = Field(default=None, min_length=1)
    temperature: float | None = None
    base_url: str | None = Field(default=None, min_length=1)
    settings: dict[str, JsonValue] = Field(
        default_factory=dict,
        exclude_if=lambda value: not value,
    )

    @field_validator("provider")
    @classmethod
    def _validate_provider(cls, value: str) -> str:
        if not value.strip() or value != value.strip() or value != value.lower():
            raise ValueError("provider must be a non-empty lowercase identifier")
        return value

    @field_validator("model", "api_key_env", "base_url")
    @classmethod
    def _validate_nonblank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("model fields must be non-empty strings")
        return value


class AgentInstructionConfig(AgentConfigBlock):
    """One normalized instruction value."""

    content: str = Field(min_length=1, pattern=r"\S")
    mode: Literal["replace"] = "replace"


class AgentInstructionsConfig(AgentConfigBlock):
    """Normalized instructions applied by the adapter target."""

    system: AgentInstructionConfig | None = None


class AgentRuntimeConfig(AgentConfigBlock):
    """Runtime behavior applied by the adapter target."""

    max_turns: int | None = Field(default=None, gt=0, le=(1 << 32) - 1)


class AgentSkillConfig(AgentConfigBlock):
    """Skill paths made available to the adapter target."""

    paths: list[str | Path] = Field(
        default_factory=list,
        exclude_if=lambda value: not value,
    )


class AgentMcpServerConfig(AgentConfigBlock):
    """One MCP server routed to the adapter target."""

    transport: str = Field(min_length=1, pattern=r"\S")
    url: str = Field(min_length=1, pattern=r"\S")
    args: list[str] = Field(default_factory=list, exclude_if=lambda value: not value)
    env: dict[str, str] = Field(
        default_factory=dict,
        exclude_if=lambda value: not value,
    )
    allowed_tools: list[str] | None = Field(
        default=None,
        description="Tool names to expose; None exposes all and an empty list exposes none.",
    )
    blocked_tools: list[str] = Field(
        default_factory=list,
        exclude_if=lambda value: not value,
    )

    @field_validator("allowed_tools", "blocked_tools")
    @classmethod
    def _validate_tool_names(cls, value: list[str] | None) -> list[str] | None:
        if value is not None and any(not tool.strip() for tool in value):
            raise ValueError("MCP tool names must not be empty")
        return value

    @model_validator(mode="after")
    def _validate_tool_policy(self) -> Self:
        if self.allowed_tools is not None:
            overlap = set(self.allowed_tools).intersection(self.blocked_tools)
            if overlap:
                name = sorted(overlap)[0]
                raise ValueError(f"MCP tool {name!r} cannot be both allowed and blocked")
        return self


class AgentMcpConfig(AgentConfigBlock):
    """Named MCP servers routed to the adapter target."""

    servers: dict[str, AgentMcpServerConfig] = Field(
        default_factory=dict,
        exclude_if=lambda value: not value,
    )


class AgentToolDefinition(AgentConfigBlock):
    """One named tool or tool-group definition resolved by the adapter."""

    kind: str = Field(min_length=1, pattern=r"\S")
    ref: str = Field(min_length=1, pattern=r"\S")
    settings: dict[str, JsonValue] = Field(
        default_factory=dict,
        exclude_if=lambda value: not value,
    )


class AgentToolsConfig(AgentConfigBlock):
    """Named tool definitions and effective target-level tool policy."""

    definitions: dict[str, AgentToolDefinition] = Field(
        default_factory=dict,
        exclude_if=lambda value: not value,
    )
    enabled: list[str] | None = Field(
        default=None,
        description="Named tools to expose; None preserves the adapter-target default.",
    )
    blocked: list[str] = Field(
        default_factory=list,
        exclude_if=lambda value: not value,
    )

    @field_validator("enabled", "blocked")
    @classmethod
    def _validate_tool_names(cls, value: list[str] | None) -> list[str] | None:
        if value is not None and any(not tool.strip() for tool in value):
            raise ValueError("tool names must not be empty")
        return value

    @model_validator(mode="after")
    def _validate_tool_policy(self) -> Self:
        if self.enabled is not None:
            overlap = set(self.enabled).intersection(self.blocked)
            if overlap:
                name = sorted(overlap)[0]
                raise ValueError(f"tool {name!r} cannot be both enabled and blocked")
        return self


class AgentWorkflowEntrypointConfig(AgentConfigBlock):
    """Adapter-declared resolution semantics for one custom agent or workflow."""

    kind: str = Field(min_length=1, pattern=r"\S")
    ref: str = Field(min_length=1, pattern=r"\S")


class AgentWorkflowConfig(AgentConfigBlock):
    """Custom agent or workflow selection and construction settings."""

    entrypoint: AgentWorkflowEntrypointConfig
    settings: dict[str, JsonValue] = Field(
        default_factory=dict,
        exclude_if=lambda value: not value,
    )


class AgentConfig(AgentConfigBlock):
    """Configuration projected southbound to one adapter target."""

    harness: AgentHarnessConfig | None = None
    models: dict[str, AgentModelConfig] = Field(
        default_factory=dict,
        exclude_if=lambda value: not value,
    )
    instructions: AgentInstructionsConfig | None = None
    runtime: AgentRuntimeConfig | None = None
    skills: AgentSkillConfig | None = None
    mcp: AgentMcpConfig | None = None
    tools: AgentToolsConfig | None = None
    workflow: AgentWorkflowConfig | None = None


__all__ = [
    "AgentConfig",
    "AgentConfigBlock",
    "AgentHarnessConfig",
    "AgentInstructionConfig",
    "AgentInstructionsConfig",
    "AgentMcpConfig",
    "AgentMcpServerConfig",
    "AgentModelConfig",
    "AgentRuntimeConfig",
    "AgentSkillConfig",
    "AgentToolDefinition",
    "AgentToolsConfig",
    "AgentWorkflowConfig",
    "AgentWorkflowEntrypointConfig",
    "extension_schema",
]
