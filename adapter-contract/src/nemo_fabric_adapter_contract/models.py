# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pydantic models for the southbound NeMo Fabric adapter contract."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
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


class ContractModel(BaseModel):
    """Base for adapter-facing contract models."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        allow_inf_nan=False,
    )


class AgentContractBlock(ContractModel):
    """Base for explicitly extensible adapter-owned contract blocks."""

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

        return self.model_dump(mode="json")


AgentConfigBlock = AgentContractBlock


class AgentHarnessConfig(AgentContractBlock):
    """Adapter-owned target settings projected from the selected harness."""

    settings: dict[str, JsonValue] = Field(
        default_factory=dict,
        exclude_if=lambda value: not value,
    )


class AgentModelConfig(AgentContractBlock):
    """Configuration for one named model role."""

    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    api_key_env: str | None = Field(
        default=None,
        min_length=1,
        exclude_if=lambda value: value is None,
    )
    temperature: float | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    base_url: str | None = Field(
        default=None,
        min_length=1,
        exclude_if=lambda value: value is None,
    )
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


class AgentInstructionConfig(AgentContractBlock):
    """One normalized instruction value."""

    content: str = Field(min_length=1, pattern=r"\S")
    mode: Literal["replace"] = "replace"


class AgentInstructionsConfig(AgentContractBlock):
    """Normalized instructions applied by the adapter target."""

    system: AgentInstructionConfig | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )


class AgentRuntimeConfig(AgentContractBlock):
    """Runtime behavior applied by the adapter target."""

    max_turns: int | None = Field(
        default=None,
        gt=0,
        le=(1 << 32) - 1,
        exclude_if=lambda value: value is None,
    )


class AgentSkillConfig(AgentContractBlock):
    """Skill paths made available to the adapter target."""

    paths: list[str | Path] = Field(
        default_factory=list,
        exclude_if=lambda value: not value,
    )


class AgentMcpServerConfig(AgentContractBlock):
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
        exclude_if=lambda value: value is None,
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


class AgentMcpConfig(AgentContractBlock):
    """Named MCP servers routed to the adapter target."""

    servers: dict[str, AgentMcpServerConfig] = Field(
        default_factory=dict,
        exclude_if=lambda value: not value,
    )


class AgentToolDefinition(AgentContractBlock):
    """One named tool or tool-group definition resolved by the adapter."""

    kind: str = Field(min_length=1, pattern=r"\S")
    ref: str = Field(min_length=1, pattern=r"\S")
    settings: dict[str, JsonValue] = Field(
        default_factory=dict,
        exclude_if=lambda value: not value,
    )


class AgentToolsConfig(AgentContractBlock):
    """Named tool definitions and effective target-level tool policy."""

    definitions: dict[str, AgentToolDefinition] = Field(
        default_factory=dict,
        exclude_if=lambda value: not value,
    )
    enabled: list[str] | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
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


class AgentWorkflowEntrypointConfig(AgentContractBlock):
    """Adapter-declared resolution semantics for one custom agent or workflow."""

    kind: str = Field(min_length=1, pattern=r"\S")
    ref: str = Field(min_length=1, pattern=r"\S")


class AgentWorkflowConfig(AgentContractBlock):
    """Custom agent or workflow selection and construction settings."""

    entrypoint: AgentWorkflowEntrypointConfig
    settings: dict[str, JsonValue] = Field(
        default_factory=dict,
        exclude_if=lambda value: not value,
    )


class AgentConfig(AgentContractBlock):
    """Configuration projected southbound to one adapter target."""

    harness: AgentHarnessConfig | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    models: dict[str, AgentModelConfig] = Field(
        default_factory=dict,
        exclude_if=lambda value: not value,
    )
    instructions: AgentInstructionsConfig | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    runtime: AgentRuntimeConfig | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    skills: AgentSkillConfig | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    mcp: AgentMcpConfig | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    tools: AgentToolsConfig | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    workflow: AgentWorkflowConfig | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )


class AgentRunRequest(AgentContractBlock):
    """One invocation request projected southbound to an adapter target."""

    input: JsonValue
    context: dict[str, JsonValue] = Field(
        default_factory=dict,
        exclude_if=lambda value: not value,
    )


class AgentRunStatus(StrEnum):
    """Completion status reported by an adapter target."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentRunError(AgentContractBlock):
    """Error reported by an adapter target."""

    code: str = Field(min_length=1, pattern=r"\S")
    message: str = Field(min_length=1, pattern=r"\S")
    retryable: bool = False


class AgentArtifact(AgentContractBlock):
    """One artifact produced by an adapter target."""

    name: str = Field(min_length=1, pattern=r"\S")
    kind: str = Field(min_length=1, pattern=r"\S")
    path: str | Path
    media_type: str | None = Field(
        default=None,
        min_length=1,
        pattern=r"\S",
        exclude_if=lambda value: value is None,
    )

    @field_validator("path", mode="before")
    @classmethod
    def _validate_path(cls, value: str | Path) -> str | Path:
        raw = str(value)
        path = Path(raw)
        components = raw.replace("\\", "/").split("/")
        windows_drive_path = len(raw) >= 2 and raw[0].isalpha() and raw[1] == ":"
        if (
            not raw
            or path.is_absolute()
            or raw.startswith(("/", "\\"))
            or windows_drive_path
            or ".." in components
        ):
            raise ValueError(
                "artifact path must be non-empty, relative, and contain no parent traversal"
            )
        return value


class AgentUsage(AgentContractBlock):
    """Normalized model usage reported by an adapter target."""

    input_tokens: int | None = Field(
        default=None,
        ge=0,
        le=(1 << 64) - 1,
        exclude_if=lambda value: value is None,
    )
    output_tokens: int | None = Field(
        default=None,
        ge=0,
        le=(1 << 64) - 1,
        exclude_if=lambda value: value is None,
    )
    total_tokens: int | None = Field(
        default=None,
        ge=0,
        le=(1 << 64) - 1,
        exclude_if=lambda value: value is None,
    )
    cost_usd: float | None = Field(
        default=None,
        ge=0,
        exclude_if=lambda value: value is None,
    )


class AgentRunResult(AgentContractBlock):
    """Terminal result returned by an adapter target."""

    status: AgentRunStatus
    output: JsonValue
    error: AgentRunError | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    usage: AgentUsage | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    artifacts: list[AgentArtifact] = Field(
        default_factory=list,
        exclude_if=lambda value: not value,
    )

    @model_validator(mode="after")
    def _validate_status_and_error(self) -> Self:
        if self.status is AgentRunStatus.FAILED and self.error is None:
            raise ValueError("failed result requires an error")
        if self.status is AgentRunStatus.SUCCEEDED and self.error is not None:
            raise ValueError("succeeded result must not include an error")
        return self


class ControlLocation(StrEnum):
    """Where Fabric control code runs relative to the task environment."""

    EXTERNAL_CONTROL = "external_control"
    IN_ENV_CONTROL = "in_env_control"


class EnvironmentOwnership(StrEnum):
    """Whether Fabric owns the underlying environment resource."""

    CALLER_OWNED = "caller_owned"
    FABRIC_OWNED = "fabric_owned"


class EnvironmentHandle(ContractModel):
    """Resolved execution environment visible to an adapter target."""

    environment_id: str = Field(min_length=1, pattern=r"\S")
    provider: str = Field(min_length=1, pattern=r"\S")
    control_location: ControlLocation
    workspace: str | Path | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    artifacts: str | Path | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    env: dict[str, str] = Field(
        default_factory=dict,
        exclude_if=lambda value: not value,
    )
    ownership: EnvironmentOwnership
    connection: dict[str, JsonValue] = Field(
        default_factory=dict,
        exclude_if=lambda value: not value,
    )
    metadata: dict[str, JsonValue] = Field(
        default_factory=dict,
        exclude_if=lambda value: not value,
    )


class ArtifactRef(ContractModel):
    """Reference to one artifact visible through RuntimeContext."""

    name: str = Field(min_length=1, pattern=r"\S")
    kind: str = Field(min_length=1, pattern=r"\S")
    path: str | Path
    media_type: str | None = Field(
        default=None,
        min_length=1,
        pattern=r"\S",
        exclude_if=lambda value: value is None,
    )


class ArtifactManifest(ContractModel):
    """Artifacts visible to an adapter at invocation start."""

    root: str | Path | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    artifacts: list[ArtifactRef] = Field(
        default_factory=list,
        exclude_if=lambda value: not value,
    )


class RuntimeTelemetryContext(ContractModel):
    """Telemetry configuration generated for one adapter invocation."""

    relay_enabled: bool
    config_path: str | Path | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    env: dict[str, str] = Field(
        default_factory=dict,
        exclude_if=lambda value: not value,
    )
    metadata: dict[str, JsonValue] = Field(
        default_factory=dict,
        exclude_if=lambda value: not value,
    )


class RuntimeContext(ContractModel):
    """Fabric-generated context for one adapter invocation."""

    runtime_id: str = Field(min_length=1, pattern=r"\S")
    invocation_id: str = Field(min_length=1, pattern=r"\S")
    request_id: str = Field(min_length=1, pattern=r"\S")
    environment: EnvironmentHandle
    artifacts: ArtifactManifest
    telemetry: RuntimeTelemetryContext | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
