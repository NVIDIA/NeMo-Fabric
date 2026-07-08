# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pydantic SDK models for NeMo Fabric configuration and requests.

The Rust core remains the source of truth for persisted schema snapshots. These
models provide the Python SDK's typed authoring surface and intentionally keep
extension fields so consumers can carry adapter- or application-owned data
without waiting for a schema release.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Self


class FabricBaseModel(BaseModel):
    """Base class for SDK-facing Pydantic models."""

    model_config = ConfigDict(
        extra="allow",
        validate_assignment=True,
        populate_by_name=True,
        use_enum_values=True,
        allow_inf_nan=False,
    )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> Self:
        """Validate a mapping using this Pydantic model."""

        return cls.model_validate(value)

    @property
    def extra_fields(self) -> dict[str, Any]:
        """Return fields preserved by the extension point for this model."""

        return dict(self.model_extra or {})

    def to_mapping(self) -> dict[str, Any]:
        """Return a detached JSON-compatible mapping for Rust/core calls."""

        data = self.model_dump(mode="json", exclude_none=True)
        return {key: item for key, item in data.items() if item not in ({}, [])}


class MetadataConfigModel(FabricBaseModel):
    """Human-readable agent identity."""

    name: str = Field(min_length=1)
    description: str | None = None


class HarnessConfigModel(FabricBaseModel):
    """Harness adapter selection plus adapter-owned settings."""

    adapter_id: str = Field(min_length=1)
    resolution: (
        Literal[
            "preinstalled",
            "image_provided",
            "pip_uv",
            "npm",
            "source",
            "service",
            "native_plugin",
        ]
        | None
    ) = None
    settings: dict[str, Any] = Field(default_factory=dict)


class RuntimeConfigModel(FabricBaseModel):
    """Runtime input/output contract."""

    input_schema: str | None = None
    output_schema: str | None = None
    artifacts: str | Path | None = None


class EnvironmentConfigModel(FabricBaseModel):
    """Execution environment metadata supplied by the consumer."""

    provider: str = Field(default="local", min_length=1)
    workspace: str | Path | None = None
    artifacts: str | Path | None = None
    settings: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    connection: dict[str, Any] = Field(default_factory=dict)
    ownership: Literal["caller_owned", "fabric_owned"] = "caller_owned"
    control_location: Literal["external_control", "in_env_control"] = "in_env_control"


class ModelConfigModel(FabricBaseModel):
    """Model alias configuration."""

    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    api_key_env: str | None = None
    temperature: float | None = None
    settings: dict[str, Any] = Field(default_factory=dict)


class SkillConfigModel(FabricBaseModel):
    """Skill capability configuration."""

    paths: list[str | Path] = Field(default_factory=list)

    def add_path(self, path: str | Path) -> Self:
        """Add a skill path if absent."""

        value = str(path)
        paths = [str(item) for item in self.paths]
        if value not in paths:
            self.paths = [*paths, value]
        return self

    def remove_path(self, path: str | Path) -> Self:
        """Remove a skill path if present."""

        value = str(path)
        self.paths = [item for item in self.paths if str(item) != value]
        return self


class McpServerConfigModel(FabricBaseModel):
    """MCP server configuration."""

    transport: str = Field(min_length=1)
    url: str = Field(min_length=1)
    exposure: Literal["harness_native", "fabric_managed"] = "harness_native"


class McpConfigModel(FabricBaseModel):
    """MCP capability configuration."""

    servers: dict[str, McpServerConfigModel] = Field(default_factory=dict)

    def add_server(
        self,
        name: str,
        *,
        transport: str,
        url: str,
        exposure: Literal["harness_native", "fabric_managed"] = "harness_native",
        extra_fields: Mapping[str, Any] | None = None,
    ) -> Self:
        """Add or replace a named MCP server."""

        self.servers[name] = McpServerConfigModel(
            transport=transport,
            url=url,
            exposure=exposure,
            **dict(extra_fields or {}),
        )
        return self

    def remove_server(self, name: str) -> Self:
        """Remove a named MCP server if present."""

        self.servers.pop(name, None)
        return self


class TelemetryConfigModel(FabricBaseModel):
    """Telemetry configuration."""

    enabled: bool = False
    provider: Literal["relay", "native"] | None = None
    project: str | None = None
    output_dir: str | Path | None = None
    config: dict[str, Any] | None = None

    def enable_relay(
        self,
        *,
        project: str | None = None,
        output_dir: str | Path | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> Self:
        """Enable NeMo Relay telemetry for subsequently started runtimes."""

        self.enabled = True
        self.provider = "relay"
        if project is not None:
            self.project = project
        if output_dir is not None:
            self.output_dir = output_dir
        if config is not None:
            self.config = dict(config)
        return self

    def enable_native(self) -> Self:
        """Let the selected adapter handle telemetry natively."""

        self.enabled = True
        self.provider = "native"
        return self

    def disable(self) -> Self:
        """Disable telemetry."""

        self.enabled = False
        return self


class ProfileRegistryConfigModel(FabricBaseModel):
    """Profile discovery config for portable file-backed agent packages."""

    directories: list[str | Path] = Field(default_factory=list)


class FabricConfigModel(FabricBaseModel):
    """SDK-facing typed Fabric agent configuration."""

    schema_version: str = "fabric.agent/v1alpha1"
    metadata: MetadataConfigModel
    harness: HarnessConfigModel
    runtime: RuntimeConfigModel = Field(default_factory=RuntimeConfigModel)
    environment: EnvironmentConfigModel | None = None
    models: dict[str, ModelConfigModel | dict[str, Any]] = Field(default_factory=dict)
    mcp: McpConfigModel | None = None
    skills: SkillConfigModel | None = None
    telemetry: TelemetryConfigModel | None = None
    profiles: ProfileRegistryConfigModel | dict[str, Any] | None = None
    tools: Any = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> Self:
        """Validate the public agent config mapping shape."""

        return cls.model_validate(value)

    def to_mapping(self) -> dict[str, Any]:
        """Return a detached mapping matching the Rust ``FabricConfig`` schema."""

        data = super().to_mapping()
        data.setdefault("schema_version", "fabric.agent/v1alpha1")
        data.setdefault("runtime", {})
        return data

    def add_mcp_server(
        self,
        name: str,
        *,
        transport: str,
        url: str,
        exposure: Literal["harness_native", "fabric_managed"] = "harness_native",
        extra_fields: Mapping[str, Any] | None = None,
    ) -> Self:
        """Add or replace a named MCP server and return this config."""

        if self.mcp is None:
            self.mcp = McpConfigModel()
        self.mcp.add_server(
            name,
            transport=transport,
            url=url,
            exposure=exposure,
            extra_fields=extra_fields,
        )
        return self

    def add_skill_path(self, path: str | Path) -> Self:
        """Add a skill path and return this config."""

        if self.skills is None:
            self.skills = SkillConfigModel()
        self.skills.add_path(path)
        return self

    def enable_relay(
        self,
        *,
        project: str | None = None,
        output_dir: str | Path | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> Self:
        """Enable NeMo Relay telemetry and return this config."""

        if self.telemetry is None:
            self.telemetry = TelemetryConfigModel()
        self.telemetry.enable_relay(
            project=project,
            output_dir=output_dir,
            config=config,
        )
        return self


class FabricProfileConfigModel(FabricBaseModel):
    """Typed profile overlay used when a Python caller wants file-style overlays."""

    schema_version: str = "fabric.profile/v1alpha1"
    name: str = Field(min_length=1)
    description: str | None = None
    harness: HarnessConfigModel | dict[str, Any] | None = None
    runtime: RuntimeConfigModel | dict[str, Any] | None = None
    environment: EnvironmentConfigModel | dict[str, Any] | None = None
    models: dict[str, ModelConfigModel | dict[str, Any]] | None = None
    mcp: McpConfigModel | dict[str, Any] | None = None
    skills: SkillConfigModel | dict[str, Any] | None = None
    telemetry: TelemetryConfigModel | dict[str, Any] | None = None
    tools: Any = None


class RunRequestModel(FabricBaseModel):
    """Pydantic authoring model for one Fabric invocation request."""

    input: Any = None
    request_id: str | None = None
    context: dict[str, Any] = Field(default_factory=dict)
    overrides: dict[str, Any] | None = None

    def to_mapping(self) -> dict[str, Any]:
        """Return a request mapping; runtime wrappers fill generated ids."""

        data = super().to_mapping()
        if "input" not in data:
            data["input"] = ""
        return data
