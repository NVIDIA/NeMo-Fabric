# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pydantic SDK models for NeMo Fabric configuration and requests.

The Rust core remains the source of truth for persisted schema snapshots. These
models provide the Python SDK's typed authoring surface and intentionally keep
extension fields so consumers can carry adapter- or application-owned data
without waiting for a schema release.
"""

from __future__ import annotations

import math
import uuid
from collections.abc import Mapping
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated
from typing import Any
from typing import Literal
from typing import Self

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator


def _json_value(value: Any, name: str) -> Any:
    """Validate and detach a JSON-compatible value."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} must contain only finite JSON numbers")
        return value
    if isinstance(value, list):
        return [_json_value(item, name) for item in value]
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{name} JSON object keys must be strings")
            result[key] = _json_value(item, name)
        return result
    raise ValueError(f"{name} must be JSON-compatible")


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


class MetadataConfig(FabricBaseModel):
    """Human-readable agent identity."""

    name: str = Field(min_length=1)
    description: str | None = None


class HarnessConfig(FabricBaseModel):
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


class RuntimeConfig(FabricBaseModel):
    """Runtime input/output contract."""

    input_schema: str | None = None
    output_schema: str | None = None
    artifacts: str | Path | None = None


class EnvironmentConfig(FabricBaseModel):
    """Execution environment configuration supplied by the consumer.

    ``provider`` selects the environment implementation. ``workspace`` is the
    path visible to the harness, while ``artifacts`` is the provider-specific
    output location. ``settings`` configures the selected provider;
    ``connection`` describes how Fabric reaches an existing environment; and
    ``metadata`` carries consumer-owned values that Fabric does not interpret.
    ``ownership`` identifies who tears the environment down, and
    ``control_location`` identifies whether Fabric control code runs inside or
    outside it.
    """

    provider: str = Field(
        default="local",
        min_length=1,
        description="Environment provider, such as local, docker, opensandbox, or k8s.",
    )
    workspace: str | Path | None = Field(
        default=None,
        description="Workspace path visible to the harness.",
    )
    artifacts: str | Path | None = Field(
        default=None,
        description="Environment-specific artifact path.",
    )
    settings: dict[str, Any] = Field(
        default_factory=dict,
        description="Provider-specific configuration interpreted by the environment provider.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Consumer-owned environment metadata passed through without Fabric semantics.",
    )
    connection: dict[str, Any] = Field(
        default_factory=dict,
        description="Connection data for an existing environment, such as URL, namespace, or credential reference.",
    )
    ownership: Literal["caller_owned", "fabric_owned"] = Field(
        default="caller_owned",
        description="Whether the caller or Fabric owns environment teardown.",
    )
    control_location: Literal["external_control", "in_env_control"] = Field(
        default="in_env_control",
        description="Whether Fabric control code runs outside or inside the environment.",
    )


class ModelConfig(FabricBaseModel):
    """Model alias configuration."""

    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    api_key_env: str | None = None
    temperature: float | None = None
    settings: dict[str, Any] = Field(default_factory=dict)


class SkillConfig(FabricBaseModel):
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


class McpServerConfig(FabricBaseModel):
    """MCP server configuration."""

    transport: str = Field(min_length=1)
    url: str = Field(min_length=1)
    exposure: Literal["harness_native", "fabric_managed"] = "harness_native"


class McpConfig(FabricBaseModel):
    """MCP capability configuration."""

    servers: dict[str, McpServerConfig] = Field(default_factory=dict)

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

        self.servers[name] = McpServerConfig(
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


class RelayConfigPolicy(FabricBaseModel):
    """NeMo Relay config validation policy."""

    unknown_component: Literal["ignore", "warn", "error"] = "warn"
    unknown_field: Literal["ignore", "warn", "error"] = "warn"
    unsupported_value: Literal["ignore", "warn", "error"] = "error"


class RelayAtofEndpointConfig(FabricBaseModel):
    """NeMo Relay ATOF remote endpoint configuration."""

    url: str
    transport: Literal["http_post", "websocket", "ndjson"] = "http_post"
    headers: dict[str, str] = Field(default_factory=dict)
    timeout_millis: int = 3000
    field_name_policy: Literal["preserve", "replace_dots"] = "preserve"


class RelayAtofConfig(FabricBaseModel):
    """NeMo Relay ATOF export configuration."""

    enabled: bool = False
    output_directory: str | Path | None = None
    filename: str | None = None
    mode: Literal["append", "overwrite"] = "append"
    endpoints: list[RelayAtofEndpointConfig | dict[str, Any]] | None = None


class RelayS3StorageConfig(FabricBaseModel):
    """NeMo Relay ATIF S3 storage configuration."""

    type: Literal["s3"] = "s3"
    bucket: str = ""
    key_prefix: str | None = None
    access_key_id: str | None = None
    secret_access_key_var: str | None = None
    session_token_var: str | None = None
    region: str | None = None
    endpoint_url: str | None = None
    allow_http: bool | None = None


class RelayHttpStorageConfig(FabricBaseModel):
    """NeMo Relay ATIF HTTP storage configuration."""

    type: Literal["http"] = "http"
    endpoint: str = ""
    headers: dict[str, str] = Field(default_factory=dict)
    header_env: dict[str, str] = Field(default_factory=dict)
    timeout_millis: int = 3000


class RelayAtifConfig(FabricBaseModel):
    """NeMo Relay ATIF export configuration."""

    enabled: bool = False
    agent_name: str = "NeMo Relay"
    agent_version: str | None = None
    model_name: str = "unknown"
    tool_definitions: list[dict[str, Any]] | None = None
    extra: dict[str, Any] | None = None
    output_directory: str | Path | None = None
    filename_template: str = "nemo-relay-atif-{session_id}.json"
    storage: (
        list[
            Annotated[
                RelayS3StorageConfig | RelayHttpStorageConfig,
                Field(discriminator="type"),
            ]
            | dict[str, Any]
        ]
        | None
    ) = None


class RelayOtlpConfig(FabricBaseModel):
    """NeMo Relay OTLP export configuration for OpenTelemetry/OpenInference."""

    enabled: bool = False
    transport: Literal["http_binary", "grpc"] = "http_binary"
    endpoint: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    resource_attributes: dict[str, str] = Field(default_factory=dict)
    service_name: str = "nemo-relay"
    service_namespace: str | None = None
    service_version: str | None = None
    instrumentation_scope: str | None = None
    timeout_millis: int = 3000


class RelayObservabilityConfig(FabricBaseModel):
    """NeMo Relay observability component configuration."""

    version: int = 1
    atof: RelayAtofConfig | dict[str, Any] | None = None
    atif: RelayAtifConfig | dict[str, Any] | None = None
    opentelemetry: RelayOtlpConfig | dict[str, Any] | None = None
    openinference: RelayOtlpConfig | dict[str, Any] | None = None
    policy: RelayConfigPolicy | dict[str, Any] | None = None


class RelayComponentConfig(FabricBaseModel):
    """Generic NeMo Relay plugin component configuration."""

    kind: str = Field(min_length=1)
    enabled: bool = True
    config: dict[str, Any] = Field(default_factory=dict)


class RelayConfig(FabricBaseModel):
    """First-class NeMo Relay integration configuration."""

    project: str | None = None
    output_dir: str | Path | None = None
    observability: RelayObservabilityConfig | dict[str, Any] | None = None
    components: list[RelayComponentConfig | dict[str, Any]] = Field(default_factory=list)
    policy: RelayConfigPolicy | dict[str, Any] | None = None


class TelemetryProviderConfig(FabricBaseModel):
    """Provider-specific telemetry configuration."""

    config: dict[str, Any] | None = None


class TelemetryConfig(FabricBaseModel):
    """Telemetry configuration."""

    providers: dict[Literal["relay", "native"], TelemetryProviderConfig | dict[str, Any]] = Field(default_factory=dict)

    def enable_relay(
        self,
    ) -> Self:
        """Enable NeMo Relay telemetry for subsequently started runtimes."""

        self.providers["relay"] = TelemetryProviderConfig()
        return self

    def enable_native(self, *, config: Mapping[str, Any] | None = None) -> Self:
        """Let the selected adapter handle telemetry natively."""

        provider_config = self.providers.get("native", TelemetryProviderConfig())
        if not isinstance(provider_config, TelemetryProviderConfig):
            provider_config = TelemetryProviderConfig.model_validate(provider_config)
        if config is not None:
            provider_config.config = dict(config)
        self.providers["native"] = provider_config
        return self

    def remove_provider(self, provider: Literal["relay", "native"]) -> Self:
        """Remove a configured telemetry provider."""

        self.providers.pop(provider, None)
        return self


class ToolsConfig(FabricBaseModel):
    """Harness-neutral tool capability configuration."""

    blocked: list[str] = Field(default_factory=list)


class FabricConfig(FabricBaseModel):
    """SDK-facing typed Fabric agent configuration."""

    schema_version: str = "fabric.agent/v1alpha1"
    metadata: MetadataConfig
    harness: HarnessConfig
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    environment: EnvironmentConfig | None = None
    models: dict[str, ModelConfig | dict[str, Any]] = Field(default_factory=dict)
    mcp: McpConfig | None = None
    skills: SkillConfig | None = None
    telemetry: TelemetryConfig | None = None
    relay: RelayConfig | dict[str, Any] | None = None
    tools: ToolsConfig | dict[str, Any] | None = None

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
            self.mcp = McpConfig()
        self.mcp.add_server(
            name,
            transport=transport,
            url=url,
            exposure=exposure,
            extra_fields=extra_fields,
        )
        return self

    def remove_mcp_server(self, name: str) -> Self:
        """Remove a named MCP server and return this config."""

        if self.mcp is not None:
            self.mcp.remove_server(name)
            if not self.mcp.servers:
                self.mcp = None
        return self

    def add_skill_path(self, path: str | Path) -> Self:
        """Add a skill path and return this config."""

        if self.skills is None:
            self.skills = SkillConfig()
        self.skills.add_path(path)
        return self

    def remove_skill_path(self, path: str | Path) -> Self:
        """Remove a skill path and return this config."""

        if self.skills is not None:
            self.skills.remove_path(path)
            if not self.skills.paths:
                self.skills = None
        return self

    def block_tools(self, *tools: str) -> Self:
        """Block adapter-native tool names or toolsets and return this config."""

        if self.tools is None or isinstance(self.tools, dict):
            self.tools = ToolsConfig.model_validate(self.tools or {})
        existing = list(self.tools.blocked)
        for tool in tools:
            if tool not in existing:
                existing.append(tool)
        self.tools.blocked = existing
        return self

    def enable_relay(
        self,
        *,
        project: str | None = None,
        output_dir: str | Path | None = None,
        observability: RelayObservabilityConfig | Mapping[str, Any] | None = None,
        components: Sequence[RelayComponentConfig | Mapping[str, Any]] | None = None,
        policy: RelayConfigPolicy | Mapping[str, Any] | None = None,
    ) -> Self:
        """Enable NeMo Relay telemetry and return this config."""

        if self.telemetry is None:
            self.telemetry = TelemetryConfig()
        self.telemetry.enable_relay()
        if self.relay is None:
            relay = RelayConfig()
        elif isinstance(self.relay, RelayConfig):
            relay = self.relay.model_copy(deep=True)
        else:
            relay = RelayConfig.from_mapping(self.relay)
        if project is not None:
            relay.project = project
        if output_dir is not None:
            relay.output_dir = output_dir
        if observability is not None:
            relay.observability = (
                observability if isinstance(observability, RelayObservabilityConfig) else dict(observability)
            )
        if components is not None:
            relay.components = [item if isinstance(item, RelayComponentConfig) else dict(item) for item in components]
        if policy is not None:
            relay.policy = policy if isinstance(policy, RelayConfigPolicy) else dict(policy)
        self.relay = relay
        return self


class RunRequest(FabricBaseModel):
    """One validated Fabric invocation request."""

    input: Any = ""
    request_id: str = Field(
        default_factory=lambda: f"request-{uuid.uuid4().hex}",
        min_length=1,
    )
    context: dict[str, Any] = Field(default_factory=dict)
    overrides: dict[str, Any] | None = None

    @field_validator("input", mode="before")
    @classmethod
    def _validate_input(cls, value: Any) -> Any:
        return _json_value("" if value is None else value, "request input")

    @field_validator("context", mode="before")
    @classmethod
    def _validate_context(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            raise ValueError("request context must be a JSON object")
        return _json_value(value, "request context")

    @field_validator("overrides", mode="before")
    @classmethod
    def _validate_overrides(cls, value: Any) -> Any:
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise ValueError("request overrides must be a JSON object")
        return _json_value(value, "request overrides")

    @model_validator(mode="after")
    def _validate_extensions(self) -> Self:
        for name, value in (self.model_extra or {}).items():
            _json_value(value, f"request extension {name!r}")
        return self

    def to_mapping(self) -> dict[str, Any]:
        """Return a detached request mapping for the Rust runtime."""

        data = _json_value(
            self.model_dump(mode="python", exclude_none=True),
            "request",
        )
        assert isinstance(data, dict)
        return data
