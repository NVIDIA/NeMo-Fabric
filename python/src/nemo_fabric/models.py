# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pydantic SDK models for NVIDIA NeMo Fabric configuration and requests.

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
from pydantic import SerializerFunctionWrapHandler
from pydantic import field_validator
from pydantic import model_serializer
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
    settings: dict[str, Any] = Field(
        default_factory=dict,
        exclude_if=lambda value: not value,
    )


class WorkflowEntrypointConfig(FabricBaseModel):
    """Adapter-owned workflow entry point."""

    kind: str = Field(min_length=1, pattern=r"\S")
    ref: str = Field(min_length=1, pattern=r"\S")

    @field_validator("kind", "ref")
    @classmethod
    def _validate_nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("workflow entrypoint values must be non-empty strings")
        return value


class WorkflowConfig(FabricBaseModel):
    """Adapter-owned workflow selection and immutable construction settings."""

    entrypoint: WorkflowEntrypointConfig
    settings: dict[str, Any] = Field(default_factory=dict)

    @model_serializer(mode="wrap")
    def _serialize_workflow(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict[str, Any]:
        data = handler(self)
        if not self.settings:
            data.pop("settings", None)
        return data


class InstructionConfig(FabricBaseModel):
    """One portable instruction value."""

    content: str = Field(min_length=1, pattern=r"\S")
    mode: Literal["replace"] = "replace"

    @field_validator("content")
    @classmethod
    def _validate_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("instruction content must be a non-empty string")
        return value


class InstructionsConfig(FabricBaseModel):
    """Harness-neutral agent instructions."""

    system: InstructionConfig | None = None


class RuntimeConfig(FabricBaseModel):
    """Invocation runtime contract."""

    input_schema: str | None = None
    output_schema: str | None = None
    artifacts: str | Path | None = None
    timeout_seconds: float | None = Field(
        default=None,
        gt=0,
        allow_inf_nan=False,
    )
    max_turns: int | None = Field(default=None, gt=0, le=(1 << 32) - 1)


class EnvironmentConfig(FabricBaseModel):
    """Execution environment configuration supplied by the consumer.

    ``provider`` selects the environment implementation. ``workspace`` is the
    path visible to the harness, while ``artifacts`` is the provider-specific
    output location. ``settings`` configures the selected provider;
    ``connection`` describes how NeMo Fabric reaches an existing environment; and
    ``metadata`` carries consumer-owned values that NeMo Fabric does not interpret.
    ``ownership`` identifies who tears the environment down, and
    ``control_location`` identifies whether NeMo Fabric control code runs inside or
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
    env: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Environment variables visible to the harness and its tools. Values are "
            "serialized into configuration and run plans; prefer api_key_env-style "
            "environment-variable-name indirection for credentials."
        ),
        json_schema_extra={"propertyNames": {"pattern": r"\S"}},
    )
    settings: dict[str, Any] = Field(
        default_factory=dict,
        description="Provider-specific configuration interpreted by the environment provider.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Consumer-owned environment metadata passed through without NeMo Fabric semantics.",
    )
    connection: dict[str, Any] = Field(
        default_factory=dict,
        description="Connection data for an existing environment, such as URL, namespace, or credential reference.",
    )
    ownership: Literal["caller_owned", "fabric_owned"] = Field(
        default="caller_owned",
        description="Whether the caller or NeMo Fabric owns environment teardown.",
    )
    control_location: Literal["external_control", "in_env_control"] = Field(
        default="in_env_control",
        description="Whether NeMo Fabric control code runs outside or inside the environment.",
    )

    @field_validator("env")
    @classmethod
    def _validate_env_names(cls, value: dict[str, str]) -> dict[str, str]:
        if any(not name.strip() for name in value):
            raise ValueError("environment.env variable names must not be empty")
        return value


class ModelConfig(FabricBaseModel):
    """Configuration for one model role."""

    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    api_key_env: str | None = None
    temperature: float | None = None
    base_url: str | None = Field(default=None, min_length=1)
    settings: dict[str, Any] = Field(default_factory=dict)

    @field_validator("provider")
    @classmethod
    def _validate_provider(cls, value: str) -> str:
        if not value.strip() or value != value.strip() or value != value.lower():
            raise ValueError("provider must be a non-empty lowercase identifier")
        return value

    @field_validator("model", "api_key_env")
    @classmethod
    def _validate_nonempty_model_fields(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("model fields must be non-empty strings")
        return value

    @field_validator("base_url")
    @classmethod
    def _validate_base_url(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("base_url must be a non-empty string")
        return value


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


class McpAuthenticationConfig(FabricBaseModel):
    """MCP server authentication configuration."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["oauth2", "service_account"]
    client_id: str | None = None
    client_secret_env: str | None = None
    scopes: list[str] = Field(
        default_factory=list, exclude_if=lambda value: not value
    )
    redirect_uri: str | None = None
    enable_dynamic_registration: bool = Field(
        default=True, exclude_if=lambda value: value
    )
    client_name: str | None = None
    token_endpoint_auth_method: (
        Literal["none", "client_secret_post", "client_secret_basic"] | None
    ) = None
    authorization_timeout_seconds: int = Field(
        default=300, gt=0, exclude_if=lambda value: value == 300
    )
    token_url: str | None = None
    token_cache_buffer_seconds: int = Field(
        default=300, ge=0, exclude_if=lambda value: value == 300
    )

    @field_validator(
        "client_id",
        "client_secret_env",
        "redirect_uri",
        "client_name",
        "token_url",
    )
    @classmethod
    def _validate_optional_nonblank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("authentication values must not be empty")
        return value

    @field_validator("scopes")
    @classmethod
    def _validate_scopes(cls, value: list[str]) -> list[str]:
        if any(not scope.strip() for scope in value):
            raise ValueError("authentication scopes must not be empty")
        return value

    @model_validator(mode="after")
    def _validate_authentication_type(self) -> Self:
        if self.type == "oauth2":
            service_account_fields = self.model_fields_set.intersection(
                {"token_url", "token_cache_buffer_seconds"}
            )
            if service_account_fields:
                name = sorted(service_account_fields)[0]
                raise ValueError(
                    f"{name} is only valid for service_account authentication"
                )
            if self.client_secret_env and not self.client_id:
                raise ValueError("client_secret_env requires client_id")
            if not self.client_id and not self.enable_dynamic_registration:
                raise ValueError(
                    "oauth2 authentication requires client_id when dynamic registration is disabled"
                )
            if (
                self.token_endpoint_auth_method
                in {
                    "client_secret_basic",
                    "client_secret_post",
                }
                and self.client_id is not None
                and not self.client_secret_env
            ):
                raise ValueError(
                    "token_endpoint_auth_method requires client_secret_env for a pre-registered client"
                )
            if (
                self.token_endpoint_auth_method == "none"
                and self.client_secret_env is not None
            ):
                raise ValueError(
                    "token_endpoint_auth_method 'none' cannot use client_secret_env"
                )
            return self

        oauth2_fields = self.model_fields_set.intersection(
            {
                "redirect_uri",
                "enable_dynamic_registration",
                "client_name",
                "authorization_timeout_seconds",
            }
        )
        if oauth2_fields:
            name = sorted(oauth2_fields)[0]
            raise ValueError(f"{name} is only valid for oauth2 authentication")
        missing = [
            name
            for name, value in (
                ("client_id", self.client_id),
                ("client_secret_env", self.client_secret_env),
                ("token_url", self.token_url),
            )
            if not value
        ]
        if missing:
            raise ValueError(
                "service_account authentication requires " + ", ".join(missing)
            )
        if self.token_endpoint_auth_method == "none":
            raise ValueError(
                "service_account authentication requires client_secret_basic or client_secret_post"
            )
        return self


class McpServerConfig(FabricBaseModel):
    """MCP server configuration."""

    transport: Literal["stdio", "sse", "streamable-http"]
    url: str = Field(
        min_length=1,
        description=(
            "MCP server URL for network transports or executable for stdio."
        ),
    )
    args: list[str] = Field(
        default_factory=list,
        exclude_if=lambda value: not value,
        description="Command-line arguments passed to an MCP stdio server process.",
    )
    env: dict[str, str] = Field(
        default_factory=dict,
        exclude_if=lambda value: not value,
        description="Environment variables passed to an MCP stdio server process.",
    )
    authentication: McpAuthenticationConfig | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    custom_headers: dict[str, str] = Field(
        default_factory=dict,
        exclude_if=lambda value: not value,
        description=(
            "HTTP headers passed to an MCP server when transport is sse or "
            "streamable-http."
        ),
    )
    exposure: Literal["harness_native", "fabric_managed"] = "harness_native"
    allowed_tools: list[str] | None = Field(
        default=None,
        description=(
            "MCP tools to expose. None exposes every discovered tool; an empty "
            "list exposes no tools."
        ),
    )
    blocked_tools: list[str] = Field(
        default_factory=list,
        description="MCP tools to block after applying the optional allowlist.",
    )

    @model_serializer(mode="wrap")
    def _serialize_tool_policy(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict[str, Any]:
        data = handler(self)
        if self.allowed_tools is None:
            data.pop("allowed_tools", None)
        if not self.blocked_tools:
            data.pop("blocked_tools", None)
        return data

    @field_validator("allowed_tools", "blocked_tools")
    @classmethod
    def _validate_tool_names(cls, value: list[str] | None) -> list[str] | None:
        if value is not None and any(not tool.strip() for tool in value):
            raise ValueError("MCP tool names must not be empty")
        return value

    @model_validator(mode="after")
    def _validate_tool_policy(self) -> Self:
        if self.transport != "stdio" and self.env:
            raise ValueError("env is only valid for stdio transport")
        if self.allowed_tools is not None:
            overlap = set(self.allowed_tools).intersection(self.blocked_tools)
            if overlap:
                name = sorted(overlap)[0]
                raise ValueError(f"MCP tool {name!r} cannot be both allowed and blocked")
        return self

    def to_mapping(self) -> dict[str, Any]:
        """Return the server mapping without collapsing an explicit empty allowlist."""

        data = super().to_mapping()
        if self.allowed_tools == []:
            data["allowed_tools"] = []
        return data


class McpConfig(FabricBaseModel):
    """MCP capability configuration."""

    servers: dict[str, McpServerConfig] = Field(default_factory=dict)

    def add_server(
        self,
        name: str,
        *,
        transport: str,
        url: str,
        args: Sequence[str] | None = None,
        env: Mapping[str, str] | None = None,
        authentication: McpAuthenticationConfig | None = None,
        custom_headers: Mapping[str, str] | None = None,
        exposure: Literal["harness_native", "fabric_managed"] = "harness_native",
        allowed_tools: Sequence[str] | None = None,
        blocked_tools: Sequence[str] = (),
        extra_fields: Mapping[str, Any] | None = None,
    ) -> Self:
        """Add or replace a named MCP server.

        For stdio, set ``url`` to the executable and pass each command-line
        argument as a separate ``args`` element.
        """

        extensions = dict(extra_fields or {})
        legacy_args = extensions.pop("args", ())
        legacy_env = extensions.pop("env", None)
        legacy_authentication = extensions.pop("authentication", None)
        legacy_custom_headers = extensions.pop("custom_headers", None)
        if isinstance(allowed_tools, str):
            raise TypeError("allowed_tools must be a sequence of strings, not a string")
        if isinstance(blocked_tools, str):
            raise TypeError("blocked_tools must be a sequence of strings, not a string")

        self.servers[name] = McpServerConfig(
            transport=transport,
            url=url,
            args=list(args if args is not None else legacy_args),
            env=env if env is not None else legacy_env or {},
            authentication=(
                authentication if authentication is not None else legacy_authentication
            ),
            custom_headers=(
                custom_headers
                if custom_headers is not None
                else legacy_custom_headers or {}
            ),
            exposure=exposure,
            allowed_tools=None if allowed_tools is None else list(allowed_tools),
            blocked_tools=list(blocked_tools),
            **extensions,
        )
        return self

    def remove_server(self, name: str) -> Self:
        """Remove a named MCP server if present."""

        self.servers.pop(name, None)
        return self


class RelayConfigPolicy(FabricBaseModel):
    """NVIDIA NeMo Relay config validation policy."""

    unknown_component: Literal["ignore", "warn", "error"] = "warn"
    unknown_field: Literal["ignore", "warn", "error"] = "warn"
    unsupported_value: Literal["ignore", "warn", "error"] = "error"


class RelayAtofFileSinkConfig(FabricBaseModel):
    """NeMo Relay ATOF file sink configuration."""

    type: Literal["file"] = "file"
    output_directory: str | Path | None = None
    filename: str | None = None
    mode: Literal["append", "overwrite"] = "append"


class RelayAtofStreamSinkConfig(FabricBaseModel):
    """NeMo Relay ATOF stream sink configuration."""

    type: Literal["stream"] = "stream"
    url: str
    transport: Literal["http_post", "websocket", "ndjson"] = "http_post"
    headers: dict[str, str] = Field(
        default_factory=dict, exclude_if=lambda value: not value
    )
    header_env: dict[str, str] = Field(
        default_factory=dict, exclude_if=lambda value: not value
    )
    timeout_millis: int = 3000
    field_name_policy: Literal["preserve", "replace_dots"] = "preserve"
    name: str | None = None


class RelayAtofConfig(FabricBaseModel):
    """NeMo Relay ATOF export configuration."""

    enabled: bool = False
    sinks: (
        list[
            Annotated[
                RelayAtofFileSinkConfig | RelayAtofStreamSinkConfig,
                Field(discriminator="type"),
            ]
            | dict[str, Any]
        ]
        | None
    ) = None


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

    version: int = 2
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
    components: list[RelayComponentConfig | dict[str, Any]] = Field(
        default_factory=list
    )
    policy: RelayConfigPolicy | dict[str, Any] | None = None


class TelemetryProviderConfig(FabricBaseModel):
    """Provider-specific telemetry configuration."""

    config: dict[str, Any] | None = None


class TelemetryConfig(FabricBaseModel):
    """Telemetry configuration."""

    providers: dict[
        Literal["relay", "native"], TelemetryProviderConfig | dict[str, Any]
    ] = Field(default_factory=dict)

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


class ToolDefinitionConfig(FabricBaseModel):
    """One named normalized tool or tool-group definition."""

    kind: str = Field(min_length=1, pattern=r"\S")
    ref: str = Field(min_length=1, pattern=r"\S")
    settings: dict[str, Any] = Field(
        default_factory=dict,
        exclude_if=lambda value: not value,
    )

    @field_validator("kind", "ref")
    @classmethod
    def _validate_identity(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("tool definition values must be non-empty strings")
        return value


class ToolsConfig(FabricBaseModel):
    """Harness-neutral tool capability configuration."""

    definitions: dict[str, ToolDefinitionConfig] = Field(
        default_factory=dict,
        exclude_if=lambda value: not value,
        description="Named normalized tool and tool-group definitions.",
    )

    enabled: list[str] | None = Field(
        default=None,
        description=(
            "Adapter-native tools to expose. None preserves the harness default; "
            "an empty list exposes no tools."
        ),
    )
    blocked: list[str] = Field(
        default_factory=list,
        exclude_if=lambda value: not value,
        description="Adapter-native tool names to deny.",
    )

    @field_validator("definitions")
    @classmethod
    def _validate_definition_names(
        cls, value: dict[str, ToolDefinitionConfig]
    ) -> dict[str, ToolDefinitionConfig]:
        if any(not name.strip() for name in value):
            raise ValueError("tool definition names must not be empty")
        return value

    @field_validator("enabled", "blocked")
    @classmethod
    def _validate_tools(cls, value: list[str] | None) -> list[str] | None:
        if value is not None and any(not tool.strip() for tool in value):
            raise ValueError("tool names must not be empty")
        return value

    @model_validator(mode="after")
    def _validate_policy(self) -> Self:
        if self.enabled is not None:
            overlap = set(self.enabled).intersection(self.blocked)
            if overlap:
                name = sorted(overlap)[0]
                raise ValueError(f"tool {name!r} cannot be both enabled and blocked")
        return self

    def add_definition(
        self,
        name: str,
        *,
        kind: str,
        ref: str,
        settings: Mapping[str, Any] | None = None,
        extra_fields: Mapping[str, Any] | None = None,
    ) -> Self:
        """Add or replace one named definition and return this tools config."""

        if not name.strip():
            raise ValueError("tool definition names must not be empty")
        value = {
            "kind": kind,
            "ref": ref,
            "settings": dict(settings or {}),
            **dict(extra_fields or {}),
        }
        self.definitions[name] = ToolDefinitionConfig.model_validate(value)
        return self

    def remove_definition(self, name: str) -> Self:
        """Remove one named definition and return this tools config."""

        self.definitions.pop(name, None)
        return self


class FabricConfig(FabricBaseModel):
    """SDK-facing typed NeMo Fabric agent configuration.

    NeMo Fabric-owned fields apply uniformly. Adapter-translated fields are
    checked against the selected descriptor; refer to the [normalized
    configuration compatibility
    table](../../../sdk/python.mdx#normalized-configuration-compatibility).
    """

    schema_version: str = "fabric.agent/v1alpha1"
    metadata: MetadataConfig
    harness: HarnessConfig
    workflow: WorkflowConfig | None = None
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    environment: EnvironmentConfig | None = None
    models: dict[str, ModelConfig] = Field(default_factory=dict)
    instructions: InstructionsConfig | None = None
    mcp: McpConfig | None = None
    skills: SkillConfig | None = None
    telemetry: TelemetryConfig | None = None
    relay: RelayConfig | dict[str, Any] | None = None
    tools: ToolsConfig | None = None

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
        args: Sequence[str] | None = None,
        env: Mapping[str, str] | None = None,
        authentication: McpAuthenticationConfig | None = None,
        custom_headers: Mapping[str, str] | None = None,
        exposure: Literal["harness_native", "fabric_managed"] = "harness_native",
        allowed_tools: Sequence[str] | None = None,
        blocked_tools: Sequence[str] = (),
        extra_fields: Mapping[str, Any] | None = None,
    ) -> Self:
        """Add or replace a named MCP server and return this config.

        For stdio, set ``url`` to the executable and pass each command-line
        argument as a separate ``args`` element.
        """

        if self.mcp is None:
            self.mcp = McpConfig()
        self.mcp.add_server(
            name,
            transport=transport,
            url=url,
            args=args,
            env=env,
            authentication=authentication,
            custom_headers=custom_headers,
            exposure=exposure,
            allowed_tools=allowed_tools,
            blocked_tools=blocked_tools,
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
        """Block adapter-native tool names and return this config."""

        if self.tools is None:
            self.tools = ToolsConfig()
        existing = list(self.tools.blocked)
        for tool in tools:
            if tool not in existing:
                existing.append(tool)
        self.tools.blocked = existing
        return self

    def add_tool_definition(
        self,
        name: str,
        *,
        kind: str,
        ref: str,
        settings: Mapping[str, Any] | None = None,
        extra_fields: Mapping[str, Any] | None = None,
    ) -> Self:
        """Add or replace one named tool definition and return this config."""

        if self.tools is None:
            self.tools = ToolsConfig()
        self.tools.add_definition(
            name,
            kind=kind,
            ref=ref,
            settings=settings,
            extra_fields=extra_fields,
        )
        return self

    def remove_tool_definition(self, name: str) -> Self:
        """Remove one named tool definition and return this config."""

        if self.tools is not None:
            self.tools.remove_definition(name)
            if (
                not self.tools.definitions
                and self.tools.enabled is None
                and not self.tools.blocked
                and not self.tools.model_extra
            ):
                self.tools = None
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
                observability
                if isinstance(observability, RelayObservabilityConfig)
                else dict(observability)
            )
        if components is not None:
            relay.components = [
                item if isinstance(item, RelayComponentConfig) else dict(item)
                for item in components
            ]
        if policy is not None:
            relay.policy = (
                policy if isinstance(policy, RelayConfigPolicy) else dict(policy)
            )
        self.relay = relay
        return self


class RunRequest(FabricBaseModel):
    """One validated NeMo Fabric invocation request."""

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
