# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free dataclasses for the southbound NeMo Fabric adapter contract."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import MISSING
from dataclasses import dataclass
from dataclasses import field
from enum import StrEnum
from pathlib import Path
from typing import Any
from typing import ClassVar
from typing import Literal
from typing import Self

from nemo_fabric_adapter_contract.codec import ContractValidationError
from nemo_fabric_adapter_contract.codec import JsonValue
from nemo_fabric_adapter_contract.codec import decode_dataclass
from nemo_fabric_adapter_contract.codec import decode_field
from nemo_fabric_adapter_contract.codec import encode_dataclass
from nemo_fabric_adapter_contract.codec import json_mapping
from nemo_fabric_adapter_contract.codec import validate_dataclass


def _optional(default: Any = None):
    return field(default=default, metadata={"omit_none": True})


def _empty_dict():
    return field(default_factory=dict, metadata={"omit_empty": True})


def _json_dict():
    return field(default_factory=dict, metadata={"json": True, "omit_empty": True})


def _empty_list():
    return field(default_factory=list, metadata={"omit_empty": True})


def _default(value: Any):
    return field(default=value, metadata={"omit_default": value})


def _json_value_field(*, default: Any = MISSING, omit_empty: bool = False):
    metadata = {"json": True}
    if omit_empty:
        metadata["omit_empty"] = True
    if default is MISSING:
        return field(metadata=metadata)
    return field(default=default, metadata=metadata)


def _nonblank(value: str, field_name: str) -> None:
    if not value.strip():
        raise ContractValidationError("must be a non-empty string", path=(field_name,))


def _bounded_int(value: int | None, field_name: str, maximum: int) -> None:
    if value is not None and not 0 <= value <= maximum:
        raise ContractValidationError(
            f"must be between 0 and {maximum}",
            path=(field_name,),
        )


@dataclass(slots=True, kw_only=True)
class ContractModel:
    """Base for adapter-facing contract dataclasses."""

    # Pydantic reads this metadata only when its optional TypeAdapter is used.
    # It does not require importing Pydantic in the base contract package.
    __pydantic_config__: ClassVar[dict[str, Any]] = {
        "extra": "forbid",
        "allow_inf_nan": False,
    }

    def __post_init__(self) -> None:
        validate_dataclass(self)
        self._validate()

    def __setattr__(self, name: str, value: Any) -> None:
        if name not in self.__dataclass_fields__:
            raise AttributeError(f"{type(self).__name__} has no field {name!r}")
        try:
            previous = getattr(self, name)
        except AttributeError:
            object.__setattr__(self, name, value)
            return

        decoded = decode_field(self, name, value)
        object.__setattr__(self, name, decoded)
        try:
            self._validate()
        except ContractValidationError:
            object.__setattr__(self, name, previous)
            raise

    def _validate(self) -> None:
        """Validate constraints that are more specific than field types."""

    @classmethod
    def from_mapping(cls, value: Any) -> Self:
        """Validate and decode one closed adapter wire mapping."""

        return decode_dataclass(cls, value)

    def to_mapping(self) -> dict[str, Any]:
        """Return a detached JSON-compatible adapter wire mapping."""

        return encode_dataclass(self)


@dataclass(slots=True, kw_only=True)
class AgentContractBlock(ContractModel):
    """Base for explicitly extensible adapter-owned contract blocks."""

    extensions: dict[str, JsonValue] = _json_dict()

    def set_extensions(self, value: Mapping[str, Any]) -> Self:
        """Replace this block's adapter-owned extensions and return the block."""

        if not isinstance(value, Mapping):
            raise ContractValidationError("extensions must be an object")
        self.extensions = json_mapping(value, path=("extensions",))
        return self


AgentConfigBlock = AgentContractBlock


@dataclass(slots=True, kw_only=True)
class AgentHarnessConfig(AgentContractBlock):
    """Adapter-owned target settings projected from the selected harness."""

    settings: dict[str, JsonValue] = _json_dict()


@dataclass(slots=True, kw_only=True)
class AgentModelConfig(AgentContractBlock):
    """Configuration for one named model role."""

    provider: str
    model: str
    api_key_env: str | None = _optional()
    temperature: float | None = _optional()
    base_url: str | None = _optional()
    settings: dict[str, JsonValue] = _json_dict()

    def _validate(self) -> None:
        if (
            not self.provider.strip()
            or self.provider != self.provider.strip()
            or self.provider != self.provider.lower()
        ):
            raise ContractValidationError(
                "must be a non-empty lowercase identifier",
                path=("provider",),
            )
        _nonblank(self.model, "model")
        if self.api_key_env is not None:
            _nonblank(self.api_key_env, "api_key_env")
        if self.base_url is not None:
            _nonblank(self.base_url, "base_url")


@dataclass(slots=True, kw_only=True)
class AgentInstructionConfig(AgentContractBlock):
    """One normalized instruction value."""

    content: str
    mode: Literal["replace"] = "replace"

    def _validate(self) -> None:
        _nonblank(self.content, "content")


@dataclass(slots=True, kw_only=True)
class AgentInstructionsConfig(AgentContractBlock):
    """Normalized instructions applied by the adapter target."""

    system: AgentInstructionConfig | None = _optional()


@dataclass(slots=True, kw_only=True)
class AgentRuntimeConfig(AgentContractBlock):
    """Runtime behavior applied by the adapter target."""

    max_turns: int | None = _optional()

    def _validate(self) -> None:
        if self.max_turns is not None and not 1 <= self.max_turns <= (1 << 32) - 1:
            raise ContractValidationError(
                f"must be between 1 and {(1 << 32) - 1}",
                path=("max_turns",),
            )


@dataclass(slots=True, kw_only=True)
class AgentSkillConfig(AgentContractBlock):
    """Skill paths made available to the adapter target."""

    paths: list[str | Path] = _empty_list()


def _validate_tool_names(value: list[str] | None, field_name: str, label: str) -> None:
    if value is not None and any(not tool.strip() for tool in value):
        raise ContractValidationError(
            f"{label} names must not be empty",
            path=(field_name,),
        )


class OAuthTokenEndpointAuthMethod(StrEnum):
    """OAuth client authentication method used at the token endpoint."""

    NONE = "none"
    CLIENT_SECRET_POST = "client_secret_post"
    CLIENT_SECRET_BASIC = "client_secret_basic"


@dataclass(slots=True, kw_only=True)
class McpOAuth2Config(ContractModel):
    """OAuth 2.0 authorization-code authentication for an MCP server."""

    type: Literal["oauth2"]
    client_id: str | None = _optional()
    client_secret_env: str | None = _optional()
    scopes: list[str] = _empty_list()
    redirect_uri: str | None = _optional()
    enable_dynamic_registration: bool = _default(True)
    client_name: str | None = _optional()
    token_endpoint_auth_method: OAuthTokenEndpointAuthMethod | None = _optional()
    authorization_timeout_seconds: int = _default(300)

    @property
    def scope(self) -> str | None:
        """Return scopes in the space-delimited form expected by OAuth clients."""

        value = " ".join(self.scopes)
        return value or None

    def _validate(self) -> None:
        for name in ("client_id", "client_secret_env", "redirect_uri", "client_name"):
            if (value := getattr(self, name)) is not None:
                _nonblank(value, name)
        _validate_tool_names(self.scopes, "scopes", "authentication scope")
        if self.client_secret_env is not None and self.client_id is None:
            raise ContractValidationError(
                "requires client_id", path=("client_secret_env",)
            )
        if not self.enable_dynamic_registration and self.client_id is None:
            raise ContractValidationError(
                "is required when dynamic registration is disabled",
                path=("client_id",),
            )
        if (
            self.token_endpoint_auth_method
            in {
                OAuthTokenEndpointAuthMethod.CLIENT_SECRET_BASIC,
                OAuthTokenEndpointAuthMethod.CLIENT_SECRET_POST,
            }
            and self.client_id is not None
            and self.client_secret_env is None
        ):
            raise ContractValidationError(
                "requires client_secret_env for a pre-registered client",
                path=("token_endpoint_auth_method",),
            )
        if (
            self.token_endpoint_auth_method is OAuthTokenEndpointAuthMethod.NONE
            and self.client_secret_env is not None
        ):
            raise ContractValidationError(
                "'none' cannot be combined with client_secret_env",
                path=("token_endpoint_auth_method",),
            )
        if not 1 <= self.authorization_timeout_seconds <= (1 << 64) - 1:
            raise ContractValidationError(
                "must be greater than zero",
                path=("authorization_timeout_seconds",),
            )


@dataclass(slots=True, kw_only=True)
class McpServiceAccountConfig(ContractModel):
    """OAuth 2.0 client-credentials authentication for an MCP server."""

    type: Literal["service_account"]
    client_id: str
    client_secret_env: str
    token_url: str
    scopes: list[str] = _empty_list()
    token_endpoint_auth_method: OAuthTokenEndpointAuthMethod | None = _optional()
    token_cache_buffer_seconds: int = _default(300)

    def _validate(self) -> None:
        for name in ("client_id", "client_secret_env", "token_url"):
            _nonblank(getattr(self, name), name)
        _validate_tool_names(self.scopes, "scopes", "authentication scope")
        if self.token_endpoint_auth_method is OAuthTokenEndpointAuthMethod.NONE:
            raise ContractValidationError(
                "service_account requires client_secret_basic or client_secret_post",
                path=("token_endpoint_auth_method",),
            )
        _bounded_int(
            self.token_cache_buffer_seconds,
            "token_cache_buffer_seconds",
            (1 << 64) - 1,
        )


McpAuthenticationConfig = McpOAuth2Config | McpServiceAccountConfig


@dataclass(slots=True, kw_only=True)
class AgentMcpServerConfig(AgentContractBlock):
    """One MCP server routed to the adapter target."""

    transport: str
    url: str
    args: list[str] = _empty_list()
    env: dict[str, str] = _empty_dict()
    authentication: McpAuthenticationConfig | None = _optional()
    custom_headers: dict[str, str] = _empty_dict()
    allowed_tools: list[str] | None = _optional()
    blocked_tools: list[str] = _empty_list()

    def _validate(self) -> None:
        _nonblank(self.transport, "transport")
        _nonblank(self.url, "url")
        _validate_tool_names(self.allowed_tools, "allowed_tools", "MCP tool")
        _validate_tool_names(self.blocked_tools, "blocked_tools", "MCP tool")
        if self.allowed_tools is not None:
            overlap = set(self.allowed_tools).intersection(self.blocked_tools)
            if overlap:
                name = sorted(overlap)[0]
                raise ContractValidationError(
                    f"MCP tool {name!r} cannot be both allowed and blocked"
                )


@dataclass(slots=True, kw_only=True)
class AgentMcpConfig(AgentContractBlock):
    """Named MCP servers routed to the adapter target."""

    servers: dict[str, AgentMcpServerConfig] = _empty_dict()


@dataclass(slots=True, kw_only=True)
class AgentToolDefinition(AgentContractBlock):
    """One named tool or tool-group definition resolved by the adapter."""

    kind: str
    ref: str
    settings: dict[str, JsonValue] = _json_dict()

    def _validate(self) -> None:
        _nonblank(self.kind, "kind")
        _nonblank(self.ref, "ref")


@dataclass(slots=True, kw_only=True)
class AgentToolsConfig(AgentContractBlock):
    """Named tool definitions and effective target-level tool policy."""

    definitions: dict[str, AgentToolDefinition] = _empty_dict()
    enabled: list[str] | None = _optional()
    blocked: list[str] = _empty_list()

    def _validate(self) -> None:
        _validate_tool_names(self.enabled, "enabled", "tool")
        _validate_tool_names(self.blocked, "blocked", "tool")
        if self.enabled is not None:
            overlap = set(self.enabled).intersection(self.blocked)
            if overlap:
                name = sorted(overlap)[0]
                raise ContractValidationError(
                    f"tool {name!r} cannot be both enabled and blocked"
                )


@dataclass(slots=True, kw_only=True)
class AgentWorkflowEntrypointConfig(AgentContractBlock):
    """Adapter-declared resolution semantics for one custom agent or workflow."""

    kind: str
    ref: str

    def _validate(self) -> None:
        _nonblank(self.kind, "kind")
        _nonblank(self.ref, "ref")


@dataclass(slots=True, kw_only=True)
class AgentWorkflowConfig(AgentContractBlock):
    """Custom agent or workflow selection and construction settings."""

    entrypoint: AgentWorkflowEntrypointConfig
    settings: dict[str, JsonValue] = _json_dict()


@dataclass(slots=True, kw_only=True)
class AgentConfig(AgentContractBlock):
    """Configuration projected southbound to one adapter target."""

    harness: AgentHarnessConfig | None = _optional()
    models: dict[str, AgentModelConfig] = _empty_dict()
    instructions: AgentInstructionsConfig | None = _optional()
    runtime: AgentRuntimeConfig | None = _optional()
    skills: AgentSkillConfig | None = _optional()
    mcp: AgentMcpConfig | None = _optional()
    tools: AgentToolsConfig | None = _optional()
    workflow: AgentWorkflowConfig | None = _optional()


@dataclass(slots=True, kw_only=True)
class AgentRunRequest(AgentContractBlock):
    """Preview southbound request for the future typed invoke transport."""

    input: JsonValue = _json_value_field()
    context: dict[str, JsonValue] = _json_dict()


class AgentRunStatus(StrEnum):
    """Completion status reported by an adapter target."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(slots=True, kw_only=True)
class AgentRunError(AgentContractBlock):
    """Error reported by an adapter target."""

    code: str
    message: str
    retryable: bool = False

    def _validate(self) -> None:
        _nonblank(self.code, "code")
        _nonblank(self.message, "message")


def _validate_artifact_path(value: str | Path) -> None:
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
        raise ContractValidationError(
            "artifact path must be non-empty, relative, and contain no parent traversal",
            path=("path",),
        )


@dataclass(slots=True, kw_only=True)
class AgentArtifact(AgentContractBlock):
    """One artifact produced by an adapter target."""

    name: str
    kind: str
    path: str | Path
    media_type: str | None = _optional()

    def _validate(self) -> None:
        _nonblank(self.name, "name")
        _nonblank(self.kind, "kind")
        _validate_artifact_path(self.path)
        if self.media_type is not None:
            _nonblank(self.media_type, "media_type")


@dataclass(slots=True, kw_only=True)
class AgentUsage(AgentContractBlock):
    """Normalized model usage reported by an adapter target."""

    input_tokens: int | None = _optional()
    output_tokens: int | None = _optional()
    total_tokens: int | None = _optional()
    cost_usd: float | None = _optional()

    def _validate(self) -> None:
        _bounded_int(self.input_tokens, "input_tokens", (1 << 64) - 1)
        _bounded_int(self.output_tokens, "output_tokens", (1 << 64) - 1)
        _bounded_int(self.total_tokens, "total_tokens", (1 << 64) - 1)
        if self.cost_usd is not None and self.cost_usd < 0:
            raise ContractValidationError(
                "must be greater than or equal to 0", path=("cost_usd",)
            )


@dataclass(slots=True, kw_only=True)
class AgentRunResult(AgentContractBlock):
    """Preview southbound result for the future typed invoke transport."""

    status: AgentRunStatus
    output: JsonValue = _json_value_field()
    error: AgentRunError | None = _optional()
    usage: AgentUsage | None = _optional()
    artifacts: list[AgentArtifact] = _empty_list()

    def _validate(self) -> None:
        if self.status is AgentRunStatus.FAILED and self.error is None:
            raise ContractValidationError("failed result requires an error")
        if self.status is AgentRunStatus.SUCCEEDED and self.error is not None:
            raise ContractValidationError("succeeded result must not include an error")


class ControlLocation(StrEnum):
    """Where Fabric control code runs relative to the task environment."""

    EXTERNAL_CONTROL = "external_control"
    IN_ENV_CONTROL = "in_env_control"


class EnvironmentOwnership(StrEnum):
    """Whether Fabric owns the underlying environment resource."""

    CALLER_OWNED = "caller_owned"
    FABRIC_OWNED = "fabric_owned"


@dataclass(slots=True, kw_only=True)
class EnvironmentHandle(ContractModel):
    """Resolved execution environment visible to an adapter target."""

    environment_id: str
    provider: str
    control_location: ControlLocation
    workspace: str | Path | None = _optional()
    artifacts: str | Path | None = _optional()
    env: dict[str, str] = _empty_dict()
    ownership: EnvironmentOwnership
    connection: dict[str, JsonValue] = _json_dict()
    metadata: dict[str, JsonValue] = _json_dict()

    def _validate(self) -> None:
        _nonblank(self.environment_id, "environment_id")
        _nonblank(self.provider, "provider")


@dataclass(slots=True, kw_only=True)
class ArtifactRef(ContractModel):
    """Reference to one artifact visible through RuntimeContext."""

    name: str
    kind: str
    path: str | Path
    media_type: str | None = _optional()
    metadata: dict[str, JsonValue] = _json_dict()

    def _validate(self) -> None:
        _nonblank(self.name, "name")
        _nonblank(self.kind, "kind")
        if self.media_type is not None:
            _nonblank(self.media_type, "media_type")


@dataclass(slots=True, kw_only=True)
class ArtifactManifest(ContractModel):
    """Artifacts visible to an adapter at invocation start."""

    root: str | Path | None = _optional()
    artifacts: list[ArtifactRef] = _empty_list()


@dataclass(slots=True, kw_only=True)
class RuntimeTelemetryContext(ContractModel):
    """Telemetry configuration generated for one adapter invocation."""

    relay_enabled: bool
    config_path: str | Path | None = _optional()
    env: dict[str, str] = _empty_dict()
    metadata: dict[str, JsonValue] = _json_dict()


@dataclass(slots=True, kw_only=True)
class RuntimeContext(ContractModel):
    """Fabric-generated context for one adapter invocation."""

    runtime_id: str
    invocation_id: str
    request_id: str
    environment: EnvironmentHandle
    artifacts: ArtifactManifest
    telemetry: RuntimeTelemetryContext | None = _optional()

    def _validate(self) -> None:
        _nonblank(self.runtime_id, "runtime_id")
        _nonblank(self.invocation_id, "invocation_id")
        _nonblank(self.request_id, "request_id")
