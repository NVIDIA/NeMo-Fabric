# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public data contracts for the NeMo Fabric Python SDK."""

from __future__ import annotations

import math
from collections.abc import Iterator
from collections.abc import Mapping
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from types import MappingProxyType
from typing import Any
from typing import TypeVar

from pydantic import BaseModel

from nemo_fabric.errors import FabricConfigError

JSONScalar = str | int | float | bool | None
JSONValue = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]

_T = TypeVar("_T")


def _plain(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _plain(value.model_dump(mode="json", exclude_none=True))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, _ConfigMapping):
        return value.to_mapping()
    if isinstance(value, FabricMapping):
        return value.to_mapping()
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise FabricConfigError("JSON object keys must be strings")
            result[key] = _plain(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise FabricConfigError("JSON numbers must be finite")
    if value is None or isinstance(value, (str, int, float, bool)):
        return deepcopy(value)
    raise FabricConfigError(f"value of type {type(value).__name__} is not JSON-compatible")


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json", exclude_none=True)
    if not isinstance(value, Mapping):
        raise FabricConfigError(f"{name} must be a JSON object")
    return _plain(value)


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FabricConfigError(f"{name} must be a non-empty string")
    return value


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise FabricConfigError(f"{name} must be a boolean")
    return value


def _coerce(model: type[_T], value: _T | Mapping[str, Any], name: str) -> _T:
    if isinstance(value, model):
        return deepcopy(value)
    if isinstance(value, BaseModel):
        return model.from_mapping(value.model_dump(mode="json", exclude_none=True))  # type: ignore[attr-defined,no-any-return]
    if isinstance(value, Mapping):
        return model.from_mapping(value)  # type: ignore[attr-defined,no-any-return]
    raise FabricConfigError(f"{name} must be a {model.__name__} or JSON object")


class _ConfigMapping(dict[str, Any]):
    """Mutable schema-shaped config with explicit extension storage."""

    _fields: frozenset[str] = frozenset()
    _omit_if_empty: frozenset[str] = frozenset()

    def __init__(
        self,
        values: Mapping[str, Any],
        *,
        extra_fields: Mapping[str, Any] | None = None,
    ) -> None:
        extras = _mapping({} if extra_fields is None else extra_fields, "extra_fields")
        overlap = self._fields.intersection(extras)
        if overlap:
            raise FabricConfigError(f"extra_fields duplicates known fields: {', '.join(sorted(overlap))}")
        stored = {
            key: deepcopy(item) if isinstance(item, _ConfigMapping) else _plain(item) for key, item in values.items()
        }
        super().__init__({**stored, **extras})

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as error:
            if name in self._fields:
                return None
            raise AttributeError(name) from error

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        if name not in self._fields:
            raise AttributeError(name)
        self[name] = deepcopy(value) if isinstance(value, _ConfigMapping) else _plain(value)

    @property
    def extra_fields(self) -> dict[str, Any]:
        """Return preserved schema-extension fields as a deep copy."""

        return {key: _plain(value) for key, value in self.items() if key not in self._fields}

    def to_mapping(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible mapping for serialization."""

        data = _plain(dict(self))
        for key in self._omit_if_empty:
            if data.get(key) in ({}, []):
                data.pop(key, None)
        return data


class _MetadataConfig(_ConfigMapping):
    """Agent identity and human-readable metadata.

    Attributes:
        name: Stable, non-empty agent name.
        description: Optional human-readable description.
        extra_fields: Preserved extension fields not recognized by this SDK.
    """

    _fields = frozenset({"name", "description"})

    def __init__(
        self,
        *,
        name: str,
        description: str | None = None,
        extra_fields: Mapping[str, Any] | None = None,
    ) -> None:
        values: dict[str, Any] = {"name": _required_text(name, "metadata name")}
        if description is not None:
            values["description"] = description
        super().__init__(values, extra_fields=extra_fields)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "_MetadataConfig":
        """Validate a metadata mapping and preserve unknown extension fields."""

        data = _mapping(value, "metadata")
        return cls(
            name=data.get("name"),
            description=data.get("description"),
            extra_fields={key: item for key, item in data.items() if key not in cls._fields},
        )


class _HarnessConfig(_ConfigMapping):
    """Harness adapter selection and adapter-owned settings.

    Attributes:
        adapter_id: Stable identifier of the Fabric adapter to resolve.
        resolution: Optional adapter resolution strategy.
        settings: JSON-compatible settings owned by the selected adapter.
        extra_fields: Preserved extension fields not recognized by this SDK.
    """

    _fields = frozenset({"adapter_id", "resolution", "settings"})
    _omit_if_empty = frozenset({"settings"})

    def __init__(
        self,
        *,
        adapter_id: str,
        resolution: str | None = None,
        settings: Mapping[str, Any] | None = None,
        extra_fields: Mapping[str, Any] | None = None,
    ) -> None:
        values: dict[str, Any] = {
            "adapter_id": _required_text(adapter_id, "adapter_id"),
            "settings": _mapping(
                {} if settings is None else settings,
                "harness settings",
            ),
        }
        if resolution is not None:
            values["resolution"] = resolution
        super().__init__(values, extra_fields=extra_fields)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "_HarnessConfig":
        """Validate a harness mapping and preserve unknown extension fields."""

        data = _mapping(value, "harness")
        return cls(
            adapter_id=data.get("adapter_id"),
            resolution=data.get("resolution"),
            settings=data.get("settings"),
            extra_fields={key: item for key, item in data.items() if key not in cls._fields},
        )


class _RuntimeConfig(_ConfigMapping):
    """Runtime input/output contract.

    Attributes:
        input_schema: Optional logical input contract identifier.
        output_schema: Optional logical output contract identifier.
        artifacts: Optional artifact-root path.
        timeout_seconds: Optional invocation deadline in seconds.
        extra_fields: Preserved extension fields not recognized by this SDK.
    """

    _fields = frozenset({"input_schema", "output_schema", "artifacts", "timeout_seconds"})

    def __init__(
        self,
        *,
        input_schema: str | None = None,
        output_schema: str | None = None,
        artifacts: str | Path | None = None,
        timeout_seconds: float | None = None,
        extra_fields: Mapping[str, Any] | None = None,
    ) -> None:
        values: dict[str, Any] = {}
        for key, item in (
            ("input_schema", input_schema),
            ("output_schema", output_schema),
            ("artifacts", artifacts),
        ):
            if item is not None:
                values[key] = item
        if timeout_seconds is not None:
            if (
                isinstance(timeout_seconds, bool)
                or not isinstance(timeout_seconds, (int, float))
                or not math.isfinite(timeout_seconds)
                or timeout_seconds <= 0
            ):
                raise FabricConfigError(
                    "runtime timeout_seconds must be a finite number greater than zero"
                )
            values["timeout_seconds"] = float(timeout_seconds)
        super().__init__(values, extra_fields=extra_fields)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "_RuntimeConfig":
        """Validate a runtime mapping and apply stable constructor defaults."""

        data = _mapping(value, "runtime")
        return cls(
            input_schema=data.get("input_schema"),
            output_schema=data.get("output_schema"),
            artifacts=data.get("artifacts"),
            timeout_seconds=data.get("timeout_seconds"),
            extra_fields={key: item for key, item in data.items() if key not in cls._fields},
        )


class _EnvironmentConfig(_ConfigMapping):
    """Execution environment configuration.

    Attributes:
        provider: Environment provider identifier; defaults to ``local``.
        workspace: Optional workspace path visible to the harness.
        artifacts: Optional environment-specific artifact path.
        env: Environment variables visible to the harness and its tools.
        settings: JSON-compatible provider settings.
        metadata: JSON-compatible caller metadata.
        extra_fields: Preserved extension fields not recognized by this SDK.
    """

    _fields = frozenset({"provider", "workspace", "artifacts", "env", "settings", "metadata"})
    _omit_if_empty = frozenset({"env", "settings", "metadata"})

    def __init__(
        self,
        *,
        provider: str = "local",
        workspace: str | Path | None = None,
        artifacts: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        settings: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        extra_fields: Mapping[str, Any] | None = None,
    ) -> None:
        values: dict[str, Any] = {
            "provider": _required_text(provider, "environment provider"),
            "env": {},
            "settings": _mapping(
                {} if settings is None else settings,
                "environment settings",
            ),
            "metadata": _mapping(
                {} if metadata is None else metadata,
                "environment metadata",
            ),
        }
        raw_env = _mapping({} if env is None else env, "environment env")
        if any(not isinstance(value, str) for value in raw_env.values()):
            raise FabricConfigError("environment env values must be strings")
        values["env"] = {
            _required_text(name, "environment variable name"): value
            for name, value in raw_env.items()
        }
        if workspace is not None:
            values["workspace"] = workspace
        if artifacts is not None:
            values["artifacts"] = artifacts
        super().__init__(values, extra_fields=extra_fields)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "_EnvironmentConfig":
        """Validate an environment mapping and preserve extension fields."""

        data = _mapping(value, "environment")
        return cls(
            provider=data.get("provider", "local"),
            workspace=data.get("workspace"),
            artifacts=data.get("artifacts"),
            env=data.get("env"),
            settings=data.get("settings"),
            metadata=data.get("metadata"),
            extra_fields={key: item for key, item in data.items() if key not in cls._fields},
        )


class _SkillConfig(_ConfigMapping):
    """Skill capability configuration.

    The shape matches the ``skills`` section of ``FabricConfig`` while
    providing small authoring helpers for Python callers.
    """

    _fields = frozenset({"paths"})
    _omit_if_empty = frozenset({"paths"})

    def __init__(
        self,
        *,
        paths: Sequence[str | Path] | None = None,
        extra_fields: Mapping[str, Any] | None = None,
    ) -> None:
        values: dict[str, Any] = {"paths": [str(path) for path in ([] if paths is None else paths)]}
        super().__init__(values, extra_fields=extra_fields)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "_SkillConfig":
        """Validate a skill mapping and preserve extension fields."""

        data = _mapping(value, "skills")
        return cls(
            paths=data.get("paths", []),
            extra_fields={key: item for key, item in data.items() if key not in cls._fields},
        )

    def add_path(self, path: str | Path) -> "_SkillConfig":
        """Add a skill path to this config if it is not already present."""

        value = str(path)
        paths = list(self.get("paths", []))
        if value not in paths:
            paths.append(value)
        self["paths"] = paths
        return self

    def remove_path(self, path: str | Path) -> "_SkillConfig":
        """Remove a skill path from this config if present."""

        value = str(path)
        self["paths"] = [item for item in self.get("paths", []) if item != value]
        return self


class _ToolsetConfig(_ConfigMapping):
    """Toolset selection and blocking policy."""

    _fields = frozenset({"enabled", "blocked"})
    _omit_if_empty = frozenset({"blocked"})

    def __init__(
        self,
        *,
        enabled: Sequence[str] | None = None,
        blocked: Sequence[str] | None = None,
        extra_fields: Mapping[str, Any] | None = None,
    ) -> None:
        for name, values in (("enabled", enabled), ("blocked", blocked)):
            if values is not None and (
                isinstance(values, (str, bytes)) or not isinstance(values, Sequence)
            ):
                raise FabricConfigError(
                    f"tools toolsets {name} must be an ordered sequence of strings"
                )
        enabled_values = (
            None
            if enabled is None
            else [_required_text(toolset, "enabled toolset") for toolset in enabled]
        )
        blocked_values = [
            _required_text(toolset, "blocked toolset") for toolset in (blocked or [])
        ]
        overlap = set(enabled_values or []).intersection(blocked_values)
        if overlap:
            name = sorted(overlap)[0]
            raise FabricConfigError(f"toolset {name!r} cannot be both enabled and blocked")
        values: dict[str, Any] = {"blocked": blocked_values}
        if enabled_values is not None:
            values["enabled"] = enabled_values
        super().__init__(values, extra_fields=extra_fields)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "_ToolsetConfig":
        """Validate a toolset mapping and preserve extension fields."""

        data = _mapping(value, "tools toolsets")
        return cls(
            enabled=data.get("enabled"),
            blocked=data.get("blocked", []),
            extra_fields={key: item for key, item in data.items() if key not in cls._fields},
        )


class _ToolsConfig(_ConfigMapping):
    """Harness-neutral tool capability configuration."""

    _fields = frozenset({"blocked", "toolsets"})
    _omit_if_empty = frozenset({"blocked"})

    def __init__(
        self,
        *,
        blocked: Sequence[str] | None = None,
        toolsets: _ToolsetConfig | Mapping[str, Any] | None = None,
        extra_fields: Mapping[str, Any] | None = None,
    ) -> None:
        if blocked is not None and (
            isinstance(blocked, (str, bytes)) or not isinstance(blocked, Sequence)
        ):
            raise FabricConfigError("tools blocked must be an ordered sequence of strings")
        values: dict[str, Any] = {
            "blocked": [_required_text(tool, "blocked tool") for tool in (blocked or [])]
        }
        if toolsets is not None:
            values["toolsets"] = _coerce(_ToolsetConfig, toolsets, "tools toolsets")
        super().__init__(values, extra_fields=extra_fields)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "_ToolsConfig":
        """Validate a tools mapping and preserve extension fields."""

        data = _mapping(value, "tools")
        blocked = data.get("blocked", [])
        if isinstance(blocked, (str, bytes)) or not isinstance(blocked, Sequence):
            raise FabricConfigError("tools blocked must be an ordered sequence of strings")
        return cls(
            blocked=blocked,
            toolsets=data.get("toolsets"),
            extra_fields={key: item for key, item in data.items() if key not in cls._fields},
        )

    def block(self, *tools: str) -> _ToolsConfig:
        """Block adapter-native tool names."""

        blocked = list(self.get("blocked", []))
        for tool in tools:
            value = _required_text(tool, "blocked tool")
            if value not in blocked:
                blocked.append(value)
        self["blocked"] = blocked
        return self


class _McpConfig(_ConfigMapping):
    """MCP capability configuration with authoring helpers."""

    _fields = frozenset({"servers"})
    _omit_if_empty = frozenset({"servers"})
    _EXPOSURES = frozenset({"harness_native", "fabric_managed"})

    def __init__(
        self,
        *,
        servers: Mapping[str, Any] | None = None,
        extra_fields: Mapping[str, Any] | None = None,
    ) -> None:
        values: dict[str, Any] = {"servers": _mapping({} if servers is None else servers, "mcp servers")}
        super().__init__(values, extra_fields=extra_fields)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "_McpConfig":
        """Validate an MCP mapping and preserve extension fields."""

        data = _mapping(value, "mcp")
        return cls(
            servers=data.get("servers", {}),
            extra_fields={key: item for key, item in data.items() if key not in cls._fields},
        )

    def add_server(
        self,
        name: str,
        *,
        transport: str,
        url: str,
        exposure: str = "harness_native",
        extra_fields: Mapping[str, Any] | None = None,
    ) -> "_McpConfig":
        """Add or replace a named MCP server."""

        if exposure not in self._EXPOSURES:
            allowed = ", ".join(sorted(self._EXPOSURES))
            raise FabricConfigError(f"mcp exposure must be one of: {allowed}")
        server = {
            "transport": _required_text(transport, "mcp transport"),
            "url": _required_text(url, "mcp url"),
            "exposure": exposure,
        }
        server.update(
            _mapping(
                {} if extra_fields is None else extra_fields,
                "mcp server extra_fields",
            )
        )
        servers = dict(self.get("servers", {}))
        servers[_required_text(name, "mcp server name")] = server
        self["servers"] = servers
        return self

    def remove_server(self, name: str) -> "_McpConfig":
        """Remove a named MCP server if present."""

        servers = dict(self.get("servers", {}))
        servers.pop(name, None)
        self["servers"] = servers
        return self


class _TelemetryConfig(_ConfigMapping):
    """Telemetry configuration with authoring helpers."""

    _fields = frozenset({"providers"})
    _PROVIDERS = frozenset({"relay", "native"})

    def __init__(
        self,
        *,
        providers: Mapping[str, Any] | None = None,
        extra_fields: Mapping[str, Any] | None = None,
    ) -> None:
        values: dict[str, Any] = {"providers": self._providers(providers or {})}
        super().__init__(values, extra_fields=extra_fields)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "_TelemetryConfig":
        """Validate a telemetry mapping and preserve extension fields."""

        data = _mapping(value, "telemetry")
        return cls(
            providers=data.get("providers", {}),
            extra_fields={key: item for key, item in data.items() if key not in cls._fields},
        )

    @classmethod
    def _providers(cls, providers: Mapping[str, Any]) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for key, value in _mapping(providers, "telemetry providers").items():
            provider = _required_text(key, "telemetry provider")
            if provider not in cls._PROVIDERS:
                allowed = ", ".join(sorted(cls._PROVIDERS))
                raise FabricConfigError(f"telemetry provider must be one of: {allowed}")
            values[provider] = _mapping(value, f"{provider} telemetry provider")
        return values

    def enable_relay(
        self,
    ) -> "_TelemetryConfig":
        """Enable NeMo Relay telemetry for subsequently started runtimes."""

        providers = dict(self.get("providers", {}))
        providers["relay"] = {}
        self["providers"] = providers
        return self

    def enable_native(self, *, config: Mapping[str, Any] | None = None) -> "_TelemetryConfig":
        """Let the selected harness adapter handle telemetry natively."""

        providers = dict(self.get("providers", {}))
        provider_config: dict[str, Any] = dict(providers.get("native", {}))
        if config is not None:
            provider_config["config"] = _mapping(config, "native telemetry config")
        providers["native"] = provider_config
        self["providers"] = providers
        return self

    def remove_provider(self, provider: str) -> "_TelemetryConfig":
        """Remove a configured telemetry provider."""

        providers = dict(self.get("providers", {}))
        providers.pop(self._provider(provider), None)
        self["providers"] = providers
        return self

    @classmethod
    def _provider(cls, provider: str) -> str:
        value = _required_text(provider, "telemetry provider")
        if value not in cls._PROVIDERS:
            allowed = ", ".join(sorted(cls._PROVIDERS))
            raise FabricConfigError(f"telemetry provider must be one of: {allowed}")
        return value


class _FabricConfigSnapshot(_ConfigMapping):
    """Typed snapshot of the Fabric configuration stored in a run plan.

    It is reconstructed from the native plan payload and exposed through the
    immutable ``RunPlan`` mapping. Unknown fields survive round trips through
    ``extra_fields``.

    Attributes:
        schema_version: Agent schema identifier.
        metadata: Required ``MetadataConfig`` agent identity.
        harness: Required ``HarnessConfig`` adapter selection.
        runtime: Runtime input/output configuration.
        environment: Optional execution environment configuration.
        models: Named, JSON-compatible model configurations.
        system_prompt: Optional agent system instructions.
        max_turns: Optional shared harness turn limit.
        mcp: Optional MCP configuration.
        skills: Optional skill configuration.
        telemetry: Optional telemetry configuration.
        relay: Optional Relay integration configuration.
        tools: Optional harness-neutral tool configuration.
        extra_fields: Preserved extension fields not recognized by this SDK.
    """

    _fields = frozenset(
        {
            "schema_version",
            "metadata",
            "harness",
            "runtime",
            "environment",
            "models",
            "system_prompt",
            "max_turns",
            "mcp",
            "skills",
            "telemetry",
            "relay",
            "tools",
        }
    )
    _omit_if_empty = frozenset({"models", "mcp", "skills"})

    def __init__(
        self,
        *,
        metadata: _MetadataConfig | Mapping[str, Any],
        harness: _HarnessConfig | Mapping[str, Any],
        runtime: _RuntimeConfig | Mapping[str, Any] | None = None,
        schema_version: str = "fabric.agent/v1alpha1",
        environment: _EnvironmentConfig | Mapping[str, Any] | None = None,
        models: Mapping[str, Any] | None = None,
        system_prompt: str | None = None,
        max_turns: int | None = None,
        mcp: Mapping[str, Any] | None = None,
        skills: Mapping[str, Any] | None = None,
        telemetry: Mapping[str, Any] | None = None,
        relay: Mapping[str, Any] | None = None,
        tools: Mapping[str, Any] | None = None,
        extra_fields: Mapping[str, Any] | None = None,
    ) -> None:
        metadata_value = _coerce(_MetadataConfig, metadata, "metadata")
        harness_value = _coerce(_HarnessConfig, harness, "harness")
        runtime_value = _coerce(
            _RuntimeConfig,
            _RuntimeConfig() if runtime is None else runtime,
            "runtime",
        )
        environment_value = None if environment is None else _coerce(_EnvironmentConfig, environment, "environment")
        mcp_value = None if mcp is None else _coerce(_McpConfig, mcp, "mcp")
        skills_value = None if skills is None else _coerce(_SkillConfig, skills, "skills")
        telemetry_value = None if telemetry is None else _coerce(_TelemetryConfig, telemetry, "telemetry")
        relay_value = None if relay is None else _mapping(relay, "relay")
        tools_value = None if tools is None else _coerce(_ToolsConfig, tools, "tools")
        values: dict[str, Any] = {
            "schema_version": _required_text(schema_version, "schema_version"),
            "metadata": metadata_value,
            "harness": harness_value,
            "runtime": runtime_value,
            "models": _mapping({} if models is None else models, "models"),
        }
        if system_prompt is not None:
            if not isinstance(system_prompt, str):
                raise FabricConfigError("system_prompt must be a string")
            values["system_prompt"] = system_prompt
        if max_turns is not None:
            if (
                isinstance(max_turns, bool)
                or not isinstance(max_turns, int)
                or max_turns <= 0
            ):
                raise FabricConfigError("max_turns must be greater than zero")
            values["max_turns"] = max_turns
        for key, item in (
            ("environment", environment_value),
            ("mcp", mcp_value),
            ("skills", skills_value),
            ("telemetry", telemetry_value),
            ("relay", relay_value),
            ("tools", tools_value),
        ):
            if item is not None:
                values[key] = item
        super().__init__(values, extra_fields=extra_fields)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "_FabricConfigSnapshot":
        """Build a typed agent config from a mapping."""

        data = _mapping(value, "FabricConfig")
        if "profiles" in data:
            raise FabricConfigError(
                "FabricConfig profiles are no longer supported; "
                "compose a complete typed config before calling the SDK"
            )
        if "metadata" not in data:
            raise FabricConfigError("FabricConfig metadata is required")
        if "harness" not in data:
            raise FabricConfigError("FabricConfig harness is required")
        return cls(
            schema_version=data.get("schema_version", "fabric.agent/v1alpha1"),
            metadata=data["metadata"],
            harness=data["harness"],
            runtime=data.get("runtime"),
            environment=data.get("environment"),
            models=data.get("models"),
            system_prompt=data.get("system_prompt"),
            max_turns=data.get("max_turns"),
            mcp=data.get("mcp"),
            skills=data.get("skills"),
            telemetry=data.get("telemetry"),
            relay=data.get("relay"),
            tools=data.get("tools"),
            extra_fields={key: item for key, item in data.items() if key not in cls._fields},
        )

    @property
    def mcp(self) -> _McpConfig:
        """Mutable MCP capability config, created on first access."""

        return self._ensure_section("mcp", _McpConfig)

    @mcp.setter
    def mcp(self, value: _McpConfig | Mapping[str, Any]) -> None:
        self["mcp"] = _coerce(_McpConfig, value, "mcp")

    @property
    def skills(self) -> _SkillConfig:
        """Mutable skill capability config, created on first access."""

        return self._ensure_section("skills", _SkillConfig)

    @skills.setter
    def skills(self, value: _SkillConfig | Mapping[str, Any]) -> None:
        self["skills"] = _coerce(_SkillConfig, value, "skills")

    @property
    def tools(self) -> _ToolsConfig:
        """Mutable tool capability config, created on first access."""

        return self._ensure_section("tools", _ToolsConfig)

    @tools.setter
    def tools(self, value: _ToolsConfig | Mapping[str, Any]) -> None:
        self["tools"] = _coerce(_ToolsConfig, value, "tools")

    @property
    def telemetry(self) -> _TelemetryConfig:
        """Mutable telemetry config, created on first access."""

        return self._ensure_section("telemetry", _TelemetryConfig)

    @telemetry.setter
    def telemetry(self, value: _TelemetryConfig | Mapping[str, Any]) -> None:
        self["telemetry"] = _coerce(_TelemetryConfig, value, "telemetry")

    def _ensure_section(self, key: str, model: type[_T]) -> _T:
        value = self.get(key)
        if value is None:
            value = model()  # type: ignore[call-arg]
            self[key] = value
        elif not isinstance(value, model):
            value = _coerce(model, value, key)
            self[key] = value
        return value

    def add_mcp_server(
        self,
        name: str,
        *,
        transport: str,
        url: str,
        exposure: str = "harness_native",
        extra_fields: Mapping[str, Any] | None = None,
    ) -> "_FabricConfigSnapshot":
        """Add or replace a named MCP server and return this config."""

        self.mcp.add_server(
            name,
            transport=transport,
            url=url,
            exposure=exposure,
            extra_fields=extra_fields,
        )
        return self

    def add_skill_path(self, path: str | Path) -> "_FabricConfigSnapshot":
        """Add a skill path and return this config."""

        self.skills.add_path(path)
        return self

    def block_tools(self, *tools: str) -> _FabricConfigSnapshot:
        """Block adapter-native tool names or toolsets and return this config."""

        self.tools.block(*tools)
        return self

    def enable_relay(
        self,
        *,
        project: str | None = None,
        output_dir: str | Path | None = None,
        observability: Mapping[str, Any] | None = None,
        components: Sequence[Mapping[str, Any]] | None = None,
        policy: Mapping[str, Any] | None = None,
    ) -> "_FabricConfigSnapshot":
        """Enable NeMo Relay telemetry and return this config."""

        self.telemetry.enable_relay()
        relay = dict(self.get("relay") or {})
        if project is not None:
            relay["project"] = project
        if output_dir is not None:
            relay["output_dir"] = str(output_dir)
        if observability is not None:
            relay["observability"] = _mapping(observability, "relay observability")
        if components is not None:
            relay["components"] = [_mapping(component, "relay component") for component in components]
        if policy is not None:
            relay["policy"] = _mapping(policy, "relay policy")
        self["relay"] = relay
        return self


def _freeze(value: Any) -> Any:
    if isinstance(value, FabricMapping):
        return value
    if isinstance(value, _ConfigMapping):
        return deepcopy(value)
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, _ConfigMapping):
        return value.to_mapping()
    if isinstance(value, FabricMapping):
        return value.to_mapping()
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return deepcopy(value)


def _snapshot_value(value: Any, *, json_value: bool) -> Any:
    if isinstance(value, _ConfigMapping):
        return deepcopy(value)
    if json_value:
        return _thaw(value)
    return value


class FabricMapping(Mapping[str, Any]):
    """Immutable mapping-compatible base for SDK snapshots and results.

    lazydocs: ignore
    """

    _fields: frozenset[str] = frozenset()
    _json_fields: frozenset[str] = frozenset()
    _omit_if_empty: frozenset[str] = frozenset()

    def __init__(self, mapping: Mapping[str, Any]) -> None:
        data = self._normalize(_mapping(mapping, type(self).__name__))
        object.__setattr__(self, "_data", _freeze(data))

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "FabricMapping":
        """Validate and copy a mapping into the requested typed model."""

        return cls(mapping)

    @classmethod
    def _normalize(cls, data: dict[str, Any]) -> dict[str, Any]:
        return data

    def __getitem__(self, key: str) -> Any:
        return _snapshot_value(
            self._data[key],
            json_value=key in self._json_fields or key not in self._fields,
        )

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __getattr__(self, name: str) -> Any:
        try:
            return _snapshot_value(
                self._data[name],
                json_value=name in self._json_fields or name not in self._fields,
            )
        except KeyError as error:
            raise AttributeError(name) from error

    @property
    def extra_fields(self) -> Mapping[str, Any]:
        """Return an immutable view of preserved extension fields."""

        return MappingProxyType({key: _thaw(value) for key, value in self._data.items() if key not in self._fields})

    def to_mapping(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible mapping for serialization."""

        data = _thaw(self._data)
        for key in self._omit_if_empty:
            if data.get(key) in ({}, []):
                data.pop(key, None)
        return data

    def to_dict(self) -> dict[str, Any]:
        """Return the same detached representation as ``to_mapping()``."""

        return self.to_mapping()


class AdapterInfo(FabricMapping):
    """Resolved adapter identity attached to a run plan.

    Attributes:
        adapter_id: Stable identifier of the Fabric adapter implementation.
        harness: Stable machine-readable harness identifier.
        adapter_kind: Execution mechanism used by the adapter.
        metadata: Adapter-specific, JSON-compatible metadata.
    """

    adapter_id: str
    harness: str
    adapter_kind: str
    metadata: Mapping[str, Any]
    _fields = frozenset({"adapter_id", "harness", "adapter_kind", "metadata"})
    _json_fields = frozenset({"metadata"})

    @classmethod
    def _normalize(cls, data: dict[str, Any]) -> dict[str, Any]:
        data["adapter_id"] = _required_text(data.get("adapter_id"), "adapter_id")
        data["harness"] = _required_text(data.get("harness"), "harness")
        data["adapter_kind"] = _required_text(data.get("adapter_kind"), "adapter_kind")
        data["metadata"] = _mapping(data.get("metadata", {}), "adapter metadata")
        return data


class RuntimeCapabilities(FabricMapping):
    """Operations declared by the resolved runtime and adapter.

    Capabilities describe what the selected runtime can support; callers should
    still expect a capability-specific error when a transport is modeled but
    not implemented.

    Attributes:
        service: Whether long-lived service handles are supported.
        streaming: Whether event streaming is supported.
        updates: Whether runtime configuration updates are supported.
        cancellation: Whether in-flight cancellation is supported.
        metadata: Additional capability details.
    """

    service: bool
    streaming: bool
    updates: bool
    cancellation: bool
    metadata: Mapping[str, Any]
    _fields = frozenset(
        {
            "service",
            "streaming",
            "updates",
            "cancellation",
            "metadata",
        }
    )
    _json_fields = frozenset({"metadata"})
    _omit_if_empty = frozenset({"metadata"})

    @classmethod
    def _normalize(cls, data: dict[str, Any]) -> dict[str, Any]:
        for field in cls._fields - {"metadata"}:
            data[field] = _boolean(data.get(field, False), f"{field} capability")
        data["metadata"] = _mapping(data.get("metadata", {}), "capability metadata")
        return data


class RunPlan(FabricMapping):
    """Immutable execution plan produced before a runtime is started.

    Attributes:
        agent_name: Resolved agent name.
        base_dir: Base directory used to resolve relative paths.
        config: Typed configuration snapshot.
        adapter: Resolved adapter identity.
        capabilities: Operations declared by the resolved runtime.
    """

    agent_name: str
    base_dir: Path
    config: _FabricConfigSnapshot
    adapter: AdapterInfo
    capabilities: RuntimeCapabilities
    _fields = frozenset({"agent_name", "base_dir", "config", "adapter", "capabilities"})

    @classmethod
    def _normalize(cls, data: dict[str, Any]) -> dict[str, Any]:
        descriptor = data.get("adapter")
        if descriptor is None:
            descriptor = (data.get("adapter_descriptor") or {}).get("descriptor", {})
        if "base_dir" not in data:
            raise FabricConfigError("RunPlan base_dir is required")
        data["base_dir"] = Path(data["base_dir"])
        data["config"] = _FabricConfigSnapshot.from_mapping(data.get("config", {}))
        data["adapter"] = AdapterInfo.from_mapping(descriptor)
        data["capabilities"] = RuntimeCapabilities.from_mapping(data.get("capabilities", {}))
        return data


class DoctorCheck(FabricMapping):
    """One diagnostic check in a ``DoctorReport``.

    Attributes:
        name: Stable check identifier.
        status: Check outcome: ``pass``, ``warn``, or ``fail``.
        message: Human-readable result.
        metadata: Structured check details.
    """

    name: str
    status: str
    message: str
    metadata: Mapping[str, Any]
    _fields = frozenset({"name", "status", "message", "metadata"})
    _json_fields = frozenset({"metadata"})

    @classmethod
    def _normalize(cls, data: dict[str, Any]) -> dict[str, Any]:
        data["metadata"] = _mapping(data.get("metadata", {}), "doctor metadata")
        return data


class DoctorReport(FabricMapping):
    """Aggregate preflight diagnostics for a resolved run plan.

    Attributes:
        agent_name: Resolved agent name.
        status: Aggregate outcome: ``pass``, ``warn``, or ``fail``.
        checks: Ordered ``DoctorCheck`` results.
    """

    agent_name: str
    status: str
    checks: Sequence[DoctorCheck]
    _fields = frozenset({"agent_name", "status", "checks"})

    @classmethod
    def _normalize(cls, data: dict[str, Any]) -> dict[str, Any]:
        data["checks"] = tuple(DoctorCheck.from_mapping(check) for check in data.get("checks", []))
        return data


class ErrorInfo(FabricMapping):
    """Structured failure returned inside a normalized ``RunResult``.

    Attributes:
        stage: Lifecycle stage that failed.
        code: Stable machine-readable error code.
        message: Human-readable failure description.
        retryable: Whether retrying may succeed without changing the request.
        metadata: Adapter- or runtime-specific details.
    """

    stage: str
    code: str
    message: str
    retryable: bool
    metadata: Mapping[str, Any]
    _fields = frozenset({"stage", "code", "message", "retryable", "metadata"})
    _json_fields = frozenset({"metadata"})

    @classmethod
    def _normalize(cls, data: dict[str, Any]) -> dict[str, Any]:
        data["metadata"] = _mapping(data.get("metadata", {}), "error metadata")
        return data


class ArtifactRef(FabricMapping):
    """Reference to one artifact produced by a run.

    Attributes:
        name: Stable artifact name.
        kind: Artifact category.
        path: Artifact path under the manifest root or workspace.
        media_type: Optional MIME type.
        metadata: Artifact-specific details.
    """

    name: str
    kind: str
    path: Path
    media_type: str | None
    metadata: Mapping[str, Any]
    _fields = frozenset({"name", "kind", "path", "media_type", "metadata"})
    _json_fields = frozenset({"metadata"})

    @classmethod
    def _normalize(cls, data: dict[str, Any]) -> dict[str, Any]:
        data["path"] = Path(data["path"])
        data["metadata"] = _mapping(data.get("metadata", {}), "artifact metadata")
        return data


class ArtifactManifest(FabricMapping):
    """Normalized collection of artifacts produced by a run.

    Attributes:
        root: Optional common artifact root.
        artifacts: Ordered ``ArtifactRef`` entries.
    """

    root: Path | None
    artifacts: Sequence[ArtifactRef]
    _fields = frozenset({"root", "artifacts"})

    @classmethod
    def _normalize(cls, data: dict[str, Any]) -> dict[str, Any]:
        data["root"] = None if data.get("root") is None else Path(data["root"])
        data["artifacts"] = tuple(ArtifactRef.from_mapping(artifact) for artifact in data.get("artifacts", []))
        return data


class TelemetryRef(FabricMapping):
    """Reference to external or persisted telemetry for a run.

    Attributes:
        provider: Telemetry provider, such as Relay.
        kind: Reference kind, such as ``trace``.
        uri: Optional location of persisted telemetry.
        trace_id: Optional provider trace identifier.
        metadata: Provider-specific details.
    """

    provider: str
    kind: str
    uri: str | None
    trace_id: str | None
    metadata: Mapping[str, Any]
    _fields = frozenset({"provider", "kind", "uri", "trace_id", "metadata"})
    _json_fields = frozenset({"metadata"})

    @classmethod
    def _normalize(cls, data: dict[str, Any]) -> dict[str, Any]:
        metadata = _mapping(data.get("metadata", {}), "telemetry metadata")
        if "relay_enabled" in data:
            providers = metadata.get("telemetry_providers")
            provider = (
                providers[0] if isinstance(providers, list) and providers and isinstance(providers[0], str) else "relay"
            )
            metadata.setdefault("relay_enabled", data["relay_enabled"])
            data = {
                "provider": provider,
                "kind": "trace",
                "uri": metadata.get("relay_output_dir"),
                "trace_id": metadata.get("trace_id"),
                "metadata": metadata,
            }
        else:
            data.setdefault("uri", None)
            data.setdefault("trace_id", None)
            data["metadata"] = metadata
        return data


class FabricEvent(FabricMapping):
    """One normalized lifecycle or invocation event.

    Attributes:
        event_id: Stable event identifier.
        timestamp_millis: Event time as Unix epoch milliseconds.
        kind: Machine-readable event kind.
        message: Human-readable event description.
        metadata: Event-specific structured details.
    """

    event_id: str
    timestamp_millis: int
    kind: str
    message: str
    metadata: Mapping[str, Any]
    _fields = frozenset({"event_id", "timestamp_millis", "kind", "message", "metadata"})
    _json_fields = frozenset({"metadata"})

    @classmethod
    def _normalize(cls, data: dict[str, Any]) -> dict[str, Any]:
        data["metadata"] = _mapping(data.get("metadata", {}), "event metadata")
        return data


class RuntimeHandle(FabricMapping):
    """Opaque identity and binding for one started runtime.

    Applications should treat ``runtime_binding`` as opaque. Fabric validates
    the handle against the run plan before invocation or shutdown.

    Attributes:
        runtime_id: Unique identifier for this runtime lifecycle.
        runtime_binding: Opaque integrity-bound runtime binding.
        agent_name: Resolved agent name.
        harness: Stable harness identifier.
        adapter_kind: Adapter execution mechanism.
        adapter_id: Optional Fabric adapter identifier.
        environment: Prepared environment snapshot.
    """

    runtime_id: str
    runtime_binding: str
    agent_name: str
    harness: str
    adapter_kind: str
    adapter_id: str | None
    environment: Mapping[str, Any]
    _fields = frozenset(
        {
            "runtime_id",
            "runtime_binding",
            "agent_name",
            "harness",
            "adapter_kind",
            "adapter_id",
            "environment",
        }
    )
    _json_fields = frozenset({"environment"})

    @classmethod
    def _normalize(cls, data: dict[str, Any]) -> dict[str, Any]:
        for field in (
            "runtime_id",
            "runtime_binding",
            "agent_name",
            "harness",
            "adapter_kind",
        ):
            data[field] = _required_text(data.get(field), field.replace("_", " "))
        if data.get("adapter_id") is not None:
            data["adapter_id"] = _required_text(data["adapter_id"], "adapter id")
        data["environment"] = _mapping(data.get("environment"), "environment")
        return data


class RunOutput(FabricMapping):
    """Normalized adapter output.

    ``response`` is a known adapter response field whose value follows the
    core Fabric JSON contract. Other keys are adapter-specific extensions.
    """

    response: JSONValue | None
    _fields = frozenset({"response"})
    _json_fields = frozenset({"response"})

    @property
    def response(self) -> JSONValue | None:
        """Return the raw ``response`` JSON value, or ``None`` when absent."""

        return _snapshot_value(self._data.get("response"), json_value=True)


class RunResult(FabricMapping):
    """Normalized terminal result from one Fabric invocation.

    The model is both attribute-accessible and mapping-compatible. A harness
    failure can be represented by ``status`` and ``error`` without raising when
    the adapter successfully returns a normalized result.

    Attributes:
        agent_name: Resolved agent name.
        harness: Stable harness identifier.
        adapter_kind: Adapter execution mechanism.
        adapter_id: Fabric adapter identifier.
        runtime_id: Runtime lifecycle identifier.
        invocation_id: Identifier for this invocation.
        request_id: Correlated request identifier.
        status: Terminal invocation status.
        output: Object-shaped adapter output as ``RunOutput``; non-object values
            are preserved as-is.
        error: Structured failure, or ``None`` on success.
        artifacts: Normalized artifact manifest.
        telemetry: Ordered telemetry references.
        events: Ordered lifecycle and invocation events.
        metadata: Result-specific structured details.
    """

    agent_name: str
    harness: str
    adapter_kind: str
    adapter_id: str
    runtime_id: str
    invocation_id: str
    request_id: str
    status: str
    output: RunOutput | JSONValue
    error: ErrorInfo | None
    artifacts: ArtifactManifest
    telemetry: Sequence[TelemetryRef]
    events: Sequence[FabricEvent]
    metadata: Mapping[str, Any]
    _fields = frozenset(
        {
            "agent_name",
            "harness",
            "adapter_kind",
            "adapter_id",
            "runtime_id",
            "invocation_id",
            "request_id",
            "status",
            "output",
            "error",
            "artifacts",
            "telemetry",
            "events",
            "metadata",
        }
    )
    _json_fields = frozenset({"metadata"})

    @classmethod
    def _normalize(cls, data: dict[str, Any]) -> dict[str, Any]:
        for field in (
            "agent_name",
            "harness",
            "adapter_kind",
            "runtime_id",
            "invocation_id",
            "request_id",
            "status",
        ):
            data[field] = _required_text(data.get(field), field.replace("_", " "))
        data["error"] = None if data.get("error") is None else ErrorInfo.from_mapping(data["error"])
        data["artifacts"] = ArtifactManifest.from_mapping(data.get("artifacts", {"artifacts": []}))
        telemetry = data.get("telemetry")
        if telemetry is None:
            data["telemetry"] = ()
        elif isinstance(telemetry, Mapping):
            data["telemetry"] = (TelemetryRef.from_mapping(telemetry),)
        else:
            data["telemetry"] = tuple(TelemetryRef.from_mapping(item) for item in telemetry)
        data["events"] = tuple(FabricEvent.from_mapping(event) for event in data.get("events", []))
        data["metadata"] = _mapping(data.get("metadata", {}), "result metadata")
        raw_output = data.get("output")
        if isinstance(raw_output, RunOutput):
            data["output"] = raw_output
        elif isinstance(raw_output, Mapping):
            data["output"] = RunOutput.from_mapping(raw_output)
        return data
