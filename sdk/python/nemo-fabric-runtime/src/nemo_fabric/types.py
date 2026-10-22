# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public data contracts for the NeMo Fabric Python SDK."""

from __future__ import annotations

import math
import re
from collections.abc import Iterator
from collections.abc import Mapping
from collections.abc import Sequence
from copy import deepcopy
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from types import MappingProxyType
from typing import Any
from typing import Self
from typing import TypeVar

from pydantic import BaseModel

from nemo_fabric.errors import FabricConfigError

JSONScalar = str | int | float | bool | None
JSONValue = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]

_T = TypeVar("_T")
_UINT64_MAX = (1 << 64) - 1
_DEFAULT_SYSTEM_STATS: Mapping[str, Any] = MappingProxyType({})


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
    raise FabricConfigError(
        f"value of type {type(value).__name__} is not JSON-compatible"
    )


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


# RFC 3339 date-time shape. fromisoformat also accepts a space separator,
# date-only values and basic ISO 8601 forms, so check the shape first.
_RFC3339_DATE_TIME = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})"
)


def _utc_timestamp(value: Any, name: str) -> str:
    text = _required_text(value, name)
    message = f"{name} must be an RFC 3339 UTC timestamp"
    if _RFC3339_DATE_TIME.fullmatch(text) is None:
        raise FabricConfigError(message)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        raise FabricConfigError(message) from None
    if parsed.utcoffset() != timedelta(0):
        raise FabricConfigError(message)
    return text


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
            raise FabricConfigError(
                f"extra_fields duplicates known fields: {', '.join(sorted(overlap))}"
            )
        stored = {
            key: deepcopy(item) if isinstance(item, _ConfigMapping) else _plain(item)
            for key, item in values.items()
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
        self[name] = (
            deepcopy(value) if isinstance(value, _ConfigMapping) else _plain(value)
        )

    @property
    def extra_fields(self) -> dict[str, Any]:
        """Return preserved schema-extension fields as a deep copy."""

        return {
            key: _plain(value) for key, value in self.items() if key not in self._fields
        }

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
            extra_fields={
                key: item for key, item in data.items() if key not in cls._fields
            },
        )


class _HarnessConfig(_ConfigMapping):
    """Harness adapter selection and adapter-owned settings.

    Attributes:
        adapter_id: Stable identifier of the NeMo Fabric adapter to resolve.
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
            extra_fields={
                key: item for key, item in data.items() if key not in cls._fields
            },
        )


class _WorkflowConfig(_ConfigMapping):
    """Registered workflow target and construction settings."""

    _fields = frozenset({"target_id", "settings"})
    _omit_if_empty = frozenset({"settings"})

    def __init__(
        self,
        *,
        target_id: str,
        settings: Mapping[str, Any] | None = None,
        extra_fields: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            {
                "target_id": _required_text(target_id, "workflow target_id"),
                "settings": _mapping(
                    {} if settings is None else settings,
                    "workflow settings",
                ),
            },
            extra_fields=extra_fields,
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "_WorkflowConfig":
        """Validate a workflow mapping."""

        data = _mapping(value, "workflow")
        if "target_id" not in data:
            raise FabricConfigError("workflow target_id is required")
        return cls(
            target_id=data["target_id"],
            settings=data.get("settings"),
            extra_fields={
                key: item for key, item in data.items() if key not in cls._fields
            },
        )


class _DiscoveryConfig(_ConfigMapping):
    """Explicit local descriptor discovery paths."""

    _fields = frozenset({"local_paths"})
    _omit_if_empty = frozenset({"local_paths"})

    def __init__(
        self,
        *,
        local_paths: Sequence[str | Path] = (),
        extra_fields: Mapping[str, Any] | None = None,
    ) -> None:
        paths = [str(path) for path in local_paths]
        if any(not path.strip() for path in paths):
            raise FabricConfigError("discovery local paths must not be empty")
        super().__init__({"local_paths": paths}, extra_fields=extra_fields)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "_DiscoveryConfig":
        """Validate descriptor discovery configuration."""

        data = _mapping(value, "discovery")
        return cls(
            local_paths=data.get("local_paths", ()),
            extra_fields={
                key: item for key, item in data.items() if key not in cls._fields
            },
        )


class _InstructionConfig(_ConfigMapping):
    """One portable instruction value."""

    _fields = frozenset({"content", "mode"})

    def __init__(
        self,
        *,
        content: str,
        mode: str = "replace",
        extra_fields: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(mode, str) or mode not in {"replace", "append"}:
            raise FabricConfigError("instruction mode must be replace or append")
        super().__init__(
            {
                "content": _required_text(content, "instruction content"),
                "mode": mode,
            },
            extra_fields=extra_fields,
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "_InstructionConfig":
        """Validate an instruction mapping and preserve extension fields."""

        data = _mapping(value, "instruction")
        return cls(
            content=data.get("content"),
            mode=data.get("mode", "replace"),
            extra_fields={
                key: item for key, item in data.items() if key not in cls._fields
            },
        )


class _InstructionsConfig(_ConfigMapping):
    """Harness-neutral agent instructions."""

    _fields = frozenset({"system"})

    def __init__(
        self,
        *,
        system: _InstructionConfig | Mapping[str, Any] | None = None,
        extra_fields: Mapping[str, Any] | None = None,
    ) -> None:
        values: dict[str, Any] = {}
        if system is not None:
            values["system"] = _coerce(_InstructionConfig, system, "system instruction")
        super().__init__(values, extra_fields=extra_fields)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "_InstructionsConfig":
        """Validate an instructions mapping and preserve extension fields."""

        data = _mapping(value, "instructions")
        return cls(
            system=data.get("system"),
            extra_fields={
                key: item for key, item in data.items() if key not in cls._fields
            },
        )


class _RuntimeConfig(_ConfigMapping):
    """Invocation runtime contract.

    Attributes:
        input_schema: Optional logical input contract identifier.
        output_schema: Optional logical output contract identifier.
        artifacts: Optional artifact-root path.
        timeout_seconds: Optional invocation deadline in seconds.
        max_turns: Optional harness turn limit.
        extra_fields: Preserved extension fields not recognized by this SDK.
    """

    _fields = frozenset(
        {"input_schema", "output_schema", "artifacts", "timeout_seconds", "max_turns"}
    )

    def __init__(
        self,
        *,
        input_schema: str | None = None,
        output_schema: str | None = None,
        artifacts: str | Path | None = None,
        timeout_seconds: float | None = None,
        max_turns: int | None = None,
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
        if max_turns is not None:
            if (
                isinstance(max_turns, bool)
                or not isinstance(max_turns, int)
                or max_turns <= 0
                or max_turns > (1 << 32) - 1
            ):
                raise FabricConfigError(
                    "runtime max_turns must be between 1 and 4294967295"
                )
            values["max_turns"] = max_turns
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
            max_turns=data.get("max_turns"),
            extra_fields={
                key: item for key, item in data.items() if key not in cls._fields
            },
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

    _fields = frozenset(
        {"provider", "workspace", "artifacts", "env", "settings", "metadata"}
    )
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
            extra_fields={
                key: item for key, item in data.items() if key not in cls._fields
            },
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
        values: dict[str, Any] = {
            "paths": [str(path) for path in ([] if paths is None else paths)]
        }
        super().__init__(values, extra_fields=extra_fields)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "_SkillConfig":
        """Validate a skill mapping and preserve extension fields."""

        data = _mapping(value, "skills")
        return cls(
            paths=data.get("paths", []),
            extra_fields={
                key: item for key, item in data.items() if key not in cls._fields
            },
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


class _ToolDefinitionConfig(_ConfigMapping):
    """One named normalized tool or tool-group definition."""

    _fields = frozenset({"kind", "ref", "settings"})
    _omit_if_empty = frozenset({"settings"})

    def __init__(
        self,
        *,
        kind: str,
        ref: str,
        settings: Mapping[str, Any] | None = None,
        extra_fields: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            {
                "kind": _required_text(kind, "tool definition kind"),
                "ref": _required_text(ref, "tool definition ref"),
                "settings": _mapping(
                    {} if settings is None else settings,
                    "tool definition settings",
                ),
            },
            extra_fields=extra_fields,
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> _ToolDefinitionConfig:
        """Validate a tool definition and preserve extension fields."""

        data = _mapping(value, "tool definition")
        return cls(
            kind=data.get("kind"),
            ref=data.get("ref"),
            settings=data.get("settings"),
            extra_fields={
                key: item for key, item in data.items() if key not in cls._fields
            },
        )


class _ToolsConfig(_ConfigMapping):
    """Harness-neutral tool capability configuration."""

    _fields = frozenset({"definitions", "enabled", "blocked"})
    _omit_if_empty = frozenset({"definitions", "blocked"})

    def __init__(
        self,
        *,
        definitions: Mapping[str, Any] | None = None,
        enabled: Sequence[str] | None = None,
        blocked: Sequence[str] | None = None,
        extra_fields: Mapping[str, Any] | None = None,
    ) -> None:
        for name, values in (("enabled", enabled), ("blocked", blocked)):
            if values is not None and (
                isinstance(values, (str, bytes)) or not isinstance(values, Sequence)
            ):
                raise FabricConfigError(
                    f"tools {name} must be an ordered sequence of strings"
                )
        enabled_values = (
            None
            if enabled is None
            else [_required_text(tool, "enabled tool") for tool in enabled]
        )
        blocked_values = [
            _required_text(tool, "blocked tool") for tool in (blocked or [])
        ]
        overlap = set(enabled_values or []).intersection(blocked_values)
        if overlap:
            name = sorted(overlap)[0]
            raise FabricConfigError(f"tool {name!r} cannot be both enabled and blocked")
        raw_definitions = _mapping(
            {} if definitions is None else definitions,
            "tool definitions",
        )
        definition_values = {
            _required_text(name, "tool definition name"): _coerce(
                _ToolDefinitionConfig,
                definition,
                f"tool definition {name}",
            )
            for name, definition in raw_definitions.items()
        }
        values: dict[str, Any] = {
            "definitions": definition_values,
            "blocked": blocked_values,
        }
        if enabled_values is not None:
            values["enabled"] = enabled_values
        super().__init__(values, extra_fields=extra_fields)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "_ToolsConfig":
        """Validate a tools mapping and preserve extension fields."""

        data = _mapping(value, "tools")
        return cls(
            definitions=data.get("definitions"),
            enabled=data.get("enabled"),
            blocked=data.get("blocked", []),
            extra_fields={
                key: item for key, item in data.items() if key not in cls._fields
            },
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

    def add_definition(
        self,
        name: str,
        *,
        kind: str,
        ref: str,
        settings: Mapping[str, Any] | None = None,
        extra_fields: Mapping[str, Any] | None = None,
    ) -> _ToolsConfig:
        """Add or replace one named definition."""

        definitions = dict(self.get("definitions", {}))
        definitions[_required_text(name, "tool definition name")] = (
            _ToolDefinitionConfig(
                kind=kind,
                ref=ref,
                settings=settings,
                extra_fields=extra_fields,
            )
        )
        self["definitions"] = definitions
        return self

    def remove_definition(self, name: str) -> _ToolsConfig:
        """Remove one named definition."""

        definitions = dict(self.get("definitions", {}))
        definitions.pop(name, None)
        self["definitions"] = definitions
        return self


class _McpConfig(_ConfigMapping):
    """MCP capability configuration with authoring helpers."""

    _fields = frozenset({"servers"})
    _omit_if_empty = frozenset({"servers"})
    _EXPOSURES = frozenset({"harness_native", "fabric_managed"})
    _SERVER_FIELDS = frozenset(
        {"transport", "url", "exposure", "allowed_tools", "blocked_tools"}
    )

    def __init__(
        self,
        *,
        servers: Mapping[str, Any] | None = None,
        extra_fields: Mapping[str, Any] | None = None,
    ) -> None:
        values: dict[str, Any] = {
            "servers": _mapping({} if servers is None else servers, "mcp servers")
        }
        super().__init__(values, extra_fields=extra_fields)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "_McpConfig":
        """Validate an MCP mapping and preserve extension fields."""

        data = _mapping(value, "mcp")
        return cls(
            servers=data.get("servers", {}),
            extra_fields={
                key: item for key, item in data.items() if key not in cls._fields
            },
        )

    def add_server(
        self,
        name: str,
        *,
        transport: str,
        url: str,
        authentication: Mapping[str, Any] | None = None,
        exposure: str = "harness_native",
        allowed_tools: Sequence[str] | None = None,
        blocked_tools: Sequence[str] = (),
        extra_fields: Mapping[str, Any] | None = None,
    ) -> "_McpConfig":
        """Add or replace a named MCP server."""

        if exposure not in self._EXPOSURES:
            allowed = ", ".join(sorted(self._EXPOSURES))
            raise FabricConfigError(f"mcp exposure must be one of: {allowed}")
        for policy, tools in (
            ("allowed_tools", allowed_tools),
            ("blocked_tools", blocked_tools),
        ):
            if tools is not None and (
                isinstance(tools, (str, bytes)) or not isinstance(tools, Sequence)
            ):
                raise FabricConfigError(
                    f"mcp {policy} must be an ordered sequence of strings"
                )
        allowed_values = (
            None
            if allowed_tools is None
            else [_required_text(tool, "allowed MCP tool") for tool in allowed_tools]
        )
        blocked_values = [
            _required_text(tool, "blocked MCP tool") for tool in blocked_tools
        ]
        overlap = set(allowed_values or []).intersection(blocked_values)
        if overlap:
            tool = sorted(overlap)[0]
            raise FabricConfigError(
                f"MCP tool {tool!r} cannot be both allowed and blocked"
            )
        server: dict[str, Any] = {
            "transport": _required_text(transport, "mcp transport"),
            "url": _required_text(url, "mcp url"),
            "exposure": exposure,
        }
        if allowed_values is not None:
            server["allowed_tools"] = allowed_values
        if blocked_values:
            server["blocked_tools"] = blocked_values
        extensions = _mapping(
            {} if extra_fields is None else extra_fields,
            "mcp server extra_fields",
        )
        legacy_authentication = extensions.pop("authentication", None)
        reserved = self._SERVER_FIELDS.intersection(extensions)
        if reserved:
            field = sorted(reserved)[0]
            raise FabricConfigError(
                f"mcp server extra_fields must not contain reserved field {field!r}"
            )
        authentication_value = (
            authentication if authentication is not None else legacy_authentication
        )
        if authentication_value is not None:
            server["authentication"] = _mapping(
                authentication_value, "mcp server authentication"
            )
        server.update(extensions)
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
            extra_fields={
                key: item for key, item in data.items() if key not in cls._fields
            },
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

    def enable_native(
        self, *, config: Mapping[str, Any] | None = None
    ) -> "_TelemetryConfig":
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
    """Typed snapshot of the NeMo Fabric configuration stored in a run plan.

    It is reconstructed from the native plan payload and exposed through the
    immutable ``RunPlan`` mapping. Unknown fields survive round trips through
    ``extra_fields``.

    Attributes:
        schema_version: Agent schema identifier.
        metadata: Required ``MetadataConfig`` agent identity.
        harness: Optional direct adapter selection and settings.
        workflow: Optional registered workflow target and construction settings.
        discovery: Optional explicit descriptor discovery paths.
        runtime: Invocation runtime configuration.
        environment: Optional execution environment configuration.
        models: Named, JSON-compatible model configurations.
        instructions: Optional portable agent instructions.
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
            "workflow",
            "discovery",
            "runtime",
            "environment",
            "models",
            "instructions",
            "mcp",
            "skills",
            "system_stats",
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
        harness: _HarnessConfig | Mapping[str, Any] | None = None,
        workflow: _WorkflowConfig | Mapping[str, Any] | None = None,
        discovery: _DiscoveryConfig | Mapping[str, Any] | None = None,
        runtime: _RuntimeConfig | Mapping[str, Any] | None = None,
        schema_version: str = "fabric.agent/v1alpha1",
        environment: _EnvironmentConfig | Mapping[str, Any] | None = None,
        models: Mapping[str, Any] | None = None,
        instructions: _InstructionsConfig | Mapping[str, Any] | None = None,
        mcp: Mapping[str, Any] | None = None,
        skills: Mapping[str, Any] | None = None,
        telemetry: Mapping[str, Any] | None = None,
        relay: Mapping[str, Any] | None = None,
        system_stats: Mapping[str, Any] | None = _DEFAULT_SYSTEM_STATS,
        tools: Mapping[str, Any] | None = None,
        extra_fields: Mapping[str, Any] | None = None,
    ) -> None:
        metadata_value = _coerce(_MetadataConfig, metadata, "metadata")
        harness_value = (
            None if harness is None else _coerce(_HarnessConfig, harness, "harness")
        )
        workflow_value = (
            None if workflow is None else _coerce(_WorkflowConfig, workflow, "workflow")
        )
        discovery_value = (
            None
            if discovery is None
            else _coerce(_DiscoveryConfig, discovery, "discovery")
        )
        if harness_value is None and workflow_value is None:
            raise FabricConfigError(
                "at least one of harness.adapter_id or workflow.target_id is required"
            )
        runtime_value = _coerce(
            _RuntimeConfig,
            _RuntimeConfig() if runtime is None else runtime,
            "runtime",
        )
        environment_value = (
            None
            if environment is None
            else _coerce(_EnvironmentConfig, environment, "environment")
        )
        instructions_value = (
            None
            if instructions is None
            else _coerce(_InstructionsConfig, instructions, "instructions")
        )
        mcp_value = None if mcp is None else _coerce(_McpConfig, mcp, "mcp")
        skills_value = (
            None if skills is None else _coerce(_SkillConfig, skills, "skills")
        )
        telemetry_value = (
            None
            if telemetry is None
            else _coerce(_TelemetryConfig, telemetry, "telemetry")
        )
        from nemo_fabric.models import SystemStatsConfig

        stats_value = (
            None
            if system_stats is None
            else SystemStatsConfig.from_mapping(system_stats).to_mapping()
        )
        relay_value = None if relay is None else _mapping(relay, "relay")
        tools_value = None if tools is None else _coerce(_ToolsConfig, tools, "tools")
        values: dict[str, Any] = {
            "schema_version": _required_text(schema_version, "schema_version"),
            "metadata": metadata_value,
            "runtime": runtime_value,
            "models": _mapping({} if models is None else models, "models"),
            "system_stats": stats_value,
        }
        for key, item in (
            ("harness", harness_value),
            ("environment", environment_value),
            ("workflow", workflow_value),
            ("discovery", discovery_value),
            ("instructions", instructions_value),
            ("mcp", mcp_value),
            ("skills", skills_value),
            ("system_stats", stats_value),
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
        if "harness" not in data and "workflow" not in data:
            raise FabricConfigError(
                "at least one of harness.adapter_id or workflow.target_id is required"
            )
        return cls(
            schema_version=data.get("schema_version", "fabric.agent/v1alpha1"),
            metadata=data["metadata"],
            harness=data.get("harness"),
            workflow=data.get("workflow"),
            discovery=data.get("discovery"),
            runtime=data.get("runtime"),
            environment=data.get("environment"),
            models=data.get("models"),
            instructions=data.get("instructions"),
            mcp=data.get("mcp"),
            skills=data.get("skills"),
            system_stats=data.get("system_stats", _DEFAULT_SYSTEM_STATS),
            telemetry=data.get("telemetry"),
            relay=data.get("relay"),
            tools=data.get("tools"),
            extra_fields={
                key: item for key, item in data.items() if key not in cls._fields
            },
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
        authentication: Mapping[str, Any] | None = None,
        exposure: str = "harness_native",
        allowed_tools: Sequence[str] | None = None,
        blocked_tools: Sequence[str] = (),
        extra_fields: Mapping[str, Any] | None = None,
    ) -> _FabricConfigSnapshot:
        """Add or replace a named MCP server and return this config."""

        self.mcp.add_server(
            name,
            transport=transport,
            url=url,
            authentication=authentication,
            exposure=exposure,
            allowed_tools=allowed_tools,
            blocked_tools=blocked_tools,
            extra_fields=extra_fields,
        )
        return self

    def add_skill_path(self, path: str | Path) -> "_FabricConfigSnapshot":
        """Add a skill path and return this config."""

        self.skills.add_path(path)
        return self

    def block_tools(self, *tools: str) -> _FabricConfigSnapshot:
        """Block adapter-native tool names and return this config."""

        self.tools.block(*tools)
        return self

    def add_tool_definition(
        self,
        name: str,
        *,
        kind: str,
        ref: str,
        settings: Mapping[str, Any] | None = None,
        extra_fields: Mapping[str, Any] | None = None,
    ) -> _FabricConfigSnapshot:
        """Add or replace one named tool definition and return this config."""

        self.tools.add_definition(
            name,
            kind=kind,
            ref=ref,
            settings=settings,
            extra_fields=extra_fields,
        )
        return self

    def remove_tool_definition(self, name: str) -> _FabricConfigSnapshot:
        """Remove one named tool definition and return this config."""

        tools = self.get("tools")
        if tools is not None:
            if not isinstance(tools, _ToolsConfig):
                raise FabricConfigError("tools must be a _ToolsConfig")
            tools.remove_definition(name)
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
            relay["components"] = [
                _mapping(component, "relay component") for component in components
            ]
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
    def from_mapping(cls, mapping: Mapping[str, Any]) -> Self:
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

        return MappingProxyType(
            {
                key: _thaw(value)
                for key, value in self._data.items()
                if key not in self._fields
            }
        )

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
        adapter_id: Stable identifier of the NeMo Fabric adapter implementation.
        adapter_kind: Execution mechanism used by the adapter.
        metadata: Adapter-specific, JSON-compatible metadata.
    """

    adapter_id: str
    adapter_kind: str
    metadata: Mapping[str, Any]
    _fields = frozenset({"adapter_id", "adapter_kind", "metadata"})
    _json_fields = frozenset({"metadata"})

    @classmethod
    def _normalize(cls, data: dict[str, Any]) -> dict[str, Any]:
        data["adapter_id"] = _required_text(data.get("adapter_id"), "adapter_id")
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
        data["capabilities"] = RuntimeCapabilities.from_mapping(
            data.get("capabilities", {})
        )
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
        data["checks"] = tuple(
            DoctorCheck.from_mapping(check) for check in data.get("checks", [])
        )
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
    _omit_if_empty = frozenset({"metadata"})

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
        data["artifacts"] = tuple(
            ArtifactRef.from_mapping(artifact) for artifact in data.get("artifacts", [])
        )
        return data


class TelemetryRef(FabricMapping):
    """Reference to external or persisted telemetry for a run.

    Attributes:
        provider: Telemetry provider, such as NVIDIA NeMo Relay.
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
        if "provider" in data and "kind" in data:
            data.pop("relay_enabled", None)
        elif "relay_enabled" in data:
            providers = metadata.get("telemetry_providers")
            provider = (
                providers[0]
                if isinstance(providers, list)
                and providers
                and isinstance(providers[0], str)
                else "relay"
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


class EnvironmentReference(FabricMapping):
    """Provider-specific identity for an existing execution environment.

    Pass this reference to ``Fabric.attach_environment()``. The provider verifies
    the resource before returning a durable ``EnvironmentHandle`` and never gains
    deletion authority over the caller-owned resource.

    Attributes:
        provider: Stable environment-provider identifier.
        resource: Provider-specific resource identity.
    """

    provider: str
    resource: Mapping[str, Any]
    _fields = frozenset({"provider", "resource"})
    _json_fields = frozenset({"resource"})
    _omit_if_empty = frozenset({"resource"})

    @classmethod
    def _normalize(cls, data: dict[str, Any]) -> dict[str, Any]:
        unknown = set(data).difference(cls._fields)
        if unknown:
            raise FabricConfigError(
                "EnvironmentReference contains unknown fields: "
                + ", ".join(sorted(unknown))
            )
        data["provider"] = _required_text(data.get("provider"), "provider")
        data["resource"] = _mapping(data.get("resource", {}), "resource")
        return data


class EnvironmentHandle(FabricMapping):
    """Durable identity and provider binding for a prepared or attached environment.

    Environment handles are independent of runtime sessions. Runtime and reuse
    limits are provider-specific. Release the environment separately after
    stopping its runtime and collecting results.

    Attributes:
        environment_id: Unique identifier for the prepared environment.
        provider: Stable environment-provider identifier.
        control_location: Whether Fabric control runs outside or inside the environment.
        workspace: Optional workspace visible to the harness runtime.
        artifacts: Optional artifact root visible to the harness runtime.
        env: Environment variables visible to the harness and its tools.
        ownership: Whether the caller or Fabric owns the environment resource.
        connection: Provider connection metadata.
        metadata: Provider-specific metadata.
    """

    environment_id: str
    provider: str
    control_location: str
    workspace: str | Path | None
    artifacts: str | Path | None
    env: Mapping[str, str]
    ownership: str
    connection: Mapping[str, Any]
    metadata: Mapping[str, Any]
    _fields = frozenset(
        {
            "environment_id",
            "provider",
            "control_location",
            "workspace",
            "artifacts",
            "env",
            "ownership",
            "connection",
            "metadata",
        }
    )
    _json_fields = frozenset({"env", "connection", "metadata"})
    _omit_if_empty = frozenset({"env", "connection", "metadata"})

    @classmethod
    def _normalize(cls, data: dict[str, Any]) -> dict[str, Any]:
        unknown = set(data).difference(cls._fields)
        if unknown:
            raise FabricConfigError(
                "EnvironmentHandle contains unknown fields: "
                + ", ".join(sorted(unknown))
            )
        for field in (
            "environment_id",
            "provider",
            "control_location",
            "ownership",
        ):
            data[field] = _required_text(data.get(field), field.replace("_", " "))
        if data["control_location"] not in ("external_control", "in_env_control"):
            raise FabricConfigError("invalid environment control_location")
        if data["ownership"] not in ("caller_owned", "fabric_owned"):
            raise FabricConfigError("invalid environment ownership")
        for field in ("workspace", "artifacts"):
            value = data.get(field)
            if value is not None:
                if data["provider"] == "local":
                    if isinstance(value, Path):
                        data[field] = value
                    else:
                        data[field] = Path(_required_text(value, field))
                else:
                    # Task paths are strings; never reinterpret them using host OS rules.
                    data[field] = _required_text(value, field)
                    if data["provider"] == "openshell" and (
                        not value.startswith("/")
                        or "\\" in value
                        or "\0" in value
                        or ".." in value.split("/")
                    ):
                        raise FabricConfigError(
                            f"{field} must be an absolute POSIX task path"
                        )
            else:
                data[field] = None
        environment = _mapping(data.get("env", {}), "environment variables")
        for key, value in environment.items():
            _required_text(key, "environment variable name")
            if not isinstance(value, str):
                raise FabricConfigError("environment variable values must be strings")
        data["env"] = environment
        data["connection"] = _mapping(data.get("connection", {}), "connection")
        data["metadata"] = _mapping(data.get("metadata", {}), "environment metadata")
        return data


class RuntimeHandle(FabricMapping):
    """Opaque identity and binding for one started runtime.

    Applications should treat ``runtime_binding`` as opaque. NeMo Fabric validates
    the handle against the run plan before invocation or shutdown.

    Attributes:
        runtime_id: Unique identifier for this runtime lifecycle.
        runtime_binding: Opaque integrity-bound runtime binding.
        agent_name: Resolved agent name.
        harness: Stable harness identifier.
        adapter_kind: Adapter execution mechanism.
        adapter_id: Optional NeMo Fabric adapter identifier.
        service_id: Optional service used by this runtime.
        environment: Prepared environment snapshot.
    """

    runtime_id: str
    runtime_binding: str
    agent_name: str
    harness: str
    adapter_kind: str
    adapter_id: str | None
    service_id: str | None
    environment: Mapping[str, Any]
    _fields = frozenset(
        {
            "runtime_id",
            "runtime_binding",
            "agent_name",
            "harness",
            "adapter_kind",
            "adapter_id",
            "service_id",
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
        if data.get("service_id") is not None:
            data["service_id"] = _required_text(data["service_id"], "service id")
        else:
            data["service_id"] = None
        data["environment"] = _mapping(data.get("environment"), "environment")
        return data


class ServiceReference(FabricMapping):
    """Reference to an already-running caller-owned service.

    ``adapter_id`` selects the adapter that understands the service type and
    connection fields.

    ``connection`` contains endpoints and credential references such as an
    environment-variable name. It must not contain credential values.
    """

    adapter_id: str
    service_type: str
    connection: Mapping[str, Any]
    metadata: Mapping[str, Any]
    _fields = frozenset({"adapter_id", "service_type", "connection", "metadata"})
    _json_fields = frozenset({"connection", "metadata"})
    _omit_if_empty = frozenset({"connection", "metadata"})

    @classmethod
    def _normalize(cls, data: dict[str, Any]) -> dict[str, Any]:
        data["adapter_id"] = _required_text(data.get("adapter_id"), "adapter id")
        data["service_type"] = _required_text(data.get("service_type"), "service type")
        data["connection"] = _mapping(data.get("connection", {}), "connection")
        data["metadata"] = _mapping(data.get("metadata", {}), "metadata")
        return data


class ServiceHandle(FabricMapping):
    """Opaque identity and binding for one prepared or attached service.

    ``service_id`` identifies the process-local Fabric service lifecycle, not
    the underlying remote deployment. ``service_binding`` is opaque and must
    be returned unchanged when the handle is used. ``adapter_id`` identifies
    the adapter that created or attached to the service, and ``service_type``
    is that adapter's stable service kind. ``ownership`` is ``fabric_owned``
    or ``caller_owned``. ``connection`` is a sanitized summary that never
    contains credential values, while ``metadata`` contains adapter-reported
    version and readiness information.
    """

    service_id: str
    service_binding: str
    adapter_id: str
    service_type: str
    ownership: str
    connection: Mapping[str, Any]
    metadata: Mapping[str, Any]
    _fields = frozenset(
        {
            "service_id",
            "service_binding",
            "adapter_id",
            "service_type",
            "ownership",
            "connection",
            "metadata",
        }
    )
    _json_fields = frozenset({"connection", "metadata"})
    _omit_if_empty = frozenset({"connection", "metadata"})

    @classmethod
    def _normalize(cls, data: dict[str, Any]) -> dict[str, Any]:
        for field in (
            "service_id",
            "service_binding",
            "adapter_id",
            "service_type",
            "ownership",
        ):
            data[field] = _required_text(data.get(field), field.replace("_", " "))
        if data["ownership"] not in {"fabric_owned", "caller_owned"}:
            raise FabricConfigError(
                "service ownership must be fabric_owned or caller_owned"
            )
        data["connection"] = _mapping(data.get("connection", {}), "connection")
        data["metadata"] = _mapping(data.get("metadata", {}), "metadata")
        return data


class RunOutput(FabricMapping):
    """Normalized adapter output.

    ``response`` is a known adapter response field whose value follows the
    core NeMo Fabric JSON contract. Other keys are adapter-specific extensions.
    """

    response: JSONValue | None
    _fields = frozenset({"response"})
    _json_fields = frozenset({"response"})

    @property
    def response(self) -> JSONValue | None:
        """Return the raw ``response`` JSON value, or ``None`` when absent."""

        return _snapshot_value(self._data.get("response"), json_value=True)


class RunUsage(FabricMapping):
    """Normalized invocation-local usage reported by an adapter target.

    Missing counters and cost are unknown, not zero. ``cached_input_tokens``
    records cache usage; ``input_tokens_include_cache`` declares whether input
    already includes that count. An absent flag means the semantics are unknown.
    Descriptor-declared detailed usage is preserved in ``metadata``.
    """

    input_tokens: int | None
    cached_input_tokens: int | None
    input_tokens_include_cache: bool | None
    output_tokens: int | None
    total_tokens: int | None
    cost_usd: float | None
    metadata: Mapping[str, Any]
    _fields = frozenset(
        {
            "input_tokens",
            "cached_input_tokens",
            "input_tokens_include_cache",
            "output_tokens",
            "total_tokens",
            "cost_usd",
            "metadata",
        }
    )
    _json_fields = frozenset({"metadata"})

    @classmethod
    def _normalize(cls, data: dict[str, Any]) -> dict[str, Any]:
        for field in (
            "input_tokens",
            "cached_input_tokens",
            "output_tokens",
            "total_tokens",
        ):
            value = data.get(field)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value <= _UINT64_MAX
            ):
                raise FabricConfigError(
                    f"{field.replace('_', ' ')} must be a nonnegative integer "
                    f"no greater than {_UINT64_MAX}"
                )
        includes_cache = data.get("input_tokens_include_cache")
        if includes_cache is not None:
            data["input_tokens_include_cache"] = _boolean(
                includes_cache, "input tokens include cache"
            )
        cost = data.get("cost_usd")
        if cost is not None and (
            isinstance(cost, bool)
            or not isinstance(cost, (int, float))
            or not math.isfinite(cost)
            or cost < 0
        ):
            raise FabricConfigError("cost usd must be a finite nonnegative number")
        if cost is not None:
            data["cost_usd"] = float(cost)
        data["metadata"] = _mapping(data.get("metadata", {}), "usage metadata")
        return data


class RuntimeResourceSnapshot(FabricMapping):
    """Timestamped resource usage of one runtime's local adapter host process tree.

    Memory, process count, and CPU time cover the adapter host process and its
    live descendants, not a shared service process. Values that cannot be read
    raise an error instead of reading as zero.

    Attributes:
        runtime_id: Runtime this snapshot describes.
        started_at: When NeMo Fabric started the adapter host, as RFC 3339 UTC.
        observed_at: When NeMo Fabric read these values, as RFC 3339 UTC.
        resident_memory_bytes: Sum of the live processes' resident set sizes.
            Pages shared between processes count once per process.
        process_count: Live processes in the tree, including the adapter host.
        cpu_time_seconds: CPU time used since the adapter host started,
            including exited descendants. It can drop when a descendant
            leaves the tree.
    """

    runtime_id: str
    started_at: str
    observed_at: str
    resident_memory_bytes: int
    process_count: int
    cpu_time_seconds: float
    _fields = frozenset(
        {
            "runtime_id",
            "started_at",
            "observed_at",
            "resident_memory_bytes",
            "process_count",
            "cpu_time_seconds",
        }
    )

    @classmethod
    def _normalize(cls, data: dict[str, Any]) -> dict[str, Any]:
        data["runtime_id"] = _required_text(data.get("runtime_id"), "runtime id")
        for field in ("started_at", "observed_at"):
            data[field] = _utc_timestamp(data.get(field), field.replace("_", " "))
        for field in ("resident_memory_bytes", "process_count"):
            value = data.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value <= _UINT64_MAX
            ):
                raise FabricConfigError(
                    f"{field.replace('_', ' ')} must be a nonnegative integer "
                    f"no greater than {_UINT64_MAX}"
                )
        cpu = data.get("cpu_time_seconds")
        if (
            isinstance(cpu, bool)
            or not isinstance(cpu, (int, float))
            or not math.isfinite(cpu)
            or cpu < 0
        ):
            raise FabricConfigError(
                "cpu time seconds must be a finite nonnegative number"
            )
        data["cpu_time_seconds"] = float(cpu)
        return data


class ResourceObservation(FabricMapping):
    """Adapter host process-tree reading, or an explicit collection failure.

    Attributes:
        status: ``available``, ``unavailable``, ``unsupported``, or ``not_collected``.
        snapshot: Present only when available; a ``RuntimeResourceSnapshot``.
        reason: Present only when unavailable or unsupported.
    """

    status: str
    _fields = frozenset({"status", "snapshot", "reason"})

    @property
    def snapshot(self) -> RuntimeResourceSnapshot | None:
        """Return the reading when available, otherwise None."""
        return self._data.get("snapshot")

    @property
    def reason(self) -> str | None:
        """Return the collection failure reason, otherwise None."""
        return self._data.get("reason")

    @classmethod
    def _normalize(cls, data: dict[str, Any]) -> dict[str, Any]:
        status = data.get("status")
        if status == "available":
            if "reason" in data:
                raise FabricConfigError("available observation cannot contain reason")
            data["snapshot"] = RuntimeResourceSnapshot.from_mapping(
                data.get("snapshot")
            )
        elif status == "not_collected":
            if "snapshot" in data or "reason" in data:
                raise FabricConfigError(
                    "not-collected observation has no snapshot or reason"
                )
        elif status in ("unavailable", "unsupported"):
            if "snapshot" in data:
                raise FabricConfigError("missing observation cannot contain snapshot")
            data["reason"] = _required_text(data.get("reason"), "observation reason")
        else:
            raise FabricConfigError(
                "observation status must be available, unavailable, unsupported, or not_collected"
            )
        return data


def _stats_number(value: Any, name: str, *, integer: bool = False) -> int | float:
    try:
        finite = (
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(value)
        )
    except OverflowError:
        finite = False
    if not finite or value < 0:
        raise FabricConfigError(f"{name} must be a finite nonnegative number")
    if integer and (not isinstance(value, int) or value > _UINT64_MAX):
        raise FabricConfigError(f"{name} must be an unsigned 64-bit integer")
    return value if integer else float(value)


class Measurement(FabricMapping):
    """Measured numeric value, an unlimited limit, or explicit missing evidence.

    Attributes:
        status: available, unlimited, unavailable, unsupported, or not_collected.
        value: Numeric reading when available, otherwise None.
        reason: Collection failure/capability reason, otherwise None.
    """

    status: str
    _fields = frozenset({"status", "value", "reason"})

    @property
    def value(self) -> int | float | None:
        """Return an available reading, otherwise None."""
        return self._data.get("value")

    @property
    def reason(self) -> str | None:
        """Return the reason for missing evidence, otherwise None."""
        return self._data.get("reason")

    @classmethod
    def _normalize(cls, data: dict[str, Any]) -> dict[str, Any]:
        status = data.get("status")
        if status == "available":
            _stats_number(data.get("value"), "measurement value")
            if "reason" in data:
                raise FabricConfigError("available measurement cannot contain reason")
        elif status in ("unsupported", "unavailable"):
            data["reason"] = _required_text(data.get("reason"), "measurement reason")
            if "value" in data:
                raise FabricConfigError("missing measurement cannot contain value")
        elif status in ("unlimited", "not_collected"):
            if "reason" in data or "value" in data:
                raise FabricConfigError("unlimited measurement has no value or reason")
        else:
            raise FabricConfigError("invalid measurement status")
        return data


def _stats_measurement(
    value: Any, *, integer: bool = True, limit: bool = False
) -> Measurement:
    reading = Measurement.from_mapping(value)
    if reading.status == "available":
        _stats_number(reading.value, "measurement value", integer=integer)
    if reading.status == "unlimited" and not limit:
        raise FabricConfigError("only resource limits may be unlimited")
    return reading


class CgroupSnapshot(FabricMapping):
    """Native adapter-host cgroup measurements, separate from process RSS.

    Counters cover this cgroup and its descendants, potentially including other
    workloads. Limits are visible ancestor bounds; hidden ancestors and actual
    exclusive capacity are unknown. memory_peak_bytes covers cgroup lifetime or
    its last external reset, never this invocation's peak.
    """

    source: str
    scope: str
    target: str
    observed_at: str
    visible_cpu_limit_cores: Measurement
    visible_memory_limit_bytes: Measurement
    memory_current_bytes: Measurement
    memory_peak_bytes: Measurement
    swap_current_bytes: Measurement
    cpu_usage_microseconds: Measurement
    cpu_throttled_microseconds: Measurement
    oom_kills: Measurement
    pressure_stall_microseconds: Mapping[str, Measurement]
    _counters = frozenset(
        {
            "memory_current_bytes",
            "memory_peak_bytes",
            "swap_current_bytes",
            "cpu_usage_microseconds",
            "cpu_throttled_microseconds",
            "oom_kills",
        }
    )
    _fields = _counters | frozenset(
        {
            "source",
            "scope",
            "target",
            "observed_at",
            "visible_cpu_limit_cores",
            "visible_memory_limit_bytes",
            "pressure_stall_microseconds",
        }
    )

    @classmethod
    def _normalize(cls, data: dict[str, Any]) -> dict[str, Any]:
        for field in ("source", "scope", "target"):
            data[field] = _required_text(data.get(field), field)
        data["observed_at"] = _utc_timestamp(data.get("observed_at"), "observed at")
        for field in cls._counters:
            data[field] = _stats_measurement(data.get(field))
        data["visible_cpu_limit_cores"] = _stats_measurement(
            data.get("visible_cpu_limit_cores"), integer=False, limit=True
        )
        data["visible_memory_limit_bytes"] = _stats_measurement(
            data.get("visible_memory_limit_bytes"), limit=True
        )
        data["pressure_stall_microseconds"] = {
            name: _stats_measurement(reading)
            for name, reading in _mapping(
                data.get("pressure_stall_microseconds"), "pressure readings"
            ).items()
        }
        return data


class CgroupObservation(FabricMapping):
    """Native cgroup snapshot or explicit unavailable/unsupported reason."""

    status: str
    _fields = frozenset({"status", "snapshot", "reason"})

    @property
    def snapshot(self) -> CgroupSnapshot | None:
        """Return the cgroup reading, otherwise None."""
        return self._data.get("snapshot")

    @property
    def reason(self) -> str | None:
        """Return the missing-evidence reason, otherwise None."""
        return self._data.get("reason")

    @classmethod
    def _normalize(cls, data: dict[str, Any]) -> dict[str, Any]:
        if data.get("status") == "available":
            if "reason" in data:
                raise FabricConfigError("available observation cannot contain reason")
            data["snapshot"] = CgroupSnapshot.from_mapping(data.get("snapshot"))
        elif data.get("status") == "not_collected":
            if "snapshot" in data or "reason" in data:
                raise FabricConfigError(
                    "not-collected observation has no snapshot or reason"
                )
        elif data.get("status") in ("unavailable", "unsupported"):
            if "snapshot" in data:
                raise FabricConfigError("missing observation cannot contain snapshot")
            data["reason"] = _required_text(data.get("reason"), "observation reason")
        else:
            raise FabricConfigError("invalid cgroup observation status")
        return data


class CgroupStats(FabricMapping):
    """Cgroup endpoint observations and counter deltas over an invocation."""

    before: CgroupObservation
    after: CgroupObservation
    cpu_usage_microseconds: Measurement
    cpu_throttled_microseconds: Measurement
    oom_kills: Measurement
    _fields = frozenset(
        {
            "before",
            "after",
            "cpu_usage_microseconds",
            "cpu_throttled_microseconds",
            "oom_kills",
        }
    )

    @classmethod
    def _normalize(cls, data: dict[str, Any]) -> dict[str, Any]:
        for field in ("before", "after"):
            data[field] = CgroupObservation.from_mapping(data.get(field))
        for field in (
            "cpu_usage_microseconds",
            "cpu_throttled_microseconds",
            "oom_kills",
        ):
            data[field] = _stats_measurement(data.get(field))
        return data


class GaugeSummary(FabricMapping):
    """Sampled gauge extrema, hold-last-value mean, and covered time.

    Sampled peak is not true high-water. Means exclude missing readings and
    gaps longer than twice the requested cadence. covered_fraction discloses
    how much of the invocation contributes to the mean.
    """

    unit: str
    final_value: int | None
    min: int | None
    sampled_peak: int | None
    time_weighted_mean: float | None
    valid_samples: int
    covered_seconds: float
    covered_fraction: float
    _fields = frozenset(
        {
            "unit",
            "final_value",
            "min",
            "sampled_peak",
            "time_weighted_mean",
            "valid_samples",
            "covered_seconds",
            "covered_fraction",
        }
    )

    @classmethod
    def _normalize(cls, data: dict[str, Any]) -> dict[str, Any]:
        data["unit"] = _required_text(data.get("unit"), "gauge unit")
        for field in ("final_value", "min", "sampled_peak", "time_weighted_mean"):
            value = data.get(field)
            data[field] = (
                None
                if value is None
                else _stats_number(value, field, integer=field != "time_weighted_mean")
            )
        for field in ("valid_samples", "covered_seconds", "covered_fraction"):
            data[field] = _stats_number(
                data.get(field), field, integer=field == "valid_samples"
            )
        if data["covered_fraction"] > 1:
            raise FabricConfigError("covered fraction must not exceed one")
        return data


class SampleSummary(FabricMapping):
    """Retained native series, reductions, and collection-quality diagnostics."""

    target_interval_ms: int
    max_samples: int
    sample_count: int
    dropped_samples: int
    largest_gap_seconds: float
    collection_seconds: float
    truncated: bool
    diagnostics: Sequence[str]
    samples: ArtifactRef | None
    cgroup_target_continuous: bool
    resident_memory_bytes: GaugeSummary | None
    process_count: GaugeSummary | None
    cpu_time_seconds_estimate: Measurement
    cpu_percent_one_core: Measurement
    _fields = frozenset(
        {
            "target_interval_ms",
            "max_samples",
            "sample_count",
            "dropped_samples",
            "largest_gap_seconds",
            "collection_seconds",
            "truncated",
            "diagnostics",
            "samples",
            "cgroup_target_continuous",
            "resident_memory_bytes",
            "process_count",
            "cpu_time_seconds_estimate",
            "cpu_percent_one_core",
        }
    )

    @classmethod
    def _normalize(cls, data: dict[str, Any]) -> dict[str, Any]:
        for field in (
            "target_interval_ms",
            "max_samples",
            "sample_count",
            "dropped_samples",
            "largest_gap_seconds",
            "collection_seconds",
        ):
            data[field] = _stats_number(
                data.get(field),
                field,
                integer=field not in ("largest_gap_seconds", "collection_seconds"),
            )
        data["cgroup_target_continuous"] = _boolean(
            data.get("cgroup_target_continuous"), "cgroup target continuity"
        )
        data["truncated"] = _boolean(data.get("truncated"), "truncated")
        data["diagnostics"] = tuple(
            _required_text(value, "diagnostic code")
            for value in data.get("diagnostics", [])
        )
        data["samples"] = (
            None
            if data.get("samples") is None
            else ArtifactRef.from_mapping(data["samples"])
        )
        for field in ("resident_memory_bytes", "process_count"):
            data[field] = (
                None
                if data.get(field) is None
                else GaugeSummary.from_mapping(data[field])
            )
        for field in ("cpu_time_seconds_estimate", "cpu_percent_one_core"):
            data[field] = _stats_measurement(data.get(field), integer=False)
        return data


class InvocationSystemStats(FabricMapping):
    """System measurements for one invocation in its executing runner.

    Observations cover the adapter host process tree, excluding the orchestrator
    and remote services. Endpoints are neither peaks nor time averages.

    Attributes:
        elapsed_seconds: Monotonic adapter exchange duration, excluding startup,
            endpoint collection, artifact preparation, and teardown.
        before: Observation immediately before the adapter exchange.
        after: Observation immediately after the exchange, under the same lock.
        cgroup: Native adapter-host cgroup readings and counter deltas, or None.
        sampled: Optional retained series, gauge reductions, and collection quality.
        cpu_time_seconds_estimate: Nonnegative CPU counter difference, or None
            if an endpoint is missing, identities differ, or the counter fell.
            Process reaping and reparenting can make this underestimate work.
    """

    cgroup: CgroupStats | None
    sampled: SampleSummary | None
    collected_groups: Sequence[str]
    elapsed_seconds: float
    before: ResourceObservation
    after: ResourceObservation
    _fields = frozenset(
        {
            "collected_groups",
            "elapsed_seconds",
            "before",
            "after",
            "cpu_time_seconds_estimate",
            "cgroup",
            "sampled",
        }
    )

    @property
    def cpu_time_seconds_estimate(self) -> float | None:
        """Return the CPU counter difference estimate, or None when unknown."""
        return self._data.get("cpu_time_seconds_estimate")

    @classmethod
    def _normalize(cls, data: dict[str, Any]) -> dict[str, Any]:
        supported = ("process", "cgroup.cpu", "cgroup.memory", "cgroup.pressure")
        groups = data.get("collected_groups", supported)
        if (
            not isinstance(groups, (list, tuple))
            or not groups
            or any(g not in supported for g in groups)
            or len(set(groups)) != len(groups)
        ):
            raise FabricConfigError(
                "collected_groups must contain unique supported groups"
            )
        data["collected_groups"] = tuple(groups)
        data["cgroup"] = (
            None
            if data.get("cgroup") is None
            else CgroupStats.from_mapping(data["cgroup"])
        )
        data["sampled"] = (
            None
            if data.get("sampled") is None
            else SampleSummary.from_mapping(data["sampled"])
        )
        data["before"] = ResourceObservation.from_mapping(data.get("before"))
        data["after"] = ResourceObservation.from_mapping(data.get("after"))
        for field in ("elapsed_seconds", "cpu_time_seconds_estimate"):
            value = data.get(field)
            if value is None and field == "cpu_time_seconds_estimate":
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise FabricConfigError(f"{field} must be a finite nonnegative number")
            data[field] = float(value)
        return data


class PricingBasis(FabricMapping):
    """Validated comparison basis for persisted catalog estimates."""

    basis: str
    currency: str
    normalization_version: str
    assumption_policy: str
    bundled_catalog_sha256: str
    user_catalog_sha256: str | None
    _fields = frozenset(
        {
            "basis",
            "currency",
            "normalization_version",
            "assumption_policy",
            "bundled_catalog_sha256",
            "user_catalog_sha256",
        }
    )

    @classmethod
    def _normalize(cls, data: dict[str, Any]) -> dict[str, Any]:
        if set(data).difference(cls._fields) or any(
            not isinstance(data.get(k), str) or not data[k]
            for k in ("basis", "currency", "normalization_version", "assumption_policy")
        ):
            raise FabricConfigError("unsupported or incomplete pricing basis")
        for key in ("bundled_catalog_sha256", "user_catalog_sha256"):
            value = data.get(key)
            if key == "user_catalog_sha256" and value is None:
                continue
            if (
                not isinstance(value, str)
                or re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None
            ):
                raise FabricConfigError("pricing basis requires catalog fingerprints")
        return data


class AppliedPricing(FabricMapping):
    """Selected price entry and integer nano-USD rates per million tokens."""

    catalog_sha256: str
    provider: str
    model_id: str
    pricing_as_of: str
    pricing_source: str
    rates: dict[str, int | None]
    _fields = frozenset(
        {
            "catalog_sha256",
            "provider",
            "model_id",
            "pricing_as_of",
            "pricing_source",
            "rates",
        }
    )

    @classmethod
    def _normalize(cls, data: dict[str, Any]) -> dict[str, Any]:
        if set(data) != cls._fields:
            raise FabricConfigError("incomplete applied pricing")
        for key in cls._fields - {"rates"}:
            _required_text(data.get(key), "pricing " + key)
        if re.fullmatch(r"sha256:[0-9a-f]{64}", data["catalog_sha256"]) is None:
            raise FabricConfigError("invalid applied pricing fingerprint")
        rates = _mapping(data["rates"], "pricing rates")
        if set(rates) != {"input", "output", "cache_read", "cache_write"}:
            raise FabricConfigError("incomplete pricing rates")
        for key, value in rates.items():
            if value is not None or key in {"input", "output"}:
                _usage_counter(value, "pricing rate")
        data["rates"] = rates
        return data


_MODEL_USAGE_TOKEN_FIELDS = frozenset(
    {
        "input_tokens",
        "cached_input_tokens",
        "cache_write_input_tokens",
        "output_tokens",
        "total_tokens",
    }
)
_MODEL_USAGE_DIMENSIONS = (
    *sorted(_MODEL_USAGE_TOKEN_FIELDS),
    "reported_cost",
    "estimated_cost",
)
_USAGE_GAP_REASONS = frozenset({"unfinished_work", "missing_usage", "scope_unverified"})


def _usage_counter(value: Any, name: str) -> int:
    if type(value) is not int or not 0 <= value <= _UINT64_MAX:
        raise FabricConfigError(f"{name} must be a nonnegative uint64 integer")
    return value


class TokenCounts(FabricMapping):
    """Independently nullable tokens in one fixed invocation/model scope."""

    input_tokens: int | None
    cached_input_tokens: int | None
    cache_write_input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    _fields = _MODEL_USAGE_TOKEN_FIELDS

    @classmethod
    def _normalize(cls, data: dict[str, Any]) -> dict[str, Any]:
        for field in cls._fields:
            value = data.setdefault(field, None)
            if value is not None:
                _usage_counter(value, field)
        return data


class UsageGap(FabricMapping):
    """Dimension-specific incompleteness with an optional stable diagnostic."""

    dimensions: tuple[str, ...]
    reason: str
    code: str | None
    _fields = frozenset({"dimensions", "reason", "code"})

    @classmethod
    def _normalize(cls, data: dict[str, Any]) -> dict[str, Any]:
        dimensions = data.get("dimensions")
        if (
            not isinstance(dimensions, list)
            or not dimensions
            or any(
                not isinstance(d, str) or d not in _MODEL_USAGE_DIMENSIONS
                for d in dimensions
            )
            or len(set(dimensions)) != len(dimensions)
        ):
            raise FabricConfigError("gap dimensions must be nonempty, known and unique")
        if data.get("reason") not in _USAGE_GAP_REASONS:
            raise FabricConfigError("unsupported usage gap reason")
        code = data.setdefault("code", None)
        if code is not None and (
            not isinstance(code, str) or not code or len(code) > 64
        ):
            raise FabricConfigError("usage diagnostic must be 1 to 64 characters")
        data["dimensions"] = tuple(dimensions)
        return data


class CostAmount(FabricMapping):
    """Complete and observed amounts in integer nano-USD, never an invoice guarantee."""

    amount_nano_usd: int | None
    observed_nano_usd: int | None
    _fields = frozenset({"amount_nano_usd", "observed_nano_usd"})

    @classmethod
    def _normalize(cls, data: dict[str, Any]) -> dict[str, Any]:
        for field in cls._fields:
            value = data.setdefault(field, None)
            if value is not None:
                _usage_counter(value, field)
        complete = data["amount_nano_usd"]
        observed = data["observed_nano_usd"]
        if complete is not None and observed is not None and observed > complete:
            raise FabricConfigError("cost observation exceeds complete amount")
        return data


class TokenCost(FabricMapping):
    """Provider-reported amounts and separate catalog estimates with their basis."""

    reported: CostAmount
    estimated: CostAmount
    pricing: PricingBasis | None
    applied_rates: tuple[AppliedPricing, ...]
    _fields = frozenset({"reported", "estimated", "pricing", "applied_rates"})

    @classmethod
    def _normalize(cls, data: dict[str, Any]) -> dict[str, Any]:
        for name in ("reported", "estimated"):
            data[name] = CostAmount.from_mapping(data.get(name, {}))
        data["pricing"] = (
            None
            if data.get("pricing") is None
            else PricingBasis.from_mapping(data["pricing"])
        )
        rates = data.get("applied_rates", [])
        if not isinstance(rates, list):
            raise FabricConfigError("applied_rates must be an array")
        data["applied_rates"] = tuple(
            AppliedPricing.from_mapping(rate) for rate in rates
        )
        for rate in data["applied_rates"]:
            basis = data["pricing"]
            if basis is None or rate.catalog_sha256 not in {
                basis.bundled_catalog_sha256,
                basis.get("user_catalog_sha256"),
            }:
                raise FabricConfigError(
                    "applied pricing must belong to the persisted basis"
                )
        return data


def _usage_value(
    data: Mapping[str, Any], dimension: str, *, observed: bool = False
) -> int | None:
    if dimension in _MODEL_USAGE_TOKEN_FIELDS:
        return data["observed"].get(dimension) if observed else data.get(dimension)
    amount = data["cost"]["reported" if dimension == "reported_cost" else "estimated"]
    return amount.observed_nano_usd if observed else amount.amount_nano_usd


def _normalize_model_usage(data: dict[str, Any]) -> dict[str, Any]:
    for field in _MODEL_USAGE_TOKEN_FIELDS:
        value = data.setdefault(field, None)
        if value is not None:
            _usage_counter(value, field)
    data["observed"] = TokenCounts.from_mapping(data.get("observed", {}))
    raw_gaps = data.get("gaps")
    if not isinstance(raw_gaps, list):
        raise FabricConfigError("gaps must be an array")
    data["gaps"] = tuple(UsageGap.from_mapping(gap) for gap in raw_gaps)
    data["cost"] = TokenCost.from_mapping(data.get("cost", {}))
    for gap in data["gaps"]:
        if any(
            _usage_value(data, dimension) is not None for dimension in gap.dimensions
        ):
            raise FabricConfigError("complete count conflicts with accounting gap")
    total = data["total_tokens"]
    if total is not None and (
        data["input_tokens"] is None
        or data["output_tokens"] is None
        or total != data["input_tokens"] + data["output_tokens"]
    ):
        raise FabricConfigError("normalized total must equal known input plus output")
    # Complete cache dimensions share the invocation's inclusive input scope.
    # Compare only the known subset; missing cache counters remain unknown.
    known_cache = sum(
        count
        for count in (data["cached_input_tokens"], data["cache_write_input_tokens"])
        if count is not None
    )
    if data["input_tokens"] is not None and known_cache > data["input_tokens"]:
        raise FabricConfigError("complete cache counts exceed inclusive input")
    observed = data["observed"]
    observed_total = observed.total_tokens
    if observed_total is not None and (
        observed.input_tokens is None
        or observed.output_tokens is None
        or observed_total != observed.input_tokens + observed.output_tokens
    ):
        raise FabricConfigError("observed total must equal known input plus output")
    for field in _MODEL_USAGE_TOKEN_FIELDS:
        complete = data[field]
        observation = observed.get(field)
        if complete is not None and observation is not None and observation > complete:
            raise FabricConfigError("token observation exceeds complete count")
    return data


class PerModelUsage(FabricMapping):
    """Normalized counts for a source-established accounting model/provider.

    A configured identity requires source proof that the included calls used it.
    An outer requested alias alone does not establish accounting attribution.
    This identity does not replace served-model provenance.
    """

    model: str
    provider: str | None
    input_tokens: int | None
    cached_input_tokens: int | None
    cache_write_input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    observed: TokenCounts
    gaps: tuple[UsageGap, ...]
    cost: TokenCost
    _fields = frozenset(
        {*_MODEL_USAGE_TOKEN_FIELDS, "model", "provider", "observed", "gaps", "cost"}
    )

    @classmethod
    def _normalize(cls, data: dict[str, Any]) -> dict[str, Any]:
        data["model"] = _required_text(data.get("model"), "model")
        if data.get("provider") is not None:
            _required_text(data["provider"], "provider")
        return _normalize_model_usage(data)


class ModelUsage(FabricMapping):
    """Complete normalized tokens, observations, gaps and separately typed costs.

    Input includes cache and output includes reasoning. All counts belong to
    this invocation, including descendants. Late work never moves to another
    invocation. Observations are not guaranteed mathematical lower bounds.
    """

    input_tokens: int | None
    cached_input_tokens: int | None
    cache_write_input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    observed: TokenCounts
    gaps: tuple[UsageGap, ...]
    models_exhaustive: bool
    models: tuple[PerModelUsage, ...]
    cost: TokenCost
    _fields = frozenset(
        {
            *_MODEL_USAGE_TOKEN_FIELDS,
            "observed",
            "gaps",
            "models_exhaustive",
            "models",
            "cost",
        }
    )

    @classmethod
    def _normalize(cls, data: dict[str, Any]) -> dict[str, Any]:
        _normalize_model_usage(data)
        if type(data.get("models_exhaustive")) is not bool:
            raise FabricConfigError("models_exhaustive must be boolean")
        models = data.get("models", [])
        if not isinstance(models, list):
            raise FabricConfigError("models must be an array")
        data["models"] = tuple(PerModelUsage.from_mapping(model) for model in models)
        identities = [(model.get("provider"), model.model) for model in data["models"]]
        if len(set(identities)) != len(identities):
            raise FabricConfigError("duplicate provider/model identity")
        for field in _MODEL_USAGE_TOKEN_FIELDS:
            complete = data[field]
            if complete is None:
                continue
            parts = [
                model[field] if model[field] is not None else model.observed.get(field)
                for model in data["models"]
            ]
            subtotal = sum(value for value in parts if value is not None)
            complete_parts = [
                model[field] for model in data["models"] if model[field] is not None
            ]
            if (
                subtotal > _UINT64_MAX
                or subtotal > complete
                or (
                    data["models_exhaustive"]
                    and data["models"]
                    and len(complete_parts) == len(data["models"])
                    and sum(complete_parts) != complete
                )
            ):
                raise FabricConfigError(
                    f"model token totals contradict invocation {field}"
                )
        return data


class RunResult(FabricMapping):
    """Normalized terminal result from one NeMo Fabric invocation.

    The model is both attribute-accessible and mapping-compatible. A harness
    failure can be represented by ``status`` and ``error`` without raising when
    the adapter successfully returns a normalized result.

    Attributes:
        agent_name: Resolved agent name.
        harness: Stable harness identifier.
        adapter_kind: Adapter execution mechanism.
        adapter_id: NeMo Fabric adapter identifier.
        runtime_id: Runtime lifecycle identifier.
        invocation_id: Identifier for this invocation.
        request_id: Correlated request identifier.
        status: Terminal invocation status.
        output: Object-shaped adapter output as ``RunOutput``; non-object values
            are preserved as-is.
        error: Structured failure, or ``None`` on success.
        usage: Normalized invocation usage, or ``None`` when unavailable.
        resource_snapshot: Runtime resource snapshot read once after the
            adapter returned this result, or ``None`` when the runtime's
            resources could not be observed.
        system_stats: Invocation duration and endpoint observations, or None
            for older results or execution backends without collection.
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
    usage: RunUsage | None
    model_usage: ModelUsage | None
    system_stats: InvocationSystemStats | None
    resource_snapshot: RuntimeResourceSnapshot | None
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
            "usage",
            "model_usage",
            "resource_snapshot",
            "system_stats",
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
        data["error"] = (
            None if data.get("error") is None else ErrorInfo.from_mapping(data["error"])
        )
        data["usage"] = (
            None if data.get("usage") is None else RunUsage.from_mapping(data["usage"])
        )
        data["model_usage"] = (
            None
            if data.get("model_usage") is None
            else ModelUsage.from_mapping(data["model_usage"])
        )
        data["system_stats"] = (
            None
            if data.get("system_stats") is None
            else InvocationSystemStats.from_mapping(data["system_stats"])
        )
        data["resource_snapshot"] = (
            None
            if data.get("resource_snapshot") is None
            else RuntimeResourceSnapshot.from_mapping(data["resource_snapshot"])
        )
        data["artifacts"] = ArtifactManifest.from_mapping(
            data.get("artifacts", {"artifacts": []})
        )
        telemetry = data.get("telemetry")
        if telemetry is None:
            data["telemetry"] = ()
        elif isinstance(telemetry, Mapping):
            data["telemetry"] = (TelemetryRef.from_mapping(telemetry),)
        else:
            data["telemetry"] = tuple(
                TelemetryRef.from_mapping(item) for item in telemetry
            )
        data["events"] = tuple(
            FabricEvent.from_mapping(event) for event in data.get("events", [])
        )
        data["metadata"] = _mapping(data.get("metadata", {}), "result metadata")
        raw_output = data.get("output")
        if isinstance(raw_output, RunOutput):
            data["output"] = raw_output
        elif isinstance(raw_output, Mapping):
            data["output"] = RunOutput.from_mapping(raw_output)
        return data
