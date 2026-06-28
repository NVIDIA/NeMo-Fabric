# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public data contracts for the NeMo Fabric Python SDK."""

from __future__ import annotations

import math
import uuid
from collections.abc import Iterator, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from types import MappingProxyType
from typing import Any, TypeVar

from nemo_fabric.errors import FabricConfigError

JSONScalar = str | int | float | bool | None
JSONValue = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]

_UNSET = object()
_T = TypeVar("_T")


def _plain(value: Any) -> Any:
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
        extras = _mapping(extra_fields or {}, "extra_fields")
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
        self[name] = deepcopy(value) if isinstance(value, _ConfigMapping) else _plain(value)

    @property
    def extra_fields(self) -> dict[str, Any]:
        return {
            key: _plain(value)
            for key, value in self.items()
            if key not in self._fields
        }

    def to_mapping(self) -> dict[str, Any]:
        data = _plain(dict(self))
        for key in self._omit_if_empty:
            if data.get(key) in ({}, []):
                data.pop(key, None)
        return data


class MetadataConfig(_ConfigMapping):
    """Agent identity and human-readable metadata."""

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
    def from_mapping(cls, value: Mapping[str, Any]) -> "MetadataConfig":
        data = _mapping(value, "metadata")
        return cls(
            name=data.get("name"),
            description=data.get("description"),
            extra_fields={key: item for key, item in data.items() if key not in cls._fields},
        )


class HarnessConfig(_ConfigMapping):
    """Harness adapter selection and adapter-owned settings."""

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
            "settings": _mapping(settings or {}, "harness settings"),
        }
        if resolution is not None:
            values["resolution"] = resolution
        super().__init__(values, extra_fields=extra_fields)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "HarnessConfig":
        data = _mapping(value, "harness")
        return cls(
            adapter_id=data.get("adapter_id"),
            resolution=data.get("resolution"),
            settings=data.get("settings"),
            extra_fields={key: item for key, item in data.items() if key not in cls._fields},
        )


class RuntimeConfig(_ConfigMapping):
    """Runtime lifecycle mode and input/output contract."""

    _fields = frozenset(
        {"mode", "transport", "input_schema", "output_schema", "artifacts"}
    )

    def __init__(
        self,
        *,
        mode: str = "oneshot",
        transport: str | None = None,
        input_schema: str | None = None,
        output_schema: str | None = None,
        artifacts: str | Path | None = None,
        extra_fields: Mapping[str, Any] | None = None,
    ) -> None:
        if mode not in {"oneshot", "session", "service"}:
            raise FabricConfigError(f"unsupported runtime mode: {mode!r}")
        values: dict[str, Any] = {"mode": mode}
        for key, item in (
            ("transport", transport),
            ("input_schema", input_schema),
            ("output_schema", output_schema),
            ("artifacts", artifacts),
        ):
            if item is not None:
                values[key] = item
        super().__init__(values, extra_fields=extra_fields)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RuntimeConfig":
        data = _mapping(value, "runtime")
        return cls(
            mode=data.get("mode", "oneshot"),
            transport=data.get("transport"),
            input_schema=data.get("input_schema"),
            output_schema=data.get("output_schema"),
            artifacts=data.get("artifacts"),
            extra_fields={key: item for key, item in data.items() if key not in cls._fields},
        )


class EnvironmentConfig(_ConfigMapping):
    """Execution environment configuration."""

    _fields = frozenset(
        {"provider", "workspace", "artifacts", "settings", "metadata"}
    )
    _omit_if_empty = frozenset({"settings", "metadata"})

    def __init__(
        self,
        *,
        provider: str = "local",
        workspace: str | Path | None = None,
        artifacts: str | Path | None = None,
        settings: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        extra_fields: Mapping[str, Any] | None = None,
    ) -> None:
        values: dict[str, Any] = {
            "provider": _required_text(provider, "environment provider"),
            "settings": _mapping(settings or {}, "environment settings"),
            "metadata": _mapping(metadata or {}, "environment metadata"),
        }
        if workspace is not None:
            values["workspace"] = workspace
        if artifacts is not None:
            values["artifacts"] = artifacts
        super().__init__(values, extra_fields=extra_fields)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EnvironmentConfig":
        data = _mapping(value, "environment")
        return cls(
            provider=data.get("provider", "local"),
            workspace=data.get("workspace"),
            artifacts=data.get("artifacts"),
            settings=data.get("settings"),
            metadata=data.get("metadata"),
            extra_fields={key: item for key, item in data.items() if key not in cls._fields},
        )


class FabricConfig(_ConfigMapping):
    """Mutable typed SDK object for a Fabric agent config."""

    _fields = frozenset(
        {
            "schema_version",
            "metadata",
            "harness",
            "runtime",
            "environment",
            "models",
            "mcp",
            "skills",
            "telemetry",
            "profiles",
            "tools",
        }
    )
    _omit_if_empty = frozenset({"models"})

    def __init__(
        self,
        *,
        metadata: MetadataConfig | Mapping[str, Any],
        harness: HarnessConfig | Mapping[str, Any],
        runtime: RuntimeConfig | Mapping[str, Any] | None = None,
        schema_version: str = "fabric.agent/v1alpha1",
        environment: EnvironmentConfig | Mapping[str, Any] | None = None,
        models: Mapping[str, Any] | None = None,
        mcp: Mapping[str, Any] | None = None,
        skills: Mapping[str, Any] | None = None,
        telemetry: Mapping[str, Any] | None = None,
        profiles: Mapping[str, Any] | None = None,
        tools: Any = None,
        extra_fields: Mapping[str, Any] | None = None,
    ) -> None:
        metadata_value = _coerce(MetadataConfig, metadata, "metadata")
        harness_value = _coerce(HarnessConfig, harness, "harness")
        runtime_value = _coerce(RuntimeConfig, runtime or RuntimeConfig(), "runtime")
        environment_value = (
            None
            if environment is None
            else _coerce(EnvironmentConfig, environment, "environment")
        )
        values: dict[str, Any] = {
            "schema_version": _required_text(schema_version, "schema_version"),
            "metadata": metadata_value,
            "harness": harness_value,
            "runtime": runtime_value,
            "models": _mapping(models or {}, "models"),
        }
        for key, item in (
            ("environment", environment_value),
            ("mcp", mcp),
            ("skills", skills),
            ("telemetry", telemetry),
            ("profiles", profiles),
            ("tools", tools),
        ):
            if item is not None:
                values[key] = item
        super().__init__(values, extra_fields=extra_fields)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FabricConfig":
        data = _mapping(value, "FabricConfig")
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
            mcp=data.get("mcp"),
            skills=data.get("skills"),
            telemetry=data.get("telemetry"),
            profiles=data.get("profiles"),
            tools=data.get("tools"),
            extra_fields={key: item for key, item in data.items() if key not in cls._fields},
        )


class FabricProfileConfig(_ConfigMapping):
    """Mutable typed SDK object for an in-memory Fabric profile."""

    _fields = frozenset(
        {
            "schema_version",
            "name",
            "description",
            "harness",
            "runtime",
            "environment",
            "models",
            "mcp",
            "skills",
            "telemetry",
            "tools",
        }
    )

    def __init__(
        self,
        *,
        name: str,
        schema_version: str = "fabric.profile/v1alpha1",
        description: str | None = None,
        harness: HarnessConfig | Mapping[str, Any] | None = None,
        runtime: RuntimeConfig | Mapping[str, Any] | None = None,
        environment: EnvironmentConfig | Mapping[str, Any] | None = None,
        models: Mapping[str, Any] | None = None,
        mcp: Mapping[str, Any] | None = None,
        skills: Mapping[str, Any] | None = None,
        telemetry: Mapping[str, Any] | None = None,
        tools: Any = None,
        extra_fields: Mapping[str, Any] | None = None,
    ) -> None:
        values: dict[str, Any] = {
            "schema_version": _required_text(schema_version, "schema_version"),
            "name": _required_text(name, "profile name"),
        }
        if description is not None:
            values["description"] = description
        for key, item in (
            ("harness", harness),
            ("runtime", runtime),
            ("environment", environment),
        ):
            if item is not None:
                values[key] = (
                    deepcopy(item)
                    if isinstance(item, _ConfigMapping)
                    else _mapping(item, key)
                )
        for key, item in (
            ("models", models),
            ("mcp", mcp),
            ("skills", skills),
            ("telemetry", telemetry),
            ("tools", tools),
        ):
            if item is not None:
                values[key] = item
        super().__init__(values, extra_fields=extra_fields)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FabricProfileConfig":
        data = _mapping(value, "FabricProfileConfig")
        return cls(
            schema_version=data.get("schema_version", "fabric.profile/v1alpha1"),
            name=data.get("name"),
            description=data.get("description"),
            harness=data.get("harness"),
            runtime=data.get("runtime"),
            environment=data.get("environment"),
            models=data.get("models"),
            mcp=data.get("mcp"),
            skills=data.get("skills"),
            telemetry=data.get("telemetry"),
            tools=data.get("tools"),
            extra_fields={key: item for key, item in data.items() if key not in cls._fields},
        )


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


class FabricMapping(Mapping[str, Any]):
    """Immutable mapping-compatible base for SDK snapshots and results."""

    _fields: frozenset[str] = frozenset()

    def __init__(self, mapping: Mapping[str, Any]) -> None:
        data = self._normalize(_mapping(mapping, type(self).__name__))
        object.__setattr__(self, "_data", _freeze(data))

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "FabricMapping":
        return cls(mapping)

    @classmethod
    def _normalize(cls, data: dict[str, Any]) -> dict[str, Any]:
        return data

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __getattr__(self, name: str) -> Any:
        try:
            return self._data[name]
        except KeyError as error:
            raise AttributeError(name) from error

    @property
    def extra_fields(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {key: value for key, value in self._data.items() if key not in self._fields}
        )

    def to_mapping(self) -> dict[str, Any]:
        return _thaw(self._data)

    def to_dict(self) -> dict[str, Any]:
        return self.to_mapping()


class AdapterInfo(FabricMapping):
    adapter_id: str
    harness: str
    adapter_kind: str
    metadata: Mapping[str, Any]
    _fields = frozenset({"adapter_id", "harness", "adapter_kind", "metadata"})

    @classmethod
    def _normalize(cls, data: dict[str, Any]) -> dict[str, Any]:
        data["adapter_id"] = _required_text(data.get("adapter_id"), "adapter_id")
        data["harness"] = _required_text(data.get("harness"), "harness")
        data["adapter_kind"] = _required_text(data.get("adapter_kind"), "adapter_kind")
        data["metadata"] = _mapping(data.get("metadata", {}), "adapter metadata")
        return data


class RuntimeCapabilities(FabricMapping):
    session: bool
    service: bool
    streaming: bool
    updates: bool
    cancellation: bool
    concurrent_invocations: bool
    metadata: Mapping[str, Any]
    _fields = frozenset(
        {
            "session",
            "service",
            "streaming",
            "updates",
            "cancellation",
            "concurrent_invocations",
            "metadata",
        }
    )

    @classmethod
    def _normalize(cls, data: dict[str, Any]) -> dict[str, Any]:
        for field in cls._fields - {"metadata"}:
            data[field] = _boolean(data.get(field, False), f"{field} capability")
        data["metadata"] = _mapping(data.get("metadata", {}), "capability metadata")
        return data


class EffectiveConfig(FabricMapping):
    agent_name: str
    profiles: Sequence[str]
    agent_root: Path
    config_path: Path | None
    config_root: Path
    config: FabricConfig
    _fields = frozenset(
        {"agent_name", "profiles", "agent_root", "config_path", "config_root", "config"}
    )

    @classmethod
    def _normalize(cls, data: dict[str, Any]) -> dict[str, Any]:
        data["profiles"] = tuple(data.get("profiles", []))
        data["agent_root"] = Path(data.get("agent_root", "."))
        data["config_root"] = Path(data.get("config_root", "."))
        data["config_path"] = (
            None if data.get("config_path") is None else Path(data["config_path"])
        )
        data["config"] = FabricConfig.from_mapping(data.get("config", {}))
        return data


class RunPlan(FabricMapping):
    effective_config: EffectiveConfig
    agent_name: str
    profiles: Sequence[str]
    adapter: AdapterInfo
    capabilities: RuntimeCapabilities
    _fields = frozenset(
        {"effective_config", "agent_name", "profiles", "adapter", "capabilities"}
    )

    @classmethod
    def _normalize(cls, data: dict[str, Any]) -> dict[str, Any]:
        descriptor = data.get("adapter")
        if descriptor is None:
            descriptor = (data.get("adapter_descriptor") or {}).get("descriptor", {})
        data["effective_config"] = EffectiveConfig.from_mapping(data["effective_config"])
        data["profiles"] = tuple(data.get("profiles", []))
        data["adapter"] = AdapterInfo.from_mapping(descriptor)
        data["capabilities"] = RuntimeCapabilities.from_mapping(data.get("capabilities", {}))
        return data


class DoctorCheck(FabricMapping):
    name: str
    status: str
    message: str
    metadata: Mapping[str, Any]
    _fields = frozenset({"name", "status", "message", "metadata"})

    @classmethod
    def _normalize(cls, data: dict[str, Any]) -> dict[str, Any]:
        data["metadata"] = _mapping(data.get("metadata", {}), "doctor metadata")
        return data


class DoctorReport(FabricMapping):
    agent_name: str
    profiles: Sequence[str]
    status: str
    checks: Sequence[DoctorCheck]
    _fields = frozenset({"agent_name", "profiles", "status", "checks"})

    @classmethod
    def _normalize(cls, data: dict[str, Any]) -> dict[str, Any]:
        data["profiles"] = tuple(data.get("profiles", []))
        data["checks"] = tuple(
            DoctorCheck.from_mapping(check) for check in data.get("checks", [])
        )
        return data


class RunRequest(FabricMapping):
    input: Any
    request_id: str
    context: Mapping[str, Any]
    overrides: Mapping[str, Any] | None
    _fields = frozenset({"input", "request_id", "context", "overrides"})

    def __init__(
        self,
        *,
        input: Any = _UNSET,
        request_id: str | None = None,
        context: Mapping[str, Any] | None = None,
        overrides: Mapping[str, Any] | None = None,
        extra_fields: Mapping[str, Any] | None = None,
    ) -> None:
        data: dict[str, Any] = {
            "input": "" if input is _UNSET or input is None else input,
            "request_id": request_id or f"request-{uuid.uuid4().hex}",
            "context": _mapping(context or {}, "request context"),
        }
        if overrides is not None:
            data["overrides"] = _mapping(overrides, "request overrides")
        extras = _mapping(extra_fields or {}, "request extra_fields")
        overlap = self._fields.intersection(extras)
        if overlap:
            raise FabricConfigError(
                f"request extra_fields duplicates known fields: {', '.join(sorted(overlap))}"
            )
        data.update(extras)
        FabricMapping.__init__(self, data)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RunRequest":
        data = _mapping(value, "RunRequest")
        return cls(
            input=data.get("input", _UNSET),
            request_id=data.get("request_id"),
            context=data.get("context"),
            overrides=data.get("overrides"),
            extra_fields={key: item for key, item in data.items() if key not in cls._fields},
        )


class ErrorInfo(FabricMapping):
    stage: str
    code: str
    message: str
    retryable: bool
    metadata: Mapping[str, Any]
    _fields = frozenset({"stage", "code", "message", "retryable", "metadata"})

    @classmethod
    def _normalize(cls, data: dict[str, Any]) -> dict[str, Any]:
        data["metadata"] = _mapping(data.get("metadata", {}), "error metadata")
        return data


class ArtifactRef(FabricMapping):
    name: str
    kind: str
    path: Path
    media_type: str | None
    metadata: Mapping[str, Any]
    _fields = frozenset({"name", "kind", "path", "media_type", "metadata"})

    @classmethod
    def _normalize(cls, data: dict[str, Any]) -> dict[str, Any]:
        data["path"] = Path(data["path"])
        data["metadata"] = _mapping(data.get("metadata", {}), "artifact metadata")
        return data


class ArtifactManifest(FabricMapping):
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
    provider: str
    kind: str
    uri: str | None
    trace_id: str | None
    metadata: Mapping[str, Any]
    _fields = frozenset({"provider", "kind", "uri", "trace_id", "metadata"})

    @classmethod
    def _normalize(cls, data: dict[str, Any]) -> dict[str, Any]:
        metadata = _mapping(data.get("metadata", {}), "telemetry metadata")
        if "relay_enabled" in data:
            metadata.setdefault("relay_enabled", data["relay_enabled"])
            data = {
                "provider": "relay",
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
    event_id: str
    timestamp_millis: int
    kind: str
    message: str
    metadata: Mapping[str, Any]
    _fields = frozenset({"event_id", "timestamp_millis", "kind", "message", "metadata"})

    @classmethod
    def _normalize(cls, data: dict[str, Any]) -> dict[str, Any]:
        data["metadata"] = _mapping(data.get("metadata", {}), "event metadata")
        return data


class RuntimeHandle(FabricMapping):
    runtime_id: str
    runtime_binding: str
    agent_name: str
    harness: str
    mode: str
    adapter_kind: str
    adapter_id: str | None
    _fields = frozenset(
        {
            "runtime_id",
            "runtime_binding",
            "agent_name",
            "harness",
            "mode",
            "adapter_kind",
            "adapter_id",
        }
    )


class RunResult(FabricMapping):
    agent_name: str
    profiles: Sequence[str]
    harness: str
    adapter_kind: str
    adapter_id: str
    runtime_id: str
    invocation_id: str
    request_id: str
    status: str
    output: Any
    error: ErrorInfo | None
    artifacts: ArtifactManifest
    telemetry: Sequence[TelemetryRef]
    events: Sequence[FabricEvent]
    metadata: Mapping[str, Any]
    _fields = frozenset(
        {
            "agent_name",
            "profiles",
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

    @classmethod
    def _normalize(cls, data: dict[str, Any]) -> dict[str, Any]:
        data["profiles"] = tuple(data.get("profiles", []))
        data["error"] = (
            None if data.get("error") is None else ErrorInfo.from_mapping(data["error"])
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
        return data


class SessionInfo(FabricMapping):
    session_id: str
    runtime_id: str
    agent_name: str
    profiles: Sequence[str]
    harness: str
    adapter_id: str
    adapter_kind: str
    status: str
    capabilities: RuntimeCapabilities
    _fields = frozenset(
        {
            "session_id",
            "runtime_id",
            "agent_name",
            "profiles",
            "harness",
            "adapter_id",
            "adapter_kind",
            "status",
            "capabilities",
        }
    )

    @classmethod
    def _normalize(cls, data: dict[str, Any]) -> dict[str, Any]:
        data["profiles"] = tuple(data.get("profiles", []))
        data["capabilities"] = RuntimeCapabilities.from_mapping(data.get("capabilities", {}))
        return data


class RuntimeUpdate(FabricMapping):
    overrides: Mapping[str, Any]
    metadata: Mapping[str, Any]
    _fields = frozenset({"overrides", "metadata"})


class RuntimeUpdateResult(FabricMapping):
    status: str
    applied: Mapping[str, Any]
    rejected: Mapping[str, Any]
    reason: str | None
    _fields = frozenset({"status", "applied", "rejected", "reason"})
