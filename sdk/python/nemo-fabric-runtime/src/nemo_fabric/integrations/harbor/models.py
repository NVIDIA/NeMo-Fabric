# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transport contracts for the Harbor integration."""

from __future__ import annotations

from pathlib import PurePosixPath
import re
from typing import Annotated
from typing import Literal
from typing import Self

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import JsonValue
from pydantic import field_validator
from pydantic import model_validator

from nemo_fabric import FabricError
from nemo_fabric import FabricConfig
from nemo_fabric import RunRequest


class HarborMcpServer(BaseModel):
    """One Harbor-provided MCP server."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    transport: Literal["stdio", "sse", "streamable-http"]
    url: str | None = None
    command: str | None = None
    args: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_target(self) -> Self:
        if self.transport == "stdio":
            if not self.command:
                raise ValueError("stdio MCP servers require command")
        elif not self.url:
            raise ValueError(f"{self.transport} MCP servers require url")
        return self


class FabricRunPayload(BaseModel):
    """Typed Fabric inputs transported into one Harbor task environment."""

    model_config = ConfigDict(extra="forbid")

    config: FabricConfig
    config_base_dir: PurePosixPath
    logs_dir: PurePosixPath = PurePosixPath("/logs/agent")
    skills_dir: PurePosixPath | None = None
    request: RunRequest
    environment_env_names: tuple[str, ...] = ()
    adapter_descriptor_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )

    @field_validator("environment_env_names")
    @classmethod
    def validate_env_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None for name in value):
            raise ValueError(
                "environment_env_names must contain environment variable names"
            )
        return value


class FabricRunnerError(BaseModel):
    """Failure before the SDK can return a normalized RunResult."""

    model_config = ConfigDict(extra="forbid")

    stage: str | None = None
    code: str | None = None
    message: str
    retryable: bool = False


_FAILURE_DIMENSIONS = (
    "identity",
    "partial_result",
    "execution",
    "system_stats",
    "resource_snapshot",
    "usage",
    "artifacts",
    "telemetry",
    "events",
)
_MAX_FAILURE_EVIDENCE_BYTES = 262144


class FabricFailureEvidence(BaseModel):
    """Bounded observations from an incomplete lifecycle, never a terminal result.

    Mapping fields use the existing SDK validators rather than a second schema.
    Coverage is partial even for a valid retained observation: failure can cut
    collection short. Invalid optional dimensions do not hide valid siblings.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["fabric.harbor.failure_evidence/v1"] = (
        "fabric.harbor.failure_evidence/v1"
    )
    runtime_id: str | None = Field(default=None, max_length=256)
    invocation_id: str | None = Field(default=None, max_length=256)
    request_id: str | None = Field(default=None, max_length=256)
    cleanup: Literal["completed", "failed", "unknown"] = "unknown"
    exception_type: str | None = Field(default=None, max_length=256)
    cause_types: list[Annotated[str, Field(max_length=256)]] = Field(
        default_factory=list, max_length=8
    )
    partial_result: dict[str, JsonValue] | None = None
    execution: dict[str, JsonValue] | None = None
    system_stats: dict[str, JsonValue] | None = None
    resource_snapshot: dict[str, JsonValue] | None = None
    usage: dict[str, JsonValue] | None = None
    artifacts: list[dict[str, JsonValue]] = Field(default_factory=list, max_length=64)
    telemetry: list[dict[str, JsonValue]] = Field(default_factory=list, max_length=64)
    events: list[dict[str, JsonValue]] = Field(default_factory=list, max_length=256)
    coverage: dict[
        Literal[
            "identity",
            "partial_result",
            "execution",
            "system_stats",
            "resource_snapshot",
            "usage",
            "artifacts",
            "telemetry",
            "events",
        ],
        Literal[
            "missing", "partial", "invalid", "truncated", "not_requested", "unsupported"
        ],
    ] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def validate_observations(cls, value):
        import json
        from nemo_fabric.types import (
            ArtifactRef,
            FabricEvent,
            InvocationSystemStats,
            RunResult,
            RunUsage,
            RuntimeResourceSnapshot,
            TelemetryRef,
        )
        from nemo_fabric_evidence.evidence import ExecutionEvidence

        if not isinstance(value, dict):
            raise ValueError("failure evidence must be an object")
        data = dict(value)
        coverage = dict(data.get("coverage") or {})
        identity_invalid = False
        for name in ("runtime_id", "invocation_id", "request_id"):
            item = data.get(name)
            if item is not None and (
                not isinstance(item, str) or not item or len(item) > 256
            ):
                data[name] = None
                identity_invalid = True
        identity_coverage = (
            "invalid"
            if identity_invalid or coverage.get("identity") == "invalid"
            else "partial"
            if any(
                data.get(name) for name in ("runtime_id", "invocation_id", "request_id")
            )
            else "missing"
        )
        coverage["identity"] = (
            coverage["identity"]
            if coverage.get("identity") in ("invalid", "unsupported", "truncated")
            else identity_coverage
        )
        validators = {
            "partial_result": RunResult,
            "execution": ExecutionEvidence,
            "system_stats": InvocationSystemStats,
            "resource_snapshot": RuntimeResourceSnapshot,
            "usage": RunUsage,
            "artifacts": ArtifactRef,
            "telemetry": TelemetryRef,
            "events": FabricEvent,
        }
        budget = _MAX_FAILURE_EVIDENCE_BYTES - 16384
        for name, validator in validators.items():
            item = data.get(name)
            sequence = name in ("artifacts", "telemetry", "events")
            if item is None or (sequence and item == []):
                coverage.setdefault(name, "missing")
                continue
            try:
                if sequence:
                    limit = 256 if name == "events" else 64
                    if not isinstance(item, list):
                        raise ValueError("expected array")
                    retained = []
                    invalid = False
                    truncated = len(item) > limit
                    for entry in item[:limit]:
                        try:
                            required = {
                                "artifacts": ("name", "kind", "path"),
                                "telemetry": ("provider", "kind"),
                                "events": ("event_id", "kind"),
                            }[name]
                            if not isinstance(entry, dict) or any(
                                not isinstance(entry.get(key), str) or not entry[key]
                                for key in required
                            ):
                                raise ValueError("invalid evidence reference")
                            if name == "events" and not isinstance(
                                entry.get("message"), str
                            ):
                                raise ValueError("invalid event message")
                            if name == "events" and (
                                type(entry.get("timestamp_millis")) is not int
                                or entry["timestamp_millis"] < 0
                            ):
                                raise ValueError("invalid event timestamp")
                            validated = validator.from_mapping(entry).to_mapping()
                            size = len(json.dumps(validated, allow_nan=False).encode())
                            if size > budget:
                                truncated = True
                                continue
                            budget -= size
                            retained.append(validated)
                        except (ValueError, TypeError, KeyError, FabricError):
                            invalid = True
                    data[name] = retained
                    coverage[name] = (
                        coverage.get(name)
                        if coverage.get(name) in ("invalid", "truncated")
                        else (
                            "truncated"
                            if truncated
                            else "invalid"
                            if invalid
                            else "partial"
                        )
                    )
                else:
                    validated = validator.from_mapping(item).to_mapping()
                    size = len(json.dumps(validated, allow_nan=False).encode())
                    if size > budget:
                        data[name] = None
                        coverage[name] = "truncated"
                    else:
                        budget -= size
                        data[name] = validated
                        coverage[name] = "partial"
            except (ValueError, TypeError, KeyError, FabricError):
                data[name] = [] if sequence else None
                coverage[name] = "invalid"
        data["coverage"] = coverage
        return data


class FabricRunnerFailure(BaseModel):
    """Machine-readable runner failure, distinct from an invocation result."""

    model_config = ConfigDict(extra="forbid")

    runner_error: FabricRunnerError
    evidence: FabricFailureEvidence | None = None

    @field_validator("evidence", mode="before")
    @classmethod
    def retain_valid_evidence(cls, value):
        return None if value is None else _validated_failure_evidence(value)


def _validated_failure_evidence(value: object) -> FabricFailureEvidence:
    """Validate optional evidence without replacing the primary failure."""
    if isinstance(value, FabricFailureEvidence):
        return value
    if (
        isinstance(value, dict)
        and value.get("schema_version", "fabric.harbor.failure_evidence/v1")
        != "fabric.harbor.failure_evidence/v1"
    ):
        return FabricFailureEvidence(
            coverage={name: "unsupported" for name in _FAILURE_DIMENSIONS}
        )
    try:
        return FabricFailureEvidence.model_validate(value)
    except (ValueError, TypeError, KeyError, FabricError):
        return FabricFailureEvidence(
            coverage={name: "invalid" for name in _FAILURE_DIMENSIONS}
        )
