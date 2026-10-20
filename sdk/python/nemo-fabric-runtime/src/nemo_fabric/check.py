# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Health requests and immutable health reports."""

from __future__ import annotations

import re
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any, Literal, Self
from types import MappingProxyType

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from nemo_fabric.errors import FabricConfigError
from nemo_fabric.types import FabricMapping

CheckLevel = Literal["live", "active", "ready"]
CheckOutcome = Literal["passed", "failed", "unknown", "unsupported"]
CheckReason = Literal[
    "server_stopping",
    "capabilities_unverified",
    "runtime_unavailable",
    "service_unavailable",
    "agent_exited",
    "session_closed",
    "endpoint_unavailable",
    "health_failed",
    "probe_timeout",
    "probe_error",
    "observation_changed",
    "unsupported_adapter",
    "unsupported_level",
    "unsupported_scope",
]
CheckSource = Literal[
    "fabric.runtime_server",
    "fabric.runtime_server.transport",
    "fabric.adapter",
    "fabric.adapter.liveness",
    "fabric.adapter.interface",
    "fabric.adapter.readiness",
]


class CheckRequest(BaseModel):
    """Select server, runtime, or registered service health.

    Runtime and service checks default to ``live``. One positive, finite timeout covers all
    requested levels.
    Server checks reject runtime IDs, service IDs, and levels, including explicit nulls.
    """

    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, allow_inf_nan=False
    )
    schema_version: Literal["fabric.check/v1alpha1"] = "fabric.check/v1alpha1"
    scope: Literal["server", "runtime", "service"]
    runtime_id: str | None = None
    service_id: str | None = None
    level: CheckLevel | None = None
    timeout_seconds: float = Field(default=3.0, gt=0)

    @model_validator(mode="after")
    def _target(self) -> Self:
        if self.scope == "server":
            if self.model_fields_set & {"runtime_id", "service_id", "level"}:
                raise ValueError(
                    "server checks cannot select a runtime, service, or level"
                )
        else:
            target, other = (
                ("service_id", "runtime_id")
                if self.scope == "service"
                else ("runtime_id", "service_id")
            )
            selected = getattr(self, target)
            if other in self.model_fields_set:
                raise ValueError("health checks must select exactly one target")
            if not selected or not selected.strip() or "\x00" in selected:
                raise ValueError(f"{self.scope} checks require a nonempty {target}")
            if self.level is None:
                if "level" in self.model_fields_set:
                    raise ValueError("health level cannot be null")
                object.__setattr__(self, "level", "live")
        return self

    def to_mapping(self) -> dict[str, Any]:
        """Return a detached request using the defaulted level and timeout."""
        return self.model_dump(mode="json", exclude_none=True)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


def _utc(value: str) -> str:
    if not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|\+00:00)", value
    ):
        raise ValueError("invalid UTC timestamp")
    datetime.fromisoformat(value)
    return value


class _Observation(_StrictModel):
    name: Literal["server", "live", "active", "ready"]
    outcome: CheckOutcome
    reason_code: CheckReason | None
    source: CheckSource
    observed_at: str | None

    @field_validator("observed_at")
    @classmethod
    def _time(cls, value: str | None) -> str | None:
        return None if value is None else _utc(value)

    @model_validator(mode="after")
    def _evidence(self) -> Self:
        if (self.outcome == "passed") != (self.reason_code is None):
            raise ValueError("inconsistent observation outcome")
        if self.outcome == "passed" and self.observed_at is None:
            raise ValueError("passing requires fresh evidence")
        if (
            self.outcome == "passed"
            and self.source
            != {
                "server": "fabric.runtime_server",
                "live": "fabric.adapter.liveness",
                "active": "fabric.adapter.interface",
                "ready": "fabric.adapter.readiness",
            }[self.name]
        ):
            raise ValueError("invalid evidence source")
        return self


class _Server(_StrictModel):
    state: Literal["responding", "stopping", "unknown"]


class _Runtime(_StrictModel):
    state: Literal["active", "stopped", "failed", "unknown"]
    activity: Literal["idle", "busy", "stopping", "stopped", "unknown"]


class _Report(_StrictModel):
    schema_version: Literal["fabric.check/v1alpha1"]
    scope: Literal["server", "runtime", "service"]
    runtime_id: str | None
    service_id: str | None = None
    requested_level: CheckLevel | None
    outcome: CheckOutcome
    reason_code: CheckReason | None
    completed_at: str
    elapsed_ms: float = Field(ge=0)
    server: _Server | None
    runtime: _Runtime | None
    checks: list[_Observation]

    @field_validator("completed_at")
    @classmethod
    def _completion(cls, value: str) -> str:
        return _utc(value)

    @model_validator(mode="after")
    def _consistency(self) -> Self:
        if (self.outcome == "passed") != (self.reason_code is None):
            raise ValueError("inconsistent health outcome")
        completed = datetime.fromisoformat(self.completed_at)
        if self.scope != "service" and self.service_id is not None:
            raise ValueError("unexpected service target")
        for observation in self.checks:
            if (
                observation.observed_at
                and datetime.fromisoformat(observation.observed_at) > completed
            ):
                raise ValueError("observation follows completion")
        if self.scope == "server":
            if (
                self.runtime_id is not None
                or self.requested_level is not None
                or self.runtime is not None
            ):
                raise ValueError("invalid server scope")
            if self.server is None:
                if (
                    self.outcome != "unsupported"
                    or self.reason_code != "unsupported_scope"
                    or self.checks
                ):
                    raise ValueError("missing server evidence")
                return self
            if self.server.state == "unknown" and self.reason_code not in (
                "probe_timeout",
                "probe_error",
            ):
                raise ValueError("invalid unobserved server diagnostic")
            expected = {
                "responding": ("passed", None),
                "stopping": ("failed", "server_stopping"),
                "unknown": ("unknown", self.reason_code),
            }[self.server.state]
            if (
                (self.outcome, self.reason_code) != expected
                or len(self.checks) != 1
                or self.checks[0].name != "server"
            ):
                raise ValueError("inconsistent server evidence")
            observation = self.checks[0]
            observed = self.server.state != "unknown"
            if (
                (observation.outcome, observation.reason_code) != expected
                or observation.source
                != (
                    "fabric.runtime_server"
                    if observed
                    else "fabric.runtime_server.transport"
                )
                or (observation.observed_at is not None) != observed
            ):
                raise ValueError("invalid server observation")
        else:
            service = self.scope == "service"
            target = self.service_id if service else self.runtime_id
            if (
                not target
                or not target.strip()
                or "\x00" in target
                or self.requested_level is None
                or self.server is not None
                or service
                and (self.runtime_id is not None or self.runtime is not None)
            ):
                raise ValueError("invalid health target")
            levels = ["live", "active", "ready"][
                : ["live", "active", "ready"].index(self.requested_level) + 1
            ]
            if [check.name for check in self.checks] != levels[: len(self.checks)]:
                raise ValueError("invalid probe order")
            if any(check.outcome != "passed" for check in self.checks[:-1]):
                raise ValueError("later checks ran after nonpassing evidence")
            prefix = "fabric.adapter.openclaw." if service else "fabric.adapter.pi."
            if any(
                check.outcome == "passed" and not check.source.startswith(prefix)
                for check in self.checks
            ):
                raise ValueError("evidence does not match the target scope")
            if self.outcome == "passed" and (
                len(self.checks) != len(levels)
                or not service
                and (
                    self.runtime is None
                    or self.runtime.state != "active"
                    or self.runtime.activity in ("stopped", "stopping", "unknown")
                )
            ):
                raise ValueError("incomplete passing evidence")
        if self.checks:
            final = self.checks[-1]
            invalidated = (
                self.scope in ("runtime", "service")
                and self.outcome == "unknown"
                and self.reason_code == "observation_changed"
                and self.runtime is None
            )
            if (
                final.outcome != "passed"
                and not invalidated
                and (self.outcome, self.reason_code)
                != (final.outcome, final.reason_code)
            ):
                raise ValueError("inconsistent final observation")
            if (
                final.outcome == "passed"
                and self.outcome != "passed"
                and self.reason_code
                not in (
                    "observation_changed",
                    "probe_timeout",
                    "unsupported_level",
                )
            ):
                raise ValueError("invalid aggregate outcome")
        elif self.outcome == "passed":
            raise ValueError("passing requires evidence")
        return self


class CheckReport(FabricMapping):
    """Immutable health check report.

    ``to_mapping()`` returns detached JSON data.
    """

    schema_version: str
    scope: str
    runtime_id: str | None
    requested_level: str | None
    outcome: str
    reason_code: str | None
    completed_at: str
    elapsed_ms: float
    server: Mapping[str, Any] | None
    runtime: Mapping[str, Any] | None
    checks: list[Mapping[str, Any]]
    _fields = frozenset(_Report.model_fields)
    _json_fields = frozenset({"server", "runtime", "checks"})

    @property
    def service_id(self) -> str | None:
        """Return the selected service, or null for other check targets."""
        return self._data.get("service_id")

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("CheckReport is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("CheckReport is immutable")

    def _complete(self, started: float) -> Self:
        # Record client completion after response decoding and validation.
        data = dict(self._data)
        data["completed_at"] = (
            datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        )
        data["elapsed_ms"] = max(0.0, (time.monotonic() - started) * 1000)
        object.__setattr__(self, "_data", MappingProxyType(data))
        return self

    @classmethod
    def _normalize(cls, data: dict[str, Any]) -> dict[str, Any]:
        try:
            value = _Report.model_validate(data).model_dump(mode="json")
            if value["service_id"] is None:
                value.pop("service_id")
            return value
        except (ValidationError, ValueError):
            raise FabricConfigError("The check report is invalid.") from None


def _timeout_report(request: CheckRequest, started: float) -> CheckReport:
    completed = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    server = request.scope == "server"
    return CheckReport.from_mapping(
        {
            "schema_version": "fabric.check/v1alpha1",
            "scope": request.scope,
            "runtime_id": request.runtime_id,
            **(
                {"service_id": request.service_id} if request.scope == "service" else {}
            ),
            "requested_level": request.level,
            "outcome": "unknown",
            "reason_code": "probe_timeout",
            "completed_at": completed,
            "elapsed_ms": max(0.0, (time.monotonic() - started) * 1000),
            "server": {"state": "unknown"} if server else None,
            "runtime": None,
            "checks": [
                {
                    "name": "server",
                    "outcome": "unknown",
                    "reason_code": "probe_timeout",
                    "source": "fabric.runtime_server.transport",
                    "observed_at": None,
                }
            ]
            if server
            else [],
        }
    )._complete(started)
