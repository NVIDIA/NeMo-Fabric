# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict, safe health requests and immutable health reports."""

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

CheckLevel = Literal["live", "active", "ready", "operational"]
CheckOutcome = Literal["passed", "failed", "unknown", "unsupported"]
CheckReason = Literal[
    "server_stopping",
    "capabilities_unverified",
    "runtime_unavailable",
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
    "fabric.adapter.pi.session",
    "fabric.adapter.pi.interface",
    "fabric.adapter.pi.setup",
]


class CheckRequest(BaseModel):
    """Select server or runtime health in a separately supplied environment.

    Runtime checks default to ``live``. One positive, finite timeout covers all
    requested levels. ``operational`` returns unsupported without probing.
    Server checks reject runtime IDs and levels, including explicit nulls.
    """

    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, allow_inf_nan=False
    )
    schema_version: Literal["fabric.check/v1alpha1"] = "fabric.check/v1alpha1"
    scope: Literal["server", "runtime"]
    runtime_id: str | None = None
    level: CheckLevel | None = None
    timeout_seconds: float = Field(default=3.0, gt=0)

    @model_validator(mode="after")
    def _target(self) -> Self:
        if self.scope == "server":
            if self.model_fields_set & {"runtime_id", "level"}:
                raise ValueError("server checks cannot select a runtime or level")
        else:
            if (
                not self.runtime_id
                or not self.runtime_id.strip()
                or "\x00" in self.runtime_id
            ):
                raise ValueError("runtime checks require a nonempty runtime ID")
            if self.level is None:
                if "level" in self.model_fields_set:
                    raise ValueError("runtime level cannot be null")
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
                "live": "fabric.adapter.pi.session",
                "active": "fabric.adapter.pi.interface",
                "ready": "fabric.adapter.pi.setup",
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
    scope: Literal["server", "runtime"]
    runtime_id: str | None
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
            if (
                not self.runtime_id
                or not self.runtime_id.strip()
                or self.requested_level is None
                or self.server is not None
            ):
                raise ValueError("invalid runtime scope")
            if self.requested_level == "operational":
                if (
                    self.outcome != "unsupported"
                    or self.reason_code != "unsupported_level"
                    or self.checks
                ):
                    raise ValueError("operational must not probe")
                return self
            levels = ["live", "active", "ready"][
                : ["live", "active", "ready"].index(self.requested_level) + 1
            ]
            if [check.name for check in self.checks] != levels[: len(self.checks)]:
                raise ValueError("invalid probe order")
            if any(check.outcome != "passed" for check in self.checks[:-1]):
                raise ValueError("later checks ran after nonpassing evidence")
            if self.outcome == "passed" and (
                len(self.checks) != len(levels)
                or self.runtime is None
                or self.runtime.state != "active"
                or self.runtime.activity in ("stopped", "stopping", "unknown")
            ):
                raise ValueError("incomplete passing evidence")
        if self.checks:
            final = self.checks[-1]
            invalidated = (
                self.scope == "runtime"
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
    """Immutable health evidence, independent of invocation outcomes.

    ``to_mapping()`` returns detached JSON data. Passing health does not promise
    model access, valid credentials, concurrent invocation, or descendant cleanup.
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

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("CheckReport is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("CheckReport is immutable")

    def _complete(self, started: float) -> Self:
        # Evidence is already validated and frozen. Only caller-owned timing
        # changes here, after decoding and validation have finished.
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
            return _Report.model_validate(data).model_dump(mode="json")
        except (ValidationError, ValueError):
            raise FabricConfigError("The check report is invalid.") from None


def _timeout_report(request: CheckRequest, started: float) -> CheckReport:
    completed = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    server = request.scope == "server"
    # The reserved level remains unsupported even if routing exhausts the budget.
    operational = request.level == "operational"
    return CheckReport.from_mapping(
        {
            "schema_version": "fabric.check/v1alpha1",
            "scope": request.scope,
            "runtime_id": request.runtime_id,
            "requested_level": request.level,
            "outcome": "unsupported" if operational else "unknown",
            "reason_code": "unsupported_level" if operational else "probe_timeout",
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
