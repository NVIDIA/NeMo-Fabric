# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Server-scope health evidence; no runtime or adapter access."""

import re
import sys
import time
from datetime import datetime, timezone
from typing import Any

CHECK_SECONDS = 3.0
SCHEMA_VERSION = "fabric.check/v1alpha1"
SERVER_STATES = {
    "responding": ("passed", None),
    "stopping": ("failed", "server_stopping"),
    "unknown": ("unknown", "probe_timeout"),
}


def server_report(state: str, started: float) -> dict[str, Any]:
    """Record a fresh server response, or an unobserved transport timeout."""
    outcome, reason = SERVER_STATES[state]
    completed = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    observed = state != "unknown"
    return {
        "schema_version": SCHEMA_VERSION,
        "scope": "server",
        "runtime_id": None,
        "requested_level": None,
        "outcome": outcome,
        "reason_code": reason,
        "completed_at": completed,
        "elapsed_ms": max(0.0, (time.monotonic() - started) * 1000),
        "server": {"state": state},
        "runtime": None,
        "checks": [
            {
                "name": "server",
                "outcome": outcome,
                "reason_code": reason,
                "source": "fabric.runtime_server"
                if observed
                else "fabric.runtime_server.transport",
                "observed_at": completed if observed else None,
            }
        ],
    }


def _timestamp(value: Any) -> datetime:
    if not isinstance(value, str) or not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|\+00:00)", value
    ):
        raise ValueError("invalid UTC timestamp")
    return datetime.fromisoformat(value)


def validate_server_response(response: dict[str, Any]) -> None:
    """Reject malformed evidence and inconsistent command outcomes."""
    if response["changed"] is not False:
        raise ValueError("health checks cannot change state")
    error = response["error"]
    if error is not None and (
        error["effects"] != "none"
        or error["stage"] not in ("request", "check", "transport")
    ):
        raise ValueError("invalid check diagnostic")
    report = response["result"]
    if report is None:
        if error is None:
            raise ValueError("missing check report")
        return
    if set(report) != {
        "schema_version",
        "scope",
        "runtime_id",
        "requested_level",
        "outcome",
        "reason_code",
        "completed_at",
        "elapsed_ms",
        "server",
        "runtime",
        "checks",
    }:
        raise ValueError("invalid check report fields")
    server = report["server"]
    if not isinstance(server, dict) or set(server) != {"state"}:
        raise ValueError("invalid server observation")
    state = server["state"]
    if not isinstance(state, str) or state not in SERVER_STATES:
        raise ValueError("invalid server state")
    outcome, reason = SERVER_STATES[state]
    if (
        report["schema_version"] != SCHEMA_VERSION
        or report["scope"] != "server"
        or report["runtime_id"] is not None
        or report["requested_level"] is not None
        or report["runtime"] is not None
        or report["outcome"] != outcome
        or report["reason_code"] != reason
        or type(report["elapsed_ms"]) not in (int, float)
        or not 0 <= report["elapsed_ms"] <= sys.float_info.max
        or response["status"] != ("succeeded" if outcome == "passed" else "failed")
        or (
            reason is not None
            and (error is None or error["code"] != reason or error["stage"] != "check")
        )
    ):
        raise ValueError("inconsistent check report")
    completed = _timestamp(report["completed_at"])
    checks = report["checks"]
    if (
        not isinstance(checks, list)
        or len(checks) != 1
        or not isinstance(checks[0], dict)
    ):
        raise ValueError("invalid server evidence")
    observation = checks[0]
    observed_at = observation.get("observed_at")
    if state != "unknown":
        if _timestamp(observed_at) > completed:
            raise ValueError("observation follows completion")
    elif observed_at is not None:
        raise ValueError("timeout has no server observation")
    if observation != {
        "name": "server",
        "outcome": outcome,
        "reason_code": reason,
        "source": "fabric.runtime_server"
        if state != "unknown"
        else "fabric.runtime_server.transport",
        "observed_at": observed_at,
    }:
        raise ValueError("invalid server evidence fields")
