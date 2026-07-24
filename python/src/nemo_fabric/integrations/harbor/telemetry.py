# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate NeMo Fabric telemetry and publish Harbor-compatible run evidence."""

from __future__ import annotations

import json
import re
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from nemo_fabric import RunResult


class TelemetryValidationError(ValueError):
    """Raised when emitted ATOF or ATIF evidence is malformed or ambiguous."""


def publish_telemetry_evidence(
    result: RunResult,
    logs_dir: Path,
    *,
    strict: bool = False,
    harbor_session_id: str | None = None,
    harbor_context_id: str | None = None,
) -> dict[str, Any]:
    """Validate telemetry, promote ATIF, and write a machine-readable summary.

    Telemetry quality is deliberately independent from Harbor task correctness.
    By default validation failures are recorded instead of changing the task reward.
    Callers performing an explicit telemetry gate can pass ``strict=True``.
    """

    summary_path = logs_dir / "telemetry-validation.json"
    try:
        summary = validate_telemetry(
            result,
            logs_dir,
            harbor_session_id=harbor_session_id,
            harbor_context_id=harbor_context_id,
        )
    except (OSError, json.JSONDecodeError, TelemetryValidationError, ValueError) as error:
        summary = _base_summary(result)
        summary.update(status="failed", error=str(error))
        _write_summary(summary_path, summary, strict=strict)
        if strict:
            raise TelemetryValidationError(str(error)) from error
        return summary

    _write_summary(summary_path, summary, strict=strict)
    return summary


def _write_summary(path: Path, summary: dict[str, Any], *, strict: bool) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    except OSError:
        if strict:
            raise


def validate_telemetry(
    result: RunResult,
    logs_dir: Path,
    *,
    harbor_session_id: str | None = None,
    harbor_context_id: str | None = None,
) -> dict[str, Any]:
    """Validate telemetry artifacts and promote exactly one valid ATIF trajectory."""

    artifacts = tuple(result.artifacts.artifacts)
    atof_paths = [
        _resolve_artifact_path(Path(artifact.path), logs_dir)
        for artifact in artifacts
        if artifact.kind == "atof"
    ]
    atif_paths = [
        _resolve_artifact_path(Path(artifact.path), logs_dir)
        for artifact in artifacts
        if artifact.kind == "atif"
    ]
    summary = _base_summary(result)
    summary["harbor_session_id"] = harbor_session_id
    summary["harbor_context_id"] = harbor_context_id
    summary["atof"] = _validate_atof(atof_paths)
    summary["atif"] = _validate_atif(
        atif_paths,
        logs_dir,
    )
    summary["status"] = "not_emitted" if not atof_paths and not atif_paths else "succeeded"
    return summary


def _resolve_artifact_path(path: Path, logs_dir: Path) -> Path:
    """Resolve collected task paths when validation runs on the Harbor host."""

    task_logs = Path("/logs/agent")
    if ".." in path.parts or not path.is_relative_to(task_logs):
        raise TelemetryValidationError(f"telemetry artifact escapes /logs/agent: {path}")
    if logs_dir != task_logs:
        relative = path.relative_to(task_logs)
        return logs_dir / relative
    return path


def _validate_atof(paths: list[Path]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    records = 0
    required = {"atof_version", "kind", "name", "timestamp", "uuid"}
    for path in paths:
        if not path.is_file():
            raise TelemetryValidationError(f"ATOF artifact does not exist: {path}")
        text = path.read_text(encoding="utf-8")
        _reject_obvious_secrets(text, path)
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TelemetryValidationError(f"ATOF record must be an object: {path}:{line_number}")
            missing = required.difference(value)
            if missing:
                raise TelemetryValidationError(f"ATOF record missing {sorted(missing)}: {path}:{line_number}")
            _validate_atof_record(value, path, line_number)
            records += 1
            counts[value["kind"]] += 1
    return {
        "files": [str(path) for path in paths],
        "records": records,
        "kinds": dict(sorted(counts.items())),
    }


def _validate_atof_record(value: dict[str, Any], path: Path, line_number: int) -> None:
    location = f"{path}:{line_number}"
    for field in ("atof_version", "kind", "name", "timestamp", "uuid"):
        if not isinstance(value[field], str) or not value[field]:
            raise TelemetryValidationError(f"ATOF record field {field} must be a non-empty string: {location}")
    if re.fullmatch(r"\d+\.\d+(?:\.\d+)?", value["atof_version"]) is None:
        raise TelemetryValidationError(f"ATOF record atof_version is invalid: {location}")
    try:
        timestamp = datetime.fromisoformat(value["timestamp"].replace("Z", "+00:00"))
    except ValueError as error:
        raise TelemetryValidationError(f"ATOF record timestamp is invalid: {location}") from error
    if timestamp.tzinfo is None:
        raise TelemetryValidationError(f"ATOF record timestamp must include a timezone: {location}")
    try:
        UUID(value["uuid"])
    except ValueError as error:
        raise TelemetryValidationError(f"ATOF record uuid is invalid: {location}") from error


def _validate_atif(
    paths: list[Path],
    logs_dir: Path,
) -> dict[str, Any]:
    if len(paths) > 1:
        raise TelemetryValidationError(f"expected at most one ATIF artifact, found {len(paths)}")
    if not paths:
        return {"files": [], "promoted": None}

    path = paths[0]
    if not path.is_file():
        raise TelemetryValidationError(f"ATIF artifact does not exist: {path}")

    text = path.read_text(encoding="utf-8")
    _reject_obvious_secrets(text, path)
    trajectory = _validate_atif_structure(json.loads(text), path)
    session_id = trajectory.get("session_id")
    canonical = logs_dir / "trajectory.json"
    if path.resolve() != canonical.resolve():
        shutil.copyfile(path, canonical)
    return {
        "files": [str(path)],
        "promoted": str(canonical),
        "validator": "fabric_structural",
        "schema_version": trajectory["schema_version"],
        "session_id": session_id,
        "agent": trajectory["agent"]["name"],
        "steps": len(trajectory["steps"]),
        "final_metrics": trajectory.get("final_metrics"),
    }


def _validate_atif_structure(value: Any, path: Path) -> dict[str, Any]:
    """Validate the portable ATIF boundary without importing Harbor in the task."""

    if not isinstance(value, dict):
        raise TelemetryValidationError(f"ATIF trajectory must be an object: {path}")
    schema_version = value.get("schema_version")
    supported_versions = {f"ATIF-v1.{minor}" for minor in range(8)}
    if schema_version not in supported_versions:
        raise TelemetryValidationError(f"unsupported ATIF schema_version {schema_version!r}: {path}")
    session_id = value.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        raise TelemetryValidationError(f"ATIF session_id must be a non-empty string: {path}")
    agent = value.get("agent")
    if not isinstance(agent, dict) or not isinstance(agent.get("name"), str) or not agent["name"]:
        raise TelemetryValidationError(f"ATIF agent.name must be a non-empty string: {path}")
    if not isinstance(agent.get("version"), str) or not agent["version"]:
        raise TelemetryValidationError(f"ATIF agent.version must be a non-empty string: {path}")
    steps = value.get("steps")
    if not isinstance(steps, list) or not steps:
        raise TelemetryValidationError(f"ATIF steps must be a non-empty array: {path}")
    for index, step in enumerate(steps, 1):
        if not isinstance(step, dict):
            raise TelemetryValidationError(f"ATIF step {index} must be an object: {path}")
        if not isinstance(step.get("step_id"), int) or step["step_id"] < 1:
            raise TelemetryValidationError(f"ATIF step {index} has an invalid step_id: {path}")
        if step.get("source") not in {"system", "user", "agent"}:
            raise TelemetryValidationError(f"ATIF step {index} has an invalid source: {path}")
        if not isinstance(step.get("message"), str | list):
            raise TelemetryValidationError(f"ATIF step {index} has an invalid message: {path}")
    final_metrics = value.get("final_metrics")
    if final_metrics is not None and not isinstance(final_metrics, dict):
        raise TelemetryValidationError(f"ATIF final_metrics must be an object or null: {path}")
    return value


def _base_summary(result: RunResult) -> dict[str, Any]:
    return {
        "schema_version": "fabric.harbor.telemetry/v1alpha1",
        "status": "pending",
        "runtime_id": result.runtime_id,
        "invocation_id": result.invocation_id,
        "request_id": result.request_id,
        "harness": result.harness,
        "adapter_id": result.adapter_id,
    }


_SECRET_PATTERNS = (
    re.compile(r"\b(?:sk|nvapi)-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r'(?i)["\'](?:api[_-]?key|access[_-]?token|authorization)["\']\s*:\s*["\'][^"\']{8,}["\']'),
)


def _reject_obvious_secrets(text: str, path: Path) -> None:
    if any(pattern.search(text) for pattern in _SECRET_PATTERNS):
        raise TelemetryValidationError(f"telemetry contains a value that resembles a credential: {path}")
