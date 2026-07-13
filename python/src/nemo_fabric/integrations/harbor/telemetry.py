# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate Fabric telemetry and publish Harbor-compatible run evidence."""

from __future__ import annotations

import json
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

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

    logs_dir.mkdir(parents=True, exist_ok=True)
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
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        if strict:
            raise TelemetryValidationError(str(error)) from error
        return summary

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def validate_telemetry(
    result: RunResult,
    logs_dir: Path,
    *,
    harbor_session_id: str | None = None,
    harbor_context_id: str | None = None,
) -> dict[str, Any]:
    """Validate telemetry artifacts and promote exactly one valid ATIF trajectory."""

    artifacts = tuple(result.artifacts.artifacts)
    atof_paths = [Path(artifact.path) for artifact in artifacts if artifact.kind == "atof"]
    atif_paths = [Path(artifact.path) for artifact in artifacts if artifact.kind == "atif"]
    summary = _base_summary(result)
    summary["harbor_session_id"] = harbor_session_id
    summary["harbor_context_id"] = harbor_context_id
    summary["atof"] = _validate_atof(atof_paths)
    summary["atif"] = _validate_atif(
        atif_paths,
        logs_dir,
        expected_session_ids={value for value in (result.runtime_id, harbor_session_id) if value},
    )
    summary["status"] = "not_emitted" if not atof_paths and not atif_paths else "succeeded"
    return summary


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
            records += 1
            counts[str(value["kind"])] += 1
    return {
        "files": [str(path) for path in paths],
        "records": records,
        "kinds": dict(sorted(counts.items())),
    }


def _validate_atif(
    paths: list[Path],
    logs_dir: Path,
    *,
    expected_session_ids: set[str],
) -> dict[str, Any]:
    if len(paths) > 1:
        raise TelemetryValidationError(f"expected at most one ATIF artifact, found {len(paths)}")
    if not paths:
        return {"files": [], "promoted": None}

    path = paths[0]
    if not path.is_file():
        raise TelemetryValidationError(f"ATIF artifact does not exist: {path}")

    from harbor.models.trajectories.trajectory import Trajectory

    text = path.read_text(encoding="utf-8")
    _reject_obvious_secrets(text, path)
    trajectory = Trajectory.model_validate_json(text)
    if trajectory.session_id and trajectory.session_id not in expected_session_ids:
        raise TelemetryValidationError(
            f"ATIF session_id does not correlate with the Fabric runtime or Harbor session: {trajectory.session_id}"
        )
    canonical = logs_dir / "trajectory.json"
    if path.resolve() != canonical.resolve():
        shutil.copyfile(path, canonical)
    return {
        "files": [str(path)],
        "promoted": str(canonical),
        "schema_version": trajectory.schema_version,
        "session_id": trajectory.session_id,
        "agent": trajectory.agent.name,
        "steps": len(trajectory.steps),
        "final_metrics": (
            trajectory.final_metrics.model_dump(mode="json", exclude_none=True)
            if trajectory.final_metrics is not None
            else None
        ),
    }


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
