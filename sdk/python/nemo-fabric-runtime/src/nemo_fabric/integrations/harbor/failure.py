# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recover only bounded, runtime-owned observations after a runner failure."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from nemo_fabric import FabricError
from nemo_fabric.integrations.harbor.models import (
    FabricRunPayload,
    _validated_failure_evidence,
)


def failure_evidence(error: BaseException, payload: FabricRunPayload | None) -> dict:
    retained = (
        error.details.get("runtime_evidence")
        if isinstance(error, FabricError)
        else None
    )
    if isinstance(error, asyncio.CancelledError):
        retained = getattr(error, "_fabric_runtime_evidence", None)
    evidence = dict(retained) if isinstance(retained, dict) else {}
    partial_result = getattr(error, "_fabric_partial_result", None)
    if partial_result is not None:
        evidence["partial_result"] = partial_result
    # Do not forward arbitrary error details, environment bindings, or raw logs.
    allowed = {
        "runtime_id",
        "invocation_id",
        "request_id",
        "cleanup",
        "partial_result",
        "execution",
        "system_stats",
        "resource_snapshot",
        "usage",
        "artifacts",
        "telemetry",
        "events",
        "coverage",
    }
    evidence = {key: value for key, value in evidence.items() if key in allowed}
    evidence["exception_type"] = type(error).__name__
    cause = error.__cause__
    causes = []
    while cause is not None and len(causes) < 8:
        causes.append(type(cause).__name__)
        cause = cause.__cause__
    evidence["cause_types"] = causes
    evidence = _validated_failure_evidence(evidence).model_dump(mode="json")
    coverage = evidence.setdefault("coverage", {})
    if not isinstance(coverage, dict):
        coverage = evidence["coverage"] = {}
    if payload is not None:
        if evidence.get("request_id") is None:
            evidence["request_id"] = payload.request.request_id
        if payload.config.system_stats is None:
            coverage["system_stats"] = "not_requested"
        if payload.config.telemetry is None or not payload.config.telemetry.providers:
            coverage["telemetry"] = "not_requested"
        _retained_files(evidence, payload)
    # Use the same independent validation and bounds as the consumer.
    return _validated_failure_evidence(evidence).model_dump(mode="json")


def _retained_files(evidence: dict, payload: FabricRunPayload) -> None:
    runtime_id = evidence.get("runtime_id")
    if (
        not isinstance(runtime_id, str)
        or not runtime_id
        or Path(runtime_id).name != runtime_id
        or runtime_id in (".", "..")
    ):
        return
    root = payload.config.runtime.artifacts
    if root is None:
        return  # No speculative scan of a shared temporary directory.
    root = Path(root)
    if not root.is_absolute():
        root = Path(payload.config_base_dir) / root
    home = root / ".fabric" / runtime_id
    if (
        root.joinpath(".fabric").is_symlink()
        or home.is_symlink()
        or not home.resolve().is_relative_to(root.resolve())
    ):
        return
    try:
        with os.scandir(home) as entries:
            candidates = []
            for index, entry in enumerate(entries):
                if index == 4:
                    evidence["coverage"]["artifacts"] = "truncated"
                    break
                if entry.is_dir(follow_symlinks=False):
                    candidate = Path(entry.path)
                    marker = candidate / "adapter-invocation.json"
                    if not marker.is_symlink() and marker.is_file():
                        candidates.append(candidate)
        # This is a one-shot runner; ambiguity must not select another turn.
        if len(candidates) != 1:
            return
        invocation_home = candidates[0]
        evidence["invocation_id"] = invocation_home.name
        references = evidence.setdefault("artifacts", [])
        for name, kind in (
            ("adapter-invocation.json", "fabric_invocation"),
            ("system-stats-summary.json", "system_stats_summary"),
            ("system-stats.jsonl", "system_stats_samples"),
        ):
            path = invocation_home / name
            if path.is_symlink() or not path.is_file():
                continue
            references.append(
                {
                    "name": kind,
                    "kind": kind,
                    "path": str(path),
                    "media_type": "application/json"
                    if name.endswith(".json")
                    else "application/x-ndjson",
                }
            )
            if kind == "system_stats_summary":
                try:
                    # Never load a time series or a raw invocation payload.
                    with path.open("rb") as stream:
                        raw = stream.read(262145)
                    if len(raw) > 262144:
                        evidence["coverage"]["system_stats"] = "truncated"
                        continue
                    summary = json.loads(raw)
                    if (
                        summary["schema_version"] != "fabric.system_stats/v1alpha1"
                        or summary["runtime_id"] != runtime_id
                        or summary["invocation_id"] != invocation_home.name
                        or summary["request_id"] != evidence.get("request_id")
                    ):
                        raise ValueError("summary identity mismatch")
                    evidence["system_stats"] = summary["system_stats"]
                except (OSError, ValueError, KeyError, TypeError):
                    evidence["coverage"]["system_stats"] = "invalid"
    except OSError:
        return
