# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inspect saved run journals and legacy results without importing traces."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

from nemo_fabric._report_format import format_bytes, format_duration_ms, format_money
from nemo_fabric.accounting import summarize_usage
from nemo_fabric.errors import FabricConfigError
from nemo_fabric.types import (
    FabricMapping,
    InvocationSystemStats,
    ResourceObservation,
    ModelUsage,
    RuntimeResourceSnapshot,
)

MAX_RESULT_BYTES = 16 * 1024 * 1024
MAX_HISTORY_EVENTS = 256
MAX_HISTORY_BYTES = 1024 * 1024
MAX_HISTORY_MILESTONE_RESERVE_BYTES = 64 * 1024
MAX_HISTORY_FIELD_CHARS = 256
_HISTORY_MILESTONES = frozenset(
    {
        "request_started",
        "runtime_ready",
        "invocation_started",
        "invocation_finished",
        "runtime_stopped",
        "summary",
    }
)
MAX_EVENT_BYTES = 64 * 1024
MAX_HISTORY_MESSAGE_CHARS = 4096


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value


def _invalid_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON number: {value}")


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non-finite JSON number")
    return number


def _load(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        data = stream.read(MAX_RESULT_BYTES + 1)
    if len(data) > MAX_RESULT_BYTES:
        raise ValueError(
            "saved evidence exceeds the summary reader's 16 MiB limit; "
            "the original file is unchanged and can be inspected separately"
        )
    try:
        document = json.loads(
            data, parse_constant=_invalid_constant, parse_float=_finite_float
        )
    except RecursionError as error:
        raise ValueError("saved evidence is nested too deeply") from error
    if not isinstance(document, dict):
        raise ValueError("saved evidence must be a JSON object")
    return document


def _validated_system_stats(
    value: Any, runtime_id: str | None
) -> tuple[dict[str, Any] | None, str | None]:
    """Keep independently usable resource evidence bound to its runtime."""
    if value is None:
        return None, None
    try:
        stats = InvocationSystemStats.from_mapping(value).to_mapping()
        if runtime_id is None or any(
            observation.get("snapshot", {}).get("runtime_id", runtime_id) != runtime_id
            for observation in (stats["before"], stats["after"])
        ):
            raise ValueError("system stats belong to another runtime")
    except (FabricConfigError, TypeError, ValueError):
        return None, "invalid_system_stats"
    return stats, None


def summarize_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Describe one returned invocation, preserving independent usable evidence.

    This is not a task verdict or proof of a complete scheduled run. Normalized
    accounting uses the shared reducer. Legacy usage remains labeled diagnostics;
    logs and Relay/ATIF totals never fill missing normalized dimensions.
    """
    document = (
        result.to_mapping() if isinstance(result, FabricMapping) else dict(result)
    )
    identity = {
        field: _text(document.get(field), field)
        for field in ("request_id", "runtime_id", "invocation_id")
    }
    status = document.get("status")
    if not isinstance(status, str) or status not in {
        "succeeded",
        "failed",
        "cancelled",
    }:
        raise ValueError("saved invocation has an invalid execution status")
    error = document.get("error")
    execution_error = (
        {
            field: error[field]
            for field in ("stage", "code", "retryable")
            if field in error
        }
        if isinstance(error, Mapping)
        else None
    )
    events = document.get("events", [])
    accounting_error = (
        "invalid_model_usage"
        if isinstance(events, list)
        and any(
            isinstance(event, Mapping) and event.get("kind") == "model_usage_invalid"
            for event in events
        )
        else None
    )
    normalized = None
    if document.get("model_usage") is not None:
        try:
            normalized = ModelUsage.from_mapping(document["model_usage"]).to_mapping()
        except (FabricConfigError, TypeError, ValueError):
            accounting_error = "invalid_model_usage"
            document.pop("model_usage", None)
    usage = summarize_usage(
        [document], expected=[(identity["runtime_id"], identity["invocation_id"])]
    )
    resource = None
    resource_error = None
    if document.get("resource_snapshot") is not None:
        try:
            resource = RuntimeResourceSnapshot.from_mapping(
                document["resource_snapshot"]
            ).to_mapping()
            if resource["runtime_id"] != identity["runtime_id"]:
                raise ValueError("resource snapshot belongs to another runtime")
        except (FabricConfigError, TypeError, ValueError):
            resource = None
            resource_error = "invalid_resource_snapshot"
    stats, stats_error = _validated_system_stats(
        document.get("system_stats"), identity["runtime_id"]
    )
    refs = document.get("artifacts")
    refs = refs.get("artifacts", []) if isinstance(refs, Mapping) else []
    artifacts = (
        [
            {
                key: ref[key]
                for key in ("name", "kind", "path", "media_type", "metadata")
                if key in ref
            }
            for ref in refs
            if isinstance(ref, Mapping)
        ]
        if isinstance(refs, list)
        else []
    )
    return {
        "schema": "nemo.fabric.invocation_summary",
        "version": "v1alpha1",
        **identity,
        "status": status,
        "recording_failed": any(
            isinstance(event, Mapping) and event.get("kind") == "record_write_failed"
            for event in document.get("events", [])
        )
        if isinstance(document.get("events", []), list)
        else False,
        "execution_error": execution_error,
        "harness": document.get("harness")
        if isinstance(document.get("harness"), str)
        else None,
        "outcome_boundary": next(
            (
                ref.get("metadata", {}).get("boundary")
                for ref in artifacts
                if ref.get("kind") in {"run_result", "run_log"}
                and isinstance(ref.get("metadata"), Mapping)
            ),
            None,
        ),
        "accounting_error": accounting_error,
        "model_usage": normalized,
        "usage": usage,
        "legacy_usage": document.get("usage"),
        "resource_snapshot": resource,
        "resource_error": resource_error,
        "system_stats": stats,
        "system_stats_error": stats_error,
        "resource_window": _resource_window(
            {"resource_scope": RESOURCE_SCOPE},
            {
                "resource_scope": RESOURCE_SCOPE,
                "resources_before": stats["before"],
                "resources": stats["after"],
            },
            identity["runtime_id"],
        )
        if stats is not None
        else None,
        "what_ran": None,
        "provenance_error": None,
        "artifacts": artifacts,
        "telemetry": document["telemetry"]
        if isinstance(document.get("telemetry"), list)
        else [],
    }


def read_result(path: Path) -> dict[str, Any]:
    """Read one saved full result without requiring task labels or a manifest."""
    path = Path(path)
    summary = summarize_result(_load(path))
    summary["result_path"] = str(path)
    _attach_invocation_interval(summary, path)
    return summary


def _attach_invocation_interval(summary: dict[str, Any], result_path: Path) -> None:
    # Only the core's fixed sibling name is eligible. Never follow a path from
    # artifact data, and never use the pre-shutdown status to replace the result.
    record_path = result_path.with_name("run-record.jsonl")
    try:
        if record_path.is_symlink() or not record_path.is_file():
            return
        record = read_invocation_record(record_path)
        if any(
            record[field] != summary[field]
            for field in ("request_id", "runtime_id", "invocation_id")
        ):
            raise ValueError("sibling invocation record identity differs")
        summary["execution_interval"] = {
            "scope": "core_invocation",
            "elapsed_ms": record["elapsed_ms"],
            "source": str(record_path),
            "truncated_tail": record["truncated_tail"],
        }
        if record["exchange_elapsed_ms"] is not None:
            summary["adapter_exchange"] = {
                "elapsed_ms": record["exchange_elapsed_ms"],
                "source": str(record_path),
            }
    except (OSError, ValueError, UnicodeError):
        summary["record_error"] = "invalid_or_unreadable_sibling_record"


def read_evidence(path: Path) -> dict[str, Any]:
    """Read a run journal or legacy result without opening referenced artifacts."""
    path = Path(path)
    if path.suffix == ".jsonl":
        return _read_journal(path)
    document = _load(path)
    summary = summarize_result(document)
    summary["result_path"] = str(path)
    _attach_invocation_interval(summary, path)
    return summary


def _utc_time(value: Any) -> datetime:
    text = _text(value, "time")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("invalid UTC timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError("timestamps must include UTC timezone")
    return parsed.astimezone(timezone.utc)


def _read_journal(path: Path) -> dict[str, Any]:
    """Read a unified run log or the original two-line invocation journal."""
    with path.open("rb") as stream:
        first = stream.readline(MAX_RESULT_BYTES + 1)
    if len(first) > MAX_RESULT_BYTES:
        raise ValueError("run record exceeds the reader's 16 MiB per-record limit")
    try:
        document = json.loads(
            first, parse_constant=_invalid_constant, parse_float=_finite_float
        )
    except (ValueError, RecursionError) as error:
        raise ValueError(
            f"run log must begin with a complete JSON record: {error}"
        ) from error
    if (
        isinstance(document, dict)
        and document.get("schema") == "nemo.fabric.run_record"
    ):
        return read_invocation_record(path)
    return _read_run_log(path)


def _journal_seconds(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError("elapsed seconds must be finite and nonnegative")
    try:
        finite = math.isfinite(value)
    except OverflowError as error:
        raise ValueError("elapsed seconds exceed the supported range") from error
    if not finite:
        raise ValueError("elapsed seconds must be finite and nonnegative")
    return value


def _journal_models(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("request models must be an object")
    for role, model in value.items():
        _text(role, "model role")
        if not isinstance(model, dict):
            raise ValueError("requested model must be an object")
        _text(model.get("model"), "model")
        _text(model.get("provider"), "provider")
    return value


def _journal_error(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("execution error must be an object or null")
    for field in ("stage", "code"):
        _text(value.get(field), "error " + field)
    return {
        field: value[field]
        for field in ("stage", "code", "retryable")
        if field in value
    }


def _history_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    """Retain bounded display fields, not arbitrary event payloads."""
    item = {key: row[key] for key in ("kind", "time", "elapsed_seconds", "source")}
    shortened = False

    def text(value: str, limit: int = MAX_HISTORY_FIELD_CHARS) -> str:
        nonlocal shortened
        shortened |= len(value) > limit
        return value[:limit]

    if len(item["time"]) > MAX_HISTORY_FIELD_CHARS:
        item["time"] = _utc_time(item["time"]).isoformat().replace("+00:00", "Z")
        shortened = True
    kind, data = row["kind"], row["data"]
    projected = {}
    keys = {
        "invocation_started": ("timeout_seconds",),
        "invocation_finished": ("status",),
        "runtime_stopped": ("succeeded",),
        "diagnostic": ("level", "severity", "stream", "scope"),
    }.get(kind, ())
    for key in keys:
        value = data.get(key)
        if isinstance(value, str):
            projected[key] = text(value)
        elif isinstance(value, bool):
            projected[key] = value
        elif isinstance(value, (int, float)):
            try:
                projected[key] = _journal_seconds(value)
            except ValueError:
                shortened = True
    if (
        kind in {"invocation_finished", "runtime_stopped"}
        and data.get("error") is not None
    ):
        error = _journal_error(data["error"])
        projected["error"] = {key: text(error[key]) for key in ("stage", "code")}
        if type(error.get("retryable")) is bool:
            projected["error"]["retryable"] = error["retryable"]
        elif "retryable" in error:
            shortened = True
    if kind == "diagnostic" and isinstance(data.get("message"), str):
        projected["message"] = text(data["message"], MAX_HISTORY_MESSAGE_CHARS)
        projected["message_truncated"] = (
            len(data["message"]) > MAX_HISTORY_MESSAGE_CHARS
        )
    item["data"] = projected
    if shortened:
        item["projection_truncated"] = True
    return item


def _read_run_log(path: Path) -> dict[str, Any]:
    """Stream journal events with bounded display history; retain terminal evidence."""
    first = None
    final = None
    invocation = None
    history = []
    history_bytes = 0
    history_truncated = 0
    omitted = 0
    truncated = False
    identities: dict[str, str] = {}
    previous_elapsed = 0.0
    milestones = set()
    diagnostics = {"omitted_bytes": 0, "read_errors": []}
    with path.open("rb") as stream:
        sequence = 0
        while line := stream.readline(MAX_RESULT_BYTES + 1):
            if len(line) > MAX_RESULT_BYTES:
                raise ValueError(
                    "run record exceeds the reader's 16 MiB per-record limit; the original file is unchanged"
                )
            if not line.endswith(b"\n"):
                truncated = True
                break
            try:
                row = json.loads(
                    line, parse_constant=_invalid_constant, parse_float=_finite_float
                )
            except (ValueError, RecursionError) as error:
                raise ValueError("invalid complete run log record") from error
            if not isinstance(row, dict) or (row.get("schema"), row.get("version")) != (
                "nemo.fabric.run",
                "v1alpha1",
            ):
                raise ValueError("unsupported run log schema or version")
            if final is not None:
                raise ValueError("run log contains records after its terminal summary")
            if type(row.get("sequence")) is not int or row["sequence"] != sequence:
                raise ValueError(
                    "run log sequence must be contiguous and start at zero"
                )
            sequence += 1
            for field in ("recording_id", "request_id"):
                value = _text(row.get(field), field)
                if field in identities and identities[field] != value:
                    raise ValueError("run log identity changed")
                identities[field] = value
            for field in ("runtime_id", "invocation_id"):
                if field in row:
                    value = _text(row[field], field)
                    if field in identities and identities[field] != value:
                        raise ValueError("run log identity changed")
                    identities[field] = value
            _utc_time(row.get("time"))
            elapsed = _journal_seconds(row.get("elapsed_seconds"))
            if elapsed < previous_elapsed:
                raise ValueError(
                    "run elapsed seconds must be finite, nonnegative and ordered"
                )
            previous_elapsed = elapsed
            if len(_text(row.get("source"), "source")) > 256:
                raise ValueError("run source exceeds 256 characters")
            kind = _text(row.get("kind"), "kind")
            if len(kind) > 128:
                raise ValueError("run kind exceeds 128 characters")
            if (
                kind in {"diagnostic", "resource_sample"}
                and len(line) > MAX_EVENT_BYTES
            ):
                raise ValueError("run event exceeds the reader's 64 KiB event limit")
            if kind in {"runtime_ready", "invocation_started", "runtime_stopped"}:
                if kind in milestones:
                    raise ValueError("duplicate lifecycle milestone")
                milestones.add(kind)
            data = row.get("data")
            if not isinstance(data, dict):
                raise ValueError("run record data must be an object")
            if first is None:
                if kind != "request_started":
                    raise ValueError("run log must begin with request_started")
                override_keys = data.get("override_keys", [])
                if not isinstance(override_keys, list) or any(
                    not isinstance(key, str) for key in override_keys
                ):
                    raise ValueError("override_keys must be a list of strings")
                _journal_models(data.get("models", {}))
                _text(data.get("harness"), "harness")
                if data.get("outcome_scope") not in {
                    "invocation",
                    "one_shot_lifecycle",
                }:
                    raise ValueError("invalid request outcome scope")
                first = row
            elif kind == "request_started":
                raise ValueError("duplicate request_started record")
            if kind == "runtime_ready" and data.get("runtime_id") != identities.get(
                "runtime_id"
            ):
                raise ValueError("runtime_ready identity differs from envelope")
            if kind == "invocation_finished":
                if invocation is not None:
                    raise ValueError("duplicate invocation_finished record")
                invocation = row
            elif kind == "summary":
                final = row
            if kind == "diagnostic_retention":
                omitted_bytes = data.get("omitted_bytes")
                read_error = data.get("read_error")
                if type(omitted_bytes) is not int or omitted_bytes < 0:
                    raise ValueError("diagnostic omitted_bytes must be nonnegative")
                if read_error is not None and not isinstance(read_error, str):
                    raise ValueError("diagnostic read_error must be text or null")
                diagnostics["omitted_bytes"] += omitted_bytes
                if read_error and len(diagnostics["read_errors"]) < 16:
                    diagnostics["read_errors"].append(
                        read_error[:MAX_HISTORY_MESSAGE_CHARS]
                    )
            if kind not in {"resource_sample", "diagnostic_retention"}:
                item = _history_projection(row)
                item_bytes = len(
                    json.dumps(item, ensure_ascii=True, separators=(",", ":")).encode(
                        "utf-8"
                    )
                )
                milestone = kind in _HISTORY_MILESTONES
                if milestone or (
                    len(history) < MAX_HISTORY_EVENTS
                    and history_bytes + item_bytes
                    <= MAX_HISTORY_BYTES - MAX_HISTORY_MILESTONE_RESERVE_BYTES
                ):
                    history.append(item)
                    history_bytes += item_bytes
                    history_truncated += item.get("projection_truncated", False)
                else:
                    omitted += 1
    if first is None:
        raise ValueError("run log has no complete request_started record")
    payload = final["data"] if final else {}
    result = payload.get("result")
    if result is not None:
        if not isinstance(result, dict):
            raise ValueError("summary result must be an object or null")
        summary = summarize_result(result)
        for field in ("request_id", "runtime_id", "invocation_id"):
            if field not in identities or summary[field] != identities[field]:
                raise ValueError("terminal result identity differs from run log")
    elif final is None and invocation is not None:
        evidence = invocation["data"]
        if not all(field in identities for field in ("runtime_id", "invocation_id")):
            raise ValueError(
                "invocation evidence has no runtime or invocation identity"
            )
        if any(
            field in evidence and evidence[field] != identities[field]
            for field in ("request_id", "runtime_id", "invocation_id")
        ):
            raise ValueError("invocation evidence identity differs from run log")
        summary = summarize_result({**evidence, **identities})
        summary.update(
            schema="nemo.fabric.invocation_record_summary",
            outcome_boundary="invocation",
            state="record_only",
            elapsed_ms=None,
            exchange_elapsed_ms=None,
        )
    else:
        summary = {
            "schema": "nemo.fabric.request_summary",
            "version": "v1alpha1",
            "request_id": identities["request_id"],
            "status": None,
            "state": "no_result" if final else "registered",
            "usage": None,
            "model_usage": None,
            "execution_error": payload.get("error"),
        }
    if final is not None and result is None and invocation is not None:
        evidence = invocation["data"]
        if not all(field in identities for field in ("runtime_id", "invocation_id")):
            raise ValueError(
                "invocation evidence has no runtime or invocation identity"
            )
        if any(
            field in evidence and evidence[field] != identities[field]
            for field in ("request_id", "runtime_id", "invocation_id")
        ):
            raise ValueError("invocation evidence identity differs from run log")
        retained = summarize_result({**evidence, **identities})
        retained.update(
            schema="nemo.fabric.invocation_record_summary",
            outcome_boundary="invocation",
        )
        summary["invocation_summary"] = retained
        for field in (
            "usage",
            "model_usage",
            "system_stats",
            "system_stats_error",
            "resource_window",
            "accounting_error",
        ):
            summary[field] = retained[field]
    if invocation is not None or final is not None:
        exchange = (
            payload.get("adapter_exchange_elapsed_seconds")
            if final is not None
            else invocation["data"].get("adapter_exchange_elapsed_seconds")
        )
        if exchange is not None:
            summary["adapter_exchange"] = {
                "elapsed_ms": _journal_seconds(exchange) * 1000
            }
    if final is not None:
        if payload.get("outcome_scope") not in {"invocation", "one_shot_lifecycle"}:
            raise ValueError("invalid final outcome scope")
        if payload["outcome_scope"] != first["data"]["outcome_scope"]:
            raise ValueError("final outcome scope differs from request")
        _utc_time(payload.get("request_started_at"))
        _utc_time(payload.get("request_finished_at"))
        duration = _journal_seconds(payload.get("elapsed_seconds"))
        error = _journal_error(payload.get("error"))
        if result is None and error is None:
            raise ValueError("final no-result summary must contain an execution error")
        summary.update(
            outcome_boundary=payload["outcome_scope"],
            started_at=payload["request_started_at"],
            finished_at=payload["request_finished_at"],
            request_elapsed_seconds=duration,
        )
        if payload.get("error") is not None:
            summary["execution_error"] = error
            summary["stage"] = error["stage"]
    else:
        summary["started_at"] = first["time"]
    if result is None and payload.get("system_stats") is not None:
        stats, error = _validated_system_stats(
            payload["system_stats"], identities.get("runtime_id")
        )
        summary["system_stats"] = stats
        summary["system_stats_error"] = error
        summary["resource_window"] = (
            _resource_window(
                {"resource_scope": RESOURCE_SCOPE},
                {
                    "resource_scope": RESOURCE_SCOPE,
                    "resources_before": stats["before"],
                    "resources": stats["after"],
                },
                identities["runtime_id"],
            )
            if stats is not None
            else None
        )
    for field in ("runtime_id", "invocation_id"):
        if field in identities:
            summary[field] = identities[field]
    summary.update(
        recording_id=identities["recording_id"],
        harness=summary.get("harness") or first["data"].get("harness"),
        requested_models=first["data"].get("models", {}),
        model_override_unrecorded="models" in first["data"].get("override_keys", []),
        history=history,
        diagnostics=diagnostics,
        history_projection_truncated_count=history_truncated,
        history_omitted_count=omitted,
        truncated_tail=truncated,
        terminal_summary_present=final is not None,
        evidence_path=str(path),
    )
    return summary


RESOURCE_SCOPE = "runtime_adapter_host_process_tree"


def _resource_observation(
    value: Any, runtime_id: str
) -> tuple[dict[str, Any] | None, str | None]:
    if value is None:
        return None, "missing_snapshot"
    if (
        isinstance(value, Mapping)
        and isinstance(value.get("error"), str)
        and value["error"]
        in {
            "resource_observation_unavailable",
            "resource_observation_unsupported",
        }
    ):
        return None, value["error"]
    try:
        if isinstance(value, Mapping) and "status" in value:
            observation = ResourceObservation.from_mapping(value)
            if observation.status != "available":
                return None, {
                    "not_collected": "not_collected",
                    "unavailable": "resource_observation_unavailable",
                    "unsupported": "resource_observation_unsupported",
                }[observation.status]
            value = observation.snapshot.to_mapping()
        snapshot = RuntimeResourceSnapshot.from_mapping(value).to_mapping()
        if snapshot["runtime_id"] != runtime_id:
            return None, "runtime_mismatch"
        for field in ("started_at", "observed_at"):
            time = datetime.fromisoformat(snapshot[field].replace("Z", "+00:00"))
            if time.tzinfo is None:
                raise ValueError("snapshot timestamps require timezone")
        cpu = snapshot["cpu_time_seconds"]
        if type(cpu) not in (int, float) or not math.isfinite(cpu) or cpu < 0:
            raise ValueError("invalid CPU counter")
        return snapshot, None
    except (FabricConfigError, TypeError, ValueError, KeyError, AttributeError):
        return None, "invalid_snapshot"


def _resource_window(
    start: Mapping[str, Any], end: Mapping[str, Any] | None, runtime_id: str
) -> dict[str, Any] | None:
    if "resources" not in start and (end is None or "resources" not in end):
        return None  # Older records have no observation pair.
    before_source = (
        end.get("resources_before")
        if end is not None and "resources_before" in end
        else start.get("resources")
    )
    before, reason = _resource_observation(before_source, runtime_id)
    after, end_reason = _resource_observation(
        end.get("resources") if end else None, runtime_id
    )
    window = {
        "scope": RESOURCE_SCOPE,
        "start": before,
        "end": after,
        "cpu_counter_change_seconds": None,
        "delta_reason": reason or end_reason,
    }
    scope = (
        end.get("resource_scope")
        if end is not None and "resources_before" in end
        else start.get("resource_scope")
    )
    if scope != RESOURCE_SCOPE or (
        end is not None and end.get("resource_scope") != RESOURCE_SCOPE
    ):
        window["delta_reason"] = "scope_unverified"
    elif before is not None and after is not None:

        def parse_time(value: str) -> datetime:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))

        if before["started_at"] != after["started_at"]:
            window["delta_reason"] = "host_changed"
        elif (
            not parse_time(before["started_at"])
            <= parse_time(before["observed_at"])
            <= parse_time(after["observed_at"])
        ):
            window["delta_reason"] = "time_reversed"
        elif after["cpu_time_seconds"] < before["cpu_time_seconds"]:
            window["delta_reason"] = "counter_decreased"
        else:
            window["cpu_counter_change_seconds"] = (
                after["cpu_time_seconds"] - before["cpu_time_seconds"]
            )
            window["delta_reason"] = None
    return window


def read_invocation_record(path: Path) -> dict[str, Any]:
    """Read legacy start/end evidence; a truncated tail never creates a result.

    This reports the core invocation window before one-shot shutdown. Only the
    typed model_usage projection supplies normalized accounting; legacy counters
    never gain a completeness guarantee.
    """
    path = Path(path)
    with path.open("rb") as stream:
        data = stream.read(MAX_RESULT_BYTES + 1)
    if len(data) > MAX_RESULT_BYTES:
        raise ValueError(
            "record exceeds the summary reader's 16 MiB limit; original file is unchanged"
        )
    rows = []
    truncated = False
    for line in data.splitlines(keepends=True):
        if not line.endswith(b"\n"):
            truncated = True
            break
        try:
            row = json.loads(
                line, parse_constant=_invalid_constant, parse_float=_finite_float
            )
        except (ValueError, RecursionError) as error:
            raise ValueError(
                f"invalid complete invocation record line: {error}"
            ) from error
        if not isinstance(row, dict) or (row.get("schema"), row.get("version")) != (
            "nemo.fabric.run_record",
            "v1alpha1",
        ):
            raise ValueError("unsupported invocation record schema or version")
        rows.append(row)
    if not rows or rows[0].get("kind") != "start":
        raise ValueError("invocation record must begin with a complete start line")
    if len(rows) > 2 or (len(rows) == 2 and rows[1].get("kind") != "end"):
        raise ValueError("conflicting or duplicate invocation record lines")
    identity = {
        field: _text(rows[0].get(field), field)
        for field in ("request_id", "runtime_id", "invocation_id")
    }
    end = rows[1] if len(rows) == 2 else None
    if end is not None:
        if any(end.get(field) != value for field, value in identity.items()):
            raise ValueError("invocation record identity changed")
        if not isinstance(end.get("status"), str) or end["status"] not in {
            "succeeded",
            "failed",
            "cancelled",
        }:
            raise ValueError("invalid invocation record status")
        elapsed = end.get("elapsed_ms")
        if type(elapsed) is not int or not 0 <= elapsed < 1 << 64:
            raise ValueError("invalid invocation elapsed time")
        if end.get("outcome_scope") != "invocation":
            raise ValueError("unsupported invocation outcome scope")
        exchange = end.get("exchange_elapsed_ms")
        if exchange is not None and (
            type(exchange) is not int or not 0 <= exchange < 1 << 64
        ):
            raise ValueError("invalid adapter exchange elapsed time")
    normalized = None
    accounting_error = None
    usage = None
    if end is not None and end.get("model_usage") is not None:
        try:
            normalized = ModelUsage.from_mapping(end["model_usage"]).to_mapping()
            usage = summarize_usage(
                [{**identity, "model_usage": normalized}],
                expected=[(identity["runtime_id"], identity["invocation_id"])],
            )
        except (FabricConfigError, TypeError, ValueError):
            normalized = None
            accounting_error = "invalid_model_usage"
    return {
        "schema": "nemo.fabric.invocation_record_summary",
        "version": "v1alpha1",
        **identity,
        "status": end["status"] if end else None,
        "execution_error": end.get("error") if end else None,
        "harness": rows[0].get("harness"),
        "started_at": rows[0].get("time"),
        "finished_at": end.get("time") if end else None,
        "model_usage": normalized,
        "usage": usage,
        "accounting_error": accounting_error,
        "legacy_usage": end.get("usage") if end else None,
        "what_ran": None,
        "elapsed_ms": end["elapsed_ms"] if end else None,
        "exchange_elapsed_ms": end.get("exchange_elapsed_ms") if end else None,
        "resource_window": None,
        "outcome_boundary": "invocation",
        "truncated_tail": truncated,
        "evidence_path": str(path),
    }


def render_invocation_record_summary(
    summary: Mapping[str, Any], *, details: bool = False
) -> str:
    """Render legacy invocation evidence without claiming one-shot completion."""
    display = dict(summary)
    if summary.get("elapsed_ms") is not None:
        display["execution_interval"] = {"elapsed_ms": summary["elapsed_ms"]}
    if summary.get("exchange_elapsed_ms") is not None:
        display["adapter_exchange"] = {"elapsed_ms": summary["exchange_elapsed_ms"]}
    return render_result_summary(display, details=details) + (
        "\nThis record covers the agent call; it does not establish the one-shot shutdown outcome."
    )


def _safe(value: Any) -> str:
    """Escape control characters without adding JSON quotes around plain text."""
    return json.dumps(str(value), ensure_ascii=True)[1:-1]


def _rows(items: list[tuple[str, str]]) -> list[str]:
    width = max(len(label) for label, _ in items)
    return [f"{label:<{width}}  {value}" for label, value in items]


def _dimension_value(
    field: Mapping[str, Any], *, money: bool = False, estimate: bool = False
) -> str:
    complete = field.get("complete_total")
    observed = field.get("observed_sum")
    value = complete if complete is not None else observed
    if value is None:
        return "Unavailable"
    rendered = format_money(value) if money else f"{value:,}"
    if complete is None:
        rendered += " recorded; " + (
            "full estimate unknown" if estimate else "total unknown"
        )
    return rendered


def _time_rows(summary: Mapping[str, Any]) -> list[tuple[str, str]]:
    start_value, end_value = summary.get("started_at"), summary.get("finished_at")
    if not start_value:
        return []
    start = _utc_time(start_value)
    end = _utc_time(end_value) if end_value else None
    same_day = end is None or start.date() == end.date()
    fmt = "%H:%M:%S.%f" if same_day else "%Y-%m-%d %H:%M:%S.%f"
    rows = [("Date:", start.date().isoformat())] if same_day else []
    rows.extend(
        [
            ("Timezone:", "UTC"),
            ("Started:", start.strftime(fmt)[:-3]),
            ("Finished:", end.strftime(fmt)[:-3] if end else "Unavailable"),
        ]
    )
    return rows


def _models(summary: Mapping[str, Any]) -> tuple[str, str]:
    facts = summary.get("what_ran") or {}
    roles = facts.get("models", {})
    requested = summary.get("requested_models", {})
    names = set(roles) | set(requested)
    if "default" in names:
        role = "default"
    elif len(names) == 1:
        role = next(iter(names))
    elif names:
        return f"Multiple roles ({len(names)})", "See details"
    else:
        return "Unavailable", "Not reported"
    model_facts = roles.get(role, {})
    name = model_facts.get("requested", {}).get("value")
    served = model_facts.get("served", {}).get("value")
    if name is None and summary.get("model_override_unrecorded"):
        requested_text = "Unavailable (model override not retained)"
    else:
        name = name or requested.get(role, {}).get("model")
        requested_text = _safe(name) if name else "Unavailable"
    return requested_text, _safe(served) if served else "Not reported"


def _resource_rows(summary: Mapping[str, Any]) -> list[tuple[str, str]]:
    stats = summary.get("system_stats") or {}
    cpu = stats.get("cpu_time_seconds_estimate")
    after = stats.get("after", {})
    memory = (
        after.get("snapshot", {}).get("resident_memory_bytes")
        if after.get("status") == "available"
        else None
    )
    if not stats and summary.get("resource_snapshot") is not None:
        memory = summary["resource_snapshot"]["resident_memory_bytes"]
    sampled = stats.get("sampled")
    peak = (sampled or {}).get("resident_memory_bytes") or {}
    return [
        ("CPU time, estimated", f"{cpu:.3f} s" if cpu is not None else "Unavailable"),
        (
            "Memory after execution",
            format_bytes(memory) if memory is not None else "Unavailable",
        ),
        (
            "Peak memory, sampled",
            format_bytes(peak["sampled_peak"])
            if peak.get("sampled_peak") is not None
            else "Unavailable"
            + (" (sampling disabled)" if stats and sampled is None else ""),
        ),
    ]


def _gap_lines(summary: Mapping[str, Any]) -> list[str]:
    explanations = set()
    labels = {
        "input_tokens": "input tokens",
        "cached_input_tokens": "cache read tokens",
        "cache_write_input_tokens": "cache write tokens",
        "output_tokens": "output tokens",
        "total_tokens": "total tokens",
        "estimated_cost": "estimated cost",
        "reported_cost": "provider-reported cost",
    }
    messages = {
        "unfinished_work": "some model work was unfinished at return",
        "scope_unverified": "the source could not establish coverage of all model work",
        "missing_usage": "the source did not report usable usage",
    }
    for gap in (summary.get("model_usage") or {}).get("gaps", []):
        dimensions = gap["dimensions"]
        # An absent provider cost is routine and already represented by omission.
        if dimensions == ["reported_cost"]:
            continue
        names = ", ".join(labels.get(name, name) for name in dimensions)
        reason = (
            "the applicable cache-write price was not established"
            if gap.get("code") == "cache_write_tariff_unverified"
            else messages.get(gap["reason"], gap["reason"].replace("_", " "))
        )
        if dimensions == ["estimated_cost"] and gap["reason"] != "unfinished_work":
            reason = "the estimate could not be established from available usage and pricing evidence"
        explanations.add(f"Accounting gap: {names}: {reason}.")
    for key, message in (
        (
            "accounting_error",
            "Saved model accounting is invalid; execution evidence is retained.",
        ),
        (
            "system_stats_error",
            "Saved resource measurements are invalid; execution and accounting are retained.",
        ),
        (
            "resource_error",
            "Saved resource snapshot is invalid; accounting is retained.",
        ),
        (
            "record_error",
            "The sibling invocation record could not be used; result accounting is retained.",
        ),
        (
            "provenance_error",
            "Saved execution facts are unreadable; measurements are retained.",
        ),
    ):
        if summary.get(key):
            explanations.add(message)
    return sorted(explanations)


def _history_lines(summary: Mapping[str, Any]) -> list[str]:
    history = summary.get("history", [])
    if not history:
        return []
    labels = {
        "request_started": "Request recording started",
        "runtime_ready": "Agent runtime ready",
        "invocation_started": "Invocation started",
        "invocation_finished": "Invocation finished",
        "runtime_stopped": "Agent runtime stopped",
        "summary": "Final summary",
    }
    lines = ["Execution history"]
    day = None
    for event in history:
        kind, data = event["kind"], event["data"]
        if kind == "summary":
            continue
        time = _utc_time(event["time"])
        if time.date() != day:
            day = time.date()
            lines.extend(
                [
                    f"Date: {day.isoformat()} · Timezone: UTC",
                    "Time          Elapsed (s)  Event",
                ]
            )
        label = labels.get(kind, _safe(kind.replace("_", " ")))
        if kind == "invocation_finished" and data.get("status"):
            label += ": " + _safe(data["status"])
        elif kind == "diagnostic":
            label = _safe(event["source"]) + ": " + _safe(data.get("message", ""))
            if data.get("message_truncated"):
                label += " (message shortened; full event retained in file)"
        if kind == "runtime_stopped" and data.get("succeeded") is False:
            label = "Agent runtime shutdown failed"
        if data.get("error"):
            label += (
                ": "
                + _safe(data["error"]["stage"])
                + "/"
                + _safe(data["error"]["code"])
            )
        if event.get("projection_truncated"):
            label += " (history details shortened; full event retained in file)"
        lines.append(
            f"{time.strftime('%H:%M:%S.%f')[:-3]}  {event['elapsed_seconds']:11.3f}  {label}"
        )
    if summary.get("history_omitted_count"):
        lines.append(
            f"Additional history entries retained in file: {summary['history_omitted_count']:,}"
        )
    return lines


def _measurement_text(field: Mapping[str, Any], unit: str) -> str:
    status = field["status"]
    if status == "available":
        value = field["value"]
        if unit == "bytes":
            return format_bytes(value)
        if unit == "microseconds":
            return f"{value / 1_000_000:.3f} s"
        if unit == "cores":
            return f"{value:g} cores"
        return f"{value:,}"
    if status == "unlimited":
        return "Unlimited"
    if status == "not_collected":
        return "Not collected"
    reason = field.get("reason")
    return "Unavailable" + (" (" + _safe(reason) + ")" if reason else "")


def _cgroup_lines(cgroup: Mapping[str, Any]) -> list[str]:
    lines = ["Cgroup measurements (may include other workloads)"]
    lines.extend(
        _rows(
            [
                (label, _measurement_text(cgroup[key], unit))
                for key, label, unit in (
                    ("cpu_usage_microseconds", "CPU time", "microseconds"),
                    (
                        "cpu_throttled_microseconds",
                        "CPU throttled time",
                        "microseconds",
                    ),
                    ("oom_kills", "OOM kills", "count"),
                )
            ]
        )
    )
    after = cgroup["after"]
    if after["status"] != "available":
        lines.append("Cgroup endpoint: " + _measurement_text(after, "count"))
        return lines
    snapshot = after["snapshot"]
    lines.extend(
        _rows(
            [
                (label, _safe(snapshot[key]))
                for key, label in (
                    ("source", "Source"),
                    ("scope", "Scope"),
                    ("target", "Target"),
                )
            ]
        )
    )
    lines.extend(
        _rows(
            [
                (label, _measurement_text(snapshot[key], unit))
                for key, label, unit in (
                    ("visible_cpu_limit_cores", "Visible CPU limit", "cores"),
                    ("visible_memory_limit_bytes", "Visible memory limit", "bytes"),
                    ("memory_current_bytes", "Memory after execution", "bytes"),
                    ("memory_peak_bytes", "Lifetime peak memory", "bytes"),
                    ("swap_current_bytes", "Swap after execution", "bytes"),
                )
            ]
        )
    )
    lines.extend(
        [
            "Limits cover visible ancestors; hidden bounds and exclusive capacity are unknown.",
            "Cgroup peak covers its lifetime or last reset, not this request.",
        ]
    )
    if snapshot["pressure_stall_microseconds"]:
        lines.append("Cumulative pressure-stall counters (not request deltas)")
        lines.extend(
            _rows(
                [
                    (
                        _safe(name.replace("_", " ")),
                        _measurement_text(field, "microseconds"),
                    )
                    for name, field in snapshot["pressure_stall_microseconds"].items()
                ]
            )
        )
    return lines


def _details_lines(summary: Mapping[str, Any]) -> list[str]:
    sections = []
    history = _history_lines(summary)
    if history:
        sections.append(history)
    stats = summary.get("system_stats")
    if stats:
        lines = ["Resource details"]
        if "process" in stats["collected_groups"]:
            lines.extend(
                [
                    "Measured processes: adapter host and its child processes",
                    "Endpoint memory is not a peak or time average.",
                    "Remote model inference is not measured.",
                    "Concurrent work can contribute to CPU time; exited processes can cause undercounting.",
                ]
            )
        else:
            lines.append("Process measurements: Not collected")
        sampled = stats.get("sampled")
        if sampled:
            memory = sampled.get("resident_memory_bytes")
            lines.extend(
                _rows(
                    [
                        (
                            "Requested sample period",
                            format_duration_ms(sampled["target_interval_ms"]),
                        ),
                        ("Samples retained", f"{sampled['sample_count']:,}"),
                        ("Samples dropped", f"{sampled['dropped_samples']:,}"),
                        (
                            "Largest sample gap",
                            f"{sampled['largest_gap_seconds']:.3f} s",
                        ),
                    ]
                )
            )
            if memory:
                mean = memory.get("time_weighted_mean")
                lines.extend(
                    _rows(
                        [
                            (
                                "Sampled memory mean",
                                format_bytes(mean)
                                if mean is not None
                                else "Unavailable",
                            ),
                            (
                                "Time covered by mean",
                                f"{memory['covered_fraction']:.1%}",
                            ),
                            ("Memory readings used", f"{memory['valid_samples']:,}"),
                            (
                                "Time included in mean",
                                f"{memory['covered_seconds']:.3f} s",
                            ),
                        ]
                    )
                )
            lines.append("Short memory spikes between samples can be missed.")
            lines.append(
                "The mean excludes missing readings and gaps longer than twice the requested sample period."
            )
            if sampled["truncated"]:
                lines.append("Sampling retention was truncated.")
            lines.extend(
                "Sampling diagnostic: " + _safe(value)
                for value in sampled["diagnostics"]
            )
        cgroup = stats.get("cgroup")
        if cgroup:
            lines.extend(_cgroup_lines(cgroup))
        sections.append(lines)
    elif summary.get("resource_snapshot") is not None:
        snapshot = summary["resource_snapshot"]
        sections.append(
            [
                "Runtime snapshot",
                "Observed at: " + _safe(snapshot["observed_at"]),
                "CPU counter since host start: "
                + f"{snapshot['cpu_time_seconds']:.3f} s",
                "Resident memory: " + format_bytes(snapshot["resident_memory_bytes"]),
                "This snapshot does not establish invocation resource consumption.",
            ]
        )
    facts = summary.get("what_ran") or {}
    roles = facts.get("models", {})
    configured = summary.get("requested_models", {})
    if len(set(roles) | set(configured)) > 1:
        lines = ["Model roles"]
        for role in sorted(set(roles) | set(configured)):
            facts_for_role = roles.get(role, {})
            requested = facts_for_role.get("requested", {}).get("value")
            if requested is None and not summary.get("model_override_unrecorded"):
                requested = configured.get(role, {}).get("model")
            served = facts_for_role.get("served", {}).get("value")
            lines.append(
                _safe(role)
                + ": requested "
                + (_safe(requested) if requested else "Unavailable")
                + "; served "
                + (_safe(served) if served else "Not reported")
            )
        sections.append(lines)
    normalized = summary.get("model_usage") or {}
    models = normalized.get("models", [])
    if models:
        lines = ["Per-model accounting"]
        for model in models:
            lines.append(
                _safe((model.get("provider") or "unspecified") + "/" + model["model"])
            )
            lines.extend(
                _rows(
                    [
                        (
                            label,
                            _dimension_value(
                                {
                                    "complete_total": model.get(key),
                                    "observed_sum": model.get("observed", {}).get(key),
                                }
                            ),
                        )
                        for key, label in (
                            ("input_tokens", "Input tokens"),
                            ("output_tokens", "Output tokens"),
                            ("total_tokens", "Total tokens"),
                        )
                    ]
                )
            )
        if not normalized.get("models_exhaustive"):
            lines.append("This model breakdown does not cover all invocation usage.")
        sections.append(lines)
    pricing = (summary.get("usage") or {}).get("pricing")
    if pricing:
        lines = [
            "Pricing",
            "Price source: "
            + (
                "Fabric packaged price table"
                if not pricing.get("user_catalog_sha256")
                else "Fabric and user price tables"
            ),
        ]
        for key, label in (
            ("bundled_catalog_sha256", "Packaged table fingerprint"),
            ("user_catalog_sha256", "User table fingerprint"),
        ):
            if pricing.get(key):
                lines.append(label + ": " + _safe(pricing[key]))
        lines.extend(
            [
                "Input includes cache; output includes reasoning.",
                "The estimate requires known token counts and applicable prices.",
            ]
        )
        applied = list(normalized.get("cost", {}).get("applied_rates", []))
        for model in models:
            applied.extend(model.get("cost", {}).get("applied_rates", []))
        seen = set()
        for rate in applied:
            key = json.dumps(rate, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            lines.extend(
                [
                    "Rates for "
                    + _safe(rate["provider"] + "/" + rate["model_id"])
                    + " (USD per 1,000,000 tokens)",
                    "  Pricing date: " + _safe(rate["pricing_as_of"]),
                    "  Source: " + _safe(rate["pricing_source"]),
                ]
            )
            for name, label in (
                ("input", "Uncached input"),
                ("cache_read", "Cache reads"),
                ("cache_write", "Cache writes"),
                ("output", "Output"),
            ):
                value = rate["rates"].get(name)
                if value is not None:
                    lines.append("  " + label + ": " + format_money(value))
        sections.append(lines)
    refs = []
    for artifact in summary.get("artifacts", []):
        if artifact.get("kind") in {"run_log", "run_record", "run_result"}:
            continue
        if isinstance(artifact.get("path"), str):
            name = {
                "stdout.txt": "Adapter output",
                "stderr.txt": "Adapter diagnostics",
            }.get(
                Path(artifact["path"]).name,
                artifact.get("name") or artifact.get("kind") or "Artifact",
            )
            refs.append((_safe(name), _safe(artifact["path"])))
    for reference in summary.get("telemetry", []):
        if isinstance(reference, Mapping) and isinstance(reference.get("uri"), str):
            refs.append(("Trace reference", _safe(reference["uri"])))
    if refs:
        sections.append(["Related artifacts", *_rows(refs)])
    return [line for section in sections for line in ["", *section]]


def render_result_summary(summary: Mapping[str, Any], *, details: bool = False) -> str:
    """Render concise measurement values; details add retained history and sources."""
    requested, served = _models(summary)
    status = summary.get("status")
    execution = (
        status.capitalize()
        if status
        else (
            "No result returned"
            if summary.get("state") == "no_result"
            else "Completion unconfirmed"
        )
    )
    if summary.get("terminal_summary_present") is False:
        execution = "Completion unconfirmed"
    lines = _rows(
        [
            (
                "Harness:",
                _safe(summary["harness"]) if summary.get("harness") else "Unavailable",
            ),
            ("Requested model:", requested),
            ("Served model:", served),
            ("Execution:", execution),
            *_time_rows(summary),
        ]
    )
    error = summary.get("execution_error")
    if isinstance(error, Mapping):
        lines.append(
            "Execution error: "
            + "; ".join(
                _safe(error[key])
                for key in ("stage", "code")
                if error.get(key) is not None
            )
        )
    stats = summary.get("system_stats") or {}
    exchange = summary.get("adapter_exchange") or {}
    execution_ms = exchange.get("elapsed_ms")
    if execution_ms is None and "elapsed_seconds" in stats:
        execution_ms = stats["elapsed_seconds"] * 1000
    request_seconds = summary.get("request_elapsed_seconds")
    measurements = [
        (
            "Request elapsed",
            f"{request_seconds:.3f} s"
            if request_seconds is not None
            else "Unavailable",
        ),
        (
            "Request execution time",
            format_duration_ms(execution_ms)
            if execution_ms is not None
            else "Unavailable",
        ),
        *_resource_rows(summary),
    ]
    dimensions = (summary.get("usage") or {}).get("dimensions", {})
    for key, label in (
        ("input_tokens", "Input tokens"),
        ("output_tokens", "Output tokens"),
        ("total_tokens", "Total tokens"),
        ("cached_input_tokens", "Cache read tokens"),
        ("cache_write_input_tokens", "Cache write tokens"),
        ("estimated_cost", "Estimated cost, USD"),
    ):
        measurements.append(
            (
                label,
                _dimension_value(
                    dimensions.get(key, {}),
                    money=key == "estimated_cost",
                    estimate=key == "estimated_cost",
                ),
            )
        )
    reported = dimensions.get("reported_cost", {})
    if (
        reported.get("complete_total") is not None
        or reported.get("observed_sum") is not None
    ):
        measurements.append(
            ("Provider-reported cost, USD", _dimension_value(reported, money=True))
        )
    lines.extend(
        [
            "",
            *_rows([("Measurement", "Value"), *measurements]),
            "",
            "Cache tokens are included in input.",
        ]
    )
    lines.extend(_gap_lines(summary))
    diagnostics = summary.get("diagnostics") or {}
    if diagnostics.get("omitted_bytes"):
        lines.append(
            f"Diagnostic capture omitted {diagnostics['omitted_bytes']:,} bytes."
        )
    for read_error in diagnostics.get("read_errors", []):
        lines.append("Diagnostic capture error: " + _safe(read_error))
    if summary.get("recording_failed"):
        lines.append(
            "Recording failed; execution and available accounting are retained. Inspect record_write_failed events."
        )
    if summary.get("legacy_usage") is not None and summary.get("model_usage") is None:
        lines.append("Raw harness usage is retained in JSON as unverified diagnostics.")
    if summary.get("truncated_tail"):
        lines.append(
            "An unfinished final line was ignored; earlier complete records were retained."
        )
    if summary.get("terminal_summary_present") is False:
        lines.append(
            "No final summary was saved; retained events do not establish the request outcome."
        )
    if summary.get("invocation_summary") is not None:
        lines.append(
            "Usage and resources describe the recorded agent call; no final result was returned."
        )
    if details:
        lines.extend(_details_lines(summary))
        interval = summary.get("execution_interval")
        if interval is not None and interval.get("elapsed_ms") is not None:
            lines.append(
                "Core invocation interval: "
                + format_duration_ms(interval["elapsed_ms"])
            )
        boundary = summary.get("outcome_boundary")
        if boundary in {"one_shot_lifecycle", "invocation"}:
            label = (
                "request through runtime shutdown"
                if boundary == "one_shot_lifecycle"
                else "this invocation; later shutdown excluded"
            )
            lines.extend(["", "Outcome covers: " + label])
        if summary.get("runtime_id") and summary.get("invocation_id"):
            lines.append(
                "Invocation: "
                + _safe(summary["runtime_id"] + "/" + summary["invocation_id"])
            )
    path = summary.get("evidence_path") or summary.get("result_path")
    if path:
        lines.extend(["", "Run file: " + _safe(path)])
    return "\n".join(lines)


def _render(summary: Mapping[str, Any], *, details: bool = False) -> str:
    if summary.get("recording_id") is not None:
        return render_result_summary(summary, details=details)
    if summary["schema"] == "nemo.fabric.invocation_record_summary":
        return render_invocation_record_summary(summary, details=details)
    return render_result_summary(summary, details=details)


def main() -> None:
    """Inspect a saved run journal or legacy result as readable text or JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument(
        "--details", action="store_true", help="include resource and pricing details"
    )
    args = parser.parse_args()
    try:
        summary = read_evidence(args.result)
    except (OSError, ValueError, UnicodeError) as error:
        parser.error(str(error))
    print(
        json.dumps(summary, indent=2, allow_nan=False)
        if args.as_json
        else _render(summary, details=args.details)
    )


if __name__ == "__main__":
    main()
