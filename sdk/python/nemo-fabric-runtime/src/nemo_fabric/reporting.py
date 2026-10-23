# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inspect saved invocation evidence without a benchmark manifest or trace import."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import datetime
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
    stats = None
    stats_error = None
    if document.get("system_stats") is not None:
        try:
            stats = InvocationSystemStats.from_mapping(
                document["system_stats"]
            ).to_mapping()
            if any(
                observation.get("snapshot", {}).get(
                    "runtime_id", identity["runtime_id"]
                )
                != identity["runtime_id"]
                for observation in (stats["before"], stats["after"])
            ):
                raise ValueError("system stats belong to another runtime")
        except (FabricConfigError, TypeError, ValueError):
            stats = None
            stats_error = "invalid_system_stats"
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
                if ref.get("kind") == "run_result"
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
        window = record["resource_window"]
        if window is not None:
            if (
                summary.get("resource_snapshot") is not None
                and window["end"] is not None
                and window["end"] != summary["resource_snapshot"]
            ):
                # The returned full result remains authoritative. A conflicting
                # bounded observation cannot supply a trustworthy delta.
                window["cpu_counter_change_seconds"] = None
                window["delta_reason"] = "end_snapshot_differs_from_result"
            if summary.get("system_stats") is None:
                summary["resource_window"] = window
    except (OSError, ValueError, UnicodeError):
        summary["record_error"] = "invalid_or_unreadable_sibling_record"


def read_evidence(path: Path) -> dict[str, Any]:
    """Read a full result or invocation record; never follow references."""
    path = Path(path)
    if path.suffix == ".jsonl":
        return read_invocation_record(path)
    document = _load(path)
    summary = summarize_result(document)
    summary["result_path"] = str(path)
    _attach_invocation_interval(summary, path)
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


def _render_resource_window(window: Mapping[str, Any]) -> list[str]:
    lines = []
    if window["cpu_counter_change_seconds"] is not None:
        lines.append(
            f"Runtime process-tree CPU counter change: {window['cpu_counter_change_seconds']:.3f} s between snapshots (not invocation CPU)."
        )
    elif window["delta_reason"]:
        lines.append(
            "Runtime process-tree CPU counter change unavailable: "
            + window["delta_reason"]
            + "."
        )
    if window["start"] is not None and window["end"] is not None:
        lines.append(
            f"RSS observations: {format_bytes(window['start']['resident_memory_bytes'])} → {format_bytes(window['end']['resident_memory_bytes'])} (not a peak)."
        )
    return lines


def read_invocation_record(path: Path) -> dict[str, Any]:
    """Read bounded start/end evidence; a truncated tail never creates a result.

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
        "model_usage": normalized,
        "usage": usage,
        "accounting_error": accounting_error,
        "legacy_usage": end.get("usage") if end else None,
        "what_ran": None,
        "elapsed_ms": end["elapsed_ms"] if end else None,
        "exchange_elapsed_ms": end.get("exchange_elapsed_ms") if end else None,
        "resource_window": _resource_window(rows[0], end, identity["runtime_id"]),
        "outcome_boundary": "invocation",
        "truncated_tail": truncated,
        "evidence_path": str(path),
    }


def render_invocation_record_summary(
    summary: Mapping[str, Any], *, details: bool = False
) -> str:
    """Describe the invocation interval without assuming whole-runtime success."""
    if summary.get("usage") is not None:
        display = dict(summary)
        display["execution_interval"] = {"elapsed_ms": summary["elapsed_ms"]}
        if summary["exchange_elapsed_ms"] is not None:
            display["adapter_exchange"] = {"elapsed_ms": summary["exchange_elapsed_ms"]}
        lines = render_result_summary(display, details=details).splitlines()
        lines[0] = "Invocation outcome before runtime shutdown: " + summary["status"]
        lines.append(
            "The separate full result is authoritative for the one-shot outcome."
        )
        if summary["truncated_tail"]:
            lines.append(
                "A truncated final line was ignored; complete lines were retained."
            )
        lines.append(
            "Saved evidence: " + json.dumps(summary["evidence_path"], ensure_ascii=True)
        )
        return "\n".join(lines)
    lines = [
        "Invocation: "
        + json.dumps(
            f"{summary['runtime_id']}/{summary['invocation_id']}", ensure_ascii=True
        ),
        "Invocation outcome before runtime shutdown: "
        + (summary["status"] or "unavailable"),
    ]
    if summary.get("provenance_error"):
        lines.append(
            "Saved execution facts were unreadable; status and accounting are retained."
        )
    if summary.get("exchange_elapsed_ms") is not None:
        lines.append(
            f"Adapter exchange: {summary['exchange_elapsed_ms']:,} ms (excludes queue and result persistence)."
        )
    elif summary["elapsed_ms"] is not None:
        lines.append(f"Core invocation interval: {summary['elapsed_ms']:,} ms")
    else:
        lines.append(
            "No complete end record; elapsed time and final usage are unknown."
        )
    if details and summary.get("resource_window") is not None:
        lines.extend(_render_resource_window(summary["resource_window"]))
    if summary["truncated_tail"]:
        lines.append(
            "A truncated final line was ignored; complete lines were retained."
        )
    if summary.get("accounting_error"):
        lines.append(
            "Saved model accounting was invalid; execution evidence is retained."
        )
    lines.append("Inspect the full result for accounting and the one-shot outcome.")
    lines.append(
        "Saved evidence: " + json.dumps(summary["evidence_path"], ensure_ascii=True)
    )
    return "\n".join(lines)


def _render_tokens(dimensions: Mapping[str, Any]) -> list[str]:
    totals = ("input_tokens", "output_tokens", "total_tokens")
    caches = ("cached_input_tokens", "cache_write_input_tokens")

    def coverage(names: tuple[str, ...]) -> str:
        if all(dimensions[name]["complete_total"] is not None for name in names):
            return "complete"
        if any(
            dimensions[name]["observed_sum"] is not None
            or dimensions[name]["complete_total"] is not None
            for name in names
        ):
            return "incomplete"
        return "unavailable"

    total_coverage, cache_coverage = coverage(totals), coverage(caches)

    def count(name: str, label: str, group_coverage: str) -> str:
        field = dimensions[name]
        if field["complete_total"] is not None:
            suffix = "" if group_coverage == "complete" else " (complete)"
            return f"{label} {field['complete_total']:,}{suffix}"
        if field["observed_sum"] is not None:
            return f"{label} {field['observed_sum']:,} (observed, incomplete)"
        return f"{label} unavailable"

    return [
        "Tokens: "
        + "; ".join(
            count(name, label, total_coverage)
            for name, label in zip(totals, ("input", "output", "total"))
        ),
        "Cache (included in input): "
        + "; ".join(
            count(name, label, cache_coverage)
            for name, label in zip(caches, ("reads", "writes"))
        ),
        f"Token coverage: input/output/total {total_coverage}; cache breakdown {cache_coverage}",
    ]


def render_result_summary(summary: Mapping[str, Any], *, details: bool = False) -> str:
    """Render a concise overview; details add resource evidence.

    Token and cost completeness remain separate. JSON quantities and accounting
    semantics are unchanged by this presentation.
    """
    dimensions = summary["usage"]["dimensions"]
    lines = [f"Execution: {summary['status']}"]
    if details:
        lines.append(
            "Invocation: "
            + json.dumps(
                f"{summary['runtime_id']}/{summary['invocation_id']}", ensure_ascii=True
            )
        )
    if summary.get("harness"):
        lines.append("Harness: " + json.dumps(summary["harness"], ensure_ascii=True))
    if summary.get("execution_error"):
        error = summary["execution_error"]
        lines.append(
            "Execution error: "
            + json.dumps(
                {field: error[field] for field in ("stage", "code") if field in error},
                ensure_ascii=True,
            )
        )
    if details and summary.get("outcome_boundary"):
        lines.append(
            "Outcome boundary: "
            + json.dumps(summary["outcome_boundary"], ensure_ascii=True)
        )
    interval = summary.get("execution_interval")
    if summary.get("adapter_exchange") is not None:
        lines.append(
            f"Elapsed (adapter exchange): {format_duration_ms(summary['adapter_exchange']['elapsed_ms'])} (excludes queue and result persistence)."
        )
    elif summary.get("system_stats") is not None:
        lines.append(
            f"Elapsed (adapter exchange): {format_duration_ms(summary['system_stats']['elapsed_seconds'] * 1000)} (excludes queue and result persistence)."
        )
    elif interval is not None and interval["elapsed_ms"] is not None:
        lines.append(
            f"Core invocation interval before shutdown: {format_duration_ms(interval['elapsed_ms'])}"
        )
    else:
        lines.append("Elapsed: unavailable")
    if details and summary.get("resource_window") is not None:
        lines.extend(_render_resource_window(summary["resource_window"]))
    resource = summary.get("resource_snapshot")
    if details and resource is not None:
        lines.append(
            f"Runtime process-tree snapshot: CPU {resource['cpu_time_seconds']:.3f} s since host start; "
            f"RSS {format_bytes(resource['resident_memory_bytes'])}; live processes {resource['process_count']}"
        )
    normalized = summary.get("model_usage")
    models = normalized.get("models", []) if normalized is not None else []
    if models:
        shown = models if details else models[:3]
        lines.append(
            "Accounted models: "
            + ", ".join(
                json.dumps(
                    f"{model.get('provider') or 'unspecified'}/{model['model']}",
                    ensure_ascii=True,
                )
                for model in shown
            )
            + (
                f"; {len(models) - len(shown)} more (see --details)"
                if len(shown) < len(models)
                else ""
            )
        )
        if not normalized["models_exhaustive"]:
            lines.append("Per-model accounting is not exhaustive.")
    lines.extend(_render_tokens(dimensions))
    gaps = normalized.get("gaps", []) if normalized is not None else []
    not_reported = {
        dimension
        for gap in gaps
        if gap["reason"] == "missing_usage"
        for dimension in gap["dimensions"]
    }
    for dimension, label in (
        ("reported_cost", "Provider-reported cost"),
        ("estimated_cost", "Catalog estimate"),
    ):
        field = dimensions[dimension]
        if field["complete_total"] is not None:
            text = format_money(field["complete_total"]) + " (complete)"
        elif field["observed_sum"] is not None:
            text = format_money(field["observed_sum"]) + " (observed, incomplete)"
        else:
            text = "unavailable" + (
                " (not reported)" if dimension in not_reported else ""
            )
        lines.append(f"{label}: {text}")
    pricing = summary["usage"].get("pricing")
    if pricing:
        if details:
            lines.append(
                "Pricing basis: "
                + json.dumps(pricing, ensure_ascii=True, sort_keys=True)
            )
        else:
            catalogs = [
                f"{label} {pricing[key].removeprefix('sha256:')[:12]}"
                for key, label in (
                    ("bundled_catalog_sha256", "bundled"),
                    ("user_catalog_sha256", "user"),
                )
                if pricing.get(key)
            ]
            lines.append(
                "Pricing basis: catalog"
                + ("; " + "; ".join(catalogs) if catalogs else "")
                + " (full basis in --details or --json)."
            )
    if summary.get("recording_failed"):
        lines.append(
            "Recording failed; execution and available accounting are retained. "
            "Inspect record_write_failed events in the original result."
        )
    explanations: set[str] = set()
    missing: set[str] = set()
    for gap in gaps:
        if gap.get("code") == "cache_write_tariff_unverified":
            explanations.add("The applicable cache-write price was not established.")
        elif gap["reason"] == "unfinished_work":
            explanations.add(
                "Some invocation-owned model work was unfinished at return."
            )
        elif gap["reason"] == "scope_unverified":
            explanations.add(
                "The source did not establish complete invocation coverage."
            )
        else:
            labels = {
                "reported_cost": "provider-reported cost",
                "estimated_cost": "catalog estimate",
            }
            missing.update(
                labels.get(dimension, dimension.replace("_", " "))
                for dimension in gap["dimensions"]
                if dimension not in {"reported_cost", "estimated_cost"}
            )
    if summary.get("accounting_error"):
        explanations.add(
            "Saved model accounting was invalid; execution evidence is retained."
        )
    if missing:
        explanations.add("Not reported: " + ", ".join(sorted(missing)) + ".")
    for explanation in sorted(explanations):
        lines.append(explanation)
    if summary.get("legacy_usage") is not None and normalized is None:
        lines.append("Raw harness usage is retained in JSON as unverified diagnostics.")
    if summary.get("record_error"):
        lines.append(
            "The sibling invocation record could not be used; result accounting is retained."
        )
    if details and summary.get("system_stats") is not None:
        stats = summary["system_stats"]
        lines.append(
            "System stats collected groups: "
            + json.dumps(stats["collected_groups"], ensure_ascii=True)
        )
        sampled = stats.get("sampled")
        if sampled is not None:
            lines.append(
                f"Resource samples: {sampled['sample_count']:,}; dropped {sampled['dropped_samples']:,}; truncated {str(sampled['truncated']).lower()} (sampled peaks are not guaranteed true peaks)."
            )
    if summary.get("system_stats_error"):
        lines.append(
            "Saved system stats were invalid; execution and accounting are retained."
        )
    if summary.get("resource_error"):
        lines.append(
            "The saved resource snapshot is invalid; result accounting is retained."
        )
    for artifact in summary.get("artifacts", []):
        if artifact.get("kind") == "log" and isinstance(artifact.get("path"), str):
            lines.append(
                "Log artifact: " + json.dumps(artifact["path"], ensure_ascii=True)
            )
    for reference in summary.get("telemetry", []):
        if isinstance(reference, Mapping) and isinstance(reference.get("uri"), str):
            lines.append(
                "Telemetry reference: "
                + json.dumps(reference["uri"], ensure_ascii=True)
            )
    if summary.get("result_path"):
        lines.append(
            "Saved result: " + json.dumps(summary["result_path"], ensure_ascii=True)
        )
    lines.append("Execution status is not a task verdict. Estimates are not invoices.")
    return "\n".join(lines)


def _render(summary: Mapping[str, Any], *, details: bool = False) -> str:
    if summary["schema"] == "nemo.fabric.invocation_record_summary":
        return render_invocation_record_summary(summary, details=details)
    return render_result_summary(summary, details=details)


def main() -> None:
    """Inspect a saved invocation result as a concise summary or JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument(
        "--details", action="store_true", help="include resource and provenance details"
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
