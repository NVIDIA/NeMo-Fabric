# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare caller-defined task attempts using saved Fabric invocation evidence."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterable, Mapping
import json
import math
from statistics import median
from pathlib import Path
from typing import Any

from nemo_fabric.accounting import summarize_usage
from nemo_fabric._report_format import format_duration_ms, format_money
from nemo_fabric.reporting import MAX_RESULT_BYTES, read_evidence


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _path(root: Path, value: Any) -> Path:
    relative = Path(_text(value, "evidence path"))
    resolved = (root / relative).resolve()
    if relative.is_absolute() or not resolved.is_relative_to(root):
        raise ValueError("evidence paths must remain within the comparison root")
    return resolved


def _evaluation(value: Any) -> dict[str, str]:
    if value is None:
        return {"status": "not_evaluated"}
    if (
        not isinstance(value, Mapping)
        or not isinstance(value.get("status"), str)
        or value.get("status") not in {"passed", "failed", "not_evaluated"}
    ):
        raise ValueError("evaluation must name passed, failed or not_evaluated")
    status = value["status"]
    if status == "not_evaluated":
        return {"status": status}
    return {
        "status": status,
        "evaluator": _text(value.get("evaluator"), "evaluator"),
        "version": _text(value.get("version"), "evaluator version"),
    }


def _accounting_source(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: summary[key]
        for key in ("request_id", "runtime_id", "invocation_id", "model_usage")
    }


def _record_accounting(
    view: dict[str, Any], summary: Mapping[str, Any]
) -> dict[str, Any] | None:
    if (
        summary.get("schema") != "nemo.fabric.invocation_record_summary"
        or summary.get("model_usage") is None
    ):
        return None
    if summary.get("outcome_boundary") != "invocation":
        view["record_error"] = "unsupported_record_boundary"
        return None
    view.update(accounting_source="record", accounting_boundary="core_invocation")
    return _accounting_source(summary)


def _request(
    root: Path, spec: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    request_id = _text(spec.get("request_id"), "request_id")
    evidence = _text(spec.get("evidence"), "evidence path")
    path = _path(root, evidence)
    record_path = (
        _path(root, spec["record"]) if spec.get("record") is not None else None
    )
    view: dict[str, Any] = {"request_id": request_id, "evidence": evidence}
    summary = None
    try:
        summary = read_evidence(path)
    except FileNotFoundError:
        view["state"] = "missing_evidence"
    except (OSError, ValueError, UnicodeError):
        view["state"] = "invalid_evidence"
    else:
        if summary["request_id"] != request_id:
            view["state"] = "identity_mismatch"
            # A valid result for a different request cannot authorize a fallback.
            return view, None
        view["summary"] = summary
        if summary["schema"] == "nemo.fabric.invocation_summary":
            # A valid full result remains authoritative, including unknown or
            # malformed accounting. Never repair it from an earlier projection.
            view.update(
                state="available",
                accounting_source="result",
                accounting_boundary="invocation",
            )
            return view, _accounting_source(summary)
        view["state"] = summary.get("state", "record_only")
        accounting = _record_accounting(view, summary)
        if accounting is not None:
            return view, accounting
    if record_path is None:
        return view, None
    view["record"] = spec["record"]
    try:
        record = read_evidence(record_path)
    except (OSError, ValueError, UnicodeError):
        view["record_error"] = "invalid_or_unreadable_record"
        return view, None
    if (
        record["request_id"] != request_id
        or record["schema"] != "nemo.fabric.invocation_record_summary"
    ):
        view["record_error"] = "identity_or_record_type_mismatch"
        return view, None
    if summary is not None:
        if any(
            field in summary and summary[field] != record[field]
            for field in ("runtime_id", "invocation_id")
        ):
            view["record_error"] = "identity_mismatch"
            return view, None
        if summary.get("state") == "no_result" and summary.get("stage") == "start":
            view["record_error"] = "conflicting_request_evidence"
            return view, None
    view["record_summary"] = record
    return view, _record_accounting(view, record)


def _usage(
    results: list[dict[str, Any]],
    listed_requests: int,
    *,
    inventory_complete: bool,
    attempt_inventory_complete: bool = True,
) -> dict[str, Any]:
    # Only real returned identities enter the shared accounting reducer. Missing
    # expected requests change coverage, never fabricate invocation identities.
    summary: dict[str, Any] = dict(
        summarize_usage(
            results,
            expected=[
                (result["runtime_id"], result["invocation_id"]) for result in results
            ],
        )
    )
    summary["unit"] = "request"
    summary["accounting_evidence_count"] = len(results)
    summary["normalized_accounting_count"] = sum(
        result.get("model_usage") is not None for result in results
    )
    summary["inventory_complete"] = inventory_complete and attempt_inventory_complete
    missing = listed_requests - len(results)
    for field in summary["dimensions"].values():
        field["expected_count"] = listed_requests
        field["missing_result_count"] = missing
        if missing:
            field["complete_total"] = None
            field["reasons"]["missing_result"] = missing
        if not inventory_complete:
            field["complete_total"] = None
            field["reasons"]["request_inventory_unverified"] = 1
        if not attempt_inventory_complete:
            field["complete_total"] = None
            field["reasons"]["attempt_inventory_unverified"] = 1
        if not listed_requests:
            field["complete_total"] = None
            field["reasons"]["no_request_evidence"] = 1

    # Comparison requests can be represented by a result or a typed record.
    # Preserve the reducer's quantities while naming this coverage boundary.
    def name_accounting_coverage(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                "missing_accounting_evidence_count"
                if key == "missing_result_count"
                else key: name_accounting_coverage(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [name_accounting_coverage(item) for item in value]
        return value

    return name_accounting_coverage(summary)


def _task_counts(rows: list[dict[str, Any]]) -> Counter[tuple[str, str]]:
    return Counter((row["task_id"], row["task_revision"]) for row in rows)


def _group(
    rows: list[dict[str, Any]], results: list[dict[str, Any]], inventory_complete: bool
) -> dict[str, Any]:
    evaluation = Counter(row["evaluation"]["status"] for row in rows)
    evaluators = sorted(
        {
            (row["evaluation"]["evaluator"], row["evaluation"]["version"])
            for row in rows
            if row["evaluation"]["status"] != "not_evaluated"
        }
    )
    evaluated = evaluation["passed"] + evaluation["failed"]
    requests = [request for row in rows for request in row["requests"]]
    available = [request for request in requests if request["state"] == "available"]
    return {
        "listed_attempt_count": len(rows),
        "request_inventory_complete_attempt_count": sum(
            row["requests_complete"] for row in rows
        ),
        "attempts_without_listed_requests": sum(not row["requests"] for row in rows),
        "task_inventory": [
            {"task_id": task, "task_revision": revision, "listed_attempt_count": count}
            for (task, revision), count in sorted(_task_counts(rows).items())
        ],
        "accounting_sources": dict(
            Counter(
                request["accounting_source"]
                for request in requests
                if "accounting_source" in request
            )
        ),
        "execution": {
            "listed_request_count": len(requests),
            "available_result_count": len(available),
            "missing_result_count": len(requests) - len(available),
            "statuses": dict(
                Counter(request["summary"]["status"] for request in available)
            ),
        },
        "evaluation": {
            "passed": evaluation["passed"],
            "failed": evaluation["failed"],
            "not_evaluated": evaluation["not_evaluated"],
            "pass_fraction_evaluated": evaluation["passed"] / evaluated
            if evaluated and len(evaluators) == 1
            else None,
            "evaluators": [
                {"name": name, "version": version} for name, version in evaluators
            ],
        },
        "timing": {
            "available_attempt_count": sum(
                row["elapsed_ms"] is not None for row in rows
            ),
            "observations": [
                {
                    "attempt_id": row["attempt_id"],
                    "scope": row["elapsed_scope"],
                    "elapsed_ms": row["elapsed_ms"],
                }
                for row in rows
                if row["elapsed_ms"] is not None
            ],
        },
        "usage": _usage(
            results,
            len(requests),
            inventory_complete=all(
                row["requests_complete"] and row["requests"] for row in rows
            ),
            attempt_inventory_complete=inventory_complete,
        ),
    }


def _pricing_alignment(configurations: Mapping[str, Any]) -> dict[str, Any]:
    # The shared reducer has already qualified and grouped contributing bases.
    # This comparison labels that evidence without defining another price policy.
    configuration_bases: dict[str, list[dict[str, Any]]] = {}
    distinct_bases: list[dict[str, Any]] = []
    for name, group in configurations.items():
        usage = group["usage"]
        priced = [usage] if "pricing" in usage else usage.get("pricing_groups", [])
        bases = [part["pricing"] for part in priced]
        configuration_bases[name] = bases
        for basis in bases:
            # Match the reducer's absent/null optional-field equivalence.
            comparable = {
                key: value for key, value in basis.items() if value is not None
            }
            if comparable not in distinct_bases:
                distinct_bases.append(comparable)
    unpriced = sorted(name for name, bases in configuration_bases.items() if not bases)
    if len(distinct_bases) > 1:
        status = "different_bases"
    elif not distinct_bases:
        status = "unavailable"
    elif unpriced:
        status = "unverified"
    else:
        status = "aligned"
    return {
        "status": status,
        "configuration_bases": configuration_bases,
        "unpriced_configurations": unpriced,
    }


def summarize_attempts(
    attempts: Iterable[Mapping[str, Any]],
    *,
    root: Path,
    inventory_complete: bool,
) -> dict[str, Any]:
    """Summarize an explicit inventory of scheduled custom-task attempts.

    Each attempt names its task/revision/configuration and expected requests.
    ``inventory_complete`` asserts all scheduled attempts, including retries,
    are listed; ``requests_complete`` makes the corresponding per-attempt claim.
    Unknown inventory never establishes a complete cohort denominator. Evaluator
    verdicts and caller elapsed boundaries remain independent of execution.
    Resources retain per-invocation scopes. Logs and traces are references only.
    ``pricing_alignment`` compares contributing shared-reducer bases, not cohort
    eligibility or accounting completeness; per-configuration totals are retained.
    """
    if type(inventory_complete) is not bool:
        raise ValueError("inventory_complete must be a boolean")
    root = Path(root).resolve()
    rows: list[dict[str, Any]] = []
    groups: dict[str, list[dict[str, Any]]] = {}
    records: dict[str, list[dict[str, Any]]] = {}
    attempts_seen: set[str] = set()
    requests_seen: set[str] = set()
    invocations_seen: set[tuple[str, str]] = set()
    for spec in attempts:
        if not isinstance(spec, Mapping):
            raise ValueError("attempt must be an object")
        identity = {
            key: _text(spec.get(key), key)
            for key in ("attempt_id", "task_id", "task_revision", "configuration_id")
        }
        if identity["attempt_id"] in attempts_seen:
            raise ValueError("duplicate attempt_id")
        attempts_seen.add(identity["attempt_id"])
        requests = spec.get("requests")
        if not isinstance(requests, list) or any(
            not isinstance(request, Mapping) for request in requests
        ):
            raise ValueError("requests must be a list of objects")
        complete = spec.get("requests_complete", False)
        if type(complete) is not bool:
            raise ValueError("requests_complete must be a boolean")
        elapsed = spec.get("elapsed_ms")
        if elapsed is not None and (
            type(elapsed) is not int or not 0 <= elapsed < 1 << 64
        ):
            raise ValueError("elapsed_ms must be a nonnegative uint64 integer or null")
        scope = (
            _text(spec.get("elapsed_scope"), "elapsed_scope")
            if elapsed is not None
            else None
        )
        views = []
        results = []
        for request in requests:
            view, result = _request(root, request)
            if view["request_id"] in requests_seen:
                raise ValueError("duplicate expected request_id")
            requests_seen.add(view["request_id"])
            views.append(view)
            if result is not None:
                key = (result["runtime_id"], result["invocation_id"])
                if key in invocations_seen:
                    raise ValueError(
                        "one invocation cannot belong to multiple requests or attempts"
                    )
                invocations_seen.add(key)
                results.append(result)
        row = {
            **identity,
            "evaluation": _evaluation(spec.get("evaluation")),
            "elapsed_ms": elapsed,
            "elapsed_scope": scope,
            "requests_complete": complete,
            "requests": views,
            "usage": _usage(
                results, len(requests), inventory_complete=complete and bool(requests)
            ),
        }
        rows.append(row)
        configuration = identity["configuration_id"]
        groups.setdefault(configuration, []).append(row)
        records.setdefault(configuration, []).extend(results)
    tasks = {name: set(_task_counts(group)) for name, group in groups.items()}
    shared = set.intersection(*tasks.values()) if tasks else set()
    task_counts = [_task_counts(group) for group in groups.values()]
    configurations = {
        name: _group(group, records[name], inventory_complete)
        for name, group in groups.items()
    }
    return {
        "schema": "nemo.fabric.comparison_summary",
        "version": "v1alpha1",
        "inventory_complete": inventory_complete,
        "attempts": rows,
        "task_alignment": {
            "shared_listed_tasks": [
                {"task_id": task, "task_revision": revision}
                for task, revision in sorted(shared)
            ],
            "unmatched_by_configuration": {
                name: [
                    {"task_id": task, "task_revision": revision}
                    for task, revision in sorted(listed - shared)
                ]
                for name, listed in tasks.items()
            },
            "listed_attempt_counts_match": bool(task_counts)
            and all(counts == task_counts[0] for counts in task_counts),
        },
        "configurations": configurations,
        "pricing_alignment": _pricing_alignment(configurations),
    }


def _invalid_constant(value: str) -> Any:
    raise ValueError(f"non-finite manifest number: {value}")


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non-finite manifest number")
    return number


def read_comparison_manifest(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    """Read a bounded manifest; evidence paths are relative to its explicit root."""
    path = Path(path)
    with path.open("rb") as stream:
        payload = stream.read(MAX_RESULT_BYTES + 1)
    if len(payload) > MAX_RESULT_BYTES:
        raise ValueError("comparison manifest exceeds 16 MiB")
    try:
        manifest = json.loads(
            payload, parse_constant=_invalid_constant, parse_float=_finite_float
        )
    except RecursionError as error:
        raise ValueError("comparison manifest is nested too deeply") from error
    if (
        not isinstance(manifest, dict)
        or (manifest.get("schema"), manifest.get("version"))
        != ("nemo.fabric.comparison_manifest", "v1alpha1")
        or not isinstance(manifest.get("attempts"), list)
    ):
        raise ValueError("unsupported comparison manifest schema or version")
    return summarize_attempts(
        manifest["attempts"],
        root=root or path.resolve().parent,
        inventory_complete=manifest.get("inventory_complete", False),
    )


def _value(field: Mapping[str, Any], *, money: bool = False) -> str:
    complete = field["complete_total"]
    observed = field["observed_sum"]
    if complete is not None:
        value = format_money(complete) if money else f"{complete:,}"
    elif observed is not None:
        value = f"{format_money(observed) if money else f'{observed:,}'} observed, incomplete"
    elif field.get("aggregation_error") == "incompatible_pricing":
        value = "incompatible pricing groups"
    else:
        value = "unavailable"
    return f"{value} ({field['complete_count']}/{field['expected_count']} complete)"


def _label(value: str) -> str:
    # JSON quoting escapes terminal controls, embedded newlines and pipes.
    return json.dumps(value, ensure_ascii=True).replace("|", "\\u007c")


def _catalog_identifier(basis: Mapping[str, Any]) -> str:
    return ", ".join(
        f"{label} {_label(basis[key].removeprefix('sha256:')[:12])}"
        for key, label in (
            ("bundled_catalog_sha256", "bundled"),
            ("user_catalog_sha256", "user"),
        )
        if basis.get(key)
    ) or _label(basis["basis"])


def _timing(group: Mapping[str, Any], scopes: set[str]) -> str:
    timing = group["timing"]
    coverage = (
        f"({timing['available_attempt_count']}/{group['listed_attempt_count']} timed)"
    )
    observations = timing["observations"]
    if not observations:
        return f"unavailable {coverage}"
    if len(scopes) != 1:
        return f"mixed scopes {coverage}"
    duration = format_duration_ms(median(row["elapsed_ms"] for row in observations))
    return f"{duration} median {coverage}; {_label(observations[0]['scope'])}"


def render_comparison_summary(
    report: Mapping[str, Any], *, details: bool = False
) -> str:
    """Render per-metric coverage; details retain identities and pricing evidence.

    Timing describes observed caller attempts with one common scope, never a sum
    of invocation durations. Neither partial spend nor task quality is ranked.
    """
    alignment = report["pricing_alignment"]
    configurations = report["configurations"]
    lines = []
    if not report["inventory_complete"]:
        lines.append(
            "Attempt inventory is unverified; the full cohort denominator and totals are unknown."
        )
    if any(
        group["request_inventory_complete_attempt_count"]
        != group["listed_attempt_count"]
        or group["attempts_without_listed_requests"]
        for group in configurations.values()
    ):
        lines.append(
            "Request inventories are unverified or empty for some attempts; full accounting totals are unknown."
        )
    if not report["task_alignment"]["listed_attempt_counts_match"]:
        lines.append(
            "Listed task/revision attempt inventories differ; no cohort was silently selected."
        )
    evaluators = {
        (item["name"], item["version"])
        for group in configurations.values()
        for item in group["evaluation"]["evaluators"]
    }
    if len(evaluators) > 1:
        lines.append(
            "Evaluator identities or versions differ; outcome fractions are not comparable."
        )
    scopes = {
        row["scope"]
        for group in configurations.values()
        for row in group["timing"]["observations"]
    }
    if len(scopes) > 1:
        lines.append("Timing scopes differ; no combined timing median is shown.")
    if alignment["status"] == "different_bases":
        lines.append(
            "Catalog pricing bases differ; do not compare estimates across different bases."
        )
    elif alignment["status"] == "unverified":
        lines.append(
            "Catalog pricing evidence is missing for some configurations; a shared basis is unverified."
        )
    lines.append(
        "Configuration | Task outcomes | Input tokens | Output tokens | Catalog estimate"
    )
    for name, group in configurations.items():
        evaluation = group["evaluation"]
        evaluated = evaluation["passed"] + evaluation["failed"]
        attempts = group["listed_attempt_count"]
        if not evaluated:
            verdict = f"not evaluated (0/{attempts} evaluated)"
        elif evaluation["pass_fraction_evaluated"] is None:
            verdict = (
                f"{evaluation['passed']} passed; {evaluation['failed']} failed; "
                f"{evaluated}/{attempts} evaluated; evaluator mismatch"
            )
        else:
            verdict = f"{evaluation['passed']}/{evaluated} passed; {evaluated}/{attempts} evaluated"
            if len(evaluators) > 1:
                verdict += "; evaluator mismatch"
        dimensions = group["usage"]["dimensions"]
        lines.append(
            f"{_label(name)} | {verdict} | {_value(dimensions['input_tokens'])} | {_value(dimensions['output_tokens'])} | {_value(dimensions['estimated_cost'], money=True)}"
        )
        execution = group["execution"]
        lines.append(
            f"  Results: {execution['available_result_count']}/{execution['listed_request_count']} returned; {execution['missing_result_count']} missing. Attempts listed: {attempts}."
        )
        if execution["statuses"]:
            lines.append(
                "  Execution: "
                + ", ".join(
                    f"{count} {status}"
                    for status, count in sorted(execution["statuses"].items())
                )
            )
        lines.append(f"  Caller attempt time: {_timing(group, scopes)}.")
        other_tokens = {
            "total_tokens": "Total tokens",
            "cached_input_tokens": "Cache reads",
            "cache_write_input_tokens": "Cache writes",
        }
        other_gaps = [
            f"{label}: {_value(dimensions[key])}"
            for key, label in other_tokens.items()
            if dimensions[key]["complete_total"] is None
        ]
        if other_gaps:
            lines.append("  Other token gaps: " + "; ".join(other_gaps))
        bases = alignment["configuration_bases"][name]
        if bases:
            label = (
                "Catalog basis" if len(bases) == 1 else f"Catalog bases ({len(bases)})"
            )
            lines.append(
                f"  {label}: "
                + "; ".join(_catalog_identifier(basis) for basis in bases)
            )
        else:
            lines.append("  No priced catalog evidence for this configuration.")
        reported = dimensions["reported_cost"]
        if (
            details
            or reported["complete_total"] is not None
            or reported["observed_sum"] is not None
        ):
            lines.append(f"  Provider-reported cost: {_value(reported, money=True)}.")
        reasons = sorted(
            {
                reason
                for key, field in dimensions.items()
                if details or key != "reported_cost"
                for reason in field["reasons"]
            }
        )
        if reasons:
            lines.append(
                "  Accounting gaps: "
                + ", ".join(reason.replace("_", " ") for reason in reasons)
            )
        if details:
            lines.append(
                "  Other tokens: "
                + "; ".join(
                    f"{label}: {_value(dimensions[key])}"
                    for key, label in other_tokens.items()
                )
                + ". Cache reads and writes are included in input."
            )
            if bases:
                lines.append(
                    "  Catalog pricing bases: "
                    + json.dumps(bases, ensure_ascii=True, sort_keys=True)
                )
            lines.append(
                "  Execution statuses: "
                + json.dumps(execution["statuses"], ensure_ascii=True)
                + "; evaluators: "
                + json.dumps(evaluation["evaluators"], ensure_ascii=True)
                + f"; {evaluation['not_evaluated']} attempts not evaluated."
            )
            lines.append(
                f"  Accounting sources: {json.dumps(group['accounting_sources'], ensure_ascii=True)}; result and record are never counted twice."
            )
            if group["timing"]["observations"]:
                lines.append(
                    "  Caller timing boundaries: "
                    + json.dumps(group["timing"]["observations"], ensure_ascii=True)
                )
            lines.append(
                "  Listed task revisions: "
                + json.dumps(group["task_inventory"], ensure_ascii=True)
            )
    lines.append(
        "Token and cost coverage counts complete / listed requests; timing covers observed caller attempts."
    )
    lines.append(
        "Evaluator verdicts are caller-supplied; execution success is not task correctness."
    )
    lines.append(
        "Observed spend is incomplete, not a bill or a ranking. Details and JSON retain evidence and pricing groups."
    )
    return "\n".join(lines)


def main() -> None:
    """Print an explicit comparison manifest as a table or JSON without running agents."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--root", type=Path, help="Evidence root; defaults to manifest directory"
    )
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument(
        "--details", action="store_true", help="Include identities and pricing evidence"
    )
    args = parser.parse_args()
    try:
        report = read_comparison_manifest(args.manifest, root=args.root)
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(
        json.dumps(report, indent=2, allow_nan=False)
        if args.as_json
        else render_comparison_summary(report, details=args.details)
    )


if __name__ == "__main__":
    main()
