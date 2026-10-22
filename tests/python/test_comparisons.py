# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from nemo_fabric.comparisons import (
    read_comparison_manifest,
    render_comparison_summary,
    summarize_attempts,
)


@pytest.fixture(name="result")
def result_fixture():
    return {
        "runtime_id": "runtime",
        "invocation_id": "invocation",
        "request_id": "request",
        "status": "succeeded",
        "model_usage": {
            "input_tokens": 100,
            "cached_input_tokens": 0,
            "cache_write_input_tokens": 0,
            "output_tokens": 2,
            "total_tokens": 102,
            "observed": {},
            "gaps": [{"dimensions": ["estimated_cost"], "reason": "missing_usage"}],
            "models_exhaustive": False,
            "models": [],
            "cost": {
                "reported": {"amount_nano_usd": 10, "observed_nano_usd": None},
                "estimated": {"amount_nano_usd": None, "observed_nano_usd": None},
            },
        },
        "artifacts": {"artifacts": []},
    }


@pytest.fixture(name="attempt")
def attempt_fixture(tmp_path, result):
    (tmp_path / "result.json").write_text(json.dumps(result))
    return {
        "attempt_id": "attempt",
        "task_id": "custom-review",
        "task_revision": "v1",
        "configuration_id": "baseline",
        "requests_complete": True,
        "requests": [{"request_id": "request", "evidence": "result.json"}],
        "elapsed_ms": 6000,
        "elapsed_scope": "caller_attempt",
    }


def report(tmp_path, attempts, complete=True):
    return summarize_attempts(attempts, root=tmp_path, inventory_complete=complete)


def test_custom_task_needs_no_evaluator_or_harbor(tmp_path, attempt):
    summary = report(tmp_path, [attempt])
    group = summary["configurations"]["baseline"]
    assert group["evaluation"]["not_evaluated"] == 1
    assert group["evaluation"]["pass_fraction_evaluated"] is None
    assert group["usage"]["dimensions"]["input_tokens"]["complete_total"] == 100
    assert summary["attempts"][0]["requests"][0]["summary"]["status"] == "succeeded"


def test_missing_attempt_keeps_expected_requests_and_available_observations(
    tmp_path, attempt
):
    missing = {
        **attempt,
        "attempt_id": "missing",
        "requests": [{"request_id": "missing", "evidence": "missing.json"}],
    }
    group = report(tmp_path, [attempt, missing])["configurations"]["baseline"]
    assert group["listed_attempt_count"] == 2
    dimension = group["usage"]["dimensions"]["input_tokens"]
    assert dimension["complete_total"] is None
    assert dimension["observed_sum"] == 100
    assert dimension["expected_count"] == 2
    assert dimension["missing_accounting_evidence_count"] == 1


def test_parallel_invocations_keep_caller_time_and_separate_resource_scopes(
    tmp_path, attempt, result
):
    second = deepcopy(result)
    second.update(
        request_id="parallel", invocation_id="parallel", runtime_id="parallel-runtime"
    )
    result["resource_snapshot"] = {
        "runtime_id": "runtime",
        "cpu_time_seconds": 10.0,
        "resident_memory_bytes": 100,
        "process_count": 1,
        "started_at": "2026-10-21T00:00:00Z",
        "observed_at": "2026-10-21T00:00:05Z",
    }
    second["resource_snapshot"] = {
        **result["resource_snapshot"],
        "runtime_id": "parallel-runtime",
        "cpu_time_seconds": 20.0,
        "resident_memory_bytes": 200,
    }
    (tmp_path / "result.json").write_text(json.dumps(result))
    (tmp_path / "parallel.json").write_text(json.dumps(second))
    attempt["requests"].append({"request_id": "parallel", "evidence": "parallel.json"})
    row = report(tmp_path, [attempt])["attempts"][0]
    assert row["elapsed_ms"] == 6000
    assert row["usage"]["dimensions"]["input_tokens"]["complete_total"] == 200
    resources = [request["summary"]["resource_snapshot"] for request in row["requests"]]
    assert [resource["cpu_time_seconds"] for resource in resources] == [10.0, 20.0]
    assert [resource["runtime_id"] for resource in resources] == [
        "runtime",
        "parallel-runtime",
    ]
    assert "resource_snapshot" not in row
    assert "cpu_time_seconds" not in row


def test_partial_accounting_does_not_erase_quality_or_independent_dimensions(
    tmp_path, attempt, result
):
    usage = result["model_usage"]
    usage["input_tokens"] = usage["total_tokens"] = None
    usage["observed"] = {"input_tokens": 50}
    usage["gaps"].append(
        {"dimensions": ["input_tokens", "total_tokens"], "reason": "unfinished_work"}
    )
    result["status"] = "failed"
    (tmp_path / "result.json").write_text(json.dumps(result))
    attempt["evaluation"] = {
        "status": "passed",
        "evaluator": "external-tests",
        "version": "v1",
    }
    group = report(tmp_path, [attempt])["configurations"]["baseline"]
    assert group["evaluation"]["passed"] == 1
    assert group["execution"]["statuses"] == {"failed": 1}
    assert group["usage"]["dimensions"]["input_tokens"]["observed_sum"] == 50
    assert group["usage"]["dimensions"]["output_tokens"]["complete_total"] == 2


@pytest.mark.parametrize("scope", ["attempt", "requests"])
def test_unknown_inventory_never_certifies_complete_cohort_total(
    tmp_path, attempt, scope
):
    if scope == "requests":
        attempt["requests_complete"] = False
    group = report(tmp_path, [attempt], complete=scope != "attempt")["configurations"][
        "baseline"
    ]
    dimension = group["usage"]["dimensions"]["input_tokens"]
    assert dimension["complete_total"] is None
    assert dimension["observed_sum"] == 100
    assert group["usage"]["inventory_complete"] is False


@pytest.mark.parametrize("path", ["../escape.json", "/tmp/escape.json"])
def test_evidence_path_cannot_escape_root(tmp_path, attempt, path):
    attempt["requests"][0]["evidence"] = path
    with pytest.raises(ValueError, match="within"):
        report(tmp_path, [attempt])


def test_symlink_evidence_cannot_escape_root(tmp_path, attempt):
    outside = tmp_path.parent / (tmp_path.name + "-outside.json")
    outside.write_text("{}")
    (tmp_path / "link.json").symlink_to(outside)
    attempt["requests"][0]["evidence"] = "link.json"
    with pytest.raises(ValueError, match="within"):
        report(tmp_path, [attempt])


def test_reused_invocation_identity_is_rejected_across_distinct_requests(
    tmp_path, attempt, result
):
    result["request_id"] = "second"
    (tmp_path / "second.json").write_text(json.dumps(result))
    other = {
        **attempt,
        "attempt_id": "other",
        "requests": [{"request_id": "second", "evidence": "second.json"}],
    }
    with pytest.raises(ValueError, match="one invocation"):
        report(tmp_path, [attempt, other])


@pytest.mark.parametrize("mutation", ["wrong_id", "truncated", "invalid_status"])
def test_bad_result_is_a_visible_gap_without_losing_evaluator(
    tmp_path, attempt, result, mutation
):
    attempt["evaluation"] = {"status": "failed", "evaluator": "tests", "version": "v1"}
    if mutation == "wrong_id":
        result["request_id"] = "another"
    if mutation == "invalid_status":
        result["status"] = "unknown"
    (tmp_path / "result.json").write_text(
        "{" if mutation == "truncated" else json.dumps(result)
    )
    row = report(tmp_path, [attempt])["attempts"][0]
    assert row["requests"][0]["state"] != "available"
    assert row["evaluation"]["status"] == "failed"
    assert (
        row["usage"]["dimensions"]["input_tokens"]["missing_accounting_evidence_count"]
        == 1
    )


def test_invalid_accounting_is_quarantined_by_canonical_reader(
    tmp_path, attempt, result
):
    result["model_usage"]["input_tokens"] = -1
    result["telemetry"] = [{"provider": "relay", "uri": "do-not-open.jsonl"}]
    (tmp_path / "result.json").write_text(json.dumps(result))
    view = report(tmp_path, [attempt])["attempts"][0]["requests"][0]
    assert view["state"] == "available"
    assert view["summary"]["status"] == "succeeded"
    assert view["summary"]["accounting_error"] == "invalid_model_usage"
    assert view["summary"]["telemetry"] == result["telemetry"]


def test_legacy_usage_stays_diagnostics_without_false_completeness(
    tmp_path, attempt, result
):
    result.pop("model_usage")
    result["usage"] = {"input_tokens": 100}
    (tmp_path / "result.json").write_text(json.dumps(result))
    row = report(tmp_path, [attempt])["attempts"][0]
    assert row["requests"][0]["summary"]["legacy_usage"] == {"input_tokens": 100}
    assert row["usage"]["dimensions"]["input_tokens"]["complete_total"] is None


def test_empty_scheduled_attempt_stays_in_inventory(tmp_path, attempt):
    attempt["requests"] = []
    group = report(tmp_path, [attempt])["configurations"]["baseline"]
    assert group["listed_attempt_count"] == 1
    assert group["attempts_without_listed_requests"] == 1
    assert group["usage"]["dimensions"]["input_tokens"]["complete_total"] is None


def test_mixed_evaluator_identities_have_no_combined_fraction(tmp_path, attempt):
    attempt["evaluation"] = {"status": "passed", "evaluator": "A", "version": "v1"}
    other = {
        **attempt,
        "attempt_id": "other",
        "requests": [],
        "evaluation": {"status": "failed", "evaluator": "B", "version": "v1"},
    }
    summary = report(tmp_path, [attempt, other])
    evaluation = summary["configurations"]["baseline"]["evaluation"]
    assert evaluation["passed"] == evaluation["failed"] == 1
    assert evaluation["pass_fraction_evaluated"] is None
    rendered = render_comparison_summary(summary)
    assert "1/2 passed" not in rendered
    assert "1 passed; 1 failed; 2/2 evaluated; evaluator mismatch" in rendered


def test_unmatched_tasks_and_repetitions_are_visible_without_selection(
    tmp_path, attempt
):
    other = {
        **attempt,
        "attempt_id": "other",
        "configuration_id": "candidate",
        "task_revision": "v2",
        "requests": [],
    }
    summary = report(tmp_path, [attempt, other])
    assert summary["task_alignment"]["shared_listed_tasks"] == []
    assert (
        len(summary["task_alignment"]["unmatched_by_configuration"]["candidate"]) == 1
    )
    assert summary["task_alignment"]["listed_attempt_counts_match"] is False
    assert "no cohort was silently selected" in render_comparison_summary(summary)


def test_labels_cannot_inject_terminal_controls_or_new_rows(tmp_path, attempt):
    attempt["configuration_id"] = "bad\x1b[31m\nlabel|extra"
    text = render_comparison_summary(report(tmp_path, [attempt]))
    assert "\x1b" not in text
    assert "bad\\u001b[31m\\nlabel\\u007cextra" in text


def test_outer_request_evidence_remains_no_result_not_zero(tmp_path, attempt):
    document = {
        "schema": "nemo.fabric.request_record",
        "version": "v1alpha1",
        "kind": "end",
        "state": "no_result",
        "stage": "start",
        "request_id": "request",
        "time": "2026-10-21T00:00:00Z",
    }
    (tmp_path / "result.json").write_text(json.dumps(document))
    row = report(tmp_path, [attempt])["attempts"][0]
    assert row["requests"][0]["state"] == "no_result"
    assert row["usage"]["dimensions"]["input_tokens"]["observed_sum"] is None


def test_cli_manifest_reader_uses_same_contract(tmp_path, attempt):
    manifest = {
        "schema": "nemo.fabric.comparison_manifest",
        "version": "v1alpha1",
        "inventory_complete": True,
        "attempts": [attempt],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    expected = report(tmp_path, [attempt])
    assert read_comparison_manifest(path) == expected
    command = [sys.executable, "-m", "nemo_fabric.comparisons", str(path)]
    output = subprocess.run(
        [*command, "--json"], check=True, capture_output=True, text=True
    )
    assert json.loads(output.stdout) == expected
    table = subprocess.run(command, check=True, capture_output=True, text=True)
    assert "100 (1/1 complete)" in table.stdout
    assert "not evaluated" in table.stdout
    detail = subprocess.run(
        [*command, "--details"], check=True, capture_output=True, text=True
    )
    assert "Listed task revisions:" in detail.stdout
    assert "Listed task revisions:" not in table.stdout
    structured_detail = subprocess.run(
        [*command, "--json", "--details"], check=True, capture_output=True, text=True
    )
    assert json.loads(structured_detail.stdout) == expected


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate_attempt",
        "duplicate_request",
        "missing_elapsed_scope",
        "boolean_elapsed",
        "malformed_evaluation",
    ],
)
def test_inventory_validation_rejects_ambiguous_inputs(tmp_path, attempt, mutation):
    attempts = [attempt]
    if mutation == "duplicate_attempt":
        attempts.append(deepcopy(attempt))
    elif mutation == "duplicate_request":
        attempt["requests"].append(deepcopy(attempt["requests"][0]))
    elif mutation == "missing_elapsed_scope":
        attempt.pop("elapsed_scope")
    elif mutation == "boolean_elapsed":
        attempt["elapsed_ms"] = True
    else:
        attempt["evaluation"] = {"status": {"bad": "shape"}}
    with pytest.raises(ValueError):
        report(tmp_path, attempts)


def test_partial_pricing_groups_keep_their_coverage_and_never_become_one_cost(
    tmp_path, attempt, result
):
    requests = []
    for index, amount, catalog in [(0, 14000, "a"), (1, 20000, "b"), (2, None, "a")]:
        value = deepcopy(result)
        request_id = f"priced-{index}"
        value.update(request_id=request_id, invocation_id=request_id)
        usage = value["model_usage"]
        usage["cost"]["estimated"] = {
            "amount_nano_usd": amount,
            "observed_nano_usd": amount,
        }
        usage["cost"]["pricing"] = {
            "basis": "catalog",
            "currency": "USD",
            "normalization_version": "cache_reasoning_inclusive_v1",
            "assumption_policy": "applicable_tariff_no_missing_counts_v1",
            "bundled_catalog_sha256": "sha256:" + catalog * 64,
        }
        usage["gaps"] = (
            []
            if amount is not None
            else [{"dimensions": ["estimated_cost"], "reason": "missing_usage"}]
        )
        filename = f"{request_id}.json"
        (tmp_path / filename).write_text(json.dumps(value))
        requests.append({"request_id": request_id, "evidence": filename})
    requests.append({"request_id": "missing-price", "evidence": "missing-price.json"})
    attempt["requests"] = requests
    usage = report(tmp_path, [attempt])["configurations"]["baseline"]["usage"]
    assert usage["dimensions"]["estimated_cost"]["observed_sum"] is None
    assert (
        usage["dimensions"]["estimated_cost"]["aggregation_error"]
        == "incompatible_pricing"
    )
    assert usage["dimensions"]["estimated_cost"]["expected_count"] == 4
    groups = {
        group["pricing"]["bundled_catalog_sha256"]: group["dimensions"][
            "estimated_cost"
        ]
        for group in usage["pricing_groups"]
    }
    first = groups["sha256:" + "a" * 64]
    assert first["expected_count"] == 2
    assert first["missing_accounting_evidence_count"] == 0
    assert "missing_result_count" not in first
    assert first["incomplete_count"] == 1
    assert first["complete_total"] is None
    assert first["observed_sum"] == 14000
    assert groups["sha256:" + "b" * 64]["complete_total"] == 20000
    assert usage["dimensions"]["input_tokens"]["observed_sum"] == 300


@pytest.fixture(name="record")
def record_fixture(tmp_path, result):
    identity = {
        key: result[key] for key in ("request_id", "runtime_id", "invocation_id")
    }
    start = {
        "schema": "nemo.fabric.run_record",
        "version": "v1alpha1",
        "kind": "start",
        **identity,
    }
    end = {
        **start,
        "kind": "end",
        "status": "succeeded",
        "elapsed_ms": 1000,
        "outcome_scope": "invocation",
        "model_usage": result["model_usage"],
    }
    path = tmp_path / "fallback.jsonl"
    path.write_text(json.dumps(start) + "\n" + json.dumps(end) + "\n")
    return path


@pytest.mark.parametrize("condition", ["missing", "invalid", "oversized", "direct"])
def test_record_accounting_survives_missing_full_result_without_creating_outer_success(
    tmp_path, attempt, record, condition
):
    from nemo_fabric.reporting import MAX_RESULT_BYTES

    request = attempt["requests"][0]
    if condition == "direct":
        request["evidence"] = "fallback.jsonl"
    else:
        request["record"] = "fallback.jsonl"
        request["evidence"] = "unavailable.json"
        if condition == "invalid":
            (tmp_path / "unavailable.json").write_text("{")
        elif condition == "oversized":
            with (tmp_path / "unavailable.json").open("wb") as stream:
                stream.seek(MAX_RESULT_BYTES)
                stream.write(b" ")
    summary = report(tmp_path, [attempt])
    row = summary["attempts"][0]
    view = row["requests"][0]
    assert view["state"] != "available"
    assert view["accounting_source"] == "record"
    assert view["accounting_boundary"] == "core_invocation"
    assert row["usage"]["dimensions"]["input_tokens"]["complete_total"] == 100
    execution = summary["configurations"]["baseline"]["execution"]
    assert execution["available_result_count"] == 0
    assert execution["missing_result_count"] == 1
    assert execution["statuses"] == {}
    assert row["evaluation"]["status"] == "not_evaluated"


@pytest.mark.parametrize("condition", ["omitted_counts", "invalid_accounting"])
def test_valid_full_result_is_authority_even_when_record_has_more_accounting(
    tmp_path, attempt, result, record, condition
):
    attempt["requests"][0]["record"] = "fallback.jsonl"
    if condition == "omitted_counts":
        result["model_usage"]["input_tokens"] = result["model_usage"][
            "total_tokens"
        ] = None
        result["model_usage"]["observed"]["input_tokens"] = 50
        result["model_usage"]["gaps"].append(
            {"dimensions": ["input_tokens", "total_tokens"], "reason": "missing_usage"}
        )
    else:
        result["model_usage"]["input_tokens"] = -1
    (tmp_path / "result.json").write_text(json.dumps(result))
    row = report(tmp_path, [attempt])["attempts"][0]
    assert row["requests"][0]["accounting_source"] == "result"
    assert "record_summary" not in row["requests"][0]
    assert row["usage"]["dimensions"]["input_tokens"]["complete_total"] is None
    assert row["usage"]["dimensions"]["input_tokens"]["observed_sum"] == (
        50 if condition == "omitted_counts" else None
    )


@pytest.mark.parametrize(
    "condition", ["wrong_request", "wrong_runtime", "startup_conflict", "legacy"]
)
def test_record_fallback_checks_canonical_identity_and_never_promotes_legacy_counts(
    tmp_path, attempt, result, record, condition
):
    request = attempt["requests"][0]
    request.update(evidence="request-end.json", record="fallback.jsonl")
    if condition == "wrong_runtime":
        outer = {
            "schema": "nemo.fabric.request_record",
            "version": "v1alpha1",
            "kind": "end",
            "state": "returned",
            "time": "2026-10-21T00:00:00Z",
            "request_id": "request",
            "runtime_id": "different",
            "invocation_id": "invocation",
            "result": "result.json",
        }
        (tmp_path / "request-end.json").write_text(json.dumps(outer))
    elif condition == "startup_conflict":
        outer = {
            "schema": "nemo.fabric.request_record",
            "version": "v1alpha1",
            "kind": "end",
            "state": "no_result",
            "stage": "start",
            "time": "2026-10-21T00:00:00Z",
            "request_id": "request",
        }
        (tmp_path / "request-end.json").write_text(json.dumps(outer))
    else:
        lines = [json.loads(line) for line in record.read_text().splitlines()]
        if condition == "wrong_request":
            for line in lines:
                line["request_id"] = "another"
        else:
            lines[1].pop("model_usage")
            lines[1]["usage"] = {"input_tokens": 100}
        record.write_text("".join(json.dumps(line) + "\n" for line in lines))
    row = report(tmp_path, [attempt])["attempts"][0]
    assert "accounting_source" not in row["requests"][0]
    assert row["usage"]["dimensions"]["input_tokens"]["complete_total"] is None
    assert row["usage"]["dimensions"]["input_tokens"]["observed_sum"] is None


def test_optional_record_path_has_same_containment_boundary(tmp_path, attempt):
    attempt["requests"][0]["record"] = "../outside.jsonl"
    with pytest.raises(ValueError, match="within"):
        report(tmp_path, [attempt])


def test_unknown_sibling_fields_do_not_abort_comparison(
    tmp_path, attempt, result, record
):
    lines = [json.loads(line) for line in record.read_text().splitlines()]
    lines[0]["resources"] = {"error": {"invalid": "shape"}}
    lines[0]["resource_scope"] = "runtime_adapter_host_process_tree"
    lines[1]["resource_scope"] = "runtime_adapter_host_process_tree"
    (tmp_path / "run-record.jsonl").write_text(
        "".join(json.dumps(line) + "\n" for line in lines)
    )
    summary = report(tmp_path, [attempt])
    group = summary["configurations"]["baseline"]
    assert group["execution"]["statuses"] == {"succeeded": 1}
    assert group["usage"]["dimensions"]["input_tokens"]["complete_total"] == 100
    view = summary["attempts"][0]["requests"][0]
    assert view["accounting_source"] == "result"


@pytest.mark.parametrize(
    ("sources", "status"),
    [
        ([("baseline", "a")], "aligned"),
        ([("baseline", "a"), ("candidate", "a")], "aligned"),
        ([("baseline", "bundled"), ("candidate", "bundled-null")], "aligned"),
        ([("baseline", "a"), ("candidate", "b")], "different_bases"),
        ([("baseline", "a"), ("baseline", "b")], "different_bases"),
        ([("baseline", None)], "unavailable"),
        ([("baseline", "a"), ("candidate", None)], "unverified"),
    ],
    ids=[
        "one",
        "same",
        "omitted-versus-null-override",
        "different",
        "mixed-within-config",
        "unpriced",
        "partly-priced",
    ],
)
def test_catalog_comparison_labels_bases_without_changing_configuration_totals(
    tmp_path, result, sources, status
):
    attempts = []
    for index, (configuration, catalog) in enumerate(sources):
        value = deepcopy(result)
        request_id = f"request-{index}"
        value.update(request_id=request_id, invocation_id=request_id)
        if catalog is not None:
            multiplier = 2 if catalog == "b" else 1
            usage = value["model_usage"]
            usage["gaps"] = []
            usage["cost"] = {
                "reported": {},
                "estimated": {
                    "amount_nano_usd": 104000 * multiplier,
                    "observed_nano_usd": 104000 * multiplier,
                },
                "pricing": {
                    "basis": "catalog",
                    "currency": "USD",
                    "normalization_version": "cache_reasoning_inclusive_v1",
                    "assumption_policy": "applicable_tariff_no_missing_counts_v1",
                    "bundled_catalog_sha256": "sha256:" + "c" * 64,
                    "user_catalog_sha256": "sha256:" + catalog * 64,
                },
            }
            if catalog == "bundled":
                usage["cost"]["pricing"].pop("user_catalog_sha256")
            elif catalog == "bundled-null":
                usage["cost"]["pricing"]["user_catalog_sha256"] = None
            usage["models_exhaustive"] = True
            usage["models"] = [
                {
                    key: deepcopy(item)
                    for key, item in usage.items()
                    if key not in {"models_exhaustive", "models"}
                }
                | {"provider": "openai", "model": "model"}
            ]
        filename = request_id + ".json"
        (tmp_path / filename).write_text(json.dumps(value))
        attempts.append(
            {
                "attempt_id": str(index),
                "task_id": "same-task",
                "task_revision": "same-revision",
                "configuration_id": configuration,
                "requests_complete": True,
                "requests": [{"request_id": request_id, "evidence": filename}],
            }
        )
    summary = report(tmp_path, attempts)
    alignment = summary["pricing_alignment"]
    assert alignment["status"] == status
    assert alignment["unpriced_configurations"] == sorted(
        {configuration for configuration, catalog in sources if catalog is None}
    )
    text = render_comparison_summary(summary, details=True)
    compact = render_comparison_summary(summary)
    for configuration in {name for name, _ in sources}:
        catalogs = {catalog for name, catalog in sources if name == configuration}
        bases = alignment["configuration_bases"][configuration]
        assert {basis.get("user_catalog_sha256") for basis in bases} == {
            None if catalog.startswith("bundled") else "sha256:" + catalog * 64
            for catalog in catalogs
            if catalog is not None
        }
        dimension = summary["configurations"][configuration]["usage"]["dimensions"]
        count = sum(name == configuration for name, _ in sources)
        assert dimension["input_tokens"]["complete_total"] == 100 * count
        if len(catalogs) == 1 and None not in catalogs:
            catalog = next(iter(catalogs))
            expected = 104000 * count * (2 if catalog == "b" else 1)
            assert dimension["estimated_cost"]["complete_total"] == expected
        else:
            assert dimension["estimated_cost"]["complete_total"] is None
        if bases:
            assert "Catalog basis:" in compact or "Catalog bases (" in compact
            if len(bases) > 1:
                assert f"Catalog bases ({len(bases)}):" in compact
        for basis in bases:
            bundled = basis["bundled_catalog_sha256"]
            assert bundled.removeprefix("sha256:")[:12] in compact
            assert bundled not in compact
            assert basis["bundled_catalog_sha256"] in text
            if basis.get("user_catalog_sha256") is not None:
                assert basis["user_catalog_sha256"] in text
    assert ("Catalog pricing bases differ" in text) == (status == "different_bases")
    if alignment["unpriced_configurations"]:
        assert "No priced catalog evidence" in text
    else:
        assert "No priced catalog evidence" not in text


def test_report_shows_independent_metric_and_evaluation_coverage(
    tmp_path, attempt, result
):
    result["model_usage"]["input_tokens"] = None
    result["model_usage"]["total_tokens"] = None
    result["model_usage"]["observed"]["input_tokens"] = 50
    result["model_usage"]["gaps"].append(
        {"dimensions": ["input_tokens", "total_tokens"], "reason": "unfinished_work"}
    )
    (tmp_path / "result.json").write_text(json.dumps(result))
    attempt["evaluation"] = {"status": "passed", "evaluator": "tests", "version": "v1"}
    text = render_comparison_summary(report(tmp_path, [attempt]))
    assert "50 observed, incomplete (0/1 complete)" in text
    assert "2 (1/1 complete)" in text
    assert "unavailable (0/1 complete)" in text
    assert "1/1 passed; 1/1 evaluated" in text
    assert "Total tokens" in text


def test_inventory_warnings_precede_table_and_missing_results_remain_visible(
    tmp_path, attempt
):
    attempt["requests_complete"] = False
    other = {
        **attempt,
        "attempt_id": "missing",
        "configuration_id": "other",
        "task_revision": "v2",
        "requests": [{"request_id": "missing", "evidence": "missing.json"}],
    }
    text = render_comparison_summary(report(tmp_path, [attempt, other], complete=False))
    header = text.index("Configuration |")
    assert text.index("Attempt inventory is unverified") < header
    assert text.index("Request inventories are unverified") < header
    assert text.index("Listed task/revision attempt inventories differ") < header
    assert "0/1 returned; 1 missing" in text


def test_timing_uses_caller_attempt_median_and_rejects_mixed_scopes(
    tmp_path, attempt, result
):
    other = {**attempt, "attempt_id": "other", "elapsed_ms": 10000, "requests": []}
    summary = report(tmp_path, [attempt, other])
    text = render_comparison_summary(summary)
    assert "8 s median (2/2 timed)" in text
    assert "caller_attempt" in text
    assert "16 s" not in text
    other["elapsed_scope"] = "adapter_exchange"
    text = render_comparison_summary(report(tmp_path, [attempt, other]))
    assert "Timing scopes differ" in text
    assert "mixed scopes (2/2 timed)" in text
    assert " s median (" not in text


def test_comparison_details_keep_verbose_evidence_out_of_default(tmp_path, attempt):
    summary = report(tmp_path, [attempt])
    text = render_comparison_summary(summary)
    assert "Listed task revisions:" not in text
    assert "Caller timing boundaries:" not in text
    assert "Accounting sources:" not in text
    detail = render_comparison_summary(summary, details=True)
    assert "Listed task revisions:" in detail
    assert "Caller timing boundaries:" in detail
    assert "Accounting sources:" in detail


@pytest.mark.parametrize("mode", ["complete", "observed", "absent"])
def test_provider_cost_remains_visible_when_available(tmp_path, attempt, result, mode):
    cost = result["model_usage"]["cost"]["reported"]
    if mode != "complete":
        cost["amount_nano_usd"] = None
        cost["observed_nano_usd"] = 10 if mode == "observed" else None
        result["model_usage"]["gaps"].append(
            {"dimensions": ["reported_cost"], "reason": "missing_usage"}
        )
    (tmp_path / "result.json").write_text(json.dumps(result))
    summary = report(tmp_path, [attempt])
    text = render_comparison_summary(summary)
    if mode == "absent":
        assert "Provider-reported cost:" not in text
    else:
        assert "Provider-reported cost: $0.000000010" in text
        assert ("$0.000000010 observed, incomplete" in text) == (mode == "observed")
    assert "Provider-reported cost:" in render_comparison_summary(summary, details=True)
