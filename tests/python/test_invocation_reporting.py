# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
import json

import pytest

from nemo_fabric.reporting import (
    read_evidence,
    read_result,
    render_result_summary,
    summarize_result,
)
from nemo_fabric.types import RunResult


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


def test_one_off_result_needs_no_task_labels_or_benchmark_manifest(tmp_path, result):
    path = tmp_path / "run-result.json"
    path.write_text(json.dumps(result))
    summary = read_result(path)
    text = render_result_summary(summary)
    assert summary["status"] == "succeeded"
    assert "input 100" in text
    assert "reads 0" in text
    assert "Provider-reported cost: $0.000000010 (complete)" in text
    assert "Catalog estimate: unavailable" in text
    assert summary["result_path"] == str(path)


@pytest.mark.parametrize("status", ["failed", "cancelled"])
def test_execution_failure_does_not_erase_settled_usage(result, status):
    result["status"] = status
    summary = summarize_result(result)
    assert summary["status"] == status
    assert summary["usage"]["dimensions"]["input_tokens"]["complete_total"] == 100


def test_partial_input_preserves_observations_and_independent_output(result):
    usage = result["model_usage"]
    usage["input_tokens"] = usage["total_tokens"] = None
    usage["observed"] = {"input_tokens": 50}
    usage["gaps"].append(
        {"dimensions": ["input_tokens", "total_tokens"], "reason": "unfinished_work"}
    )
    summary = summarize_result(result)
    dimensions = summary["usage"]["dimensions"]
    assert dimensions["input_tokens"]["complete_total"] is None
    assert dimensions["input_tokens"]["observed_sum"] == 50
    assert dimensions["output_tokens"]["complete_total"] == 2
    text = render_result_summary(summary)
    assert "input 50 (observed, incomplete)" in text
    assert "output 2 (complete)" in text
    assert "Some invocation-owned model work was unfinished at return." in text


def test_legacy_usage_does_not_become_normalized_or_zero(result):
    result.pop("model_usage")
    result["usage"] = {"input_tokens": 100, "output_tokens": 0}
    summary = summarize_result(result)
    assert summary["legacy_usage"] == result["usage"]
    assert summary["usage"]["dimensions"]["input_tokens"]["complete_total"] is None
    assert "unverified diagnostics" in render_result_summary(summary)


def test_read_result_does_not_open_log_or_trace_payloads(tmp_path, result):
    result["artifacts"] = {
        "artifacts": [{"kind": "log", "path": "missing-sensitive-log.txt"}]
    }
    result["telemetry"] = [{"provider": "relay", "uri": "missing-trace.jsonl"}]
    path = tmp_path / "result.json"
    path.write_text(json.dumps(result))
    summary = read_result(path)
    assert summary["artifacts"] == result["artifacts"]["artifacts"]
    assert summary["usage"]["dimensions"]["input_tokens"]["complete_total"] == 100


@pytest.mark.parametrize("mutation", ["status", "identity", "not_object", "non_finite"])
def test_malformed_saved_result_is_rejected(tmp_path, result, mutation):
    payload = deepcopy(result)
    if mutation == "status":
        payload["status"] = []
    elif mutation == "identity":
        payload["request_id"] = ""
    elif mutation == "not_object":
        payload = []
    else:
        payload["unrelated"] = float("nan")
    path = tmp_path / "result.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        read_result(path)


def test_result_outcome_boundary_survives_the_reader(result):
    result["artifacts"]["artifacts"] = [
        {
            "kind": "run_result",
            "path": "run-result.json",
            "metadata": {"boundary": "one_shot_lifecycle"},
        }
    ]
    summary = summarize_result(result)
    assert summary["outcome_boundary"] == "one_shot_lifecycle"
    assert 'Outcome boundary: "one_shot_lifecycle"' in render_result_summary(
        summary, details=True
    )


@pytest.fixture(name="record_lines")
def record_lines_fixture():
    common = {
        "schema": "nemo.fabric.run_record",
        "version": "v1alpha1",
        "request_id": "request",
        "runtime_id": "runtime",
        "invocation_id": "invocation",
    }
    return [
        {**common, "kind": "start"},
        {
            **common,
            "kind": "end",
            "status": "succeeded",
            "elapsed_ms": 123,
            "outcome_scope": "invocation",
        },
    ]


def test_invocation_record_keeps_its_pre_shutdown_scope(tmp_path, record_lines):
    path = tmp_path / "run-record.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in record_lines))
    summary = read_evidence(path)
    assert summary["elapsed_ms"] == 123
    assert summary["outcome_boundary"] == "invocation"
    assert summary["model_usage"] is None and summary["usage"] is None


def test_truncated_tail_keeps_start_without_fabricating_a_result(
    tmp_path, record_lines
):
    path = tmp_path / "run-record.jsonl"
    path.write_text(json.dumps(record_lines[0]) + '\n{"kind":')
    summary = read_evidence(path)
    assert summary["truncated_tail"]
    assert summary["status"] is None and summary["elapsed_ms"] is None


@pytest.mark.parametrize(
    "condition", ["duplicate", "identity", "negative_elapsed", "scope"]
)
def test_conflicting_records_are_rejected(tmp_path, record_lines, condition):
    if condition == "duplicate":
        record_lines.append(record_lines[-1])
    elif condition == "identity":
        record_lines[1]["runtime_id"] = "other"
    elif condition == "negative_elapsed":
        record_lines[1]["elapsed_ms"] = -1
    else:
        record_lines[1]["outcome_scope"] = "unknown"
    path = tmp_path / "run-record.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in record_lines))
    with pytest.raises(ValueError):
        read_evidence(path)


@pytest.mark.parametrize("suffix", ["json", "jsonl"])
def test_overflowing_json_exponent_is_rejected_before_summary(
    tmp_path, result, record_lines, suffix
):
    path = tmp_path / ("result." + suffix)
    if suffix == "json":
        text = json.dumps(result)[:-1] + ', "extra": 1e400}'
    else:
        text = json.dumps(record_lines[0])[:-1] + ', "extra": 1e400}\n'
    path.write_text(text)
    with pytest.raises(ValueError, match="non-finite"):
        read_evidence(path)


def test_result_reads_only_matching_core_sibling_for_timing(
    tmp_path, result, record_lines
):
    path = tmp_path / "run-result.json"
    path.write_text(json.dumps(result))
    record = tmp_path / "run-record.jsonl"
    record.write_text("".join(json.dumps(row) + "\n" for row in record_lines))
    summary = read_result(path)
    assert summary["execution_interval"]["elapsed_ms"] == 123
    assert summary["execution_interval"]["scope"] == "core_invocation"
    assert "before shutdown: 123 ms" in render_result_summary(summary)
    assert summary["status"] == result["status"]
    record_lines[1]["runtime_id"] = "other"
    record.write_text("".join(json.dumps(row) + "\n" for row in record_lines))
    summary = read_evidence(path)
    assert summary["record_error"] == "invalid_or_unreadable_sibling_record"
    assert summary["usage"]["dimensions"]["input_tokens"]["complete_total"] == 100


def test_sibling_pre_shutdown_status_cannot_override_final_result(
    tmp_path, result, record_lines
):
    result["status"] = "failed"
    path = tmp_path / "run-result.json"
    path.write_text(json.dumps(result))
    (tmp_path / "run-record.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in record_lines)
    )
    summary = read_evidence(path)
    assert (
        summary["status"] == "failed"
        and summary["execution_interval"]["elapsed_ms"] == 123
    )


def test_sibling_symlink_is_not_followed(tmp_path, result):
    path = tmp_path / "run-result.json"
    path.write_text(json.dumps(result))
    (tmp_path / "run-record.jsonl").symlink_to(tmp_path / "sensitive-not-a-record.json")
    summary = read_evidence(path)
    assert "execution_interval" not in summary and "record_error" not in summary


def test_failed_result_shows_structured_error_without_losing_usage_or_dumping_message(
    result,
):
    result["status"] = "failed"
    result["error"] = {
        "stage": "invoke",
        "code": "adapter_failure",
        "retryable": False,
        "message": "private model request details",
    }
    summary = summarize_result(result)
    text = render_result_summary(summary)
    assert summary["execution_error"] == {
        "stage": "invoke",
        "code": "adapter_failure",
        "retryable": False,
    }
    assert "adapter_failure" in text
    assert "private model request details" not in text
    assert "input 100" in text


def test_foreign_runtime_snapshot_is_rejected_without_erasing_usage(result):
    result["resource_snapshot"] = {
        "runtime_id": "other-runtime",
        "started_at": "2026-10-21T00:00:00Z",
        "observed_at": "2026-10-21T00:00:01Z",
        "resident_memory_bytes": 4096,
        "process_count": 1,
        "cpu_time_seconds": 0.25,
    }
    summary = summarize_result(result)
    assert summary["resource_snapshot"] is None
    assert summary["resource_error"] == "invalid_resource_snapshot"
    assert summary["usage"]["dimensions"]["input_tokens"]["complete_total"] == 100
    assert "Runtime process-tree snapshot" not in render_result_summary(summary)


@pytest.mark.parametrize("saved", [False, True])
def test_recording_failure_remains_visible_without_changing_execution(
    tmp_path, result, saved
):
    result["events"] = [
        {
            "kind": "record_write_failed",
            "message": "sensitive untrusted path\nINJECTED",
            "metadata": {"artifact": "run-record.jsonl"},
        }
    ]
    if saved:
        path = tmp_path / "run-result.json"
        path.write_text(json.dumps(result))
        summary = read_result(path)
    else:
        summary = summarize_result(result)
    assert summary["recording_failed"] is True
    assert summary["status"] == "succeeded"
    assert summary["usage"]["dimensions"]["input_tokens"]["complete_total"] == 100
    text = render_result_summary(summary)
    assert "Recording failed" in text
    assert "record_write_failed" in text
    assert "sensitive untrusted path" not in text
    assert "INJECTED" not in text


def test_unrelated_event_does_not_report_a_recording_failure(result):
    result["events"] = [{"kind": "model_warning", "message": "record_write_failed"}]
    summary = summarize_result(result)
    assert summary["recording_failed"] is False
    assert "Recording failed" not in render_result_summary(summary)


def test_legacy_record_does_not_invent_exchange_or_resource_observations(
    tmp_path, record_lines
):
    path = tmp_path / "run-record.jsonl"
    write_record(path, record_lines)
    summary = read_evidence(path)
    assert summary["elapsed_ms"] == 123
    assert summary["exchange_elapsed_ms"] is None
    assert summary["resource_window"] is None


def test_core_accounting_failure_event_is_visible_without_replacing_execution(result):
    result.pop("model_usage")
    result["usage"] = {"input_tokens": 100}
    result["events"] = [{"kind": "model_usage_invalid"}]
    summary = summarize_result(result)
    assert summary["status"] == "succeeded"
    assert summary["model_usage"] is None
    assert summary["accounting_error"] == "invalid_model_usage"
    assert summary["legacy_usage"] == {"input_tokens": 100}
    assert "accounting was invalid" in render_result_summary(summary)


def test_overview_separates_token_and_cost_coverage_and_keeps_exact_json(result):
    summary = summarize_result(result)
    text = render_result_summary(summary)
    assert (
        "Token coverage: input/output/total complete; cache breakdown complete" in text
    )
    assert "Tokens: input 100; output 2; total 102" in text
    assert "Provider-reported cost: $0.000000010 (complete)" in text
    assert "Catalog estimate: unavailable" in text
    assert "Invocation:" not in text
    assert summary["usage"]["dimensions"]["reported_cost"]["complete_total"] == 10


def test_overview_names_missing_token_dimensions_without_hiding_complete_output(result):
    usage = result["model_usage"]
    usage["input_tokens"] = usage["total_tokens"] = None
    usage["observed"] = {"input_tokens": 50}
    usage["gaps"].append(
        {"dimensions": ["input_tokens", "total_tokens"], "reason": "unfinished_work"}
    )
    text = render_result_summary(summarize_result(result))
    assert "input 50 (observed, incomplete)" in text
    assert "total unavailable" in text
    assert "output 2 (complete)" in text
    assert "Token coverage: input/output/total incomplete" in text
    assert "unfinished at return" in text


def test_accounted_models_are_not_presented_as_served_model_provenance(result):
    summary = summarize_result(result)
    summary["model_usage"]["models"] = [
        {"provider": "openai", "model": "main-model"},
        {"provider": "anthropic", "model": "helper\x1b[2J"},
    ]
    text = render_result_summary(summary)
    assert 'Accounted models: "openai/main-model", "anthropic/helper\\u001b[2J"' in text
    assert '"served":' not in text
    assert "\x1b" not in text


def test_reporting_cli_details_and_json_keep_independent_output_modes(
    tmp_path, result, monkeypatch, capsys
):
    from nemo_fabric.reporting import main

    path = tmp_path / "run-result.json"
    path.write_text(json.dumps(result))
    monkeypatch.setattr("sys.argv", ["reporting", str(path), "--details"])
    main()
    assert "Invocation:" in capsys.readouterr().out
    monkeypatch.setattr("sys.argv", ["reporting", str(path), "--details", "--json"])
    main()
    assert json.loads(capsys.readouterr().out)["status"] == "succeeded"


def test_missing_cache_breakdown_does_not_label_complete_token_totals_incomplete(
    result,
):
    usage = result["model_usage"]
    usage["cached_input_tokens"] = usage["cache_write_input_tokens"] = None
    usage["gaps"].append(
        {
            "dimensions": ["cached_input_tokens", "cache_write_input_tokens"],
            "reason": "missing_usage",
        }
    )
    text = render_result_summary(summarize_result(result))
    assert "Tokens: input 100; output 2; total 102" in text
    assert (
        "Token coverage: input/output/total complete; cache breakdown unavailable"
        in text
    )
    assert "Cache (included in input): reads unavailable; writes unavailable" in text


def test_catalog_identity_is_visible_without_dumping_the_basis(result):
    pricing = {
        "basis": "catalog",
        "currency": "USD",
        "normalization_version": "cache_reasoning_inclusive_v1",
        "assumption_policy": "applicable_tariff_no_missing_counts_v1",
        "bundled_catalog_sha256": "sha256:" + "a" * 64,
    }
    result["model_usage"]["cost"]["pricing"] = pricing
    result["model_usage"]["cost"]["estimated"] = {
        "amount_nano_usd": 280000,
        "observed_nano_usd": 280000,
    }
    result["model_usage"]["gaps"] = []
    summary = summarize_result(result)
    overview = render_result_summary(summary)
    detail = render_result_summary(summary, details=True)
    assert "Catalog estimate: $0.000280 (complete)" in overview
    assert "Pricing basis: catalog; bundled aaaaaaaaaaaa" in overview
    assert "applicable_tariff_no_missing_counts_v1" not in overview
    assert "applicable_tariff_no_missing_counts_v1" in detail
    assert summary["usage"]["pricing"] == pricing


def write_record(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


@pytest.mark.parametrize("partial", [False, True])
def test_record_projection_retains_normalized_accounting_without_full_result(
    tmp_path, record_lines, result, partial
):
    usage = result["model_usage"]
    if partial:
        usage["input_tokens"] = usage["total_tokens"] = None
        usage["observed"] = {"input_tokens": 50}
        usage["gaps"].append(
            {
                "dimensions": ["input_tokens", "total_tokens"],
                "reason": "unfinished_work",
            }
        )
    record_lines[1]["model_usage"] = usage
    path = tmp_path / "run-record.jsonl"
    write_record(path, record_lines)
    summary = read_evidence(path)
    assert summary["outcome_boundary"] == "invocation"
    assert summary["model_usage"]["output_tokens"] == 2
    dimension = summary["usage"]["dimensions"]["input_tokens"]
    assert dimension["complete_total"] == (None if partial else 100)
    assert dimension["observed_sum"] == (50 if partial else 100)
    from nemo_fabric.reporting import render_invocation_record_summary

    text = render_invocation_record_summary(summary)
    assert "input 50 (observed, incomplete)" in text if partial else "input 100" in text
    assert "Invocation outcome before runtime shutdown: succeeded" in text
    assert not (tmp_path / "run-result.json").exists()


def test_invalid_record_accounting_keeps_valid_timing(tmp_path, record_lines, result):
    record_lines[1]["model_usage"] = result["model_usage"]
    record_lines[1]["model_usage"]["input_tokens"] = -1
    path = tmp_path / "run-record.jsonl"
    write_record(path, record_lines)
    summary = read_evidence(path)
    assert summary["accounting_error"] == "invalid_model_usage"
    assert summary["elapsed_ms"] == 123
    assert summary["status"] == "succeeded"
    assert summary["usage"] is None


def resource_snapshot(
    cpu,
    *,
    observed="2026-10-21T00:00:01Z",
    started="2026-10-21T00:00:00Z",
    runtime="runtime",
    rss=10,
):
    return {
        "runtime_id": runtime,
        "started_at": started,
        "observed_at": observed,
        "resident_memory_bytes": rss,
        "process_count": 1,
        "cpu_time_seconds": cpu,
    }


def invocation_stats(before, after):
    return {
        "collected_groups": ["process"],
        "elapsed_seconds": 0.123,
        "before": {"status": "available", "snapshot": before},
        "after": {"status": "available", "snapshot": after},
        "cpu_time_seconds_estimate": after["cpu_time_seconds"]
        - before["cpu_time_seconds"],
    }


def test_ordinary_full_result_preserves_system_stats_without_recording(result):
    result["system_stats"] = invocation_stats(
        resource_snapshot(1), resource_snapshot(3)
    )
    summary = summarize_result(result)
    assert summary["system_stats"]["before"] == result["system_stats"]["before"]
    assert summary["system_stats"]["after"] == result["system_stats"]["after"]
    assert summary["resource_window"]["cpu_counter_change_seconds"] == 2
    text = render_result_summary(summary, details=True)
    assert "Elapsed (adapter exchange): 123 ms" in text
    assert "not invocation CPU" in text
    assert "System stats collected groups" in text


def test_process_stats_not_collected_never_implies_zero(result):
    result["system_stats"] = {
        "collected_groups": ["cgroup.cpu"],
        "elapsed_seconds": 0.123,
        "before": {"status": "not_collected"},
        "after": {"status": "not_collected"},
    }
    summary = summarize_result(result)
    assert summary["system_stats"] is not None
    assert summary["resource_window"]["cpu_counter_change_seconds"] is None
    assert summary["resource_window"]["delta_reason"] == "not_collected"


def test_invalid_system_stats_preserve_execution_and_accounting(result):
    result["system_stats"] = {"elapsed_seconds": -1}
    summary = summarize_result(result)
    assert summary["system_stats"] is None
    assert summary["system_stats_error"] == "invalid_system_stats"
    assert summary["status"] == "succeeded"
    assert summary["usage"]["dimensions"]["input_tokens"]["complete_total"] == 100


def test_details_retain_resource_scope_while_overview_stays_compact(result):
    result["system_stats"] = invocation_stats(
        resource_snapshot(1), resource_snapshot(3)
    )
    summary = summarize_result(result)
    overview = render_result_summary(summary)
    detail = render_result_summary(summary, details=True)
    assert "Elapsed (adapter exchange): 123 ms" in overview
    assert "process-tree" not in overview
    assert "not invocation CPU" in detail
    assert "not a peak" in detail
    assert "Invocation:" in detail
    assert "System stats collected groups" in detail
    assert "bytes" not in detail  # Human-readable memory uses binary units.
