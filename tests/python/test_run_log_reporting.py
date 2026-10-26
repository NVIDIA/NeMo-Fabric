# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
import json

import pytest

from nemo_fabric.reporting import read_evidence, render_result_summary


@pytest.fixture(name="run_result")
def run_result_fixture():
    return {
        "request_id": "request",
        "runtime_id": "runtime",
        "invocation_id": "invocation",
        "status": "succeeded",
        "model_usage": {
            "input_tokens": 100,
            "cached_input_tokens": 0,
            "cache_write_input_tokens": 0,
            "output_tokens": 2,
            "total_tokens": 102,
            "observed": {},
            "gaps": [],
            "models": [],
            "models_exhaustive": False,
            "cost": {"reported": {"amount_nano_usd": 1}},
        },
        "artifacts": {"artifacts": []},
    }


@pytest.fixture(name="run_rows")
def run_rows_fixture(run_result):
    def row(sequence, kind, seconds, data):
        return {
            "schema": "nemo.fabric.run",
            "version": "v1alpha1",
            "recording_id": "recording",
            "request_id": "request",
            "sequence": sequence,
            "time": f"2026-10-25T19:01:{10 + int(seconds):02}.000Z",
            "elapsed_seconds": seconds,
            "source": "fabric",
            "kind": kind,
            "data": data,
            **(
                {"runtime_id": "runtime", "invocation_id": "invocation"}
                if sequence
                else {}
            ),
        }

    return [
        row(
            0,
            "request_started",
            0,
            {
                "harness": "Claude Code",
                "agent_name": "example",
                "outcome_scope": "one_shot_lifecycle",
                "models": {
                    "default": {"provider": "example", "model": "example-model"}
                },
            },
        ),
        row(1, "runtime_ready", 1, {"runtime_id": "runtime"}),
        row(2, "invocation_started", 2, {"timeout_seconds": 300}),
        row(
            3,
            "invocation_finished",
            43,
            {key: run_result[key] for key in ("status", "model_usage")},
        ),
        row(
            4,
            "summary",
            44,
            {
                "outcome_scope": "one_shot_lifecycle",
                "result": run_result,
                "error": None,
                "request_started_at": "2026-10-25T19:01:10.000Z",
                "request_finished_at": "2026-10-25T19:01:54.000Z",
                "elapsed_seconds": 44,
            },
        ),
    ]


def write_log(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_run_summary_and_timeline_keep_same_utc_times_and_seconds(tmp_path, run_rows):
    path = tmp_path / "run.jsonl"
    write_log(path, run_rows)
    summary = read_evidence(path)
    assert summary["schema"] == "nemo.fabric.invocation_summary"
    assert summary["usage"]["dimensions"]["input_tokens"]["complete_total"] == 100
    text = render_result_summary(summary)
    assert "19:01:10.000" in text and "19:01:54.000" in text
    assert "44.000 s" in text
    assert "example-model" in text
    assert "Token coverage" not in text and "Accounting: " not in text
    assert "$0.000000001" in text
    assert "2026-10-25T" not in text
    details = render_result_summary(summary, details=True)
    assert (
        "Execution history" in details and "Invocation finished: succeeded" in details
    )
    assert "43.000" in details
    assert details.count("Input tokens") == 1


def test_cross_midnight_summary_keeps_each_date(tmp_path, run_rows):
    run_rows[-1]["data"].update(
        request_started_at="2026-10-25T23:59:59.000Z",
        request_finished_at="2026-10-26T00:00:43.000Z",
    )
    path = tmp_path / "run.jsonl"
    write_log(path, run_rows)
    text = render_result_summary(read_evidence(path))
    assert "2026-10-25 23:59:59.000" in text
    assert "2026-10-26 00:00:43.000" in text


def test_missing_final_summary_retains_invocation_accounting_without_outer_success(
    tmp_path, run_rows
):
    path = tmp_path / "run.jsonl"
    write_log(path, run_rows[:-1])
    with path.open("a") as stream:
        stream.write('{"kind":')
    summary = read_evidence(path)
    assert summary["schema"] == "nemo.fabric.invocation_record_summary"
    assert summary["state"] == "record_only"
    assert summary["truncated_tail"]
    assert summary["usage"]["dimensions"]["input_tokens"]["complete_total"] == 100
    assert "Completion unconfirmed" in render_result_summary(summary)


def test_final_unknown_usage_is_not_repaired_from_invocation_event(tmp_path, run_rows):
    result = deepcopy(run_rows[-1]["data"]["result"])
    result.pop("model_usage")
    run_rows[-1]["data"]["result"] = result
    path = tmp_path / "run.jsonl"
    write_log(path, run_rows)
    summary = read_evidence(path)
    assert summary["model_usage"] is None
    assert summary["usage"]["dimensions"]["input_tokens"]["complete_total"] is None


def test_final_no_result_retains_error_and_independent_invocation_evidence(
    tmp_path, run_rows
):
    final = run_rows[-1]["data"]
    final["result"] = None
    final["error"] = {
        "stage": "stop",
        "code": "shutdown_failed",
        "retryable": False,
        "message": "private diagnostic",
    }
    path = tmp_path / "run.jsonl"
    write_log(path, run_rows)
    summary = read_evidence(path)
    assert summary["schema"] == "nemo.fabric.request_summary"
    assert summary["state"] == "no_result"
    assert (
        summary["invocation_summary"]["usage"]["dimensions"]["input_tokens"][
            "complete_total"
        ]
        == 100
    )
    text = render_result_summary(summary)
    assert "No result returned" in text
    assert "shutdown_failed" in text and "stop" in text
    assert "private diagnostic" not in text


@pytest.mark.parametrize(
    "mutation",
    [
        "sequence",
        "identity",
        "result_identity",
        "duplicate_final",
        "after_final",
        "naive_time",
        "negative_elapsed",
        "reverse_elapsed",
        "nonfinite",
        "malformed_complete",
    ],
)
def test_conflicting_or_malformed_complete_run_evidence_is_rejected(
    tmp_path, run_rows, mutation
):
    if mutation == "sequence":
        run_rows[1]["sequence"] = 4
    elif mutation == "identity":
        run_rows[2]["request_id"] = "other"
    elif mutation == "result_identity":
        run_rows[-1]["data"]["result"]["runtime_id"] = "other"
    elif mutation in {"duplicate_final", "after_final"}:
        row = deepcopy(run_rows[-1])
        row["sequence"] += 1
        if mutation == "after_final":
            row["kind"] = "diagnostic"
        run_rows.append(row)
    elif mutation == "naive_time":
        run_rows[1]["time"] = "2026-10-25T19:01:11"
    elif mutation == "negative_elapsed":
        run_rows[0]["elapsed_seconds"] = -1
    elif mutation == "reverse_elapsed":
        run_rows[3]["elapsed_seconds"] = 0
    elif mutation == "nonfinite":
        run_rows[1]["elapsed_seconds"] = float("inf")
    path = tmp_path / "run.jsonl"
    write_log(path, run_rows)
    if mutation == "malformed_complete":
        with path.open("a") as stream:
            stream.write('{"kind":\n')
    with pytest.raises(ValueError):
        read_evidence(path)


def test_journal_streams_files_larger_than_one_result_limit(tmp_path, run_rows):
    path = tmp_path / "run.jsonl"
    rows = run_rows[:-1]
    for sequence in range(4, 304):
        rows.append(
            {
                **run_rows[0],
                "sequence": sequence,
                "elapsed_seconds": 43,
                "kind": "resource_sample",
                "data": {"diagnostic": "x" * 60_000},
            }
        )
    final = deepcopy(run_rows[-1])
    final["sequence"] = len(rows)
    rows.append(final)
    write_log(path, rows)
    assert path.stat().st_size > 16 * 1024 * 1024
    summary = read_evidence(path)
    assert summary["terminal_summary_present"]
    assert all(row["kind"] != "resource_sample" for row in summary["history"])


def test_oversized_record_is_explicit_error_without_modifying_file(tmp_path, run_rows):
    path = tmp_path / "run.jsonl"
    run_rows[-1]["data"]["result"]["output"] = "x" * (16 * 1024 * 1024)
    write_log(path, run_rows)
    original_size = path.stat().st_size
    with pytest.raises(ValueError, match="per-record limit"):
        read_evidence(path)
    assert path.stat().st_size == original_size


def test_diagnostics_and_names_cannot_inject_terminal_controls(tmp_path, run_rows):
    run_rows[0]["data"]["harness"] = "agent\x1b[2J\nforged"
    diagnostic = {
        **run_rows[2],
        "sequence": 3,
        "kind": "diagnostic",
        "data": {"message": "message\x1b[2J\nforged"},
    }
    run_rows.insert(3, diagnostic)
    for sequence, row in enumerate(run_rows):
        row["sequence"] = sequence
    path = tmp_path / "run.jsonl"
    write_log(path, run_rows)
    text = render_result_summary(read_evidence(path), details=True)
    assert "\x1b" not in text
    assert "\\u001b[2J\\nforged" in text


@pytest.mark.parametrize(
    "models",
    ["invalid", {"default": "invalid"}, {"default": {"model": "missing-provider"}}],
)
def test_malformed_request_model_shapes_are_rejected(tmp_path, run_rows, models):
    run_rows[0]["data"]["models"] = models
    path = tmp_path / "run.jsonl"
    write_log(path, run_rows)
    with pytest.raises(ValueError, match="model|provider"):
        read_evidence(path)


@pytest.mark.parametrize("seconds", [True, 10**400])
def test_invalid_elapsed_values_have_explicit_validation_errors(
    tmp_path, run_rows, seconds
):
    run_rows[1]["elapsed_seconds"] = seconds
    path = tmp_path / "run.jsonl"
    write_log(path, run_rows)
    with pytest.raises(ValueError, match="seconds"):
        read_evidence(path)


def test_history_does_not_duplicate_terminal_result_or_full_diagnostics(
    tmp_path, run_rows
):
    run_rows[-1]["data"]["result"]["output"] = {"text": "x" * 100000}
    diagnostic = {
        **run_rows[2],
        "sequence": 3,
        "kind": "diagnostic",
        "data": {"message": "m" * 5000},
    }
    run_rows.insert(3, diagnostic)
    for sequence, row in enumerate(run_rows):
        row["sequence"] = sequence
    path = tmp_path / "run.jsonl"
    write_log(path, run_rows)
    summary = read_evidence(path)
    assert "result" not in summary["history"][-1]["data"]
    assert "model_usage" not in summary["history"][-2]["data"]
    message = summary["history"][3]["data"]
    assert len(message["message"]) == 4096
    assert message["message_truncated"]
    assert len(json.dumps(summary["history"])) < 10000
    assert "message shortened" in render_result_summary(summary, details=True)


def test_pricing_details_show_actual_rates_and_basis_without_overview_noise(run_result):
    from nemo_fabric.reporting import summarize_result

    fingerprint = "sha256:" + "a" * 64
    cost = run_result["model_usage"]["cost"]
    cost["pricing"] = {
        "basis": "catalog",
        "currency": "USD",
        "normalization_version": "cache_reasoning_inclusive_v1",
        "assumption_policy": "applicable_tariff_no_missing_counts_v1",
        "bundled_catalog_sha256": fingerprint,
    }
    cost["estimated"] = {"amount_nano_usd": 1000}
    cost["applied_rates"] = [
        {
            "catalog_sha256": fingerprint,
            "provider": "example",
            "model_id": "example-model",
            "pricing_as_of": "2026-10-25",
            "pricing_source": "https://example.com/pricing",
            "rates": {
                "input": 5_000_000_000,
                "output": 8_000_000_000,
                "cache_read": 1_000_000_000,
            },
        }
    ]
    summary = summarize_result(run_result)
    overview = render_result_summary(summary)
    details = render_result_summary(summary, details=True)
    assert "sha256" not in overview
    assert "USD per 1,000,000 tokens" in details
    assert "Uncached input: $5.00" in details
    assert "Cache reads: $1.00" in details
    assert "Output: $8.00" in details
    assert "2026-10-25" in details and "https://example.com/pricing" in details


def test_recorded_provider_cost_does_not_claim_to_be_an_estimate(run_result):
    from nemo_fabric.reporting import summarize_result

    cost = run_result["model_usage"]["cost"]["reported"]
    cost.update(amount_nano_usd=None, observed_nano_usd=1)
    run_result["model_usage"]["gaps"] = [
        {"dimensions": ["reported_cost"], "reason": "unfinished_work"}
    ]
    text = render_result_summary(summarize_result(run_result))
    provider_row = next(
        line for line in text.splitlines() if line.startswith("Provider-reported cost")
    )
    assert "$0.000000001 recorded; total unknown" in provider_row
    assert "estimate" not in provider_row


def test_runtime_snapshot_supplies_memory_but_never_cumulative_cpu(run_result):
    from nemo_fabric.reporting import summarize_result

    run_result["resource_snapshot"] = {
        "runtime_id": "runtime",
        "started_at": "2026-10-25T19:01:10Z",
        "observed_at": "2026-10-25T19:01:54Z",
        "resident_memory_bytes": 1024,
        "process_count": 1,
        "cpu_time_seconds": 100,
    }
    text = render_result_summary(summarize_result(run_result))
    memory = next(
        line for line in text.splitlines() if line.startswith("Memory after execution")
    )
    cpu = next(
        line for line in text.splitlines() if line.startswith("CPU time, estimated")
    )
    assert "1.0 KiB" in memory
    assert "Unavailable" in cpu


def test_multiple_nondefault_roles_are_not_presented_as_one_model(tmp_path, run_rows):
    run_rows[0]["data"]["models"] = {
        "planner": {"provider": "example", "model": "planner-model"},
        "coder": {"provider": "example", "model": "coder-model"},
    }
    path = tmp_path / "run.jsonl"
    write_log(path, run_rows)
    summary = read_evidence(path)
    text = render_result_summary(summary)
    assert "Multiple roles (2)" in text
    assert "planner-model" not in text and "coder-model" not in text
    details = render_result_summary(summary, details=True)
    assert "planner: requested planner-model" in details
    assert "coder: requested coder-model" in details


def test_unretained_model_override_does_not_turn_config_into_requested_fact(
    tmp_path, run_rows
):
    run_rows[0]["data"]["override_keys"] = ["models"]
    path = tmp_path / "run.jsonl"
    write_log(path, run_rows)
    text = render_result_summary(read_evidence(path))
    assert "Unavailable (model override not retained)" in text
    assert "example-model" not in text


@pytest.mark.parametrize("keys", [None, 1, [None]])
def test_malformed_override_key_lists_are_rejected(tmp_path, run_rows, keys):
    run_rows[0]["data"]["override_keys"] = keys
    path = tmp_path / "run.jsonl"
    write_log(path, run_rows)
    with pytest.raises(ValueError, match="override_keys"):
        read_evidence(path)


def test_shutdown_failure_is_visible_even_after_successful_invocation(
    tmp_path, run_rows
):
    error = {"stage": "stop", "code": "shutdown_failed", "retryable": False}
    run_rows.insert(
        -1,
        {
            **run_rows[-1],
            "kind": "runtime_stopped",
            "data": {"succeeded": False, "error": error},
        },
    )
    run_rows[-1]["data"]["result"].update(status="failed", error=error)
    for sequence, row in enumerate(run_rows):
        row["sequence"] = sequence
    path = tmp_path / "run.jsonl"
    write_log(path, run_rows)
    text = render_result_summary(read_evidence(path), details=True)
    assert "Invocation finished: succeeded" in text
    assert "Agent runtime shutdown failed: stop/shutdown_failed" in text
    assert "Execution:" in text and "Failed" in text


def test_diagnostic_retention_does_not_hide_terminal_milestones(tmp_path, run_rows):
    diagnostics = [
        {**run_rows[2], "kind": "diagnostic", "data": {"message": "progress"}}
        for _ in range(300)
    ]
    run_rows[3:3] = diagnostics
    for sequence, row in enumerate(run_rows):
        row["sequence"] = sequence
    path = tmp_path / "run.jsonl"
    write_log(path, run_rows)
    summary = read_evidence(path)
    assert summary["history_omitted_count"] > 0
    assert len(summary["history"]) < 265
    text = render_result_summary(summary, details=True)
    assert "Invocation finished: succeeded" in text
    assert "Additional history entries retained in file:" in text


def test_streamed_history_has_byte_budget_and_keeps_terminal_evidence(
    tmp_path, run_rows
):
    diagnostics = [
        {
            **run_rows[2],
            "kind": "diagnostic",
            "data": {
                "message": "\U0001f680" * 4096,
                "level": "L" * 4000,
                "severity": "S" * 4000,
                "stream": "stderr",
                "scope": "runtime",
            },
        }
        for _ in range(40)
    ]
    # Future kinds keep their envelope; their unknown payload is not copied.
    future = {
        **run_rows[2],
        "kind": "future_event",
        "data": {"status": "X" * (2 * 1024 * 1024)},
    }
    run_rows[3:3] = [future, *diagnostics]
    stopped = {
        **run_rows[-2],
        "kind": "runtime_stopped",
        "data": {"succeeded": False, "error": {"stage": "stop", "code": "E" * 100_000}},
    }
    run_rows.insert(-1, stopped)
    final_result = run_rows[-1]["data"]["result"]
    final_result["status"] = "failed"
    final_result["error"] = {"stage": "stop", "code": "shutdown_failed"}
    for sequence, row in enumerate(run_rows):
        row["sequence"] = sequence
    path = tmp_path / "run.jsonl"
    write_log(path, run_rows)
    summary = read_evidence(path)
    assert len(json.dumps(summary["history"]).encode("utf-8")) <= 1024 * 1024
    assert summary["history_omitted_count"] > 0
    assert summary["history_projection_truncated_count"] > 0
    assert summary["status"] == "failed"
    assert summary["execution_error"]["code"] == "shutdown_failed"
    assert summary["usage"]["dimensions"]["input_tokens"]["complete_total"] == 100
    assert [row["kind"] for row in summary["history"]][-3:] == [
        "invocation_finished",
        "runtime_stopped",
        "summary",
    ]
    future_projection = next(
        row for row in summary["history"] if row["kind"] == "future_event"
    )
    assert future_projection["data"] == {}
    stopped_projection = summary["history"][-2]
    assert len(stopped_projection["data"]["error"]["code"]) == 256
    text = render_result_summary(summary, details=True)
    assert "history details shortened; full event retained in file" in text
    assert "Agent runtime shutdown failed" in text


def test_diagnostic_retention_sums_markers_after_the_display_history_is_full(
    tmp_path, run_rows
):
    events = [
        {
            **run_rows[2],
            "kind": "diagnostic",
            "data": {"stream": "stderr", "scope": "runtime", "message": "progress"},
        }
        for _ in range(260)
    ]
    events.extend(
        {
            **run_rows[2],
            "kind": "diagnostic_retention",
            "data": {"omitted_bytes": count, "read_error": error},
        }
        for count, error in ((5, None), (6, "read_failed"))
    )
    run_rows[3:3] = events
    for sequence, row in enumerate(run_rows):
        row["sequence"] = sequence
    path = tmp_path / "run.jsonl"
    write_log(path, run_rows)
    summary = read_evidence(path)
    assert summary["diagnostics"] == {
        "omitted_bytes": 11,
        "read_errors": ["read_failed"],
    }
    assert summary["history_omitted_count"] > 0
    assert summary["terminal_summary_present"]
    assert summary["usage"]["dimensions"]["input_tokens"]["complete_total"] == 100
    text = render_result_summary(summary)
    assert "Diagnostic capture omitted 11 bytes." in text
    assert "Diagnostic capture error: read_failed" in text
    assert "diagnostic retention" not in render_result_summary(summary, details=True)


def test_diagnostic_retention_errors_are_bounded_and_zero_loss_is_quiet(
    tmp_path, run_rows
):
    from nemo_fabric.reporting import MAX_HISTORY_MESSAGE_CHARS

    markers = [
        {
            **run_rows[2],
            "kind": "diagnostic_retention",
            "data": {"omitted_bytes": 0, "read_error": None},
        }
    ]
    path = tmp_path / "run.jsonl"
    run_rows[3:3] = markers
    for sequence, row in enumerate(run_rows):
        row["sequence"] = sequence
    write_log(path, run_rows)
    summary = read_evidence(path)
    assert summary["diagnostics"] == {"omitted_bytes": 0, "read_errors": []}
    assert "Diagnostic capture" not in render_result_summary(summary)

    markers = [
        {
            **run_rows[2],
            "kind": "diagnostic_retention",
            "data": {
                "omitted_bytes": 0,
                "read_error": "e" * (MAX_HISTORY_MESSAGE_CHARS + 1),
            },
        }
        for _ in range(17)
    ]
    run_rows[3:3] = markers
    for sequence, row in enumerate(run_rows):
        row["sequence"] = sequence
    write_log(path, run_rows)
    summary = read_evidence(path)
    assert summary["diagnostics"]["omitted_bytes"] == 0
    assert (
        summary["diagnostics"]["read_errors"] == ["e" * MAX_HISTORY_MESSAGE_CHARS] * 16
    )
