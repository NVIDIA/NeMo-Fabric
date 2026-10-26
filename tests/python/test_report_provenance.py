# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
import json
from pathlib import Path

import pytest

from nemo_fabric.reporting import read_evidence, render_result_summary, summarize_result


@pytest.fixture(name="facts")
def facts_fixture():
    return json.loads(
        (
            Path(__file__).parents[1] / "fixtures/what_ran/reader-vectors.json"
        ).read_text()
    )["base"]


@pytest.fixture(name="result")
def result_fixture():
    return {
        "request_id": "request",
        "runtime_id": "runtime",
        "invocation_id": "invocation",
        "harness": "Claude Code",
        "status": "succeeded",
        "model_usage": {
            "input_tokens": 10,
            "output_tokens": 2,
            "total_tokens": 12,
            "gaps": [],
            "models": [],
            "models_exhaustive": False,
            "cost": {},
        },
    }


def test_result_provenance_shows_every_role_and_fact_source_without_usage_inference(
    result, facts
):
    roles = facts["models"]
    model = roles.pop("default")
    roles["reviewer"] = deepcopy(model)
    roles["worker"] = deepcopy(model)
    roles["reviewer"]["served"] = {
        "value": "served-review-model",
        "source": "reported",
        "collected_by": "adapter",
        "method": "adapter_report",
    }
    facts["harness"]["version"] = {
        "value": "1.2.3",
        "source": "reported",
        "collected_by": "adapter",
        "method": "adapter_report",
    }
    result["what_ran"] = facts
    summary = summarize_result(result)
    overview = render_result_summary(summary)
    details = render_result_summary(summary, details=True)
    assert "Multiple roles (2)" in overview
    assert "served-review-model" not in overview
    for text in (
        "Model roles",
        "reviewer (provider: anthropic)",
        "worker (provider: anthropic)",
        "served-review-model",
        "Unsupported (key_not_declared)",
        "reported; collected by adapter; method: adapter_report",
        "Fabric core version",
        "observed; collected by host_core; method: crate_version",
        "Not applicable (local_run)",
        "Adapter ID",
        "Execution environment",
        "MCP servers",
        "Endpoint",
        "Protocol version",
        "Tools",
    ):
        assert text in details
    assert '"requested":' not in details
    assert details.count("Model roles") == 1


def test_invalid_provenance_preserves_valid_status_and_accounting(result, facts):
    facts["fabric"]["core_version"]["state"] = "unavailable"
    result["what_ran"] = facts
    summary = summarize_result(result)
    assert summary["what_ran"] is None
    assert summary["provenance_error"] == "unreadable_what_ran"
    assert summary["status"] == "succeeded"
    assert summary["usage"]["dimensions"]["input_tokens"]["complete_total"] == 10
    assert (
        "Saved execution facts are unreadable; measurements are retained."
        in render_result_summary(summary)
    )


def test_final_unknown_facts_are_authoritative_over_sibling_and_request_configuration(
    tmp_path, result, facts
):
    fact = {
        "state": "collection_failed",
        "reason": "fact_omitted",
        "source": "reported",
        "collected_by": "adapter",
        "method": "adapter_report",
    }
    facts["models"]["default"]["requested"] = deepcopy(fact)
    facts["models"]["default"]["served"] = deepcopy(fact)
    result["what_ran"] = facts
    identity = {
        key: result[key] for key in ("request_id", "runtime_id", "invocation_id")
    }
    base = {"schema": "nemo.fabric.run_record", "version": "v1alpha1", **identity}
    earlier = deepcopy(facts)
    earlier["models"]["default"]["served"] = {
        "value": "earlier-model",
        "source": "reported",
        "collected_by": "adapter",
        "method": "adapter_report",
    }
    (tmp_path / "run-record.jsonl").write_text(
        json.dumps({**base, "kind": "start"})
        + "\n"
        + json.dumps(
            {
                **base,
                "kind": "end",
                "status": "succeeded",
                "outcome_scope": "invocation",
                "elapsed_ms": 2000,
                "what_ran": earlier,
            }
        )
        + "\n"
    )
    path = tmp_path / "run-result.json"
    path.write_text(json.dumps(result))
    summary = read_evidence(path)
    summary["requested_models"] = {"default": {"model": "earlier-configured-model"}}
    text = render_result_summary(summary, details=True)
    assert summary["what_ran"] == facts
    assert "earlier-model" not in text
    assert "earlier-configured-model" not in text
    assert "Collection failed (fact_omitted)" in text


def test_legacy_record_retains_valid_execution_facts(tmp_path, facts):
    base = {
        "schema": "nemo.fabric.run_record",
        "version": "v1alpha1",
        "request_id": "request",
        "runtime_id": "runtime",
        "invocation_id": "invocation",
    }
    path = tmp_path / "run-record.jsonl"
    path.write_text(
        json.dumps({**base, "kind": "start"})
        + "\n"
        + json.dumps(
            {
                **base,
                "kind": "end",
                "status": "succeeded",
                "outcome_scope": "invocation",
                "elapsed_ms": 2000,
                "what_ran": facts,
            }
        )
        + "\n"
    )
    summary = read_evidence(path)
    assert summary["what_ran"] == facts
    assert summary["provenance_error"] is None
    assert "Fabric core version" in render_result_summary(summary, details=True)


def test_fact_truncation_counts_and_omission_reasons_remain_visible(result, facts):
    facts["models_truncated"] = {"reason": "entry_cap", "dropped": 2}
    facts["mcp"]["servers_truncated"] = {"reason": "byte_budget", "dropped": 3}
    facts["mcp"]["servers"]["docs"]["runtime"]["tool_names"] = {
        "state": "unavailable",
        "reason": "fact_omitted",
        "source": "reported",
        "collected_by": "adapter",
        "method": "adapter_report",
    }
    result["what_ran"] = facts
    summary = summarize_result(result)
    overview = render_result_summary(summary)
    assert "Execution facts omitted: 2 model roles (entry_cap)." in overview
    assert "Execution facts omitted: 3 MCP servers (byte_budget)." in overview
    assert "Unavailable (fact_omitted)" in render_result_summary(summary, details=True)
    result["what_ran"] = {
        "schema": "nemo.fabric.what_ran",
        "version": "v1alpha1",
        "what_ran_truncated": {"reason": "byte_budget", "dropped": 1},
    }
    summary = summarize_result(result)
    assert summary["provenance_error"] is None
    assert (
        "Execution facts were truncated to fit a byte budget."
        in render_result_summary(summary)
    )


def test_unreadable_final_facts_are_not_repaired_from_legacy_sibling(
    tmp_path, result, facts
):
    base = {
        "schema": "nemo.fabric.run_record",
        "version": "v1alpha1",
        **{key: result[key] for key in ("request_id", "runtime_id", "invocation_id")},
    }
    (tmp_path / "run-record.jsonl").write_text(
        json.dumps({**base, "kind": "start"})
        + "\n"
        + json.dumps(
            {
                **base,
                "kind": "end",
                "status": "succeeded",
                "outcome_scope": "invocation",
                "elapsed_ms": 2000,
                "what_ran": facts,
            }
        )
        + "\n"
    )
    result["what_ran"] = {"schema": "unreadable"}
    path = tmp_path / "run-result.json"
    path.write_text(json.dumps(result))
    summary = read_evidence(path)
    assert summary["what_ran"] is None
    assert summary["provenance_error"] == "unreadable_what_ran"


def test_absent_optional_provenance_is_not_malformed(result):
    summary = summarize_result(result)
    assert summary["what_ran"] is None
    assert summary["provenance_error"] is None
    assert "facts are unreadable" not in render_result_summary(summary)


def test_run_journal_inspection_keeps_final_provenance(tmp_path, result, facts):
    result["what_ran"] = facts
    envelope = {
        "schema": "nemo.fabric.run",
        "version": "v1alpha1",
        "recording_id": "recording",
        "request_id": "request",
        "source": "fabric",
    }
    start = {
        **envelope,
        "sequence": 0,
        "time": "2026-10-25T19:01:10.000Z",
        "elapsed_seconds": 0,
        "kind": "request_started",
        "data": {
            "harness": "Claude Code",
            "outcome_scope": "one_shot_lifecycle",
            "models": {},
        },
    }
    end = {
        **envelope,
        "sequence": 1,
        "time": "2026-10-25T19:01:54.000Z",
        "elapsed_seconds": 44,
        "runtime_id": "runtime",
        "invocation_id": "invocation",
        "kind": "summary",
        "data": {
            "outcome_scope": "one_shot_lifecycle",
            "result": result,
            "error": None,
            "request_started_at": "2026-10-25T19:01:10.000Z",
            "request_finished_at": "2026-10-25T19:01:54.000Z",
            "elapsed_seconds": 44,
        },
    }
    path = tmp_path / "run.jsonl"
    path.write_text(json.dumps(start) + "\n" + json.dumps(end) + "\n")
    summary = read_evidence(path)
    assert summary["what_ran"] == facts
    assert summary["provenance_error"] is None
    assert summary["usage"]["dimensions"]["input_tokens"]["complete_total"] == 10
    assert "Fabric core version" in render_result_summary(summary, details=True)


def test_truncated_tool_list_shows_retained_names_and_omitted_count(result, facts):
    runtime = facts["mcp"]["servers"]["docs"]["runtime"]
    runtime["tool_names"] = {
        "value": ["read_file"],
        "source": "reported",
        "collected_by": "adapter",
        "method": "adapter_report",
    }
    runtime["tool_names_truncated"] = {"reason": "entry_cap", "dropped": 2}
    result["what_ran"] = facts
    text = render_result_summary(summarize_result(result), details=True)
    assert "read_file" in text
    assert "Tools omitted" in text
    assert "2 (entry_cap)" in text
