# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[2] / ".agents/skills/qualify-harbor/scripts"),
)
import qualify  # noqa: E402
from prepare import adapter_dockerfile  # noqa: E402

pytestmark = pytest.mark.skipif(
    os.name != "posix",
    reason="Qualification uses POSIX process groups and Linux task images",
)


@pytest.fixture(name="qualification_plan")
def qualification_plan_fixture() -> dict:
    return qualify.make_plan(
        argparse.Namespace(
            task=qualify.ROOT / "examples/harbor/calculator/task",
            mode="source",
            version=qualify.load(qualify.ROOT / qualify.CATALOG)["catalog_version"],
            harbor_version="0.24.0",
            bridge="native",
            model="nvidia/local-model",
            base_url="http://127.0.0.1:8000/v1",
            api_key_env="MODEL_API_KEY",
            endpoint_kind="openai-and-anthropic",
            trusted_http_host=[],
            estimated_cost_usd=0,
            cost_basis="local",
            timeout=60,
        )
    )


@pytest.fixture(name="clean_qualification_checkout")
def clean_qualification_checkout_fixture():
    real_run = subprocess.run

    def run_with_clean_checkout(command, *args, **kwargs):
        if command[:2] == ["git", "diff"] or command[1:] == [
            str(qualify.ROOT / "scripts/ci/generate_adapter_catalog.py"),
            "--check",
        ]:
            return subprocess.CompletedProcess(command, 0)
        return real_run(command, *args, **kwargs)

    with patch.object(qualify.subprocess, "run", side_effect=run_with_clean_checkout):
        yield


@pytest.mark.usefixtures("clean_qualification_checkout")
def test_plan_uses_complete_catalog_and_distinguishes_prerequisites(qualification_plan):
    qualify.validate_plan(qualification_plan)
    catalog = qualify.load(qualify.ROOT / qualify.CATALOG)
    rows = {row["adapter_id"]: row for row in qualification_plan["adapters"]}
    assert set(rows) == set(catalog["adapters"])
    assert rows["nvidia.fabric.claude"]["action"] == "run"
    assert rows["nvidia.fabric.kiro"]["action"] == "blocked"
    assert rows["nvidia.fabric.kiro"]["api_key_env"] == "KIRO_API_KEY"
    assert "BYOM is unsupported" in rows["nvidia.fabric.kiro"]["reason"]
    assert rows["nvidia.fabric.remote-agent"]["reason"]
    assert rows["nvidia.fabric.pi"]["capabilities"]["mcp"] == "untested"


@pytest.fixture(name="workflow_row")
def workflow_row_fixture(qualification_plan):
    qualification_plan["bridge"] = "sdk"
    row = next(
        r
        for r in qualification_plan["adapters"]
        if r["adapter_id"] == "nvidia.fabric.nooa"
    )
    row.update(
        workflow_target_id="nvidia.nooa.coding-agent",
        workflow_settings={},
        action="run",
        reason="",
    )
    return row


@pytest.mark.usefixtures("clean_qualification_checkout")
def test_workflow_selection_keeps_the_full_catalog(qualification_plan, workflow_row):
    qualify.validate_plan(qualification_plan)
    assert len(qualification_plan["adapters"]) == 18
    assert workflow_row["action"] == "run"
    assert "nvidia.nooa.coding-agent" in workflow_row["registered_targets"]


@pytest.mark.usefixtures("clean_qualification_checkout")
@pytest.mark.parametrize(
    "mutation",
    [
        "native",
        "empty",
        "spaces",
        "type",
        "missing",
        "wrong_adapter",
        "settings_without_target",
        "settings_type",
        "settings_secret",
        "nonfinite",
        "target_drift",
    ],
)
def test_workflow_selection_rejects_unattributable_routes(
    qualification_plan, workflow_row, mutation
):
    if mutation == "native":
        qualification_plan["bridge"] = "native"
    elif mutation == "empty":
        workflow_row["workflow_target_id"] = ""
    elif mutation == "spaces":
        workflow_row["workflow_target_id"] = " nvidia.nooa.coding-agent"
    elif mutation == "type":
        workflow_row["workflow_target_id"] = 1
    elif mutation == "missing":
        workflow_row["workflow_target_id"] = "not.registered"
    elif mutation == "wrong_adapter":
        other = next(
            r
            for r in qualification_plan["adapters"]
            if r["adapter_id"] == "nvidia.fabric.nooa.bench-agent"
        )
        other.update(
            workflow_target_id=workflow_row.pop("workflow_target_id"),
            workflow_settings=workflow_row.pop("workflow_settings"),
        )
        workflow_row.update(action="unsupported", reason="Not selected")
    elif mutation == "settings_without_target":
        workflow_row.pop("workflow_target_id")
    elif mutation == "settings_type":
        workflow_row["workflow_settings"] = []
    elif mutation == "settings_secret":
        workflow_row["workflow_settings"] = {"nested": {"api_key": "secret"}}
    elif mutation == "nonfinite":
        workflow_row["workflow_settings"] = {"value": float("nan")}
    else:
        qualification_plan["mode"] = "registry"
        qualification_plan["catalog_snapshot"]["targets"][
            workflow_row["workflow_target_id"]
        ]["descriptor"]["spec"]["entrypoint"]["ref"] = "changed:factory"
    with pytest.raises(ValueError):
        qualify.validate_plan(qualification_plan)


@pytest.mark.usefixtures("requires_harbor", "clean_qualification_checkout")
def test_workflow_command_payload_and_report_agree(
    qualification_plan, workflow_row, tmp_path
):
    command = qualify.harbor_command(
        qualification_plan, workflow_row, "python", tmp_path, tmp_path, "trial"
    )
    assert "fabric_workflow_target_id=nvidia.nooa.coding-agent" in command
    assert "fabric_workflow_settings={}" in command
    payload = qualify.selection_payload(qualification_plan, workflow_row)
    assert payload["config"]["workflow"] == {
        "target_id": workflow_row["workflow_target_id"]
    }
    target = qualification_plan["catalog_snapshot"]["targets"][
        workflow_row["workflow_target_id"]
    ]
    combined = qualify.admission_fingerprint(
        payload, workflow_row["catalog"]["descriptor"], target["descriptor"]
    )
    assert isinstance(combined, str) and len(combined) == 64
    changed = copy.deepcopy(target["descriptor"])
    changed["spec"]["entrypoint"]["ref"] = "different:factory"
    assert (
        qualify.admission_fingerprint(
            payload, workflow_row["catalog"]["descriptor"], changed
        )
        != combined
    )
    assert (
        qualify.run(
            qualification_plan,
            {"plan": qualification_plan},
            tmp_path / "report",
            [workflow_row["adapter_id"]],
        )
        == 1
    )
    report = qualify.load(tmp_path / "report/report.json")
    selected = next(
        r["selected"]
        for r in report["adapters"]
        if r["adapter_id"] == workflow_row["adapter_id"]
    )
    assert selected["workflow_target_id"] == workflow_row["workflow_target_id"]
    assert selected["workflow_settings"] == {}
    assert len(report["adapters"]) == 18


@pytest.mark.parametrize(
    "mutation", [None, "missing", "factory", "source", "target_id"]
)
def test_workflow_provenance_requires_installed_matching_target(
    qualification_plan, workflow_row, mutation
):
    target = qualification_plan["catalog_snapshot"]["targets"][
        workflow_row["workflow_target_id"]
    ]
    selected = {
        "adapter_id": workflow_row["adapter_id"],
        "python_prefix": "/usr/local",
        "provenance": [
            {
                "source": "installed_package",
                "path": "/usr/local/share/nemo-fabric/adapters/nooa/nooa.fabric-adapter.json",
            }
        ],
        "workflow": {
            "target_id": workflow_row["workflow_target_id"],
            "entrypoint": copy.deepcopy(target["descriptor"]["spec"]["entrypoint"]),
            "provenance": [
                {
                    "source": "installed_package",
                    "path": "/usr/local/share/nemo-fabric/adapters/nooa/targets/coding-agent.fabric-target.json",
                }
            ],
        },
    }
    if mutation == "missing":
        selected.pop("workflow")
    elif mutation == "factory":
        selected["workflow"]["entrypoint"]["ref"] = "other:factory"
    elif mutation == "source":
        selected["workflow"]["provenance"][0]["source"] = "bundled"
    elif mutation == "target_id":
        selected["workflow"]["target_id"] = "other.target"
    assert qualify.valid_selection(selected, workflow_row, target) is (mutation is None)


@pytest.mark.usefixtures("requires_harbor")
def test_workflow_settings_are_transported_and_admitted_by_target_schema(
    qualification_plan, workflow_row, tmp_path
):
    from nemo_fabric import FabricConfigError

    workflow_row.update(
        workflow_target_id="nvidia.nooa.arc-solver",
        workflow_settings={"reflect_every": 3},
    )
    command = qualify.harbor_command(
        qualification_plan, workflow_row, "python", tmp_path, tmp_path, "trial"
    )
    assert 'fabric_workflow_settings={"reflect_every": 3}' in command
    payload = qualify.selection_payload(qualification_plan, workflow_row)
    assert payload["config"]["workflow"]["settings"] == {"reflect_every": 3}
    target = qualification_plan["catalog_snapshot"]["targets"][
        workflow_row["workflow_target_id"]
    ]
    assert qualify.admission_fingerprint(
        payload, workflow_row["catalog"]["descriptor"], target["descriptor"]
    )
    payload["config"]["workflow"]["settings"] = {"reflect_every": 0}
    with pytest.raises(FabricConfigError):
        qualify.admission_fingerprint(
            payload, workflow_row["catalog"]["descriptor"], target["descriptor"]
        )


@pytest.mark.usefixtures("requires_harbor")
def test_workflow_identity_retains_target_without_importing_factory(
    qualification_plan, workflow_row, monkeypatch
):
    from identity import selection

    target = qualification_plan["catalog_snapshot"]["targets"][
        workflow_row["workflow_target_id"]
    ]
    payload = qualify.selection_payload(qualification_plan, workflow_row)
    payload["config"]["discovery"] = {
        "local_paths": [str(qualify.ROOT / "adapters/python/nooa")]
    }
    monkeypatch.setitem(
        sys.modules, "nemo_fabric_adapters.nooa.targets.coding_agent", None
    )
    selected = selection(payload)
    assert selected["workflow"]["target_id"] == workflow_row["workflow_target_id"]
    assert (
        selected["workflow"]["entrypoint"] == target["descriptor"]["spec"]["entrypoint"]
    )
    assert selected["workflow"]["provenance"]
    assert selected["admission_sha256"] == qualify.admission_fingerprint(
        payload, workflow_row["catalog"]["descriptor"], target["descriptor"]
    )


@pytest.mark.usefixtures("clean_qualification_checkout")
def test_registry_prerelease_keeps_python_and_npm_versions_distinct(qualification_plan):
    plan = copy.deepcopy(qualification_plan)
    base = plan["version"]
    plan["mode"] = "registry"
    plan["version"] = base + "rc1"
    plan["catalog_snapshot"]["catalog_version"] = plan["version"]
    for row in plan["adapters"]:
        row["catalog"]["package"]["version"] = base + (
            "rc1" if row["catalog"]["ecosystem"] == "python" else "-rc.1"
        )
    qualify.validate_plan(plan)
    row = next(
        row for row in plan["adapters"] if row["adapter_id"] == "nvidia.fabric.pi"
    )
    assert f"nemo-fabric-adapters-pi@{base}-rc.1" in adapter_dockerfile(
        plan, row, "base:exact"
    )
    plan["mode"] = "source"
    with pytest.raises(ValueError, match="unmodified catalog"):
        qualify.validate_plan(plan)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "duplicate",
        "descriptor",
        "source",
        "version",
        "credential_url",
        "untrusted_http",
        "inline_secret",
        "checkpoint",
        "catalog_extra",
    ],
)
@pytest.mark.usefixtures("clean_qualification_checkout")
def test_plan_rejects_incomplete_or_unattributable_coverage(
    qualification_plan, mutation
):
    plan = copy.deepcopy(qualification_plan)
    row = plan["adapters"][0]
    if mutation == "missing":
        plan["adapters"].pop()
    elif mutation == "duplicate":
        plan["adapters"].append(row)
    elif mutation == "descriptor":
        row["catalog"]["sha256"] = "0" * 64
    elif mutation == "source":
        plan["source_sha"] = "0" * 40
    elif mutation == "version":
        plan["version"] = "999.0.0"
    elif mutation == "credential_url":
        row["base_url"] = "https://user:secret@example.com/v1"
    elif mutation == "untrusted_http":
        row["base_url"] = "http://10.0.0.1:8000/v1"
    elif mutation == "inline_secret":
        row["harness_settings"] = {"config_overrides": {"api_key": "literal-secret"}}
    elif mutation == "checkpoint":
        row["model"] = "/mnt/checkpoint"
    else:
        plan["mode"] = "registry"
        plan["catalog_snapshot"]["adapters"]["extra.adapter"] = copy.deepcopy(
            row["catalog"]
        )
    with pytest.raises(ValueError):
        qualify.validate_plan(plan)


@pytest.mark.parametrize("outcome", ["blocked", "failed", "unsupported", "excluded"])
def test_nonpasses_never_turn_full_catalog_green(tmp_path, qualification_plan, outcome):
    report = {
        "scope": "full_catalog",
        "plan": qualification_plan,
        "adapters": [
            {"adapter_id": "one", "outcome": "passed"},
            {
                "adapter_id": "two",
                "outcome": outcome,
                "reason": "coverage gap",
            },
        ],
    }
    qualify.write_report(tmp_path, report)
    assert qualify.load(tmp_path / "report.json")["full_catalog_passed"] is False
    assert "denominator: 2" in (tmp_path / "summary.md").read_text()


def test_subset_is_never_a_full_catalog_pass(tmp_path, qualification_plan):
    report = {
        "scope": "developer_subset",
        "plan": qualification_plan,
        "adapters": [{"adapter_id": "one", "outcome": "passed"}],
    }
    qualify.write_report(tmp_path, report)
    assert report["all_harnesses_passed"] is True
    assert report["full_catalog_passed"] is False


@pytest.mark.parametrize("state", ["completed", "running", "cancelled"])
def test_full_catalog_pass_requires_completed_run(tmp_path, qualification_plan, state):
    report = {
        "scope": "full_catalog",
        "state": state,
        "plan": qualification_plan,
        "adapters": [{"adapter_id": "one", "outcome": "passed"}],
    }
    qualify.write_report(tmp_path, report)
    assert report["full_catalog_passed"] is (state == "completed")
    assert "release_qualified" not in report


def test_credential_is_transported_as_reference_and_logs_are_redacted(
    tmp_path, qualification_plan
):
    os.environ["MODEL_API_KEY"] = "qualification-secret-value"
    row = qualification_plan["adapters"][0]
    command = qualify.harbor_command(
        qualification_plan, row, "python", tmp_path, tmp_path, "trial"
    )
    assert "MODEL_API_KEY=${MODEL_API_KEY}" in command
    assert "qualification-secret-value" not in " ".join(command)
    log = tmp_path / "run.log"
    code = qualify.execute(
        [sys.executable, "-c", "import os; print(os.environ['MODEL_API_KEY'])"],
        log,
        10,
        key_env="MODEL_API_KEY",
    )
    assert code == 0
    assert "qualification-secret-value" not in log.read_text()
    assert "[REDACTED]" in log.read_text()
    assert "MODEL_API_KEY" not in qualify.process_env()


def test_timeout_terminates_process_and_retains_partial_log(tmp_path):
    log = tmp_path / "timeout.log"
    code = qualify.execute(
        [sys.executable, "-u", "-c", "import time; print('started'); time.sleep(60)"],
        log,
        1,
    )
    assert code == 124
    assert "started" in log.read_text()


def test_interrupt_terminates_only_its_process_group(tmp_path):
    with (
        patch("qualify.subprocess.Popen", autospec=True) as mock_popen,
        patch("qualify.os.killpg") as kill,
    ):
        process = mock_popen.return_value
        process.pid = 12345
        process.stdout = iter([])
        process.wait.side_effect = [KeyboardInterrupt, 0]
        assert qualify.execute(["test-process"], tmp_path / "cancel.log", 10) == 130
    kill.assert_called_once_with(process.pid, qualify.signal.SIGINT)


def test_cleanup_does_not_remove_another_jobs_resources(tmp_path):
    with (
        patch("qualify.subprocess.run") as run,
        patch("qualify.subprocess.check_output") as output,
        patch("qualify.execute") as execute,
    ):
        run.side_effect = [
            subprocess.CompletedProcess([], 0, stdout="probe"),
            subprocess.CompletedProcess([], 0, stdout="owned unrelated"),
            subprocess.CompletedProcess([], 0, stdout="owned unrelated"),
        ]
        output.side_effect = [
            json.dumps({"com.docker.compose.project": "fq-unique-0__trial"}),
            json.dumps({"com.docker.compose.project": "another-user"}),
            json.dumps({"com.docker.compose.project": "fq-unique-0__trial"}),
            json.dumps({"com.docker.compose.project": "another-user"}),
        ]
        execute.return_value = 0
        assert qualify.cleanup("fq-unique", tmp_path / "cleanup.log") is True
        assert [call.args[0] for call in execute.call_args_list] == [
            ["docker", "rm", "-f", "probe"],
            ["docker", "rm", "-f", "owned"],
            ["docker", "network", "rm", "owned"],
        ]


@pytest.mark.parametrize(
    "message",
    [
        "Provider List: nvidia; ModuleNotFoundError: No module named 'mcp'",
        "APIConnectionError: Nvidia_nimException - No module named 'mcp'",
    ],
)
def test_missing_dependency_is_not_classified_as_a_provider_failure(message):
    assert qualify.classify(message) == "install"


@pytest.fixture(name="qualification_job")
def qualification_job_fixture(tmp_path, qualification_plan):
    from nemo_fabric.integrations.harbor.fabric_agent import build_harbor_config

    row = next(
        row
        for row in qualification_plan["adapters"]
        if row["adapter_id"] == "nvidia.fabric.codex"
    )
    trial = tmp_path / "trial"
    (trial / "agent").mkdir(parents=True)
    (trial / "verifier").mkdir()
    qualify.save(
        tmp_path / "result.json",
        {"stats": {"n_completed_trials": 1, "n_errored_trials": 0}},
    )
    result = {
        "status": "succeeded",
        "adapter_id": row["adapter_id"],
        "harness": "codex",
        "agent_name": "qualification",
        "adapter_kind": "python",
        "runtime_id": "runtime",
        "invocation_id": "invocation",
        "request_id": "request",
        "output": {"response": "fixed"},
        "events": [],
        "metadata": {},
        "artifacts": {"artifacts": []},
        "telemetry": [],
        "usage": {
            "input_tokens": 12,
            "input_tokens_include_cache": True,
            "output_tokens": 3,
        },
        "error": None,
    }
    qualify.save(trial / "agent/fabric-result-one.json", result)
    spec = {
        "config": build_harbor_config(
            adapter_id=row["adapter_id"],
            workspace="/app",
            model_name=row["model"],
            model_base_url=row["base_url"],
            model_api_key_env=row["api_key_env"],
        ).to_mapping()
    }
    spec["adapter_descriptor_sha256"] = qualify.admission_fingerprint(
        spec, row["catalog"]["descriptor"]
    )
    qualify.save(trial / "agent/fabric-run-one.json", spec)
    (trial / "verifier/reward.txt").write_text("1\n")
    identity = {"editable": [], "packages": {"nemo-fabric-runtime": "0.6.0"}}
    qualify.save(trial / "verifier/qualification-identity.json", identity)
    selection = {
        "adapter_id": row["adapter_id"],
        "python_prefix": "/usr/local",
        "provenance": [
            {
                "source": "installed_package",
                "path": "/usr/local/share/nemo-fabric/adapters/codex/codex.fabric-adapter.json",
            }
        ],
    }
    qualify.save(trial / "verifier/qualification-selection.json", selection)
    qualify.save(
        trial / "result.json",
        {
            "agent_result": {
                "n_input_tokens": 12,
                "n_output_tokens": 3,
                "metadata": {
                    "fabric": {
                        "adapter_id": row["adapter_id"],
                        "invocation_id": "invocation",
                    }
                },
            }
        },
    )
    return tmp_path, row, identity, result, selection


@pytest.mark.parametrize(
    "mutation,category",
    [
        ("none", None),
        ("reward", None),
        ("missing_reward", "harbor_verifier"),
        ("invalid_reward", "harbor_verifier"),
        ("nonfinite_reward", "harbor_verifier"),
        ("identity", "adapter_admission"),
        ("usage", None),
        ("counter", "usage"),
        ("source_origin", "adapter_admission"),
        ("fingerprint", "adapter_admission"),
        ("missing_result", "transport_runner"),
        ("runner_error", "adapter_admission"),
    ],
)
@pytest.mark.usefixtures("requires_harbor")
def test_result_checks_remain_independent(qualification_job, mutation, category):
    job, row, identity, result, selection = qualification_job
    if mutation == "reward":
        (job / "trial/verifier/reward.txt").write_text("0")
    elif mutation == "missing_reward":
        (job / "trial/verifier/reward.txt").unlink()
    elif mutation == "invalid_reward":
        (job / "trial/verifier/reward.txt").write_text("invalid")
    elif mutation == "nonfinite_reward":
        (job / "trial/verifier/reward.txt").write_text("nan")
    elif mutation == "identity":
        identity = {"editable": ["runtime"]}
    elif mutation == "usage":
        result["usage"] = None
        qualify.save(job / "trial/agent/fabric-result-one.json", result)
        trial = qualify.load(job / "trial/result.json")
        trial["agent_result"]["n_input_tokens"] = None
        trial["agent_result"]["n_output_tokens"] = None
        qualify.save(job / "trial/result.json", trial)
    elif mutation == "counter":
        trial = qualify.load(job / "trial/result.json")
        trial["agent_result"]["n_input_tokens"] = 999
        qualify.save(job / "trial/result.json", trial)
    elif mutation == "source_origin":
        observed = copy.deepcopy(selection)
        observed["provenance"][0] = {
            "source": "bundled",
            "path": "/src/adapters/python/codex/codex.fabric-adapter.json",
        }
        qualify.save(job / "trial/verifier/qualification-selection.json", observed)
    elif mutation == "fingerprint":
        spec = qualify.load(job / "trial/agent/fabric-run-one.json")
        spec["adapter_descriptor_sha256"] = "wrong"
        qualify.save(job / "trial/agent/fabric-run-one.json", spec)
    elif mutation == "missing_result":
        (job / "trial/agent/fabric-result-one.json").unlink()
    elif mutation == "runner_error":
        qualify.save(
            job / "trial/agent/fabric-result-one.json",
            {"runner_error": {"message": "descriptor fingerprint mismatch"}},
        )
    summary = qualify.inspect_results(job, row, identity, selection)
    assert {failure["category"] for failure in summary["failures"]} == (
        {category} if category else set()
    )


@pytest.mark.parametrize("known_usage", [True, False])
@pytest.mark.parametrize("transport_matches", [True, False])
@pytest.mark.usefixtures("requires_harbor")
def test_task_failure_and_usage_limit_do_not_mask_transport_failure(
    qualification_job, qualification_plan, known_usage, transport_matches
):
    job, row, identity, result, selection = qualification_job
    (job / "trial/verifier/reward.txt").write_text("0")
    if not known_usage:
        result["usage"] = None
        qualify.save(job / "trial/agent/fabric-result-one.json", result)
    trial = qualify.load(job / "trial/result.json")
    trial["agent_result"]["n_input_tokens"] = 12 if known_usage else None
    trial["agent_result"]["n_output_tokens"] = 3 if known_usage else None
    if not transport_matches:
        trial["agent_result"]["n_input_tokens"] = 999
    qualify.save(job / "trial/result.json", trial)
    summary = qualify.inspect_results(job, row, identity, selection)
    assert summary["checks"]["task_success"] is False
    assert summary["checks"]["verifier_completed"] is True
    assert summary["checks"]["usage_available"] is known_usage
    assert bool(summary["limitations"]) is not known_usage
    assert {f["category"] for f in summary["failures"]} == (
        set() if transport_matches else {"usage"}
    )
    summary.update(
        adapter_id=row["adapter_id"],
        outcome="failed" if summary["failures"] else "passed",
    )
    report = {
        "scope": "developer_subset",
        "plan": qualification_plan,
        "adapters": [summary],
    }
    qualify.write_report(job, report)
    assert report["all_harnesses_passed"] is transport_matches
    assert report["full_catalog_passed"] is False
    text = (job / "summary.md").read_text()
    assert "Task reward" in text
    assert "| 0.0 |" in text


@pytest.mark.usefixtures("requires_harbor")
def test_admission_compares_normalized_descriptor_not_raw_catalog_hash(
    qualification_job,
):
    job, row, identity, _, selection = qualification_job
    row["catalog"]["sha256"] = "0" * 64
    summary = qualify.inspect_results(job, row, identity, selection)
    assert summary["checks"]["descriptor_fingerprint"] is True
    assert summary["failures"] == []


def test_source_task_uses_packed_installs_and_exact_runtime(qualification_plan):
    row = next(
        row
        for row in qualification_plan["adapters"]
        if row["adapter_id"] == "nvidia.fabric.pi"
    )
    dockerfile = adapter_dockerfile(qualification_plan, row, "qualification-base:exact")
    assert "npm pack" in dockerfile
    assert "npm install --prefix /opt/harness /packs/*.tgz" in dockerfile
    assert "@earendil-works/pi-ai@1.0.3" in dockerfile
    assert "-e /src" not in dockerfile


def test_droid_install_uses_pinned_native_cli_and_sdk(qualification_plan):
    row = next(
        r
        for r in qualification_plan["adapters"]
        if r["adapter_id"] == "nvidia.fabric.droid"
    )
    dockerfile = adapter_dockerfile(qualification_plan, row, "qualification-base:exact")
    assert "npm install --global droid@0.233.0" in dockerfile
    assert "@factory/droid-sdk@0.9.1" in dockerfile


@pytest.mark.usefixtures("clean_qualification_checkout")
@pytest.mark.parametrize("mode", ["source", "registry"])
def test_openhands_install_uses_the_documented_compatible_sdk_stack(
    qualification_plan, mode
):
    qualification_plan["mode"] = mode
    row = next(
        row
        for row in qualification_plan["adapters"]
        if row["adapter_id"] == "nvidia.fabric.openhands"
    )
    dockerfile = adapter_dockerfile(qualification_plan, row, "qualification-base:exact")
    assert (
        'pip install "openhands-sdk==1.50.0" "openhands-tools==1.50.0" '
        '"litellm==1.100.1" "mcp==1.28.1" && pip check'
    ) in dockerfile
    assert '"grpcio==' not in dockerfile
    assert '"protobuf==' not in dockerfile


@pytest.mark.usefixtures("clean_qualification_checkout")
def test_failed_host_environment_preserves_complete_report(
    tmp_path, qualification_plan
):
    output = tmp_path / "report"
    assert (
        qualify.run(
            qualification_plan, {"plan": qualification_plan, "adapters": {}}, output, []
        )
        == 1
    )
    report = qualify.load(output / "report.json")
    assert len(report["adapters"]) == len(qualification_plan["adapters"])
    for row, planned in zip(report["adapters"], qualification_plan["adapters"]):
        if planned["action"] == "run":
            assert row["outcome"] == "blocked"
            assert row["reason"].startswith("Host installation unavailable")
        else:
            assert row["outcome"] == planned["action"]
            assert row["reason"] == planned["reason"]
    assert report["full_catalog_passed"] is False


@pytest.mark.usefixtures("clean_qualification_checkout")
@pytest.mark.parametrize("key_present", [False, True])
def test_kiro_missing_native_key_skips_before_task_install_or_launch(
    tmp_path, qualification_plan, monkeypatch, key_present
):
    row = next(
        r
        for r in qualification_plan["adapters"]
        if r["adapter_id"] == "nvidia.fabric.kiro"
    )
    row.update(action="run", reason="", model="kiro/advertised-model", base_url=None)
    monkeypatch.setenv("MODEL_API_KEY", "dummy-local-model-key")
    if key_present:
        monkeypatch.setenv("KIRO_API_KEY", "native-test-key")
    else:
        monkeypatch.delenv("KIRO_API_KEY", raising=False)
    identity = {
        "editable": [],
        "packages": {
            "nemo-fabric-runtime": qualification_plan["version"],
            "harbor": qualification_plan["harbor_version"],
        },
        "catalog_sha256": qualify.fingerprint(qualify.plan_catalog(qualification_plan)),
        "wheel_sha256": {"nemo-fabric-runtime": "test-wheel-hash"},
    }
    environment = {
        "plan": qualification_plan,
        "python": sys.executable,
        "wheel_sha256": {"nemo_fabric_runtime-test.whl": "test-wheel-hash"},
        "adapters": {},
    }
    monkeypatch.setattr(qualify, "execute", lambda *args, **kwargs: 0)
    real_check_output = qualify.subprocess.check_output

    def check_output(command, *args, **kwargs):
        if str(command[-1]).endswith("identity.py"):
            return json.dumps(identity)
        return real_check_output(command, *args, **kwargs)

    monkeypatch.setattr(qualify.subprocess, "check_output", check_output)
    output = tmp_path / "report"
    assert (
        qualify.run(qualification_plan, environment, output, [row["adapter_id"]]) == 1
    )
    report = qualify.load(output / "report.json")
    kiro = next(r for r in report["adapters"] if r["adapter_id"] == row["adapter_id"])
    assert kiro["outcome"] == ("failed" if key_present else "blocked")
    assert kiro["failures"][0]["category"] == ("install" if key_present else "auth")
    if not key_present:
        assert kiro["reason"] == "Missing credential environment variable KIRO_API_KEY"
    assert len(report["adapters"]) == len(qualification_plan["adapters"])
    assert report["full_catalog_passed"] is False
    assert not (output / str(qualification_plan["adapters"].index(row))).exists()


@pytest.mark.usefixtures("clean_qualification_checkout")
def test_single_task_is_pinned_and_dataset_roots_are_rejected(
    tmp_path, qualification_plan
):
    with pytest.raises(ValueError, match="single Harbor task missing"):
        qualify.task_identity(tmp_path)
    task = tmp_path / "task"
    qualify.stage_task(qualification_plan, task, "prepared:exact")
    wrapper = (task / "tests/test.sh").read_text()
    assert wrapper.index("qualification-identity.json") < wrapper.index(
        "bash /tests/upstream-test.sh"
    )
    assert not (task / "solution").exists()
    assert "docker_image" not in (task / "task.toml").read_text()
    assert (task / "environment/Dockerfile").read_text() == "FROM prepared:exact\n"
    assert (task / "tests/upstream-test.sh").read_bytes() == (
        Path(qualification_plan["task"]["path"]) / "tests/test.sh"
    ).read_bytes()
    changed = copy.deepcopy(qualification_plan)
    changed["task"]["sha256"] = "wrong"
    with pytest.raises(ValueError, match="task changed"):
        qualify.validate_plan(changed)


@pytest.mark.parametrize("changed", ["nemo-fabric-runtime", "openhands-sdk"])
def test_task_dependencies_can_change_but_fabric_and_harness_identity_cannot(changed):
    expected = {
        "editable": [],
        "packages": {
            "nemo-fabric-runtime": "0.6.0",
            "openhands-sdk": "1.50.0",
            "protobuf": "6.33.0",
        },
    }
    observed = copy.deepcopy(expected)
    observed["packages"]["grpcio-tools"] = "1.73.0"
    assert qualify.unchanged_install(expected, observed)
    observed["packages"]["protobuf"] = "5.0.0"
    assert qualify.unchanged_install(expected, observed)
    observed["packages"][changed] = "different"
    assert not qualify.unchanged_install(expected, observed)


@pytest.mark.parametrize(
    "mutation", [None, "fabric_editable", "dirty", "revision", "unrecorded"]
)
def test_pinned_hermes_source_install_is_allowed_without_editable_fabric(mutation):
    revision = "a" * 40
    expected = {
        "editable": ["hermes-agent"],
        "editable_sources": {"hermes-agent": {"revision": revision, "dirty": False}},
        "harness_sources": {"hermes-agent": revision},
        "packages": {"nemo-fabric-runtime": "0.6.0", "hermes-agent": "0.0.0"},
    }
    observed = copy.deepcopy(expected)
    if mutation == "fabric_editable":
        observed["editable"].append("nemo-fabric-runtime")
    elif mutation == "dirty":
        observed["editable_sources"]["hermes-agent"]["dirty"] = True
    elif mutation == "revision":
        observed["editable_sources"]["hermes-agent"]["revision"] = "b" * 40
    elif mutation == "unrecorded":
        observed["harness_sources"] = {}
    assert qualify.unchanged_install(expected, observed) is (mutation is None)


@pytest.mark.usefixtures("clean_qualification_checkout")
def test_task_interpreter_reaches_runner_and_identity_without_rewriting_verifier(
    qualification_plan, tmp_path
):
    row = next(
        r
        for r in qualification_plan["adapters"]
        if r["adapter_id"] == "nvidia.fabric.hermes"
    )
    row["task_python"] = "/opt/hermes-runtime/bin/python"
    qualify.validate_plan(qualification_plan)
    command = qualify.harbor_command(
        qualification_plan, row, "host-python", tmp_path, tmp_path, "trial"
    )
    assert "fabric_python=/opt/hermes-runtime/bin/python" in command
    task = tmp_path / "task"
    qualify.stage_task(qualification_plan, task, "prepared:exact", row["task_python"])
    wrapper = (task / "tests/test.sh").read_text()
    assert "/opt/hermes-runtime/bin/python /qualification/identity.py" in wrapper
    assert "bash /tests/upstream-test.sh" in wrapper
    assert (task / "tests/upstream-test.sh").read_bytes() == (
        Path(qualification_plan["task"]["path"]) / "tests/test.sh"
    ).read_bytes()


def test_dirty_checkout_is_rejected(tmp_path, qualification_plan):
    repository = tmp_path / "repo"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    (repository / "crates").mkdir()
    probe = repository / "crates/probe"
    probe.write_text("committed\n")
    subprocess.run(["git", "add", "crates/probe"], cwd=repository, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Qualification test",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "core.hooksPath=/dev/null",
            "commit",
            "-q",
            "-m",
            "fixture",
        ],
        cwd=repository,
        check=True,
    )
    plan = copy.deepcopy(qualification_plan)
    plan["source_sha"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True
    ).strip()
    probe.write_text("dirty\n")
    catalog = qualify.ROOT / qualify.CATALOG
    with (
        patch.object(qualify, "ROOT", repository),
        patch.object(qualify, "CATALOG", catalog),
    ):
        with pytest.raises(subprocess.CalledProcessError) as error:
            qualify.validate_plan(plan)
    assert error.value.cmd[:2] == ["git", "diff"]


def test_report_redaction_preserves_types_and_custom_credentials(
    tmp_path, qualification_plan
):
    os.environ["MODEL_API_KEY"] = "true"
    os.environ["AWS_ACCESS_KEY_ID"] = "fake-access-key-id"
    os.environ["AZURE_STORAGE_CONNECTION_STRING"] = "fake-connection-string"
    os.environ["BROKER_CREDENTIAL_HANDLE"] = "opaque-custom-secret"
    os.environ["DOCKER_AUTH_CONFIG"] = (
        '{"auths":{"registry":{"auth":"encoded-secret"}}}'
    )
    report = {
        "scope": "full_catalog",
        "state": "completed",
        "plan": qualification_plan,
        "adapters": [
            {
                "adapter_id": "one",
                "outcome": "passed",
                "selected": {"api_key_env": "BROKER_CREDENTIAL_HANDLE"},
                "reason": "opaque-custom-secret fake-access-key-id fake-connection-string "
                + os.environ["DOCKER_AUTH_CONFIG"],
                "nested": [True, False, None, 1, "true"],
            }
        ],
    }
    qualify.write_report(tmp_path, report)
    text = (tmp_path / "report.json").read_text()
    redacted = json.loads(text)
    assert redacted["full_catalog_passed"] is True
    assert redacted["adapters"][0]["nested"] == [True, False, None, 1, "[REDACTED]"]
    assert "fake-access-key-id" not in text
    assert "fake-connection-string" not in text
    assert "opaque-custom-secret" not in text
    assert os.environ["DOCKER_AUTH_CONFIG"] not in text
    assert "opaque-custom-secret" not in (tmp_path / "summary.md").read_text()


def test_custom_credential_is_redacted_before_log_write(tmp_path):
    os.environ["BROKER_CREDENTIAL_HANDLE"] = "opaque-custom-secret"
    log = tmp_path / "process.log"
    assert (
        qualify.execute(
            [
                sys.executable,
                "-c",
                "import os; print(os.environ['BROKER_CREDENTIAL_HANDLE'])",
            ],
            log,
            10,
            key_env="BROKER_CREDENTIAL_HANDLE",
        )
        == 0
    )
    assert "opaque-custom-secret" not in log.read_text()
    assert "[REDACTED]" in log.read_text()


@pytest.mark.parametrize(
    "text,category",
    [
        ("status=401", "auth"),
        ("HTTP 403", "auth"),
        ("HTTP 404", "provider_model"),
        ("HTTP 429", "infrastructure"),
        ("HTTP 502", "infrastructure"),
        ("HTTP 503", "infrastructure"),
        ("run-1401-1403-1404-1429-1502-1503", "transport_runner"),
        ("id_401_403_404_429_502_503", "transport_runner"),
    ],
)
def test_http_codes_do_not_match_inside_identifiers(text, category):
    assert qualify.classify(text) == category


@pytest.mark.parametrize("suffix", [".json", ".jsonl"])
def test_artifact_redaction_preserves_machine_readable_evidence(suffix):
    os.environ["MODEL_API_KEY"] = "true"
    original = {"flag": True, "value": None, "credential": "true"}
    text = json.dumps(original) + "\n"
    if suffix == ".jsonl":
        text += json.dumps({"flag": False, "count": 1}) + "\n"
    result = qualify.sanitize_artifact(text, suffix, ("MODEL_API_KEY",))
    objects = [json.loads(line) for line in result.splitlines()]
    assert objects[0] == {"flag": True, "value": None, "credential": "[REDACTED]"}
    if suffix == ".jsonl":
        assert objects[1] == {"flag": False, "count": 1}
    os.environ["MODEL_API_KEY"] = "different-secret"
    assert qualify.sanitize_artifact(text, suffix, ("MODEL_API_KEY",)) == text


@pytest.mark.parametrize("suffix", [".jsonl", ".ndjson"])
def test_jsonl_redaction_keeps_unicode_separators_inside_records(suffix):
    secret = "prefix\u2028suffix"
    os.environ["MODEL_API_KEY"] = secret
    records = [{"credential": secret, "flag": True}, {"text": "other\u2028text"}]
    original = "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n"
    cleaned = qualify.sanitize_artifact(original, suffix, ("MODEL_API_KEY",))
    assert secret not in cleaned
    assert cleaned.endswith("\n")
    parsed = [json.loads(line) for line in cleaned.split("\n") if line]
    assert parsed == [{"credential": "[REDACTED]", "flag": True}, records[1]]


@pytest.mark.usefixtures("clean_qualification_checkout")
@pytest.mark.parametrize("provider", ["", " ", " openai", "openai ", 1])
def test_provider_override_rejects_invalid_values(qualification_plan, provider):
    qualification_plan["adapters"][0]["model_provider"] = provider
    with pytest.raises(ValueError, match="model_provider must be"):
        qualify.validate_plan(qualification_plan)


@pytest.mark.usefixtures("clean_qualification_checkout")
@pytest.mark.parametrize("max_turns", [0, -1, True, 1.5, "10"])
def test_rejects_invalid_turn_bound(qualification_plan, max_turns):
    qualification_plan["adapters"][0]["max_turns"] = max_turns
    with pytest.raises(ValueError, match="max_turns must be"):
        qualify.validate_plan(qualification_plan)


@pytest.mark.usefixtures("clean_qualification_checkout")
def test_model_settings_require_sdk_and_reject_credentials(qualification_plan):
    row = qualification_plan["adapters"][0]
    row["model_settings"] = {"cost_tracking": "disabled"}
    with pytest.raises(ValueError, match="model_settings requires"):
        qualify.validate_plan(qualification_plan)
    qualification_plan["bridge"] = "sdk"
    row["model_settings"] = {"api_key": "credential"}
    with pytest.raises(ValueError, match="credentials belong"):
        qualify.validate_plan(qualification_plan)


@pytest.mark.usefixtures("clean_qualification_checkout")
def test_model_settings_and_bound_match_command_selection_and_report(
    qualification_plan, tmp_path
):
    qualification_plan["bridge"] = "sdk"
    row = next(
        r
        for r in qualification_plan["adapters"]
        if r["adapter_id"] == "nvidia.fabric.mini-swe-agent"
    )
    row.update(model_settings={"cost_tracking": "disabled"}, max_turns=100)
    qualify.validate_plan(qualification_plan)
    command = qualify.harbor_command(
        qualification_plan, row, "python", tmp_path, tmp_path, "trial"
    )
    assert 'fabric_model_settings={"cost_tracking": "disabled"}' in command
    assert "fabric_max_turns=100" in command
    payload = qualify.selection_payload(qualification_plan, row)
    assert payload["config"]["models"]["default"]["settings"] == row["model_settings"]
    assert payload["config"]["runtime"]["max_turns"] == 100
    assert (
        qualify.run(
            qualification_plan,
            {"plan": qualification_plan},
            tmp_path / "out",
            [row["adapter_id"]],
        )
        == 1
    )
    report = qualify.load(tmp_path / "out/report.json")
    selected = next(
        r["selected"]
        for r in report["adapters"]
        if r["adapter_id"] == row["adapter_id"]
    )
    assert selected["model_settings"] == row["model_settings"]
    assert selected["max_turns"] == 100


@pytest.mark.usefixtures("clean_qualification_checkout")
def test_provider_override_preserves_model_in_command_selection_and_report(
    qualification_plan, tmp_path, monkeypatch
):
    from nemo_fabric.integrations.harbor import fabric_agent

    row = next(
        row
        for row in qualification_plan["adapters"]
        if row["adapter_id"] == "nvidia.fabric.qwen"
    )
    row["model_provider"] = "openai"
    model = row["model"]
    command = qualify.harbor_command(
        qualification_plan, row, "python", tmp_path, tmp_path, "trial"
    )
    assert command[command.index("--model") + 1] == model
    assert "fabric_model_provider=openai" in command

    original = fabric_agent.build_harbor_config
    received = {}

    def compatible_bridge(**kwargs):
        # Emulate the optional argument while keeping this gate usable with older SDKs.
        received.update(kwargs)
        provider = kwargs.pop("model_provider_name")
        config = original(**kwargs)
        config.models["default"].provider = provider
        return config

    monkeypatch.setattr(fabric_agent, "build_harbor_config", compatible_bridge)
    payload = qualify.selection_payload(qualification_plan, row)
    assert received["model_provider_name"] == "openai"
    assert payload["config"]["models"]["default"]["provider"] == "openai"
    assert payload["config"]["models"]["default"]["model"] == model

    output = tmp_path / "qualification"
    assert (
        qualify.run(
            qualification_plan,
            {"plan": qualification_plan},
            output,
            [row["adapter_id"]],
        )
        == 1
    )
    report = qualify.load(output / "report.json")
    selected = next(
        item for item in report["adapters"] if item["adapter_id"] == row["adapter_id"]
    )["selected"]
    assert selected["provider"] == "openai"
    assert selected["model"] == model
