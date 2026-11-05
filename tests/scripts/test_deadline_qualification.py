# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Release gate rejects incomplete, mismatched or changed qualification evidence."""

import importlib.util
import itertools
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "deadline_qualification",
    ROOT / ".agents/skills/qualify-runtime-deadlines/scripts/qualify.py",
)
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


@pytest.fixture(name="report")
def report_fixture(tmp_path):
    revision = "a" * 40
    install = dict(revision=revision, hosts={}, wheels={})
    for index in range(len(gate.PROJECTS)):
        wheel = tmp_path / f"package-{index}.whl"
        wheel.write_bytes(b"wheel")
        install["wheels"][str(wheel)] = gate.digest(wheel)
    for harness in gate.HARNESSES:
        host = tmp_path / harness
        host.mkdir()
        modules = {}
        for index in range(7):
            module = host / f"module-{index}.py"
            module.write_text("# module")
            modules[str(index)] = str(module)
        install["hosts"][harness] = dict(
            python=str(host / "bin/python"),
            modules=modules,
            sha256={path: gate.digest(path) for path in modules.values()},
            direct_url={},
        )
    installation = tmp_path / "installation.json"
    gate.save(installation, install)
    rows = []
    for harness, location, mode in itertools.product(
        gate.HARNESSES, gate.LOCATIONS, gate.CASES
    ):
        row = dict(
            harness=harness,
            location=location,
            mode=mode,
            status="passed",
            cleanup="passed",
            timeout_seconds={
                "smoke": 180,
                "finite": 900,
                "unlimited": None,
                "timeout": 15,
            }[mode],
            elapsed_seconds=16 if mode == "timeout" else 335,
            expected_receipt="test-receipt",
        )
        if mode == "timeout":
            row["error"] = dict(code="timeout", stage="invoke")
        else:
            row["result"] = dict(
                status="succeeded",
                output=dict(
                    completed=True, response="test-receipt answer() returns 41"
                ),
            )
        rows.append(row)
    inference = tmp_path / "inference.json"
    gate.save(inference, dict(records=rows))
    report = dict(
        schema_version=1,
        revision=revision,
        status="passed",
        cases=list(gate.CASES),
        locations=list(gate.LOCATIONS),
        installation=str(installation),
        inference=str(inference),
        artifacts={},
    )
    for name, tests in {
        "transport": [
            "test_live_invocations_outlast_300_seconds_with_finite_and_no_timeout",
            "test_live_finite_deadline_keeps_timeout_reason",
        ],
        "inference": ["test_code_review_deadline_matrix"],
    }.items():
        path = tmp_path / f"{name}.xml"
        path.write_text(
            "<testsuites><testsuite>"
            + "".join(f'<testcase name="{test}"/>' for test in tests)
            + "</testsuite></testsuites>"
        )
        report[name + "_junit"] = str(path)
    report["artifacts"] = {
        report[key]: gate.digest(report[key])
        for key in ("installation", "inference", "transport_junit", "inference_junit")
    }
    return report


def test_full_report_passes_only_for_the_selected_revision(report):
    assert gate.validate_report(report, report["revision"], release=True)
    with pytest.raises(ValueError, match="revision mismatch"):
        gate.validate_report(report, "b" * 40, release=True)


@pytest.mark.parametrize(
    "fault",
    ["missing", "duplicate", "cleanup", "short_long_run", "wrong_timeout", "receipt"],
)
def test_release_rejects_invalid_matrix_evidence(report, fault):
    path = Path(report["inference"])
    rows = json.loads(path.read_text())["records"]
    if fault == "missing":
        rows.pop()
    elif fault == "duplicate":
        rows[-1] = rows[0]
    elif fault == "cleanup":
        rows[0]["cleanup"] = "failed"
    elif fault == "short_long_run":
        next(row for row in rows if row["mode"] == "finite")["elapsed_seconds"] = 320
    elif fault == "wrong_timeout":
        next(row for row in rows if row["mode"] == "timeout")["error"]["code"] = "other"
    else:
        rows[0]["result"]["output"]["response"] = "41"
    gate.save(path, dict(records=rows))
    report["artifacts"][str(path)] = gate.digest(path)
    with pytest.raises(ValueError):
        gate.validate_report(report, report["revision"], release=True)


def test_developer_subset_does_not_pass_release_gate(report):
    report["cases"] = ["smoke"]
    with pytest.raises(ValueError, match="subsets"):
        gate.validate_report(report, report["revision"], release=True)


@pytest.mark.parametrize(
    "fault",
    [
        "artifact_changed",
        "hash_missing",
        "editable_sdk",
        "wheel_changed",
        "skipped_test",
    ],
)
def test_release_rejects_invalid_installation_or_test_evidence(report, fault):
    if fault == "artifact_changed":
        Path(report["inference"]).write_text("{}")
    elif fault == "hash_missing":
        report["artifacts"].clear()
    elif fault == "skipped_test":
        path = Path(report["transport_junit"])
        path.write_text(path.read_text().replace("/>", "><skipped/></testcase>", 1))
        report["artifacts"][str(path)] = gate.digest(path)
    else:
        path = Path(report["installation"])
        installation = json.loads(path.read_text())
        if fault == "editable_sdk":
            installation["hosts"]["hermes"]["direct_url"] = {
                "dir_info": {"editable": True}
            }
            gate.save(path, installation)
            report["artifacts"][str(path)] = gate.digest(path)
        else:
            Path(next(iter(installation["wheels"]))).write_bytes(b"changed")
    with pytest.raises(ValueError):
        gate.validate_report(report, report["revision"], release=True)
