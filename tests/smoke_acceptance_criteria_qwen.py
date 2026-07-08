#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E acceptance smoke for react-optimize and calculator-optimize parity against Qwen IGW.

Runs the eval datasets from nemo-platform acceptance-criteria examples through the
Fabric langchain-react adapter with native tool calling enabled on Qwen.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NEMO_PLATFORM = ROOT.parent / "nemo-platform"
AGENT = ROOT / "examples" / "react-optimize-agent"
COMMAND = ("cargo", "run", "-q", "-p", "fabric-cli", "--")
LANGCHAIN_SRC = ROOT / "adapters" / "langchain-react" / "src"
if str(LANGCHAIN_SRC) not in sys.path:
    sys.path.insert(0, str(LANGCHAIN_SRC))

from nemo_fabric_adapters.langchain_react.react.text import remove_r1_think_tags  # noqa: E402

REACT_DATA = (
    NEMO_PLATFORM / "plugins/nemo-agents/examples/react-agent/react-eval-data.json"
)
CALC_DATA = (
    NEMO_PLATFORM
    / "plugins/nemo-agents/examples/calculator-agent/src/calculator_agent/calculator-eval-data.json"
)


@dataclass(frozen=True)
class EvalRow:
    row_id: str
    question: str
    answer_criteria: str
    expect_any: tuple[str, ...]
    expect_all: tuple[str, ...] = ()


@dataclass
class RowResult:
    row: EvalRow
    response: str
    status: str
    failed: bool
    error: str | None
    passed: bool
    detail: str
    atif_paths: list[str]


def _react_expectations() -> dict[str, EvalRow]:
    return {
        "1": EvalRow("1", "", "Bell + current time", ("bell", "graham"), ("telephone",)),
        "2": EvalRow("2", "", "Paris + day of week", ("paris",), ("france",)),
        "3": EvalRow("3", "", "1915/relativity + date", ("1915", "relativity", "einstein"), ()),
    }


def _calc_expectations() -> dict[str, EvalRow]:
    return {
        "1": EvalRow("1", "", "21 + hour comparison", ("21",), ()),
        "2": EvalRow("2", "", "20 + day comparison", ("20",), ()),
        "3": EvalRow("3", "", "15 + minute comparison", ("15",), ()),
    }


def main() -> None:
    if os.environ.get("RUN_FABRIC_ACCEPTANCE_QWEN") != "1":
        print("skipping: set RUN_FABRIC_ACCEPTANCE_QWEN=1 to run")
        return

    for path in (REACT_DATA, CALC_DATA):
        if not path.is_file():
            raise FileNotFoundError(f"missing acceptance dataset: {path}")

    react_rows = _rows_from_json(REACT_DATA, _react_expectations())
    calc_rows = _rows_from_json(CALC_DATA, _calc_expectations())
    env = _python_env()
    react_results = _run_suite(
        name="react-optimize",
        rows=react_rows,
        profiles=("qwen-igw-local", "qwen-react-native", "relay"),
        env=env,
    )
    calc_results = _run_suite(
        name="calculator-optimize",
        rows=calc_rows,
        profiles=("qwen-igw-local", "qwen-calculator-native", "relay"),
        env=env,
    )

    _print_report("react-optimize", react_results)
    _print_report("calculator-optimize", calc_results)

    failures = [r for r in react_results + calc_results if not r.passed]
    if failures:
        raise SystemExit(f"{len(failures)} acceptance row(s) failed")

    print("acceptance criteria qwen e2e passed (react-optimize + calculator-optimize)")


def _rows_from_json(path: Path, expectations: dict[str, EvalRow]) -> list[EvalRow]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: list[EvalRow] = []
    for item in payload:
        row_id = str(item["id"])
        template = expectations[row_id]
        rows.append(
            EvalRow(
                row_id=row_id,
                question=item["question"],
                answer_criteria=item.get("answer", template.answer_criteria),
                expect_any=template.expect_any,
                expect_all=template.expect_all,
            )
        )
    return rows


def _run_suite(
    *,
    name: str,
    rows: list[EvalRow],
    profiles: tuple[str, ...],
    env: dict[str, str],
) -> list[RowResult]:
    results: list[RowResult] = []
    for row in rows:
        profile_args = [arg for profile in profiles for arg in ("--profile", profile)]
        result = call_json(
            "run",
            AGENT,
            *profile_args,
            "--input",
            row.question,
            env=env,
        )
        output = result.get("output") or {}
        response = str(output.get("response") or "")
        failed = bool(output.get("failed"))
        error = output.get("error")
        passed, detail = _score_response(row, response)
        atif_paths = _promoted_atif_paths(result)
        if result.get("status") != "succeeded":
            passed = False
            detail = f"fabric status={result.get('status')}"
        if failed:
            passed = False
            detail = f"adapter failed: {error}"
        if not atif_paths:
            passed = False
            detail = "missing promoted relay_atif artifact"
        results.append(
            RowResult(
                row=row,
                response=response,
                status=str(result.get("status")),
                failed=failed,
                error=str(error) if error else None,
                passed=passed,
                detail=detail,
                atif_paths=atif_paths,
            )
        )
        mark = "PASS" if passed else "FAIL"
        print(f"[{name}] row {row.row_id} {mark}: {detail}")
    return results


def _score_response(row: EvalRow, response: str) -> tuple[bool, str]:
    text = _normalize(response)
    if not text.strip():
        return False, "empty response"
    if row.expect_all and not all(token in text for token in row.expect_all):
        missing = [t for t in row.expect_all if t not in text]
        return False, f"missing required tokens: {missing}"
    if row.expect_any and not any(token in text for token in row.expect_any):
        return False, f"none of {row.expect_any} found in response"
    return True, "matched acceptance heuristics"


def _promoted_atif_paths(result: dict) -> list[str]:
    artifacts = (result.get("artifacts") or {}).get("artifacts") or []
    paths = [artifact.get("path") for artifact in artifacts if artifact.get("kind") == "atif"]
    return [str(path) for path in paths if path]


def _normalize(text: str) -> str:
    text = remove_r1_think_tags(text)
    return " ".join(text.split()).lower()


def _print_report(name: str, results: list[RowResult]) -> None:
    passed = sum(1 for r in results if r.passed)
    print(f"\n=== {name}: {passed}/{len(results)} rows passed ===")
    for result in results:
        snippet = result.response.replace("\n", " ")[:160]
        print(f"  id={result.row.row_id} passed={result.passed} detail={result.detail}")
        print(f"    atif: {len(result.atif_paths)} promoted trace(s)")
        print(f"    response: {snippet}")


def _python_env() -> dict[str, str]:
    env = os.environ.copy()
    venv_python = ROOT / ".venv" / "bin" / "python"
    if venv_python.is_file():
        env["FABRIC_LANGCHAIN_PYTHON"] = str(venv_python)
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(ROOT / "adapters" / "langchain-react" / "src"),
            str(ROOT / "adapters" / "common" / "src"),
            env.get("PYTHONPATH", ""),
        ]
    ).strip(os.pathsep)
    return env


def call_json(*args: str, env: dict[str, str] | None = None) -> dict:
    completed = subprocess.run(
        [*COMMAND, *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"fabric command failed ({completed.returncode}): {completed.stderr.strip() or completed.stdout}"
        )
    return json.loads(completed.stdout)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"FAILED: {error}", file=sys.stderr)
        raise
