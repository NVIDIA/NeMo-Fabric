#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in smoke for PR #564: AgentEvaluator + FabricAgentRuntime + Qwen IGW.

This validates the evaluator-owned Fabric runner path that promotes Relay ATIF
as standard trace evidence.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
NEMO_PLATFORM_SDK = ROOT.parent / "nemo-platform" / "packages" / "nemo_evaluator_sdk" / "src"
if NEMO_PLATFORM_SDK.is_dir() and str(NEMO_PLATFORM_SDK) not in sys.path:
    sys.path.insert(0, str(NEMO_PLATFORM_SDK))

from nemo_evaluator_sdk.agent_eval.evaluator import AgentEvaluator  # type: ignore[import-not-found]
from nemo_evaluator_sdk.agent_eval.runtimes.fabric.runtime import FabricAgentRuntime  # type: ignore[import-not-found]
from nemo_evaluator_sdk.agent_eval.tasks import AgentEvalRunConfig, AgentEvalTask  # type: ignore[import-not-found]
from nemo_evaluator_sdk.metrics.protocol import (  # type: ignore[import-not-found]
    MetricInput,
    MetricOutput,
    MetricOutputSpec,
    MetricResult,
)
from nemo_evaluator_sdk.values.evidence import EVIDENCE_FORMAT_ATIF, EVIDENCE_TRACE  # type: ignore[import-not-found]

AGENT = ROOT / "examples" / "react-optimize-agent"
PROFILES = AGENT / "profiles"


class _TraceAndContentMetric:
    """Scores true iff output contains an expected token and ATIF trace evidence exists."""

    def __init__(self, *, expected: tuple[str, ...]) -> None:
        self.expected = expected

    @property
    def type(self) -> str:
        return "trace-and-content"

    def output_spec(self) -> list[MetricOutputSpec]:
        return [MetricOutputSpec.boolean("passed")]

    async def compute_scores(self, input: MetricInput) -> MetricResult:  # noqa: A002 - protocol name
        text = str(input.candidate.output_text or input.candidate.response or "").lower()
        content_ok = any(token in text for token in self.expected)

        trace_ok = False
        evidence = input.candidate.evidence
        if evidence is not None:
            trace = evidence.descriptors.get(EVIDENCE_TRACE)
            if trace is not None and trace.format == EVIDENCE_FORMAT_ATIF and trace.ref:
                path = Path(trace.ref)
                trace_ok = path.exists() and bool(json.loads(path.read_text(encoding="utf-8")).get("steps"))

        return MetricResult(outputs=[MetricOutput(name="passed", value=content_ok and trace_ok)])


def main() -> None:
    if os.environ.get("RUN_FABRIC_AGENT_EVALUATOR_QWEN") != "1":
        print("skipping: set RUN_FABRIC_AGENT_EVALUATOR_QWEN=1 to run")
        return

    os.environ.setdefault("FABRIC_LANGCHAIN_PYTHON", str(ROOT / ".venv" / "bin" / "python"))

    react_result = _run_suite(
        name="react-optimize",
        profiles=["qwen-igw-local", "qwen-react-native"],
        task=AgentEvalTask(
            id="react-parity-1",
            intent="react-optimize parity smoke",
            inputs={
                "question": (
                    "What is the capital of France, and what day of the week is it today? "
                    "Use the current datetime tool to determine today's day of the week."
                ),
            },
            reference={"answer": "Answer must state Paris and include the current day of week."},
            metrics=[_TraceAndContentMetric(expected=("paris",))],
        ),
    )
    calc_result = _run_suite(
        name="calculator-optimize",
        profiles=["qwen-igw-local", "qwen-calculator-native"],
        task=AgentEvalTask(
            id="calculator-parity-1",
            intent="calculator-optimize parity smoke",
            inputs={
                "question": (
                    "What is the product of 3 and 7, and is it greater than the current hour? "
                    "Use the calculator tool for arithmetic and the current datetime tool for the hour."
                ),
            },
            reference={"answer": "Answer must contain 21 and compare it to the current hour."},
            metrics=[_TraceAndContentMetric(expected=("21",))],
        ),
    )

    for name, result in (("react-optimize", react_result), ("calculator-optimize", calc_result)):
        trial = result.trials[0]
        score = result.scores[0].outputs[0].value
        trace = trial.evidence.descriptors[EVIDENCE_TRACE]
        print(f"[{name}] trial={trial.status} score={score} trace={trace.ref}")
        assert trial.status == "completed", trial
        assert score in (True, 1.0), result.scores
        assert Path(trace.ref).exists(), trace

    print("agent evaluator qwen fabric runtime smoke passed")


def _run_suite(*, name: str, profiles: list[str], task: AgentEvalTask):
    runtime = FabricAgentRuntime(
        config=_load_yaml(AGENT / "agent.yaml"),
        profiles=[_load_yaml(PROFILES / f"{profile}.yaml") for profile in profiles],
        base_dir=ROOT,
        work_root=ROOT / "artifacts" / "agent-evaluator" / name,
        capture_trajectory=True,
    )
    return AgentEvaluator().run_sync(
        tasks=[task],
        target=runtime,
        config=AgentEvalRunConfig(
            output_dir=ROOT / "artifacts" / "agent-evaluator-results" / name,
            parallelism=1,
            write_dashboard=False,
        ),
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    if "name" not in payload and metadata.get("name"):
        payload["name"] = metadata["name"]
    return payload


if __name__ == "__main__":
    main()
