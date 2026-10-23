# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Actual SDK/core publication to caller inventory, without a model or network."""

import asyncio
import json
import os
from pathlib import Path
import sys
from time import monotonic_ns

import pytest

from nemo_fabric import Fabric, FabricConfig, RunRequest
from nemo_fabric.comparisons import read_comparison_manifest, render_comparison_summary
from nemo_fabric_adapter_contract.models import GapReason, ReportedUsage
from nemo_fabric_adapter_contract.usage import UsageObservation, usage_from_observations


@pytest.mark.integration
def test_real_sdk_capture_preserves_normal_partial_and_missing_attempts(tmp_path):
    repository = Path(
        os.environ.get("NEMO_FABRIC_TEST_REPO", Path(__file__).resolve().parents[2])
    )
    schema = json.loads(
        (repository / "schemas/extensions/usage-accounting.schema.json").read_text()
    )
    definitions = schema.pop("$defs")
    descriptor = {
        "contract_version": "fabric.adapter/v1alpha2",
        "adapter_id": "test.comparison-capture",
        "adapter_kind": "process",
        "runner": {"command": sys.executable, "script": "adapter.py"},
        "settings_schema": {"type": "object", "additionalProperties": False},
        "extension_schemas": {
            "usage": {
                "type": "object",
                "$defs": definitions,
                "properties": {
                    "nemo_fabric.usage_accounting.v1": schema,
                    "nemo_fabric.detailed_token_usage.v1": {
                        "$ref": "#/$defs/ReportedUsage"
                    },
                },
                "additionalProperties": False,
            }
        },
    }
    (tmp_path / "comparison.fabric-adapter.json").write_text(json.dumps(descriptor))
    for mode, input_tokens, gap in [
        ("normal", 10, None),
        ("partial", 7, GapReason.UNFINISHED_WORK),
    ]:
        usage = usage_from_observations(
            [
                UsageObservation(
                    model="controlled-fixture",
                    is_model_call=True,
                    tokens_aligned=True,
                    gap=gap,
                    reported=ReportedUsage(
                        input_tokens=input_tokens,
                        input_tokens_include_cache=True,
                        cached_input_tokens=0,
                        cache_write_input_tokens=0,
                        output_tokens=2,
                        output_tokens_include_reasoning=True,
                        reasoning_tokens=0,
                        total_tokens=input_tokens + 2,
                    ),
                )
            ]
        )
        (tmp_path / f"{mode}.json").write_text(json.dumps(usage.to_mapping()))
    (tmp_path / "adapter.py").write_text("""import json,sys
from pathlib import Path
for line in sys.stdin:
    message=json.loads(line)
    operation=message["operation"]
    output=None
    if operation=="invoke":
        mode=message["payload"]["request"]["input"]
        output={"status":"failed" if mode=="partial" else "succeeded",
                "output":{"response":"controlled fixture"},
                "usage":json.loads(Path(mode+".json").read_text())}
        if mode=="partial":
            output["error"]={"code":"controlled_timeout","message":"controlled unfinished work","retryable":False}
    print(json.dumps({"operation":operation,"outcome":{"status":"succeeded","output":output}}),flush=True)
    if operation=="stop":break
""")
    config = FabricConfig.from_mapping(
        {
            "metadata": {"name": "comparison-capture"},
            "harness": {"adapter_id": "test.comparison-capture"},
            "discovery": {"local_paths": ["comparison.fabric-adapter.json"]},
            "environment": {"provider": "local"},
            "runtime": {"artifacts": "./artifacts", "record": {"enabled": True}},
        }
    )

    async def capture():
        attempts = []
        for mode in ("normal", "partial"):
            started = monotonic_ns()
            result = await Fabric().run(
                config,
                base_dir=tmp_path,
                request=RunRequest(request_id=mode, input=mode),
            )
            elapsed = (monotonic_ns() - started) // 1_000_000
            saved = next(
                artifact.path
                for artifact in result.artifacts.artifacts
                if artifact.kind == "run_result"
            )
            attempts.append(
                {
                    "attempt_id": mode,
                    "task_id": "custom-change-review",
                    "task_revision": "v1",
                    "configuration_id": mode,
                    "requests_complete": True,
                    "requests": [
                        {
                            "request_id": mode,
                            "evidence": str(Path(saved).relative_to(tmp_path)),
                        }
                    ],
                    "elapsed_ms": elapsed,
                    "elapsed_scope": "caller_one_shot",
                }
            )
        attempts.append(
            {
                "attempt_id": "scheduled",
                "task_id": "custom-change-review",
                "task_revision": "v1",
                "configuration_id": "partial",
                "requests_complete": True,
                "requests": [
                    {"request_id": "scheduled", "evidence": "never-returned.json"}
                ],
            }
        )
        return attempts

    attempts = asyncio.run(capture())
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema": "nemo.fabric.comparison_manifest",
                "version": "v1alpha1",
                "inventory_complete": True,
                "attempts": attempts,
            }
        )
    )
    summary = read_comparison_manifest(path)
    normal = summary["configurations"]["normal"]["usage"]["dimensions"]["input_tokens"]
    partial = summary["configurations"]["partial"]["usage"]["dimensions"][
        "input_tokens"
    ]
    assert normal["complete_total"] == 10
    assert partial["complete_total"] is None
    assert partial["observed_sum"] == 7
    assert partial["missing_accounting_evidence_count"] == 1
    assert summary["configurations"]["partial"]["listed_attempt_count"] == 2
    assert summary["configurations"]["partial"]["execution"]["statuses"] == {
        "failed": 1
    }
    assert summary["configurations"]["normal"]["evaluation"]["not_evaluated"] == 1
    assert "7 observed, incomplete" in render_comparison_summary(summary)
