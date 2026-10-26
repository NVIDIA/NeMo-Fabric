# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reducer parity requires Node and a compiled TypeScript adapter contract.

From adapter-contract/typescript, run:
    npm exec tsc -- -p tsconfig.build.json
"""

import json
from pathlib import Path
import shutil
import subprocess

import pytest

from nemo_fabric_adapter_contract.models import GapReason, ReportedUsage, UsageGap
from nemo_fabric_adapter_contract.usage import UsageObservation, usage_from_observations

ROOT = Path(__file__).resolve().parents[2]
VECTORS = json.loads(
    (ROOT / "tests/fixtures/usage_accounting/reducer-cases.json").read_text()
)
KEY = "nemo_fabric.usage_accounting.v1"


@pytest.fixture(name="reduced")
def reduced_fixture():
    return [
        usage_from_observations(
            [
                UsageObservation(
                    reported=ReportedUsage.from_mapping(record["reported"]),
                    model=record.get("model"),
                    provider=record.get("provider"),
                    gap=GapReason(record["gap"]) if "gap" in record else None,
                    tokens_aligned=record.get("tokens_aligned", False),
                    is_model_call=record.get("is_model_call", False),
                    gaps=tuple(
                        UsageGap.from_mapping(gap) for gap in record.get("gaps", [])
                    ),
                )
                for record in vector["records"]
            ]
        ).to_mapping()
        for vector in VECTORS
    ]


def test_python_reducer_matches_shared_independent_expectations(reduced):
    for vector, usage in zip(VECTORS, reduced, strict=True):
        accounting = usage["extensions"][KEY]
        for field, value in vector["expected"].items():
            if field == "gap_reason":
                assert [gap["reason"] for gap in accounting["gaps"]] == [value]
                continue
            if field == "exhaustive":
                actual = accounting["models_exhaustive"]
            elif field.startswith("observed_"):
                actual = accounting["observed"].get(field[9:] + "_tokens")
            elif field == "peak_request_input_tokens":
                actual = usage["extensions"]["nemo_fabric.detailed_token_usage.v1"].get(
                    field
                )
            else:
                actual = usage.get(field)
            assert actual == value, (vector["name"], field)


def test_python_and_typescript_reducers_emit_the_same_wire_evidence(reduced):
    node = shutil.which("node")
    module = ROOT / "adapter-contract/typescript/dist/usage.js"
    if node is None or not module.exists():
        pytest.skip("build the TypeScript contract to run cross-binding reducer parity")
    source = """
import { pathToFileURL } from 'node:url';
import { readFileSync } from 'node:fs';
const { usageFromObservations } = await import(pathToFileURL(process.argv[1]).href);
console.log(JSON.stringify(JSON.parse(readFileSync(0, 'utf8')).map(v => usageFromObservations(v.records))));
"""
    result = subprocess.run(
        [node, "--input-type=module", "-e", source, str(module)],
        input=json.dumps(VECTORS),
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout) == reduced


@pytest.mark.integration
async def test_actual_calls_with_mixed_conventions_recover_complete_counts_in_core(
    reduced, accounting_runtime
):
    index = next(
        i
        for i, vector in enumerate(VECTORS)
        if vector["name"] == "mixed_actual_call_conventions"
    )
    result = await accounting_runtime.invoke(input=reduced[index])
    assert result.status == "succeeded"
    usage = result.model_usage
    assert usage.input_tokens == 210
    assert usage.cached_input_tokens == 20
    assert usage.output_tokens == 4
    assert usage.total_tokens == 214


@pytest.mark.integration
async def test_scoped_request_gap_preserves_cache_uncertainty_in_core(
    reduced, accounting_runtime
):
    index = next(
        i
        for i, vector in enumerate(VECTORS)
        if vector["name"] == "actual_call_with_scoped_cache_gap"
    )
    evidence = reduced[index]["extensions"][KEY]
    model = evidence["models"][0]
    request = model["requests"][0]
    assert model["observed"]["cached_input_tokens"] == 20
    assert "cached_input_tokens" not in model["reported"]
    assert "cached_input_tokens" not in request["usage"]
    assert request["usage"]["input_tokens"] == 100
    assert request["usage"]["output_tokens"] == 5
    assert request["tokens_aligned"] is True
    assert "gap" not in request

    result = await accounting_runtime.invoke(input=reduced[index])
    assert result.status == "succeeded"
    usage = result.model_usage
    assert usage.input_tokens == 100
    assert usage.output_tokens == 5
    assert usage.total_tokens == 105
    assert usage.cached_input_tokens is None
    assert usage.observed.cached_input_tokens == 20
    assert usage.models[0].cached_input_tokens is None
