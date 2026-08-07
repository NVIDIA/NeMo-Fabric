# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for adapter-facing request and result structures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from nemo_fabric_adapter_contract.models import AgentArtifact
from nemo_fabric_adapter_contract.models import AgentRunError
from nemo_fabric_adapter_contract.models import AgentRunRequest
from nemo_fabric_adapter_contract.models import AgentRunResult
from nemo_fabric_adapter_contract.models import AgentRunStatus
from nemo_fabric_adapter_contract.models import AgentUsage
from nemo_fabric_adapter_contract.models import RuntimeContext
from pydantic import ValidationError


ROOT = Path(__file__).resolve().parents[2]


def test_agent_run_request_contains_only_southbound_request_fields():
    request = AgentRunRequest(
        input={"messages": [{"role": "user", "content": "hello"}]},
        context={"task": "sample"},
    )

    assert request.to_mapping() == {
        "input": {"messages": [{"role": "user", "content": "hello"}]},
        "context": {"task": "sample"},
    }
    assert set(AgentRunRequest.model_fields) == {"input", "context", "extensions"}


def test_agent_run_result_contains_only_adapter_owned_result_fields():
    result = AgentRunResult(
        status=AgentRunStatus.FAILED,
        output=None,
        error=AgentRunError(
            code="model_error",
            message="model invocation failed",
            retryable=True,
        ),
        usage=AgentUsage(input_tokens=12, output_tokens=4, total_tokens=16),
        artifacts=[
            AgentArtifact(
                name="trace",
                kind="atof",
                path="trace.jsonl",
                media_type="application/x-ndjson",
            )
        ],
    )

    assert result.to_mapping() == {
        "status": "failed",
        "output": None,
        "error": {
            "code": "model_error",
            "message": "model invocation failed",
            "retryable": True,
        },
        "usage": {
            "input_tokens": 12,
            "output_tokens": 4,
            "total_tokens": 16,
        },
        "artifacts": [
            {
                "name": "trace",
                "kind": "atof",
                "path": "trace.jsonl",
                "media_type": "application/x-ndjson",
            }
        ],
    }
    assert set(AgentRunResult.model_fields) == {
        "status",
        "output",
        "error",
        "usage",
        "artifacts",
        "extensions",
    }


@pytest.mark.parametrize(
    ("model", "filename"),
    [
        (AgentRunRequest, "agent-run-request.schema.json"),
        (AgentRunResult, "agent-run-result.schema.json"),
        (RuntimeContext, "runtime-context.schema.json"),
    ],
)
def test_agent_execution_models_track_rust_schema_root_fields(model, filename):
    rust_schema = json.loads(
        (ROOT / "schemas" / "adapter-contract" / filename).read_text(encoding="utf-8")
    )
    pydantic_schema = model.model_json_schema()

    assert rust_schema["additionalProperties"] is False
    assert pydantic_schema["additionalProperties"] is False
    assert set(pydantic_schema["properties"]) == set(rust_schema["properties"])


def test_failed_agent_run_result_requires_error():
    with pytest.raises(ValidationError, match="failed result requires an error"):
        AgentRunResult(status=AgentRunStatus.FAILED, output=None)


@pytest.mark.parametrize(
    "path",
    [
        "",
        "/tmp/output",
        "../output",
        "nested/../output",
        r"C:\tmp\output",
        r"C:output",
        r"\\server\output",
        r"nested\..\output",
    ],
)
def test_agent_artifact_rejects_unsafe_paths(path: str):
    with pytest.raises(ValidationError, match="artifact path must be"):
        AgentArtifact(name="output", kind="file", path=path)
