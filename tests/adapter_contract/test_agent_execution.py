# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for adapter-facing request and result structures."""

from __future__ import annotations

import json
from dataclasses import MISSING
from dataclasses import fields
from pathlib import Path

import pytest
from nemo_fabric_adapter_contract.models import AgentArtifact
from nemo_fabric_adapter_contract.models import AgentRunError
from nemo_fabric_adapter_contract.models import AgentRunRequest
from nemo_fabric_adapter_contract.models import AgentRunResult
from nemo_fabric_adapter_contract.models import AgentRunStatus
from nemo_fabric_adapter_contract.models import AgentUsage
from nemo_fabric_adapter_contract.models import ArtifactManifest
from nemo_fabric_adapter_contract.models import ArtifactRef
from nemo_fabric_adapter_contract.models import EnvironmentHandle
from nemo_fabric_adapter_contract.models import RuntimeContext
from nemo_fabric_adapter_contract.models import RuntimeTelemetryContext
from nemo_fabric_adapter_contract.codec import ContractValidationError
from nemo_fabric_adapter_contract.pydantic_support import type_adapter


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("includes_cache", [True, False, None])
def test_usage_cache_semantics_round_trip(includes_cache):
    usage = AgentUsage(
        input_tokens=12,
        cached_input_tokens=4,
        input_tokens_include_cache=includes_cache,
    )
    restored = AgentUsage.from_mapping(usage.to_mapping())
    assert restored.input_tokens == 12
    assert restored.cached_input_tokens == 4
    assert restored.input_tokens_include_cache is includes_cache


@pytest.mark.parametrize("value", [-1, 1 << 64, True, 1.5])
def test_usage_rejects_invalid_cache_counter(value):
    with pytest.raises(ContractValidationError):
        AgentUsage(cached_input_tokens=value)


@pytest.mark.parametrize("value", [0, 1, "true"])
def test_usage_rejects_invalid_cache_semantics(value):
    with pytest.raises(ContractValidationError):
        AgentUsage(input_tokens_include_cache=value)


def test_agent_run_request_contains_only_southbound_request_fields():
    request = AgentRunRequest(
        input={"messages": [{"role": "user", "content": "hello"}]},
        context={"task": "sample"},
    )

    assert request.to_mapping() == {
        "input": {"messages": [{"role": "user", "content": "hello"}]},
        "context": {"task": "sample"},
    }
    assert {item.name for item in fields(AgentRunRequest)} == {
        "input",
        "relay_session_root",
        "context",
        "extensions",
    }


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
    assert {item.name for item in fields(AgentRunResult)} == {
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
    pydantic_schema = type_adapter(model).json_schema()

    assert rust_schema["additionalProperties"] is False
    assert pydantic_schema["additionalProperties"] is False
    assert set(pydantic_schema["properties"]) == set(rust_schema["properties"])


@pytest.mark.parametrize(
    ("filename", "models"),
    [
        (
            "agent-run-result.schema.json",
            (AgentArtifact, AgentRunError, AgentUsage),
        ),
        (
            "runtime-context.schema.json",
            (
                ArtifactManifest,
                ArtifactRef,
                EnvironmentHandle,
                RuntimeTelemetryContext,
            ),
        ),
    ],
)
def test_agent_execution_dataclasses_track_rust_schema_block_fields(
    filename,
    models,
):
    rust_schema = json.loads(
        (ROOT / "schemas" / "adapter-contract" / filename).read_text(encoding="utf-8")
    )

    for model in models:
        assert {item.name for item in fields(model)} == set(
            rust_schema["$defs"][model.__name__]["properties"]
        )
        required = {
            item.name
            for item in fields(model)
            if item.default is MISSING and item.default_factory is MISSING
        }
        assert required == set(rust_schema["$defs"][model.__name__].get("required", []))


def test_failed_agent_run_result_requires_error():
    with pytest.raises(
        ContractValidationError, match="failed result requires an error"
    ):
        AgentRunResult(status=AgentRunStatus.FAILED, output=None)


def test_contract_dataclasses_validate_assignment():
    usage = AgentUsage(input_tokens=1)
    with pytest.raises(AttributeError, match="AgentUsage has no field 'unknown'"):
        usage.unknown = 1  # type: ignore[attr-defined]

    with pytest.raises(
        ContractValidationError,
        match="input_tokens: must be between 0",
    ):
        usage.input_tokens = -5
    assert usage.input_tokens == 1

    result = AgentRunResult(status=AgentRunStatus.SUCCEEDED, output=None)
    with pytest.raises(
        ContractValidationError,
        match="failed result requires an error",
    ):
        result.status = AgentRunStatus.FAILED
    assert result.status is AgentRunStatus.SUCCEEDED


@pytest.mark.parametrize(
    "path",
    [
        "",
        "   ",
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
    with pytest.raises(ContractValidationError, match="artifact path must be"):
        AgentArtifact(name="output", kind="file", path=path)


def test_typed_session_root_round_trips_in_adapter_request():
    root = "018f47a4-3af7-7d94-8e61-9f0f89b5d312"
    request = AgentRunRequest(input="hello", relay_session_root=root)
    assert request.to_mapping() == {"input": "hello", "relay_session_root": root}
    assert AgentRunRequest.from_mapping(request.to_mapping()) == request
    assert "relay_session_root" not in AgentRunRequest(input="hello").to_mapping()
    with pytest.raises(ContractValidationError):
        AgentRunRequest(input="hello", relay_session_root=123)
