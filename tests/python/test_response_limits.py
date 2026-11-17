# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Normalized response-limit authoring and task transport budgets."""

import json
from pathlib import Path
import sys

import pytest
from pydantic import ValidationError

from nemo_fabric import ResponseLimits, RuntimeConfig
from nemo_fabric.runtime_server.protocol import encode_response
from nemo_fabric.runtime_server.protocol import response_transport_limit

FIELDS = (
    "max_response_bytes",
    "max_artifact_bytes",
    "max_artifact_files",
    "max_artifact_file_bytes",
)


def test_transport_formula_matches_the_shared_rust_fixture():
    cases = json.loads(
        (
            Path(__file__).parents[1] / "fixtures/response-transport-limits.json"
        ).read_text()
    )
    for case in cases:
        assert response_transport_limit(case["max_response_bytes"]) == min(
            case["transport_bytes"], sys.maxsize * 2 + 1
        )


@pytest.fixture
def local_response_limit_config(tmp_path):
    from nemo_fabric import FabricConfig

    root = Path(__file__).parents[2]
    descriptor = json.loads(
        (root / "tests/e2e/response-limits.fabric-adapter.json").read_text()
    )
    descriptor["runner"] = {
        "module": "response_limits_adapter",
        "python": sys.executable,
        "cwd": str(root / "tests/e2e"),
        "env": {"PYTHONPATH": str(root / "tests/e2e")},
    }
    descriptor_path = tmp_path / "limits.fabric-adapter.json"
    descriptor_path.write_text(json.dumps(descriptor))
    config = FabricConfig.from_mapping(
        {
            "metadata": {"name": "local-limits"},
            "harness": {
                "adapter_id": descriptor["adapter_id"],
                "resolution": "preinstalled",
            },
            "discovery": {"local_paths": [str(descriptor_path)]},
            "environment": {"provider": "local", "workspace": str(tmp_path)},
            "runtime": {
                "artifacts": str(tmp_path / "artifacts"),
                "response_limits": None,
            },
        }
    )
    return config


@pytest.mark.parametrize("artifact_sizes", [[1] * 17, [600_000] * 4])
@pytest.mark.parametrize("configured", [False, True])
async def test_local_artifact_budgets_require_explicit_configuration(
    tmp_path, artifact_sizes, configured, local_response_limit_config
):
    from nemo_fabric import Fabric, RunRequest

    config = local_response_limit_config
    config.runtime.response_limits = ResponseLimits() if configured else None
    async with await Fabric().start_runtime(config, base_dir=tmp_path) as runtime:
        result = await runtime.invoke(
            request=RunRequest(
                input="files", context={"artifact_sizes": artifact_sizes}
            )
        )
    assert result.status == ("failed" if configured else "succeeded")
    if configured:
        assert result.error.code == "response_limit_exceeded"
    else:
        assert len(result.artifacts.artifacts) >= len(artifact_sizes)


@pytest.mark.parametrize("one_shot", [False, True])
@pytest.mark.parametrize(
    "budget,payload_bytes,status",
    [(512, 16, "failed"), (8 * 1024 * 1024, 5 * 1024 * 1024, "succeeded")],
)
async def test_saved_result_matches_configured_response_budget(
    local_response_limit_config, tmp_path, one_shot, budget, payload_bytes, status
):
    from nemo_fabric import Fabric, FabricRuntimeError, RunRequest

    config = local_response_limit_config
    config.runtime.response_limits = ResponseLimits(max_response_bytes=budget)
    client = Fabric()
    request = RunRequest(input="response", context={"response_bytes": payload_bytes})
    if one_shot:
        result = await client.run(config, base_dir=tmp_path, request=request)
    else:
        async with await client.start_runtime(config, base_dir=tmp_path) as runtime:
            result = await runtime.invoke(request=request)
    assert result.status == status
    if status == "failed":
        assert result.error.code == "response_limit_exceeded"
    saved = await Fabric().invocation_result(
        agent_name=result.agent_name,
        runtime_id=result.runtime_id,
        request_id=result.request_id,
        artifacts=config.runtime.artifacts,
    )
    assert saved == result
    if payload_bytes > 4 * 1024 * 1024:
        stored = (
            Path(config.runtime.artifacts)
            / ".fabric"
            / result.runtime_id
            / result.invocation_id
            / "run-result.json"
        )
        oversized = saved.to_mapping()
        oversized["output"] = {"response": "x" * (budget + 1)}
        stored.write_text(json.dumps(oversized), encoding="utf-8")
        with pytest.raises(FabricRuntimeError) as caught:
            await Fabric().invocation_result(
                agent_name=result.agent_name,
                runtime_id=result.runtime_id,
                request_id=result.request_id,
                artifacts=config.runtime.artifacts,
            )
        assert caught.value.code == "result_unavailable"


@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("value", [0, -1, 1.5, True, "12", None, 1 << 32])
def test_response_limits_reject_invalid_values(field, value):
    with pytest.raises(ValidationError):
        ResponseLimits(**{field: value})


def test_response_limits_omitted_fields_retain_defaults():
    assert RuntimeConfig().response_limits is None
    limits = RuntimeConfig(
        response_limits={"max_response_bytes": 8_000_000}
    ).response_limits
    assert limits == ResponseLimits(max_response_bytes=8_000_000)
    assert limits.max_artifact_bytes == 2 * 1024 * 1024
    assert limits.max_artifact_files == 16
    assert limits.max_artifact_file_bytes == 1024 * 1024
    with pytest.raises(ValidationError):
        ResponseLimits(max_response_byte=100)


@pytest.mark.parametrize("delta", [-1, 0, 1])
def test_bridge_encoding_uses_the_configured_transport_budget(delta):
    response = {"operation": "invoke", "value": "é😀"}
    encoded = (
        json.dumps(response, ensure_ascii=False, separators=(",", ":")).encode() + b"\n"
    )
    wire = encode_response(response, limit=len(encoded) + delta)
    if delta >= 0:
        assert wire == encoded
    else:
        assert json.loads(wire)["error"]["code"] == "response_too_large"


def test_bridge_can_carry_response_above_old_four_mib_cap():
    response = {"operation": "invoke", "value": "x" * (4 * 1024 * 1024)}
    wire = encode_response(response, limit=8 * 1024 * 1024)
    assert json.loads(wire) == response


@pytest.mark.parametrize("stop_failure", [False, True])
async def test_python_convenience_run_rechecks_final_native_result(stop_failure):
    from unittest.mock import MagicMock
    from nemo_fabric.runtime import _run_native_lifecycle

    native = pytest.importorskip("nemo_fabric._native")
    module = MagicMock()
    result = dict(
        agent_name="test",
        harness="echo",
        adapter_kind="python",
        runtime_id="runtime-test",
        invocation_id="invocation-test",
        request_id="request-test",
        status="succeeded",
        output="done",
    )
    module.start_runtime.return_value = "{}"
    module.invoke_runtime.return_value = json.dumps(result)
    if stop_failure:
        module.stop_runtime.side_effect = RuntimeError("x" * 4096)
    else:
        module.stop_runtime.return_value = json.dumps(
            [
                dict(
                    event_id="stop-test",
                    timestamp_millis=0,
                    kind="stop",
                    message="x" * 4096,
                )
            ]
        )
    module.finalize_run_result.side_effect = native.finalize_run_result
    final = await _run_native_lifecycle(
        module,
        {"config": {"runtime": {"response_limits": {"max_response_bytes": 2048}}}},
        {},
    )
    assert final["status"] == "failed"
    assert final["error"]["code"] == "response_limit_exceeded"
    assert final["error"]["metadata"]["observed"] > 2048
    module.stop_runtime.assert_called_once()
