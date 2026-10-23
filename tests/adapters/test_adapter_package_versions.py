# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python adapters without a harness version source still report their package version."""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import jsonschema
import pytest
from nemo_fabric_adapter_contract.models import AgentRunRequest
from nemo_fabric_adapter_contract.models import AgentRunResult
from nemo_fabric_adapter_contract.models import AgentRunStatus
from nemo_fabric_adapter_contract.models import RuntimeContext

ROOT = Path(__file__).resolve().parents[2]
VERSIONS_REPORT_KEY = "nemo_fabric.versions.v1"
NO_SOURCE = {
    "state": "unsupported",
    "reason": "no_source",
    "source": "reported",
    "collected_by": "adapter",
    "method": "adapter_report",
}


@pytest.mark.parametrize(
    ("module", "runtime_type", "distribution", "descriptor"),
    [
        (
            "nemo_fabric_adapters.nooa.adapter",
            "NooaRuntime",
            "nemo-fabric-adapters-nooa",
            "adapters/python/nooa/nooa.fabric-adapter.json",
        ),
        (
            "nemo_fabric_adapters.nooa.bench_adapter",
            "BenchRuntime",
            "nemo-fabric-adapters-nooa",
            "adapters/python/nooa/nooa-bench.fabric-adapter.json",
        ),
        (
            "nemo_fabric_adapters.openhands.adapter",
            "OpenHandsRuntime",
            "nemo-fabric-adapters-openhands",
            "adapters/python/openhands/openhands.fabric-adapter.json",
        ),
        (
            "nemo_fabric_adapters.remote_agent.adapter",
            "RemoteAgentRuntime",
            "nemo-fabric-adapters-remote-agent",
            "adapters/python/remote-agent/remote-agent.fabric-adapter.json",
        ),
    ],
)
async def test_adapter_reports_its_package_version_on_each_result(
    module, runtime_type, distribution, descriptor, monkeypatch
):
    runtime = getattr(pytest.importorskip(module), runtime_type)()
    runtime._model_roles = ["default"]
    returned = AgentRunResult(
        status=AgentRunStatus.FAILED,
        output={"response": None},
        error={"code": "target_failed", "message": "failed"},
    )
    monkeypatch.setattr(runtime, "_invoke", AsyncMock(return_value=returned))

    result = await runtime.invoke(
        AgentRunRequest(input="hello"), MagicMock(spec=RuntimeContext)
    )

    assert result.status is AgentRunStatus.FAILED
    assert result.extensions[VERSIONS_REPORT_KEY] == {
        "adapter": {
            "package_version": {
                "value": importlib.metadata.version(distribution),
                "source": "reported",
                "collected_by": "adapter",
                "method": "package_metadata",
            }
        },
        "harness": {"name": NO_SOURCE, "version": NO_SOURCE},
        "models": {"default": {"served": NO_SOURCE}},
    }
    schema = json.loads((ROOT / descriptor).read_text(encoding="utf-8"))
    jsonschema.validate(
        result.to_mapping()["extensions"], schema["extension_schemas"]["run_result"]
    )
