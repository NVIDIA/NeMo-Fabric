# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Code Review OpenShell example uses the public SDK lifecycle."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.code_review_agent import openshell_sdk_consumer as consumer


@pytest.mark.parametrize("ownership", ["caller_owned", "fabric_owned"])
def test_code_review_consumer_uses_sdk_lifecycle_and_preserves_two_turn_context(
    ownership, tmp_path, monkeypatch
):
    policy = tmp_path / "policy.yaml"
    policy.write_text("version: 1\n", encoding="utf-8")
    results = tmp_path / ownership / "report.json"
    args = SimpleNamespace(
        ownership=ownership,
        gateway="http://127.0.0.1:18080",
        workspace="default",
        image="sha256:" + "a" * 64,
        policy_file=str(policy),
        results_file=str(results),
        sandbox_name=f"review-{ownership}",
        sandbox_id="caller-sandbox-id" if ownership == "caller_owned" else None,
        model="nvidia/test-model",
        base_url="https://integrate.api.nvidia.com/v1",
        inputs=["Review calculator.py", "Summarize and remember cedar-4827"],
    )
    events = []
    requests = []

    class FakeRuntime:
        runtime_id = "runtime-review"

        async def invoke(self, *, request):
            requests.append(request)
            events.append("invoke")
            value = {
                "request_id": request.request_id,
                "input": request.input,
                "context": request.context,
            }
            return SimpleNamespace(
                status="succeeded",
                request_id=request.request_id,
                to_mapping=lambda: value,
            )

        async def stop(self):
            events.append("stop")

    class FakeFabric:
        async def attach_environment(self, config, reference):
            events.append("attach")
            assert reference.provider == "openshell"
            assert reference.resource["sandbox_id"] == "caller-sandbox-id"
            return "environment"

        async def prepare_environment(self, config):
            events.append("prepare")
            return "environment"

        async def start_runtime_in(self, config, environment):
            events.append("start")
            assert environment == "environment"
            assert config.harness.adapter_id == "nvidia.fabric.openclaw"
            assert config.environment.provider == "openshell"
            assert config.environment.settings["providers"] == ["hosted"]
            assert config.models["default"].api_key_env == "OPENAI_API_KEY"
            assert config.skills.paths == ["/sandbox/code-review/skills/code-review"]
            return FakeRuntime()

        async def release_environment(self, environment):
            events.append("release")

    monkeypatch.setattr(consumer, "Fabric", FakeFabric)
    report = asyncio.run(consumer.run_demo(args))

    assert report["status"] == "succeeded"
    assert report["runtime_id"] == "runtime-review"
    assert events == [
        "attach" if ownership == "caller_owned" else "prepare",
        "start",
        "invoke",
        "invoke",
        "stop",
        "release",
    ]
    assert [(request.request_id, request.context) for request in requests] == [
        ("code-review-turn-1", {"turn": 1}),
        ("code-review-turn-2", {"turn": 2}),
    ]
    assert json.loads(results.read_text(encoding="utf-8")) == report


def test_failed_invocation_still_stops_runtime_and_releases_environment(tmp_path, monkeypatch):
    policy = tmp_path / "policy.yaml"
    policy.write_text("version: 1\n", encoding="utf-8")
    args = SimpleNamespace(
        ownership="fabric_owned",
        gateway="http://127.0.0.1:18080",
        workspace="default",
        image="sha256:" + "a" * 64,
        policy_file=str(policy),
        results_file=str(tmp_path / "report.json"),
        sandbox_name="review-failure",
        sandbox_id=None,
        model="nvidia/test-model",
        base_url="https://integrate.api.nvidia.com/v1",
        inputs=["Review", "Follow up"],
    )
    events = []

    class FakeRuntime:
        runtime_id = "runtime-review"

        async def invoke(self, *, request):
            return SimpleNamespace(
                status="failed",
                request_id=request.request_id,
                to_mapping=lambda: {"status": "failed"},
            )

        async def stop(self):
            events.append("stop")

    class FakeFabric:
        async def prepare_environment(self, config):
            return "environment"

        async def start_runtime_in(self, config, environment):
            return FakeRuntime()

        async def release_environment(self, environment):
            events.append("release")

    monkeypatch.setattr(consumer, "Fabric", FakeFabric)
    report = asyncio.run(consumer.run_demo(args))

    assert report["status"] == "failed"
    assert events == ["stop", "release"]
    assert json.loads(Path(args.results_file).read_text(encoding="utf-8"))["status"] == "failed"
