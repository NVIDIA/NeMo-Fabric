# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify the OpenClaw code-review consumer's SDK and OpenShell wiring.

The tests cover both sandbox ownership modes, request context, and artifact
handling without starting a gateway or model endpoint.
"""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import get_type_hints

import pytest

from nemo_fabric import FabricRuntimeError

from examples.code_review_agent import openshell, openshell_sdk_consumer as consumer


@pytest.mark.parametrize("ownership", ["caller_owned", "fabric_owned"])
@pytest.mark.parametrize(
    ("harness", "relay"),
    [("openclaw", False), ("hermes", False), ("hermes", True)],
)
def test_code_review_consumer_uses_sdk_lifecycle_and_preserves_two_turn_context(
    ownership, harness, relay, tmp_path, monkeypatch
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
        harness=harness,
        relay=relay,
        model="nvidia/test-model",
        base_url="https://integrate.api.nvidia.com/v1",
        inputs=["Review calculator.py", "Summarize and remember cedar-4827"],
    )
    events = []
    requests = []
    state_file = tmp_path / "fabric-owned-sandbox.json"
    monkeypatch.setenv("OPENSHELL_FABRIC_SANDBOX_STATE_FILE", str(state_file))
    environment = SimpleNamespace(connection={"sandbox_id": "fabric-sandbox-id"})

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
            return environment

        async def prepare_environment(self, config):
            events.append("prepare")
            return environment

        async def start_runtime_in(self, config, environment):
            events.append("start")
            assert environment.connection["sandbox_id"] == "fabric-sandbox-id"
            assert config.harness.adapter_id == f"nvidia.fabric.{args.harness}"
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
    if ownership == "fabric_owned":
        assert json.loads(state_file.read_text(encoding="utf-8")) == {
            "name": args.sandbox_name,
            "id": "fabric-sandbox-id",
        }
    else:
        assert not state_file.exists()


@pytest.mark.parametrize("relay", [False, True])
def test_hermes_config_enables_code_review_tools_and_optional_relay(relay, tmp_path):
    policy = tmp_path / "policy.yaml"
    policy.write_text("version: 1\n", encoding="utf-8")
    args = SimpleNamespace(
        ownership="fabric_owned",
        gateway="http://127.0.0.1:18080",
        workspace="default",
        image="sha256:" + "a" * 64,
        policy_file=str(policy),
        results_file=str(tmp_path / "report.json"),
        sandbox_name="review-hermes",
        sandbox_id=None,
        harness="hermes",
        relay=relay,
        model="nvidia/test-model",
        base_url="https://integrate.api.nvidia.com/v1",
    )

    config = consumer.build_config(args)

    assert config.harness.adapter_id == "nvidia.fabric.hermes"
    assert config.tools.enabled == ["terminal", "read_file", "search_files"]
    assert config.models["default"].api_key_env == "OPENAI_API_KEY"
    assert config.runtime.artifacts == "/sandbox/code-review/artifacts"
    assert config.environment.artifacts == "/sandbox/code-review/artifacts"
    assert (config.relay is not None) is relay
    if relay:
        assert config.relay.output_dir == "/sandbox/code-review/artifacts/relay"
        assert config.relay.observability.atif.output_directory == (
            "/sandbox/code-review/artifacts/relay"
        )


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
        harness="openclaw",
        relay=False,
        model="nvidia/test-model",
        base_url="https://integrate.api.nvidia.com/v1",
        inputs=["Review", "Follow up"],
    )
    events = []
    state_file = tmp_path / "fabric-owned-sandbox.json"
    monkeypatch.setenv("OPENSHELL_FABRIC_SANDBOX_STATE_FILE", str(state_file))
    environment = SimpleNamespace(connection={"sandbox_id": "fabric-sandbox-id"})

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
            return environment

        async def start_runtime_in(self, config, environment):
            return FakeRuntime()

        async def release_environment(self, environment):
            events.append("release")

    monkeypatch.setattr(consumer, "Fabric", FakeFabric)
    report = asyncio.run(consumer.run_demo(args))

    assert report["status"] == "failed"
    assert events == ["stop", "release"]
    assert json.loads(Path(args.results_file).read_text(encoding="utf-8"))["status"] == "failed"
    assert json.loads(state_file.read_text(encoding="utf-8")) == {
        "name": args.sandbox_name,
        "id": "fabric-sandbox-id",
    }


def test_code_review_type_annotations_resolve():
    get_type_hints(openshell.OpenShellCodeReview._command)
    get_type_hints(consumer.run_demo)


def test_sdk_error_report_keeps_structured_stage_without_diagnostic_text(
    tmp_path, monkeypatch
):
    policy = tmp_path / "policy.yaml"
    policy.write_text("version: 1\n", encoding="utf-8")
    args = SimpleNamespace(
        ownership="fabric_owned",
        gateway="http://127.0.0.1:18080",
        workspace="default",
        image="sha256:" + "a" * 64,
        policy_file=str(policy),
        results_file=str(tmp_path / "report.json"),
        sandbox_name="review-error",
        sandbox_id=None,
        harness="openclaw",
        relay=False,
        model="nvidia/test-model",
        base_url="https://integrate.api.nvidia.com/v1",
        inputs=["Review", "Follow up"],
    )

    class FakeFabric:
        async def prepare_environment(self, config):
            raise FabricRuntimeError(
                "private provider diagnostic",
                stage="environment_prepare",
                code="provider_error",
            )

    monkeypatch.setattr(consumer, "Fabric", FakeFabric)
    report = asyncio.run(consumer.run_demo(args))

    assert report["error"] == {"code": "provider_error", "stage": "environment_prepare"}
    assert "private provider diagnostic" not in json.dumps(report)
