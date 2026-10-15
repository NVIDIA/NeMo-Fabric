# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify the echo example's host SDK configuration and lifecycle.

The tests exercise the public consumer path and its OpenShell provider wiring
without requiring a live gateway or inference endpoint.
"""

import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest

from examples.openshell_sdk.consumer import run_demo


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/fixtures/openshell_echo"
EXAMPLE_ADAPTER = ROOT / "examples/openshell_sdk/adapter"


@pytest.fixture
def sdk_example_environment(tmp_path, monkeypatch):
    task = tmp_path / "task"
    task.mkdir()
    shutil.copyfile(EXAMPLE_ADAPTER / "echo_adapter.py", task / "echo_adapter.py")
    descriptor = json.loads(
        (EXAMPLE_ADAPTER / "echo.fabric-adapter.json").read_text(encoding="utf-8")
    )
    descriptor["runner"] = {
        "module": "echo_adapter",
        "python": sys.executable,
        "cwd": str(task),
        "env": {"PYTHONPATH": str(task)},
    }
    (task / "echo.fabric-adapter.json").write_text(json.dumps(descriptor))
    workspace = task / "workspace"
    artifacts = task / "artifacts"
    workspace.mkdir()
    artifacts.mkdir()

    provider = tmp_path / "provider"
    provider.write_text(
        f"#!{sys.executable}\n" + (FIXTURE / "sdk_provider.py").read_text()
    )
    provider.chmod(0o700)
    provider_log = tmp_path / "provider.jsonl"
    monkeypatch.setenv("NEMO_FABRIC_OPEN_SHELL_PROVIDER", str(provider))
    monkeypatch.setenv(
        "FABRIC_TEST_ARTIFACT_READER",
        str(ROOT / "environment-providers/openshell/src/read_artifacts.py"),
    )
    monkeypatch.setenv("FABRIC_TEST_TASK_ROOT", str(task))
    monkeypatch.setenv("FABRIC_TEST_TASK_ARTIFACTS_ROOT", str(artifacts))
    monkeypatch.setenv("FABRIC_TEST_PROVIDER_LOG", str(provider_log))
    return task, workspace, artifacts, provider_log


@pytest.mark.parametrize("ownership", ["caller_owned", "fabric_owned"])
@pytest.mark.skipif(
    sys.platform == "win32",
    reason="The task runtime fixture uses POSIX sandbox paths and subprocesses.",
)
async def test_sdk_example_runs_task_agent_and_collects_receipt(
    ownership, sdk_example_environment, tmp_path, monkeypatch
):
    task, workspace, task_artifacts, provider_log = sdk_example_environment
    sandbox_name = f"sdk-{ownership}"
    args = SimpleNamespace(
        ownership=ownership,
        gateway="https://fixture.invalid",
        image="sha256:" + "a" * 64,
        policy_file=str(ROOT / "examples/openshell_sdk/policy.yaml"),
        results_file=str(tmp_path / ownership / "results.json"),
        input='{"message":"hello from the consumer"}',
        sandbox_name=sandbox_name,
        sandbox_id="caller-sandbox-id" if ownership == "caller_owned" else None,
    )
    sandbox_state = tmp_path / f"{ownership}-sandbox.json"
    monkeypatch.setenv("OPENSHELL_FABRIC_SANDBOX_STATE_FILE", str(sandbox_state))

    summary = await run_demo(
        args,
        task_workspace=str(workspace),
        task_artifacts=str(task_artifacts),
        task_discovery_path=str(task),
    )

    assert summary["first"]["request_id"] == "openshell-echo-first"
    assert summary["second"]["request_id"] == "openshell-echo-second"
    assert summary["first"]["output"] == {
        "first_input": {"message": "hello from the consumer"},
        "turn": 1,
    }
    assert summary["second"]["output"] == {
        "first_input": {"message": "hello from the consumer"},
        "turn": 2,
    }
    receipt = Path(summary["receipt_paths"][0])
    assert receipt.is_file()
    assert json.loads(receipt.read_text()) == summary["second"]["output"]
    assert json.loads(Path(args.results_file).read_text()) == summary

    records = [json.loads(line) for line in provider_log.read_text().splitlines()]
    lifecycle_operation = "attach" if ownership == "caller_owned" else "prepare"
    assert records[0]["frame"]["operation"] == lifecycle_operation
    runtime_commands = [
        record for record in records if record["frame"]["operation"] == "runtime_command"
    ]
    assert [record["frame"]["command"]["operation"] for record in runtime_commands] == [
        "start",
        "invoke",
        "invoke",
        "stop",
    ]
    first_request = runtime_commands[1]["frame"]["command"]["request"]
    assert first_request["request_id"] == "openshell-echo-first"
    assert first_request["context"] == {"turn": 1}
    assert runtime_commands[-1]["frame"]["command"]["operation"] == "stop"
    assert records[-1]["frame"]["operation"] == "release"
    assert records[-1]["frame"]["environment"]["ownership"] == ownership
    if ownership == "fabric_owned":
        assert json.loads(sandbox_state.read_text()) == {
            "name": sandbox_name,
            "id": "fixture-created-id",
        }
