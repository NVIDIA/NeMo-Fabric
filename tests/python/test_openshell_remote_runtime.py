# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise host-to-task runtime control through the public SDK.

A model-free stdio provider stands in for OpenShell while a separate task SDK
runs the fixture adapter. The tests cover lifecycle, runtime identity, request
forwarding, artifacts, and uncertain command results.
"""

import asyncio
import json
from pathlib import Path
import shutil
import sys

import pytest
from nemo_fabric import (
    DiscoveryConfig,
    EnvironmentConfig,
    EnvironmentReference,
    Fabric,
    FabricConfig,
    FabricRuntimeError,
    HarnessConfig,
    MetadataConfig,
    RunRequest,
    RuntimeConfig,
)

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="Unix OpenShell host provider"
)
FIXTURE = Path(__file__).resolve().parents[1] / "fixtures/openshell_echo"


@pytest.fixture(name="remote_config")
def remote_config_fixture(tmp_path, monkeypatch):
    task = tmp_path / "task"
    task.mkdir()
    shutil.copy(FIXTURE / "agent.py", task / "echo_adapter.py")
    descriptor = json.loads((FIXTURE / "echo.fabric-adapter.json").read_text())
    descriptor["runner"] = {
        "module": "echo_adapter",
        "python": sys.executable,
        "cwd": str(task),
        "env": {"PYTHONPATH": str(task)},
    }
    (task / "echo.fabric-adapter.json").write_text(json.dumps(descriptor))
    (task / "workspace").mkdir()
    provider = tmp_path / "provider"
    provider.write_text(
        f"#!{sys.executable}\n" + (FIXTURE / "sdk_provider.py").read_text()
    )
    provider.chmod(0o700)
    monkeypatch.setenv("NEMO_FABRIC_OPEN_SHELL_PROVIDER", str(provider))
    monkeypatch.setenv(
        "FABRIC_TEST_ARTIFACT_READER",
        str(
            Path(__file__).resolve().parents[2]
            / "environment-providers/openshell/src/read_artifacts.py"
        ),
    )
    monkeypatch.setenv("FABRIC_TEST_PROVIDER_LOG", str(tmp_path / "provider.jsonl"))
    return FabricConfig(
        metadata=MetadataConfig(name="echo"),
        harness=HarnessConfig(
            adapter_id=descriptor["adapter_id"], resolution="preinstalled"
        ),
        discovery=DiscoveryConfig(local_paths=[task]),
        runtime=RuntimeConfig(artifacts=tmp_path / "host-artifacts"),
        environment=EnvironmentConfig(
            provider="openshell",
            ownership="caller_owned",
            workspace=str(task / "workspace"),
            artifacts=str(task / "artifacts"),
            connection={
                "gateway": "https://fixture.invalid",
                "workspace": "default",
                "sandbox_id": f"fixture-{tmp_path.name}",
            },
            settings={"task_discovery_paths": [str(task)]},
        ),
    )


async def test_real_task_sdk_retains_state_and_collects_declared_artifacts(
    remote_config, tmp_path
):
    fabric = Fabric()
    environment = await fabric.attach_environment(
        remote_config,
        EnvironmentReference.from_mapping(
            {
                "provider": "openshell",
                "resource": {
                    "sandbox_name": "echo",
                    "sandbox_id": remote_config.environment.connection["sandbox_id"],
                },
            }
        ),
    )
    runtime = await fabric.start_runtime_in(remote_config, environment)
    try:
        with pytest.raises(FabricRuntimeError):
            await fabric.release_environment(environment)
        first_input = ["review-tag-fir-724", {"stream": True}, None]
        first = await runtime.invoke(
            request=RunRequest(
                input=first_input, request_id="host-first", context={"turn": 1}
            )
        )
        second = await runtime.invoke(
            request=RunRequest(
                input={"follow_up": True}, request_id="host-second", context={"turn": 2}
            )
        )
        assert first.runtime_id == second.runtime_id == runtime.runtime_id
        assert first.request_id == "host-first"
        assert second.request_id == "host-second"
        assert first.output == {"first_input": first_input, "turn": 1}
        assert second.output == {"first_input": first_input, "turn": 2}
        receipt = next(
            artifact
            for artifact in second.artifacts.artifacts
            if artifact.name == "echo"
        )
        assert receipt.path.is_file()
        assert json.loads(receipt.path.read_text()) == dict(second.output)
        assert str(receipt.path).startswith(str(tmp_path / "host-artifacts"))
        records = [
            json.loads(line)
            for line in (tmp_path / "provider.jsonl").read_text().splitlines()
        ]
        commands = [
            record
            for record in records
            if record["frame"]["operation"] == "runtime_command"
        ]
        assert commands[0]["output"]["value"]["runtime_id"] == runtime.runtime_id
        task_config = commands[0]["frame"]["command"]["config"]
        assert task_config["environment"]["provider"] == "local"
        assert task_config["environment"].get("connection", {}) == {}
        assert (
            task_config["runtime"]["artifacts"] == remote_config.environment.artifacts
        )
        assert commands[1]["frame"]["command"]["request"]["input"] == first_input
        assert commands[1]["frame"]["command"]["request"]["context"] == {"turn": 1}
    finally:
        await runtime.stop()
        await fabric.release_environment(environment)
    rebound = await fabric.attach_environment(
        remote_config,
        EnvironmentReference.from_mapping(
            {"provider": "openshell", "resource": {"sandbox_name": "echo"}}
        ),
    )
    restarted = await fabric.start_runtime_in(remote_config, rebound)
    try:
        result = await restarted.invoke(input="after-reattach")
        assert result.status == "succeeded"
        assert result.output == {"first_input": "after-reattach", "turn": 1}
    finally:
        await restarted.stop()
        await fabric.release_environment(rebound)


@pytest.mark.parametrize("partial", [None, {"partial": "review"}])
async def test_failed_task_result_preserves_output_error_and_artifacts(
    remote_config, partial
):
    fabric = Fabric()
    environment = await fabric.attach_environment(
        remote_config,
        EnvironmentReference.from_mapping(
            {"provider": "openshell", "resource": {"sandbox_name": "echo"}}
        ),
    )
    runtime = await fabric.start_runtime_in(remote_config, environment)
    try:
        await runtime.invoke(input="first")
        result = await runtime.invoke(
            request=RunRequest(
                input="second",
                request_id="failed-turn",
                context={"fail": True, "partial_output": partial},
            )
        )
        assert result.status == "failed"
        assert result.output == partial
        assert result.request_id == "failed-turn"
        assert result.error.code == "echo_failed"
        receipt = next(
            artifact
            for artifact in result.artifacts.artifacts
            if artifact.name == "echo"
        )
        assert receipt.path.is_file()
    finally:
        await runtime.stop()
        await fabric.release_environment(environment)


async def test_internal_sdk_files_are_excluded_from_declared_artifact_collection(
    remote_config,
):
    fabric = Fabric()
    environment = await fabric.attach_environment(
        remote_config,
        EnvironmentReference.from_mapping(
            {"provider": "openshell", "resource": {"sandbox_name": "echo"}}
        ),
    )
    runtime = await fabric.start_runtime_in(remote_config, environment)
    try:
        await runtime.invoke(input="first")
        result = await runtime.invoke(
            request=RunRequest(
                input="second",
                request_id="internal-output",
                context={"large_internal_artifact": True},
            )
        )
        assert result.status == "succeeded"
        assert [artifact.name for artifact in result.artifacts.artifacts] == ["echo"]
        assert result.artifacts.artifacts[0].path.is_file()
    finally:
        await runtime.stop()
        await fabric.release_environment(environment)


async def test_runtime_workers_invoke_concurrently_and_stop_independently(
    remote_config, tmp_path
):
    fabric = Fabric()
    environment = await fabric.attach_environment(
        remote_config,
        EnvironmentReference.from_mapping(
            {"provider": "openshell", "resource": {"sandbox_name": "echo"}}
        ),
    )
    first = await fabric.start_runtime_in(remote_config, environment)
    second = await fabric.start_runtime_in(remote_config, environment)
    third = None
    first_stopped = False
    barrier = str(tmp_path / "parallel-invoke")
    try:
        first_result, second_result = await asyncio.gather(
            first.invoke(
                request=RunRequest(
                    input="first",
                    request_id="parallel-first",
                    context={"parallel_barrier": barrier},
                )
            ),
            second.invoke(
                request=RunRequest(
                    input="second",
                    request_id="parallel-second",
                    context={"parallel_barrier": barrier},
                )
            ),
        )
        assert first_result.output["both_runtimes_entered"] is True
        assert second_result.output["both_runtimes_entered"] is True
        assert first_result.runtime_id == first.runtime_id
        assert second_result.runtime_id == second.runtime_id

        await first.stop()
        first_stopped = True
        after_stop = await second.invoke(input="still active")
        assert after_stop.status == "succeeded"
        third = await fabric.start_runtime_in(remote_config, environment)
        assert third.runtime_id not in {first.runtime_id, second.runtime_id}
        await third.stop()
        third = None
    finally:
        if third is not None:
            try:
                await third.stop()
            except FabricRuntimeError:
                pass
        if not first_stopped:
            try:
                await first.stop()
            except FabricRuntimeError:
                pass
        try:
            await second.stop()
        except FabricRuntimeError:
            pass
        await fabric.release_environment(environment)


def test_canonical_telemetry_survives_native_compatibility_fields():
    from nemo_fabric import TelemetryRef

    public = {
        "provider": "relay",
        "kind": "trace",
        "uri": "https://fixture.invalid/trace",
        "trace_id": "trace-1",
        "metadata": {"relay_enabled": True},
    }
    result = TelemetryRef.from_mapping({**public, "relay_enabled": False})
    assert result.to_mapping() == public
    native = TelemetryRef.from_mapping(
        {
            "relay_enabled": True,
            "metadata": {
                "telemetry_providers": ["relay"],
                "relay_output_dir": "/sandbox/relay",
                "trace_id": "trace-2",
            },
        }
    )
    assert native.provider == "relay"
    assert native.uri == "/sandbox/relay"
    assert native.trace_id == "trace-2"


@pytest.mark.parametrize("fault", ["start", "invoke", "stop"])
async def test_uncertain_outcomes_are_not_replayed_after_reattachment(
    remote_config, tmp_path, fault, monkeypatch
):
    monkeypatch.setenv("FABRIC_TEST_PROVIDER_FAULT", fault)
    fabric = Fabric()
    reference = EnvironmentReference.from_mapping(
        {"provider": "openshell", "resource": {"sandbox_name": "echo"}}
    )
    environment = await fabric.attach_environment(remote_config, reference)
    runtime = None
    runtime_id = None
    try:
        if fault == "start":
            with pytest.raises(FabricRuntimeError):
                await fabric.start_runtime_in(remote_config, environment)
        else:
            runtime = await fabric.start_runtime_in(remote_config, environment)
            runtime_id = runtime.runtime_id
            if fault == "invoke":
                with pytest.raises(FabricRuntimeError):
                    await runtime.invoke(input="apply once")
            else:
                with pytest.raises(FabricRuntimeError):
                    await runtime.stop()
        if fault == "stop":
            # The stop may have taken effect despite its malformed response; do
            # not retry it from test cleanup.
            runtime = None
        await fabric.release_environment(environment)
        if fault != "start":
            monkeypatch.setenv("FABRIC_TEST_PROVIDER_FAULT", "")
        rebound = await fabric.attach_environment(remote_config, reference)
        if fault == "start":
            with pytest.raises(FabricRuntimeError):
                await fabric.start_runtime_in(remote_config, rebound)
        else:
            restarted = await fabric.start_runtime_in(remote_config, rebound)
            assert restarted.runtime_id != runtime_id
            await restarted.stop()
        await fabric.release_environment(rebound)
        records = [
            json.loads(line)
            for line in (tmp_path / "provider.jsonl").read_text().splitlines()
        ]
        operations = [
            record["frame"]["command"]["operation"]
            for record in records
            if record["frame"]["operation"] == "runtime_command"
        ]
        assert operations.count("start") == (1 if fault == "start" else 2)
        assert operations.count("invoke") == (1 if fault == "invoke" else 0)
        assert operations.count("stop") == {
            "start": 0,
            "invoke": 1,
            "stop": 2,
        }[fault]
    finally:
        if runtime is not None:
            try:
                await runtime.stop()
            except FabricRuntimeError:
                pass
        try:
            await fabric.release_environment(environment)
        except FabricRuntimeError:
            # An unresolved operation deliberately retains the environment binding.
            pass
