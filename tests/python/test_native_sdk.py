# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke test for the installed native Python SDK."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import nemo_fabric._native as native
import pytest
from _utils.configs import hermes_shim_config
from examples.code_review_agent import BASE_DIR
from examples.code_review_agent import base_config
from nemo_fabric import Fabric
from nemo_fabric import FabricConfig
from nemo_fabric import FabricConfigError
from nemo_fabric import FabricRuntimeError
from nemo_fabric import FabricStateError
from nemo_fabric import RunRequest
from nemo_fabric import RuntimeStatus


@pytest.mark.parametrize("stop_fails", [False, True])
async def test_one_shot_lookup_matches_cleanup_result(
    hermes_shim_agent_dir, monkeypatch, stop_fails
):
    monkeypatch.setenv("ADAPTER_PYTHON", sys.executable)
    original_stop = native.stop_runtime

    def stop(*args):
        events = original_stop(*args)
        if stop_fails:
            raise RuntimeError("test-only stop failure after confirmed cleanup")
        return events

    monkeypatch.setattr(native, "stop_runtime", stop)
    client = Fabric()
    result = await client.run(
        hermes_shim_config(), base_dir=hermes_shim_agent_dir, input="final result"
    )
    saved = await client.invocation_result(
        artifacts=hermes_shim_agent_dir / "artifacts",
        agent_name=result.agent_name,
        runtime_id=result.runtime_id,
        request_id=result.request_id,
    )
    assert result.status == ("failed" if stop_fails else "succeeded")
    assert saved == result


@pytest.mark.skipif(sys.platform == "win32", reason="Unix umask permissions")
@pytest.mark.parametrize("existing_root", [False, True])
def test_result_storage_supports_group_writable_umask(
    hermes_shim_agent_dir, monkeypatch, existing_root
):
    monkeypatch.setenv("ADAPTER_PYTHON", sys.executable)
    artifacts = hermes_shim_agent_dir / "artifacts"
    if existing_root:
        artifacts.mkdir(mode=0o700)
    script = """
import asyncio, json, os, sys
from nemo_fabric import Fabric, FabricConfig
os.umask(0o002)
async def main():
    config = FabricConfig.from_mapping(json.loads(sys.argv[1]))
    runtime = await Fabric().start_runtime(config, base_dir=sys.argv[2])
    try:
        result = await runtime.invoke(input="private result storage")
        assert result.status == "succeeded"
    finally:
        await runtime.stop()
asyncio.run(main())
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            hermes_shim_config().model_dump_json(exclude_none=True),
            str(hermes_shim_agent_dir),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr


async def test_native_sdk(hermes_shim_agent_dir: Path):
    assert native.version()

    await smoke(Fabric(), hermes_shim_agent_dir)


@pytest.mark.parametrize("new_value", ["rotated-secret", None])
async def test_named_environment_changes_require_runtime_restart(
    hermes_shim_agent_dir, new_value
):
    name = "NEMO_FABRIC_TEST_NAMED_TOKEN"
    os.environ[name] = "startup-secret"
    os.environ["ADAPTER_PYTHON"] = sys.executable
    config = FabricConfig.from_mapping(
        {
            "metadata": {"name": "named-environment-probe"},
            "harness": {
                "adapter_id": "test.fabric.hermes_shim",
                "resolution": "preinstalled",
            },
            "discovery": {"local_paths": ["adapters"]},
            "runtime": {"artifacts": "./artifacts"},
            "environment": {"env_names": [name]},
        }
    )
    client = Fabric()
    runtime = await client.start_runtime(config, base_dir=hermes_shim_agent_dir)
    try:
        first = await runtime.invoke(input="before rotation")
        assert first.status == "succeeded"
        invocation_path = Path(first.metadata["fabric_invocation"])
        invocation = json.loads(invocation_path.read_text())
        assert invocation["runtime_context"]["environment"]["env"][name] == "[REDACTED]"
        if new_value is None:
            del os.environ[name]
            expected = "is not set"
        else:
            os.environ[name] = new_value
            expected = "changed after adapter start; restart the runtime"
        with pytest.raises(FabricRuntimeError, match=expected) as caught:
            await runtime.invoke(input="after rotation")
        assert caught.value.stage == "invoke"
        assert runtime.status is RuntimeStatus.FAILED
        assert name in str(caught.value)
        assert "startup-secret" not in str(caught.value)
        assert "rotated-secret" not in str(caught.value)
    finally:
        await runtime.stop()

    if new_value is not None:
        restarted = await client.start_runtime(config, base_dir=hermes_shim_agent_dir)
        try:
            result = await restarted.invoke(input="after restart")
            assert result.status == "succeeded"
        finally:
            await restarted.stop()


def _process_running(pid: int) -> bool:
    if sys.platform == "win32":
        tasks = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            check=True,
        )
        return f'"{pid}"' in tasks.stdout
    state = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True
    ).stdout.strip()
    # Orphans can linger as zombies until reaped; they no longer run.
    return bool(state) and not state.startswith("Z")


async def test_native_deadline_keeps_type_and_evicts_runtime(
    hermes_shim_agent_dir, tmp_path
):
    config = FabricConfig.from_mapping(
        {
            "metadata": {"name": "deadline-probe"},
            "harness": {
                "adapter_id": "test.fabric.hermes_shim",
                "resolution": "preinstalled",
            },
            "discovery": {"local_paths": ["adapters"]},
            # Leave the host time to start its children on slow runners.
            "runtime": {"timeout_seconds": 5},
        }
    )
    runtime = await Fabric().start_runtime(config, base_dir=hermes_shim_agent_dir)
    pid_file = tmp_path / "children.pids"
    with pytest.raises(FabricRuntimeError) as caught:
        await runtime.invoke(
            request=RunRequest(
                input="hang",
                context={"delay_seconds": 30, "child_pid_file": str(pid_file)},
            )
        )
    child, grandchild = (int(pid) for pid in pid_file.read_text().split())
    # Only a Windows job object reaches a grandchild whose launcher exited.
    killed = [child, grandchild] if sys.platform == "win32" else [child]
    deadline = time.monotonic() + 5
    while any(map(_process_running, killed)) and time.monotonic() < deadline:
        time.sleep(0.05)
    survivors = [pid for pid in (child, grandchild) if _process_running(pid)]
    for pid in survivors:
        os.kill(pid, 9)
    assert not set(survivors) & set(killed), f"host children survived: {survivors}"
    assert caught.value.code == "timeout"
    assert caught.value.stage == "invoke"
    assert isinstance(caught.value.__cause__, TimeoutError)
    assert runtime.status == RuntimeStatus.FAILED
    with pytest.raises(FabricStateError):
        await runtime.invoke(input="retry")
    await runtime.stop()
    assert runtime.status == RuntimeStatus.STOPPED


@pytest.mark.parametrize("sampling", [False, True])
def test_runner_real_inner_deadline_preserves_failure_evidence(
    hermes_shim_agent_dir, tmp_path, sampling
):
    from nemo_fabric.integrations.harbor.models import FabricRunPayload
    from nemo_fabric.integrations.harbor.models import FabricRunnerFailure

    config = FabricConfig.from_mapping(
        {
            "metadata": {"name": "runner-deadline-probe"},
            "harness": {
                "adapter_id": "test.fabric.hermes_shim",
                "resolution": "preinstalled",
            },
            "discovery": {"local_paths": ["adapters"]},
            "runtime": {"timeout_seconds": 1, "artifacts": str(tmp_path / "artifacts")},
            "system_stats": {"sampling": {"interval_ms": 100, "max_samples": 8}}
            if sampling
            else {},
        }
    )
    payload = FabricRunPayload(
        config=config,
        config_base_dir=str(hermes_shim_agent_dir),
        logs_dir=str(tmp_path),
        request=RunRequest(input="hang", context={"delay_seconds": 30}),
    )
    spec = tmp_path / "spec.json"
    result = tmp_path / "result.json"
    spec.write_text(payload.model_dump_json(), encoding="utf-8")
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "nemo_fabric.integrations.harbor.runner",
            "--spec",
            str(spec),
            "--result",
            str(result),
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert process.returncode == 1
    failure = FabricRunnerFailure.model_validate_json(result.read_text())
    assert failure.runner_error.code == "timeout"
    assert failure.runner_error.stage == "invoke"
    assert not failure.runner_error.retryable

    assert failure.evidence.runtime_id.startswith("runtime-")
    assert failure.evidence.request_id == payload.request.request_id
    assert failure.evidence.cleanup == "completed"
    assert failure.evidence.usage is None
    assert failure.evidence.coverage["usage"] == "missing"
    assert failure.evidence.execution is not None
    assert failure.evidence.partial_result is None
    assert "status" not in json.loads(result.read_text())
    if sampling:
        assert failure.evidence.system_stats["elapsed_seconds"] > 0
        assert failure.evidence.system_stats["sampled"]["sample_count"] > 0
        assert failure.evidence.coverage["system_stats"] == "partial"
        assert {ref["kind"] for ref in failure.evidence.artifacts} >= {
            "system_stats_summary",
            "system_stats_samples",
        }
    else:
        assert failure.evidence.system_stats is None
        assert failure.evidence.coverage["system_stats"] == "missing"


async def test_adapter_python_selects_python_adapter_interpreter(
    hermes_shim_agent_dir: Path,
):
    os.environ["ADAPTER_PYTHON"] = sys.executable

    result = await Fabric().run(
        hermes_shim_config(),
        base_dir=hermes_shim_agent_dir,
        input="hello adapter python",
    )

    assert result["status"] == "succeeded"


def test_adapter_python_rejects_invalid_path_during_plan(
    hermes_shim_agent_dir: Path,
):
    invalid_python = hermes_shim_agent_dir / "missing-python"
    os.environ["ADAPTER_PYTHON"] = str(invalid_python)

    with pytest.raises(FabricConfigError, match="ADAPTER_PYTHON"):
        Fabric().plan(
            hermes_shim_config(),
            base_dir=hermes_shim_agent_dir,
        )


def test_native_run_rejects_multiple_request_sources(hermes_shim_agent_dir: Path):
    with pytest.raises(RuntimeError, match="mutually exclusive"):
        native.run_config(
            json.dumps(hermes_shim_config().model_dump(mode="json", exclude_none=True)),
            str(hermes_shim_agent_dir),
            "text",
            None,
            "{}",
            None,
        )


def test_plan_rejects_adapter_incompatible_normalized_tool_policy():
    config = base_config()
    config.harness.adapter_id = "nvidia.fabric.codex"
    config.models["default"].provider = "openai"
    config.models["default"].model = "openai/gpt-5.4"
    config.models["default"].temperature = None
    config.block_tools("Bash")

    with pytest.raises(
        FabricConfigError,
        match=r"nvidia\.fabric\.codex.*tools\.blocked",
    ):
        Fabric().plan(config, base_dir=BASE_DIR)


@pytest.mark.parametrize(
    ("authentication", "capability"),
    [
        ({"type": "oauth2"}, "mcp.auth.oauth2"),
        (
            {
                "type": "service_account",
                "client_id": "fabric-client",
                "client_secret_env": "MCP_CLIENT_SECRET",
                "token_url": "https://auth.example/token",
                "token_endpoint_auth_method": "client_secret_basic",
            },
            "mcp.auth.service_account",
        ),
    ],
)
def test_plan_rejects_adapter_incompatible_mcp_authentication(
    authentication,
    capability,
):
    config = FabricConfig.from_mapping(
        {
            "metadata": {"name": "unsupported-mcp-authentication"},
            "harness": {"adapter_id": "nvidia.fabric.claude"},
            "mcp": {
                "servers": {
                    "docs": {
                        "transport": "streamable-http",
                        "url": "https://mcp.example/docs",
                        "authentication": authentication,
                    }
                }
            },
        }
    )

    with pytest.raises(FabricConfigError) as exc_info:
        Fabric().plan(config, base_dir=BASE_DIR)

    message = str(exc_info.value)
    assert "mcp.servers.docs.authentication" in message
    assert capability in message


async def smoke(client: Fabric, fixture_agent: Path) -> None:
    example_config = base_config()

    plan = client.plan(example_config, base_dir=BASE_DIR)
    assert plan["agent_name"] == "code-review-agent"
    assert plan.base_dir == BASE_DIR
    assert plan.config.metadata.name == "code-review-agent"
    assert (
        plan["adapter_descriptor"]["descriptor"]["adapter_id"] == "nvidia.fabric.hermes"
    )
    assert "mcp_servers" not in plan["capability_plan"]["native"]
    assert plan["capability_plan"]["native"]["skill_paths"]

    minimal = FabricConfig.from_mapping(
        {
            "metadata": {"name": "minimal-typed-agent"},
            "harness": {"adapter_id": "nvidia.fabric.hermes"},
        }
    )
    minimal_plan = client.plan(minimal)
    assert minimal_plan.config.runtime.input_schema == "text"
    assert minimal_plan.config.runtime.output_schema == "text"

    typed_config = FabricConfig.from_mapping(
        {
            "schema_version": "fabric.agent/v1alpha1",
            "metadata": {"name": "typed-hermes-shim-agent"},
            "harness": {
                "adapter_id": "test.fabric.hermes_shim",
                "resolution": "preinstalled",
                "settings": {},
            },
            "discovery": {"local_paths": ["adapters"]},
            "models": {
                "default": {
                    "provider": "test",
                    "model": "test-model",
                    "temperature": 0.0,
                }
            },
            "runtime": {
                "input_schema": "chat",
                "output_schema": "message",
                "artifacts": "./artifacts",
                "timeout_seconds": 30,
            },
            "environment": {
                "provider": "local",
                "workspace": "./repos/my-service",
                "artifacts": "./artifacts/local",
            },
            "mcp": {
                "servers": {
                    "github": {
                        "transport": "streamable-http",
                        "url": "${GITHUB_MCP_URL}",
                        "exposure": "harness_native",
                        "custom_headers": {"X-Tenant": "fabric"},
                        "authentication": {
                            "type": "oauth2",
                            "client_id": "fabric-client",
                            "scopes": ["repo"],
                        },
                    }
                }
            },
            "telemetry": {"providers": {"relay": {}}},
            "relay": {"output_dir": "./artifacts/relay"},
            "consumer_extension": {
                "base": True,
                "custom": True,
                "nested": {"first": 1, "second": 2},
            },
        }
    )
    typed_plan = client.plan(
        typed_config,
        base_dir=fixture_agent,
    )
    assert typed_plan["agent_name"] == "typed-hermes-shim-agent"
    assert (
        typed_plan["adapter_descriptor"]["provenance"][0]["source"] == "explicit_local"
    )
    assert typed_plan["telemetry_plan"]["relay_enabled"] is True
    native_mcp = typed_plan["capability_plan"]["native"]["mcp_servers"]["github"]
    assert native_mcp["custom_headers"] == {"X-Tenant": "fabric"}
    assert native_mcp["authentication"] == {
        "type": "oauth2",
        "client_id": "fabric-client",
        "scopes": ["repo"],
    }
    resolved_config = typed_plan.config.to_mapping()
    assert resolved_config["harness"]["adapter_id"] == "test.fabric.hermes_shim"
    assert "settings" not in resolved_config["harness"]
    assert resolved_config["runtime"]["timeout_seconds"] == 30
    assert resolved_config["consumer_extension"] == {
        "base": True,
        "custom": True,
        "nested": {"first": 1, "second": 2},
    }

    result = await client.run(
        hermes_shim_config(),
        base_dir=fixture_agent,
        input="hello native",
    )
    async with await client.start_runtime(
        hermes_shim_config(),
        base_dir=fixture_agent,
    ) as runtime:
        first = await runtime.invoke(input="hello runtime one")
        second = await runtime.invoke(input="hello runtime two")
        openai_stream = runtime.invoke_openai_stream(input="hello streaming")
        openai_chunks = [chunk async for chunk in openai_stream]
        openai_result = await openai_stream.result()
        empty_stream = runtime.invoke_openai_stream(
            request=RunRequest(
                input="hello empty streaming",
                context={"openai_stream_mode": "empty"},
            )
        )
        empty_chunks = [chunk async for chunk in empty_stream]
        empty_result = await empty_stream.result()

    assert result["status"] == "succeeded"
    assert result.harness == "test.fabric.hermes_shim"
    assert result["adapter_kind"] == "python"
    assert result["metadata"]["adapter_runner"] == "persistent_local_host"
    assert result["output"]["received"] == "hello native"
    assert result.output["native_mcp_servers"] == ["github"]
    assert any(artifact.name == "stdout" for artifact in result.artifacts.artifacts)
    assert first["status"] == "succeeded"
    assert second["status"] == "succeeded"
    assert first.harness == "test.fabric.hermes_shim"
    assert first["runtime_id"] == second["runtime_id"]
    assert runtime.handle["runtime_id"] == first["runtime_id"]
    assert runtime.supports_openai_streaming is True
    assert [
        choice["delta"]["content"]
        for chunk in openai_chunks
        for choice in chunk["choices"]
    ] == ["hel", "lo"]
    assert openai_result.status == "succeeded"
    assert openai_result.output["received"] == "hello streaming"
    assert openai_result.output["openai_stream_invocation_count"] == 1
    assert empty_chunks == []
    assert empty_result.status == "succeeded"
    assert empty_result.output["openai_stream_invocation_count"] == 2


async def test_failed_final_save_never_exposes_pre_cleanup_success(
    hermes_shim_agent_dir, monkeypatch
):
    os.environ["ADAPTER_PYTHON"] = sys.executable
    original_stop = native.stop_runtime

    def stop(*args):
        original_stop(*args)
        raise RuntimeError("test-only stop failure")

    def save(*args):
        raise RuntimeError("test-only final save failure")

    monkeypatch.setattr(native, "stop_runtime", stop)
    monkeypatch.setattr(native, "_save_run_result", save)
    client = Fabric()
    with pytest.raises(FabricRuntimeError) as caught:
        await client.run(
            hermes_shim_config(), base_dir=hermes_shim_agent_dir, input="hello"
        )
    assert caught.value.code == "result_storage_failed"
    partial = caught.value.details["runtime_evidence"]["partial_result"]
    assert partial["status"] == "failed"
    saved = await client.invocation_result(
        artifacts=hermes_shim_agent_dir / "artifacts",
        agent_name=partial["agent_name"],
        runtime_id=partial["runtime_id"],
        request_id=partial["request_id"],
    )
    assert saved is None


async def test_default_result_root_matches_native_writes_and_lookup(
    hermes_shim_agent_dir,
):
    import shutil

    os.environ["ADAPTER_PYTHON"] = sys.executable
    config = hermes_shim_config()
    config.runtime.artifacts = None
    config.environment.artifacts = None
    client = Fabric()
    result = await client.run(
        config, base_dir=hermes_shim_agent_dir, input="default storage"
    )
    try:
        saved = await client.invocation_result(
            agent_name=result.agent_name,
            runtime_id=result.runtime_id,
            request_id=result.request_id,
        )
        assert saved == result
    finally:
        directory = Path(result.metadata["fabric_home"]).parent
        assert directory.name == result.runtime_id
        shutil.rmtree(directory)


@pytest.mark.parametrize("artifact_source", ["runtime", "environment"])
@pytest.mark.parametrize("select_environment", [False, True])
async def test_stopped_runtime_result_lookup_retains_storage_and_environment(
    hermes_shim_agent_dir, artifact_source, select_environment
):
    from nemo_fabric import EnvironmentHandle

    os.environ["ADAPTER_PYTHON"] = sys.executable
    config = hermes_shim_config()
    config.runtime.artifacts = None
    config.environment.artifacts = None
    getattr(config, artifact_source).artifacts = "retained-results"
    client = Fabric()
    runtime = await client.start_runtime(config, base_dir=hermes_shim_agent_dir)
    try:
        result = await runtime.invoke(input="retain lookup context")
    finally:
        await runtime.stop()
    environment = EnvironmentHandle.from_mapping(
        runtime.handle.to_mapping()["environment"]
    )
    saved = await client.invocation_result(
        agent_name=result.agent_name,
        runtime_id=result.runtime_id,
        request_id=result.request_id,
        environment=environment if select_environment else None,
    )
    assert saved == result
    assert runtime.runtime_id not in client._runtimes
    if select_environment:
        wrong = environment.to_mapping()
        wrong["environment_id"] = "another-environment"
        with pytest.raises(FabricStateError) as caught:
            await client.invocation_result(
                agent_name=result.agent_name,
                runtime_id=result.runtime_id,
                request_id=result.request_id,
                environment=EnvironmentHandle.from_mapping(wrong),
                artifacts=hermes_shim_agent_dir / "retained-results",
            )
        assert caught.value.code == "runtime_unavailable"


@pytest.mark.skipif(sys.platform == "win32", reason="Unix shared temporary root")
def test_default_storage_ignores_a_precreated_shared_temporary_name(
    hermes_shim_agent_dir, tmp_path
):
    shared = tmp_path / "shared"
    shared.mkdir(mode=0o777)
    # Another user can occupy this predictable name before Fabric's first run.
    (shared / f"nemo-fabric-{os.geteuid()}").write_text("occupied")
    config = hermes_shim_config()
    config.runtime.artifacts = None
    config.environment.artifacts = None
    script = """
import asyncio, json, os, shutil, sys
from pathlib import Path
from nemo_fabric import Fabric, FabricConfig
async def main():
    client = Fabric()
    result = await client.run(FabricConfig.from_mapping(json.loads(sys.argv[1])), base_dir=sys.argv[2], input="private default storage")
    directory = Path(result.metadata["fabric_home"]).parent
    try:
        assert Path(sys.argv[3]) not in directory.parents
        saved = await client.invocation_result(agent_name=result.agent_name, runtime_id=result.runtime_id, request_id=result.request_id)
        assert saved == result
    finally:
        shutil.rmtree(directory)
asyncio.run(main())
"""
    environment = dict(os.environ, TMPDIR=str(shared), ADAPTER_PYTHON=sys.executable)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            config.model_dump_json(exclude_none=True),
            str(hermes_shim_agent_dir),
            str(shared),
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("one_shot", [False, True])
@pytest.mark.parametrize("failure", ["crash", "timeout", "invalid"])
async def test_dispatched_failure_result_is_retrievable(
    blocking_adapter_config, tmp_path, one_shot, failure
):
    config = blocking_adapter_config
    config.runtime.timeout_seconds = 0.2
    adapter = tmp_path / "adapter.py"
    source = adapter.read_text().replace(
        "        if operation=='invoke':",
        "        if operation=='invoke':\n"
        "            with (root/'executions').open('a') as count: count.write('once\\n')\n"
        "            if (root/'crash').exists(): os._exit(17)",
    )
    if failure == "invalid":
        source = source.replace(
            "output={'status':'succeeded','output':{'response':'released'}}",
            "output={'status':'invalid','private':'synthetic-sensitive-value'}",
        )
        (tmp_path / "release").touch()
    elif failure == "crash":
        (tmp_path / "crash").touch()
    adapter.write_text(source)
    client = Fabric()
    request = RunRequest(request_id="terminal-failure", input="synthetic test")
    runtime = None
    try:
        if one_shot:
            with pytest.raises(FabricRuntimeError) as caught:
                await client.run(config, base_dir=tmp_path, request=request)
            runtime_id = caught.value.details["runtime_evidence"]["runtime_id"]
        else:
            runtime = await client.start_runtime(config, base_dir=tmp_path)
            runtime_id = runtime.runtime_id
            with pytest.raises(FabricRuntimeError):
                await runtime.invoke(request=request)
        saved = await client.invocation_result(
            agent_name=config.metadata.name,
            runtime_id=runtime_id,
            request_id=request.request_id,
            artifacts=tmp_path / "artifacts",
        )
        assert saved is not None
        assert saved.status == "failed"
        assert saved.output is None
        assert (
            saved.error.code
            == {
                "crash": "host_crashed",
                "timeout": "host_timeout",
                "invalid": "invalid_agent_run_result",
            }[failure]
        )
        assert saved.error.retryable is False
        assert saved.error.metadata["effects"] == "unknown"
        assert "synthetic-sensitive-value" not in json.dumps(saved.to_mapping())
        if runtime is not None:
            replayed = json.loads(
                native.invoke_runtime(
                    json.dumps(runtime._plan.to_mapping()),
                    json.dumps(runtime.handle.to_mapping()),
                    json.dumps(request.to_mapping()),
                )
            )
            assert replayed["invocation_id"] == saved.invocation_id
            assert replayed["error"] == saved.error.to_mapping()
            with pytest.raises(FabricStateError):
                await runtime.invoke(request=request)
        assert (tmp_path / "executions").read_text() == "once\n"
    finally:
        if runtime is not None:
            await runtime.stop()


@pytest.mark.skipif(sys.platform == "win32", reason="Unix storage ownership")
@pytest.mark.parametrize("one_shot", [False, True])
def test_unusable_home_uses_private_temporary_storage(
    hermes_shim_agent_dir, tmp_path, one_shot
):
    temporary = tmp_path / "fallback"
    temporary.mkdir()
    config = hermes_shim_config()
    config.runtime.artifacts = None
    config.environment.artifacts = None
    script = """
import asyncio, json, os, pathlib, sys
from nemo_fabric import Fabric, FabricConfig
async def main():
    client = Fabric()
    config = FabricConfig.from_mapping(json.loads(sys.argv[1]))
    if sys.argv[3] == "True":
        result = await client.run(config, base_dir=sys.argv[2], input="home fallback")
    else:
        runtime = await client.start_runtime(config, base_dir=sys.argv[2])
        try:
            result = await runtime.invoke(input="home fallback")
        finally:
            await runtime.stop()
    saved = await client.invocation_result(agent_name=result.agent_name, runtime_id=result.runtime_id, request_id=result.request_id)
    assert saved == result
    root = pathlib.Path(os.environ['TMPDIR']) / f'nemo-fabric-{os.geteuid()}-private'
    assert root.stat().st_mode & 0o777 == 0o700
    assert pathlib.Path(result.metadata['fabric_home']).is_relative_to(root)
asyncio.run(main())
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            config.model_dump_json(exclude_none=True),
            str(hermes_shim_agent_dir),
            str(one_shot),
        ],
        env=dict(
            os.environ, HOME="/", TMPDIR=str(temporary), ADAPTER_PYTHON=sys.executable
        ),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
