# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SDK server checks use the same health contract as the command bridge."""

import asyncio
import json
import os
import time
import sys
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from nemo_fabric import CheckReport, CheckRequest, EnvironmentHandle, Fabric
from nemo_fabric.runtime_server.health import server_report

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="Unix OpenShell command bridge"
)


@pytest.fixture(name="server_environment")
def server_environment_fixture():
    return EnvironmentHandle.from_mapping(
        {
            "environment_id": "check-environment",
            "provider": "openshell",
            "control_location": "in_env_control",
            "ownership": "caller_owned",
            "connection": {
                "gateway": "https://fixture.invalid",
                "sandbox_name": "check-server",
                "sandbox_id": "immutable-check-server",
            },
        }
    )


def test_sdk_request_rejects_removed_operational_level():
    with pytest.raises(ValidationError, match="level"):
        CheckRequest(scope="runtime", runtime_id="runtime-1", level="operational")


@pytest.mark.parametrize("state", ["responding", "stopping", "unknown"])
async def test_sdk_server_reports_preserve_command_health(
    state, server_environment, monkeypatch
):
    native = MagicMock()

    def check(environment_json, request_json):
        assert json.loads(environment_json)["provider"] == "openshell"
        request = json.loads(request_json)
        assert request["scope"] == "server"
        assert 0 < request["timeout_seconds"] <= 3
        time.sleep(0.02)
        return json.dumps(server_report(state, time.monotonic()))

    native.check_environment.side_effect = check
    monkeypatch.setattr("nemo_fabric.client._native", native)
    report = await Fabric().check(
        environment=server_environment, request=CheckRequest(scope="server")
    )
    assert isinstance(report, CheckReport)
    assert report.server["state"] == state
    assert report.elapsed_ms >= 20
    native.start_runtime.assert_not_called()
    native.stop_runtime.assert_not_called()
    native.invoke_runtime.assert_not_called()


async def test_sdk_server_timeout_has_no_lifecycle_effects(
    server_environment, monkeypatch
):
    native = MagicMock()

    def check(*args):
        time.sleep(0.06)
        return json.dumps(server_report("responding", time.monotonic()))

    native.check_environment.side_effect = check
    monkeypatch.setattr("nemo_fabric.client._native", native)
    report = await Fabric().check(
        environment=server_environment,
        request=CheckRequest(scope="server", timeout_seconds=0.01),
    )
    assert report.outcome == "unknown"
    assert report.reason_code == "probe_timeout"
    assert report.elapsed_ms >= 10
    native.stop_runtime.assert_not_called()
    await asyncio.sleep(0.07)


async def test_sdk_server_transport_errors_preserve_unknown_evidence(
    server_environment, monkeypatch
):
    report = server_report("unknown", time.monotonic())
    report["reason_code"] = "probe_error"
    report["checks"][0]["reason_code"] = "probe_error"
    native = MagicMock()
    native.check_environment.return_value = json.dumps(report)
    monkeypatch.setattr("nemo_fabric.client._native", native)
    result = await Fabric().check(
        environment=server_environment, request=CheckRequest(scope="server")
    )
    assert result.outcome == "unknown"
    assert result.reason_code == "probe_error"
    assert result.server["state"] == "unknown"


@pytest.mark.parametrize("state", ["responding", "stopping"])
async def test_native_sdk_routes_server_checks_without_starting_a_runtime(
    state, server_environment, tmp_path, monkeypatch
):
    record = tmp_path / "command.json"
    provider = tmp_path / "provider"
    provider.write_text(
        f"#!{sys.executable}\n"
        "import json, sys, time\n"
        "from pathlib import Path\n"
        "from nemo_fabric.runtime_server.health import server_report\n"
        "for line in sys.stdin:\n"
        "    frame = json.loads(line)\n"
        f"    Path({str(record)!r}).write_text(json.dumps(frame))\n"
        f"    report = server_report({state!r}, time.monotonic())\n"
        "    print(json.dumps({'protocol_version': frame['protocol_version'], "
        "'request_id': frame['request_id'], 'status': 'succeeded', "
        "'output': {'status': 'returned', 'value': report}}), flush=True)\n"
    )
    provider.chmod(0o700)
    monkeypatch.setenv("NEMO_FABRIC_OPEN_SHELL_PROVIDER", str(provider))
    report = await Fabric().check(
        environment=server_environment, request=CheckRequest(scope="server")
    )
    assert report.server["state"] == state
    frame = json.loads(record.read_text())
    assert frame["operation"] == "runtime_command"
    command = frame["command"]
    assert command["operation"] == "check"
    assert set(command) == {"operation", "timeout_seconds"}
    assert 0 < command["timeout_seconds"] <= 3


@pytest.mark.parametrize(
    ("code", "reason"),
    [
        ("probe_timeout", "probe_timeout"),
        ("gateway_timeout", "probe_timeout"),
        ("provider_unavailable", "probe_error"),
    ],
)
async def test_native_provider_preserves_early_timeout_reason(
    code, reason, server_environment, tmp_path
):
    provider = tmp_path / "provider"
    provider.write_text(
        f"#!{sys.executable}\n"
        "import json, sys\n"
        "request = json.loads(sys.stdin.readline())\n"
        "print(json.dumps({'protocol_version': request['protocol_version'], "
        "'request_id': request['request_id'], 'status': 'failed', "
        f"'error': {{'code': {code!r}, 'message': 'TEST_ONLY_PRIVATE'}}}}), flush=True)\n"
    )
    provider.chmod(0o700)
    os.environ["NEMO_FABRIC_OPEN_SHELL_PROVIDER"] = str(provider)
    report = await Fabric().check(
        environment=server_environment,
        request=CheckRequest(scope="server", timeout_seconds=3),
    )
    assert report.outcome == "unknown"
    assert report.reason_code == reason
    assert report.elapsed_ms < 3000
    assert "TEST_ONLY_PRIVATE" not in json.dumps(report.to_mapping())


@pytest.mark.parametrize("stopping", [False, True])
async def test_sdk_server_check_reaches_the_real_command_bridge(
    stopping, server_environment, server_sdk_bridge
):
    host, log = server_sdk_bridge
    host.shutting_down = stopping
    report = await Fabric().check(
        environment=server_environment, request=CheckRequest(scope="server")
    )
    assert report.outcome == ("failed" if stopping else "passed")
    assert report.server["state"] == ("stopping" if stopping else "responding")
    assert report.runtime is None
    commands = [json.loads(line) for line in log.read_text().splitlines()]
    assert [command["operation"] for command in commands] == ["check"]
    assert "agent" not in commands[0]
    assert "runtime_id" not in commands[0]


async def test_runtime_status_remote_reaches_sdk_in_the_original_server(
    blocking_adapter_config,
    server_environment,
    server_sdk_bridge,
    tmp_path,
    runtime_status_validator,
):
    host, log = server_sdk_bridge
    config = blocking_adapter_config.model_copy(deep=True)
    config.environment.workspace = str(tmp_path)
    config.discovery.local_paths = [str(tmp_path / "blocking.fabric-adapter.json")]
    started = await host.handle(
        {
            "interface_version": 2,
            "operation": "start",
            "agent": "blocking",
            "config": config.to_mapping(),
        }
    )
    assert started["status"] == "succeeded", started
    runtime_id = started["result"]["runtime_id"]
    try:
        later_client = Fabric()
        active = await later_client.runtime_status(
            environment=server_environment, runtime_id=runtime_id, agent_name="blocking"
        )
        assert active.status == "active"
        assert active.server_state == "running"
        assert active.handle.runtime_id == runtime_id
        assert active.handle.environment["provider"] == "local"
        runtime_status_validator.validate(active.to_mapping())
        stopped = await host.handle(
            {
                "interface_version": 2,
                "operation": "stop",
                "agent": "blocking",
                "runtime_id": runtime_id,
            }
        )
        assert stopped["status"] == "succeeded"
        report = await later_client.runtime_status(
            environment=server_environment, runtime_id=runtime_id, agent_name="blocking"
        )
        assert report.status == "stopped"
        assert report.handle is None
        runtime_status_validator.validate(report.to_mapping())
        commands = [json.loads(line) for line in log.read_text().splitlines()]
        assert commands == [
            {"operation": "status", "agent": "blocking", "runtime_id": runtime_id},
            {"operation": "status", "agent": "blocking", "runtime_id": runtime_id},
        ]
        assert not (tmp_path / "entered").exists()
    finally:
        await host.shutdown()


async def test_server_runtime_limit_has_no_effects_and_start_recovers_after_stop(
    blocking_adapter_config, server_sdk_bridge, tmp_path, monkeypatch
):
    monkeypatch.setattr("nemo_fabric.client._MAX_RUNTIME_RECORDS", 1)
    host, _ = server_sdk_bridge
    config = blocking_adapter_config.model_copy(deep=True)
    config.environment.workspace = str(tmp_path)
    config.discovery.local_paths = [str(tmp_path / "blocking.fabric-adapter.json")]
    command = {
        "interface_version": 2,
        "operation": "start",
        "agent": "blocking",
        "config": config.to_mapping(),
    }
    try:
        first = await host.handle(command)
        assert first["status"] == "succeeded", first
        rejected = await host.handle(command)
        assert rejected["error"]["code"] == "runtime_limit_reached", rejected
        assert rejected["error"]["effects"] == "none"
        assert rejected["changed"] is False
        assert not host.start_uncertain
        stopped = await host.handle(
            {
                "interface_version": 2,
                "operation": "stop",
                "agent": "blocking",
                "runtime_id": first["result"]["runtime_id"],
            }
        )
        assert stopped["status"] == "succeeded", stopped
        replacement = await host.handle(command)
        assert replacement["status"] == "succeeded", replacement
        assert replacement["result"]["runtime_id"] != first["result"]["runtime_id"]
    finally:
        await host.shutdown()


@pytest.mark.parametrize(
    "change",
    [
        {"runtime_id": "other"},
        {"agent_name": "other"},
        {"status": "unrecognized"},
        {"surprise": "TEST_ONLY_SECRET"},
    ],
)
async def test_runtime_status_remote_rejects_invalid_or_mismatched_reports(
    server_environment, monkeypatch, change
):
    from nemo_fabric import FabricConfigError

    native = MagicMock()
    native.runtime_status_environment.return_value = json.dumps(
        {
            "runtime_id": "runtime-1",
            "agent_name": "demo",
            "status": "stopped",
            "handle": None,
            "server_state": "stopped",
            **change,
        }
    )
    monkeypatch.setattr("nemo_fabric.client._native", native)
    with pytest.raises(FabricConfigError):
        await Fabric().runtime_status(
            environment=server_environment, runtime_id="runtime-1", agent_name="demo"
        )
    native.start_runtime.assert_not_called()
    native.check_environment.assert_not_called()


async def test_runtime_status_remote_during_invocation_keeps_the_original_runtime(
    blocking_adapter_config, server_environment, server_sdk_bridge, tmp_path
):
    host, log = server_sdk_bridge
    config = blocking_adapter_config.model_copy(deep=True)
    config.environment.workspace = str(tmp_path)
    config.discovery.local_paths = [str(tmp_path / "blocking.fabric-adapter.json")]
    started = await host.handle(
        {
            "interface_version": 2,
            "operation": "start",
            "agent": "blocking",
            "config": config.to_mapping(),
        }
    )
    assert started["status"] == "succeeded", started
    runtime_id = started["result"]["runtime_id"]
    invoking = asyncio.create_task(
        host.handle(
            {
                "interface_version": 2,
                "operation": "invoke",
                "agent": "blocking",
                "runtime_id": runtime_id,
                "input": "work",
            }
        )
    )
    try:
        async with asyncio.timeout(5):
            while not (tmp_path / "entered").exists():
                await asyncio.sleep(0.01)
        report = await asyncio.wait_for(
            Fabric().runtime_status(
                environment=server_environment,
                runtime_id=runtime_id,
                agent_name="blocking",
            ),
            3,
        )
        assert report.status == "active"
        assert report.handle.runtime_id == runtime_id
        assert not invoking.done()
        assert not (tmp_path / "stopped").exists()
        for agent_name, selected_id, expected in (
            ("other", runtime_id, "wrong_agent"),
            ("blocking", "missing", "runtime_unavailable"),
        ):
            from nemo_fabric import FabricStateError

            with pytest.raises(FabricStateError) as caught:
                await Fabric().runtime_status(
                    environment=server_environment,
                    runtime_id=selected_id,
                    agent_name=agent_name,
                )
            assert caught.value.code == expected
    finally:
        (tmp_path / "release").write_text("release")
        assert (await asyncio.wait_for(invoking, 5))["status"] == "succeeded"
        await host.shutdown()


@pytest.mark.parametrize(
    "raw",
    [
        '{"runtime_id":"runtime-1","runtime_id":"other"}',
        "null",
        "[]",
        "{",
        "x" * (4 * 1024 * 1024 + 1),
    ],
    ids=["duplicate-fields", "null", "array", "malformed", "oversized"],
)
async def test_runtime_status_remote_rejects_malformed_or_oversized_json(
    server_environment, monkeypatch, raw
):
    from nemo_fabric import FabricConfigError

    native = MagicMock()
    native.runtime_status_environment.return_value = raw
    monkeypatch.setattr("nemo_fabric.client._native", native)
    with pytest.raises(FabricConfigError, match="runtime status report is invalid"):
        await Fabric().runtime_status(
            environment=server_environment, runtime_id="runtime-1", agent_name="demo"
        )
    native.start_runtime.assert_not_called()
    native.invoke_runtime.assert_not_called()
    native.stop_runtime.assert_not_called()
    native.check_environment.assert_not_called()


@pytest.mark.parametrize(
    "report_json",
    [
        '{"runtime_id":"wrong-target","runtime_id":"runtime-1","agent_name":"demo","status":"stopped","handle":null,"server_state":"stopped"}',
        '{"runtime_id":"runtime-1","agent_name":"demo","status":"failed","status":"stopped","handle":null,"server_state":"stopped"}',
    ],
)
async def test_runtime_status_native_transport_rejects_duplicate_fields(
    server_environment, tmp_path, report_json
):
    from nemo_fabric import FabricRuntimeError

    provider = tmp_path / "duplicate-provider"
    provider.write_text(
        f"#!{sys.executable}\n"
        + "import json, sys\n"
        + "frame = json.loads(sys.stdin.readline())\n"
        + "prefix = json.dumps({'protocol_version': frame['protocol_version'], 'request_id': frame['request_id'], 'status': 'succeeded'})[:-1]\n"
        + f'print(prefix + \', "output": {{"status": "returned", "value": \' + {report_json!r} + \'}}}}\', flush=True)\n'
    )
    provider.chmod(0o700)
    os.environ["NEMO_FABRIC_OPEN_SHELL_PROVIDER"] = str(provider)
    with pytest.raises(FabricRuntimeError, match="runtime status transport failed"):
        await Fabric().runtime_status(
            environment=server_environment, runtime_id="runtime-1", agent_name="demo"
        )
