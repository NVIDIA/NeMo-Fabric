# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SDK server checks use the same health contract as the command bridge."""

import asyncio
import json
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
