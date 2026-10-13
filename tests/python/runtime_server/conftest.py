# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from nemo_fabric import (
    Fabric,
    RunResult,
    Runtime,
    RuntimeHandle,
    RuntimeStatus,
)


@pytest.fixture(name="socket_path")
def socket_path_fixture():
    # macOS Unix socket paths have a short platform limit.
    with tempfile.TemporaryDirectory(prefix="fh-", dir="/tmp") as directory:
        yield Path(directory) / "fabric.sock"


@pytest.fixture(name="process_is_running")
def process_is_running_fixture():
    def process_is_running(pid):
        result = subprocess.run(
            ["ps", "-o", "stat=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert result.returncode in (0, 1), result.stderr
        state = result.stdout.strip()
        # A terminated orphan may remain a zombie until its parent reaps it.
        return bool(state) and not state.startswith("Z")

    return process_is_running


@pytest.fixture(name="config")
def config_fixture():
    return {"metadata": {"name": "researcher"}, "harness": {"adapter_id": "test.host"}}


@pytest.fixture(name="mock_runtime")
def mock_runtime_fixture():
    runtime = MagicMock(spec=Runtime)
    runtime.runtime_id = "runtime-1"
    runtime.status = RuntimeStatus.ACTIVE
    runtime.handle = RuntimeHandle.from_mapping(
        {
            "runtime_id": "runtime-1",
            "runtime_binding": "opaque-binding",
            "agent_name": "researcher",
            "harness": "test",
            "adapter_kind": "local_process",
            "adapter_id": "test.host",
            "environment": {},
        }
    )
    runtime.stop = AsyncMock()
    runtime.invoke = AsyncMock(return_value=MagicMock(spec=RunResult))
    runtime.invoke.return_value.to_mapping.return_value = {
        "status": "succeeded",
        "runtime_id": "runtime-1",
        "output": {"response": "FOUR"},
        "agent_name": "researcher",
        "harness": "test",
        "adapter_kind": "local_process",
        "adapter_id": "test.host",
        "invocation_id": "invocation-1",
        "request_id": "request-1",
    }
    return runtime


@pytest.fixture(name="mock_fabric")
def mock_fabric_fixture(mock_runtime):
    fabric = MagicMock(spec=Fabric)
    fabric.start_runtime = AsyncMock(return_value=mock_runtime)
    return fabric


@pytest.fixture(name="host")
def host_fixture(tmp_path, mock_fabric):
    from nemo_fabric.runtime_server.state import RuntimeServer

    return RuntimeServer(directory=tmp_path, fabric=mock_fabric)


@pytest.fixture(name="host_request")
def request_fixture(host, config):
    def host_request(operation, **fields):
        value = {"interface_version": 2, "operation": operation}
        if operation in ("start", "invoke", "stop") or "runtime_id" in fields:
            value["agent"] = "researcher"
        if operation in ("invoke", "stop"):
            value["runtime_id"] = "runtime-1"
        if operation == "start":
            value["config"] = config
        return {**value, **fields}

    return host_request
