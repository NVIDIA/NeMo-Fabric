# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lifecycle and configuration tests for the OpenClaw adapter."""

from __future__ import annotations

import asyncio
import ctypes
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from nemo_fabric import Fabric
from nemo_fabric import FabricConfig
from nemo_fabric.errors import FabricConfigError
from nemo_fabric_adapter_contract.models import AgentConfig
from nemo_fabric_adapter_contract.models import AgentRunRequest
from nemo_fabric_adapter_contract.models import RuntimeContext
from nemo_fabric_adapters.openclaw import adapter
from nemo_fabric_adapters.openclaw import _windows_job


def _context(workspace: Path) -> RuntimeContext:
    payload = {
        "runtime_id": "openclaw-runtime",
        "invocation_id": "openclaw-invocation",
        "request_id": "openclaw-request",
        "environment": {
            "environment_id": "openclaw-environment",
            "provider": "local",
            "control_location": "in_env_control",
            "workspace": str(workspace),
            "env": {},
            "ownership": "caller_owned",
        },
        "artifacts": {},
    }
    return RuntimeContext.from_mapping(payload)


def _config(command: Path, *, port: int | None = None) -> AgentConfig:
    settings: dict[str, object] = {"openclaw_command": str(command)}
    if port is not None:
        settings["port"] = port
    return AgentConfig.from_mapping(
        {
            "harness": {"settings": settings},
            "instructions": {"system": {"content": "Be concise."}},
            "models": {
                "default": {
                    "provider": "test",
                    "model": "fabric-echo",
                    "base_url": "https://models.example.test/v1",
                    "temperature": 0.2,
                    "top_p": 0.8,
                    "max_tokens": 64,
                }
            },
            "skills": {"paths": ["skills"]},
            "tools": {
                "enabled": ["browser", "web_search"],
                "blocked": ["exec"],
            },
            "mcp": {
                "servers": {
                    "local": {
                        "transport": "stdio",
                        "url": "python",
                        "args": ["server.py"],
                        "env": {"MODE": "test"},
                        "allowed_tools": ["read_*"],
                    },
                    "remote": {
                        "transport": "http",
                        "url": "https://mcp.example.test/mcp",
                        "custom_headers": {"X-Test": "value"},
                        "blocked_tools": ["delete_*"],
                    },
                }
            },
        }
    )


@pytest.fixture(name="mock_openclaw")
def mock_openclaw_fixture(repo_root: Path) -> Path:
    return repo_root / "tests/_utils/mock_openclaw.py"


def test_openclaw_port_availability_checks_one_port(
    monkeypatch: pytest.MonkeyPatch,
):
    mock_socket = MagicMock(spec=adapter.socket.socket)
    monkeypatch.setattr(adapter.socket, "socket", MagicMock(return_value=mock_socket))

    assert adapter._port_available(20_000)
    mock_socket.bind.assert_called_once_with(("127.0.0.1", 20_000))
    mock_socket.close.assert_called_once_with()


def test_openclaw_selects_os_assigned_base_port(monkeypatch: pytest.MonkeyPatch):
    mock_server = MagicMock(spec=adapter.socket.socket)
    mock_server.__enter__.return_value = mock_server
    mock_server.getsockname.return_value = ("127.0.0.1", 20_000)
    mock_port_available = MagicMock(return_value=True)
    monkeypatch.setattr(
        adapter.socket, "create_server", MagicMock(return_value=mock_server)
    )
    monkeypatch.setattr(adapter, "_port_available", mock_port_available)

    assert adapter._select_port({}) == 20_000
    mock_port_available.assert_called_once_with(20_002)


def test_openclaw_checks_configured_base_and_control_ports(
    monkeypatch: pytest.MonkeyPatch,
):
    mock_port_available = MagicMock(return_value=True)
    monkeypatch.setattr(adapter, "_port_available", mock_port_available)

    assert adapter._select_port({"port": 20_000}) == 20_000
    assert [item.args[0] for item in mock_port_available.call_args_list] == [
        20_000,
        20_002,
    ]


@pytest.mark.parametrize(
    ("platform", "setpriv", "expected_prefix"),
    [
        pytest.param(
            "linux",
            "/usr/bin/setpriv",
            ["/usr/bin/setpriv", "--pdeathsig", "SIGTERM", "--"],
            id="linux-with-setpriv",
        ),
        pytest.param("linux", None, [], id="linux-without-setpriv"),
        pytest.param("darwin", "/usr/bin/setpriv", [], id="non-linux"),
    ],
)
def test_openclaw_gateway_command_uses_setpriv_on_linux_when_available(
    monkeypatch: pytest.MonkeyPatch,
    platform: str,
    setpriv: str | None,
    expected_prefix: list[str],
):
    mock_which = MagicMock(return_value=setpriv)
    monkeypatch.setattr(adapter.sys, "platform", platform)
    monkeypatch.setattr(adapter.shutil, "which", mock_which)

    openclaw = Path("/usr/bin/openclaw")
    command = adapter._gateway_command(openclaw, port=20_000)

    assert command == [
        *expected_prefix,
        str(openclaw),
        "gateway",
        "run",
        "--port",
        "20000",
        "--bind",
        "loopback",
        "--auth",
        "token",
    ]
    if platform == "linux":
        mock_which.assert_called_once_with("setpriv")
    else:
        mock_which.assert_not_called()


@pytest.mark.skipif(
    sys.platform in {"darwin", "win32"}, reason="Workes locally, fails in CI"
)
async def test_openclaw_runtime_generates_config_invokes_and_cleans_up(
    mock_openclaw: Path, tmp_path: Path
):
    capture = tmp_path / "config.json"
    request_capture = tmp_path / "request.json"
    readonly_capture = tmp_path / "config-readonly.txt"
    os.environ["FAKE_OPENCLAW_CAPTURE"] = str(capture)
    os.environ["FAKE_OPENCLAW_REQUEST"] = str(request_capture)
    os.environ["FAKE_OPENCLAW_CONFIG_READONLY"] = str(readonly_capture)
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy"):
        os.environ[name] = "http://127.0.0.1:9"
    os.environ["NO_PROXY"] = ""
    os.environ["no_proxy"] = ""
    context = _context(tmp_path)
    runtime = adapter.OpenClawRuntime()

    await runtime.start(
        {
            "config": _config(mock_openclaw),
            "runtime_context": context.to_mapping(),
            "base_dir": str(tmp_path),
        }
    )
    state_root = Path(runtime._temp_dir.name)
    gateway_token = runtime._token
    assert gateway_token is not None
    try:
        result = await runtime.invoke(AgentRunRequest(input="Hello."), context)
        assert runtime._command == mock_openclaw.resolve()
        assert runtime._process is not None and runtime._process.returncode is None
    finally:
        await runtime.stop()

    generated = json.loads(capture.read_text(encoding="utf-8"))
    request = json.loads(request_capture.read_text(encoding="utf-8"))
    assert result.status == "succeeded"
    assert result.output == {"response": "OpenClaw response"}
    assert result.usage.total_tokens == 5
    assert request == {
        "model": "openclaw/default",
        "messages": [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hello."},
        ],
        "temperature": 0.2,
        "top_p": 0.8,
        "max_completion_tokens": 64,
        "stream": True,
        "stream_options": {"include_usage": True},
        "user": "openclaw-runtime",
    }
    assert generated["gateway"]["bind"] == "loopback"
    assert generated["gateway"]["auth"]["token"] == {
        "source": "env",
        "provider": "default",
        "id": "OPENCLAW_GATEWAY_TOKEN",
    }
    assert gateway_token not in capture.read_text(encoding="utf-8")
    assert readonly_capture.read_text(encoding="utf-8") == "1"
    assert generated["gateway"]["http"]["endpoints"]["chatCompletions"] == {
        "enabled": True
    }
    assert generated["agents"]["defaults"]["workspace"] == str(tmp_path)
    assert generated["agents"]["defaults"]["skipBootstrap"] is True
    assert generated["agents"]["defaults"]["contextInjection"] == "never"
    assert generated["agents"]["defaults"]["model"] == {"primary": "test/fabric-echo"}
    assert generated["agents"]["defaults"]["models"]["test/fabric-echo"] == {
        "params": {"temperature": 0.2, "topP": 0.8, "maxTokens": 64},
        "agentRuntime": {"id": "openclaw"},
    }
    assert generated["skills"]["load"]["extraDirs"] == [
        str((tmp_path / "skills").resolve())
    ]
    assert generated["models"]["providers"]["test"]["baseUrl"] == (
        "https://models.example.test/v1"
    )
    assert generated["tools"] == {
        "allow": ["browser", "web_search"],
        "deny": ["exec"],
    }
    assert "plugins" not in generated
    assert generated["mcp"]["servers"]["local"]["command"] == "python"
    assert generated["mcp"]["servers"]["remote"]["transport"] == ("streamable-http")
    assert "auth" not in generated["mcp"]["servers"]["remote"]
    assert not state_root.exists()


def test_openclaw_preserves_context_injection_without_system_instruction(
    mock_openclaw: Path, tmp_path: Path
):
    config = _config(mock_openclaw)
    config.instructions = None

    generated = adapter._openclaw_config(
        config,
        _context(tmp_path),
        base_dir=tmp_path,
        port=20_000,
        token_env="OPENCLAW_GATEWAY_TOKEN",
    )

    assert "contextInjection" not in generated["agents"]["defaults"]


def test_openclaw_explicit_empty_enabled_tools_denies_all(
    mock_openclaw: Path, tmp_path: Path
):
    config = _config(mock_openclaw)
    assert config.tools is not None
    config.tools.enabled = []
    config.tools.blocked = []

    generated = adapter._openclaw_config(
        config,
        _context(tmp_path),
        base_dir=tmp_path,
        port=20_000,
        token_env="OPENCLAW_GATEWAY_TOKEN",
    )

    assert generated["tools"] == {"deny": ["*"]}


def test_openclaw_explicit_empty_mcp_allowed_tools_excludes_all():
    config = AgentConfig.from_mapping(
        {
            "models": {"default": {"provider": "test", "model": "fabric-echo"}},
            "mcp": {
                "servers": {
                    "docs": {
                        "transport": "streamable-http",
                        "url": "https://mcp.example.test/mcp",
                        "allowed_tools": [],
                    }
                }
            },
        }
    )

    generated = adapter._mcp_config(config)

    assert generated["docs"]["toolFilter"] == {"exclude": ["*"]}


async def test_openclaw_invoke_rejects_exited_gateway(tmp_path: Path):
    context = _context(tmp_path)
    runtime = adapter.OpenClawRuntime()
    mock_process = MagicMock(spec=asyncio.subprocess.Process)
    mock_process.returncode = 17
    runtime._client = MagicMock()
    runtime._config = _config(tmp_path / "openclaw")
    runtime._context = context
    runtime._port = 12345
    runtime._process = mock_process

    with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
        await runtime.invoke(AgentRunRequest(input="Hello."), context)

    assert caught.value.code == "openclaw_gateway_exited"
    assert (
        caught.value.message
        == "OpenClaw Gateway exited unexpectedly with exit status 17"
    )
    assert caught.value.metadata == {"exit_code": 17}
    runtime._client.stream.assert_not_called()


async def test_openclaw_read_failure_is_not_retryable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    context = _context(tmp_path)
    runtime = adapter.OpenClawRuntime()
    mock_process = MagicMock(spec=asyncio.subprocess.Process)
    mock_process.returncode = None
    runtime._client = MagicMock()
    runtime._config = _config(tmp_path / "openclaw")
    runtime._context = context
    runtime._port = 12345
    runtime._process = mock_process
    request = adapter.httpx.Request(
        "POST", "http://127.0.0.1:12345/v1/chat/completions"
    )
    mock_invoke = AsyncMock(
        side_effect=adapter.httpx.ReadError("response disconnected", request=request)
    )
    monkeypatch.setattr(adapter, "_invoke_gateway", mock_invoke)

    with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
        await runtime.invoke(AgentRunRequest(input="Hello."), context)

    assert caught.value.code == "openclaw_transport_failed"
    assert caught.value.retryable is False


async def test_openclaw_process_exit_fails_active_invocation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    context = _context(tmp_path)
    runtime = adapter.OpenClawRuntime()
    process_exited = asyncio.Event()
    invocation_started = asyncio.Event()
    mock_process = MagicMock(spec=asyncio.subprocess.Process)
    mock_process.returncode = None

    async def wait_for_process():
        await process_exited.wait()
        return 17

    async def wait_for_gateway(*_args, **_kwargs):
        invocation_started.set()
        await asyncio.Event().wait()

    mock_process.wait = AsyncMock(side_effect=wait_for_process)
    mock_client = MagicMock(spec=adapter.httpx.AsyncClient)
    mock_client.get = AsyncMock()
    mock_invoke = AsyncMock(side_effect=wait_for_gateway)
    monkeypatch.setattr(adapter, "_invoke_gateway", mock_invoke)
    runtime._client = mock_client
    runtime._config = _config(tmp_path / "openclaw")
    runtime._context = context
    runtime._port = 12345
    runtime._process = mock_process
    runtime._stderr_tail.append("gateway crashed")
    runtime._monitor_task = asyncio.create_task(
        runtime._monitor_gateway(mock_process, mock_client, 12345)
    )

    invoke_task = asyncio.create_task(
        runtime.invoke(AgentRunRequest(input="Hello."), context)
    )
    await invocation_started.wait()
    process_exited.set()

    with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
        await asyncio.wait_for(invoke_task, timeout=1)

    assert caught.value.code == "openclaw_gateway_exited"
    assert caught.value.metadata == {
        "exit_code": 17,
        "detail": "gateway crashed",
    }
    assert mock_invoke.await_count == 1


async def test_openclaw_health_monitor_detects_unresponsive_gateway(
    monkeypatch: pytest.MonkeyPatch,
):
    runtime = adapter.OpenClawRuntime()
    mock_process = MagicMock(spec=asyncio.subprocess.Process)

    async def block_forever(*_args, **_kwargs):
        await asyncio.Event().wait()

    mock_process.wait = AsyncMock(side_effect=block_forever)
    mock_client = MagicMock(spec=adapter.httpx.AsyncClient)
    mock_client.get = AsyncMock(side_effect=block_forever)
    monkeypatch.setattr(adapter, "HEALTH_CHECK_INTERVAL_SECONDS", 0)
    monkeypatch.setattr(adapter, "HEALTH_CHECK_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(adapter, "HEALTH_CHECK_FAILURE_THRESHOLD", 1)

    await asyncio.wait_for(
        runtime._monitor_gateway(mock_process, mock_client, 12345), timeout=1
    )

    assert runtime._gateway_failure is not None
    assert runtime._gateway_failure.code == "openclaw_gateway_unhealthy"
    assert runtime._gateway_failure.retryable is False
    assert runtime._gateway_failure.metadata == {"failed_health_checks": 1}


async def test_openclaw_stop_continues_after_cleanup_failures(
    monkeypatch: pytest.MonkeyPatch,
):
    runtime = adapter.OpenClawRuntime()
    mock_client = MagicMock()
    mock_client.aclose = AsyncMock(side_effect=OSError("close failed"))
    mock_process = MagicMock(spec=asyncio.subprocess.Process)
    mock_process.pid = 1234
    mock_process.returncode = None
    mock_process.wait = AsyncMock()
    mock_temp_dir = MagicMock()
    log_task = asyncio.create_task(asyncio.Event().wait())
    mock_close_job = MagicMock(side_effect=OSError("close failed"))
    if os.name != "nt":
        monkeypatch.setattr(adapter.os, "killpg", MagicMock())
    monkeypatch.setattr(adapter._windows_job, "close_job", mock_close_job)
    runtime._client = mock_client
    runtime._process = mock_process
    runtime._windows_job = 1234
    runtime._log_tasks = [log_task]
    runtime._temp_dir = mock_temp_dir

    await runtime.stop()

    mock_client.aclose.assert_awaited_once_with()
    mock_process.wait.assert_awaited_once_with()
    mock_close_job.assert_called_once_with(1234)
    assert log_task.cancelled()
    mock_temp_dir.cleanup.assert_called_once_with()
    assert runtime._client is None
    assert runtime._process is None
    assert runtime._windows_job is None
    assert runtime._log_tasks == []
    assert runtime._temp_dir is None


@pytest.mark.parametrize("phase", ["client", "process", "logs"])
async def test_openclaw_stop_completes_cleanup_before_propagating_cancellation(
    monkeypatch: pytest.MonkeyPatch, phase: str
):
    runtime = adapter.OpenClawRuntime()
    entered = asyncio.Event()
    release = asyncio.Event()

    async def block_cleanup(*_args, **_kwargs):
        entered.set()
        await release.wait()

    mock_client = MagicMock()
    mock_client.aclose = AsyncMock(
        side_effect=block_cleanup if phase == "client" else None
    )
    mock_process = MagicMock(spec=asyncio.subprocess.Process)
    mock_process.pid = 1234
    mock_process.returncode = None
    process_wait_calls = 0

    async def wait_for_process():
        nonlocal process_wait_calls
        process_wait_calls += 1
        if phase == "process" and process_wait_calls == 1:
            await block_cleanup()

    mock_process.wait = AsyncMock(side_effect=wait_for_process)
    if os.name != "nt":
        monkeypatch.setattr(adapter.os, "killpg", MagicMock())

    if phase == "logs":
        monkeypatch.setattr(adapter.asyncio, "gather", block_cleanup)

    log_task = asyncio.create_task(asyncio.Event().wait())
    await asyncio.sleep(0)
    mock_temp_dir = MagicMock()
    runtime._client = mock_client
    runtime._process = mock_process
    runtime._log_tasks = [log_task]
    runtime._temp_dir = mock_temp_dir

    stop_task = asyncio.create_task(runtime.stop())
    await asyncio.wait_for(entered.wait(), timeout=1)
    stop_task.cancel()
    try:
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(asyncio.shield(stop_task), timeout=0.2)
    finally:
        release.set()
        with suppress(asyncio.CancelledError):
            await stop_task

    mock_client.aclose.assert_awaited_once_with()
    assert mock_process.wait.await_count >= 1
    if phase == "process":
        if os.name == "nt":
            mock_process.kill.assert_called_once_with()
        else:
            adapter.os.killpg.assert_any_call(1234, signal.SIGKILL)
    assert log_task.cancelled()
    mock_temp_dir.cleanup.assert_called_once_with()
    assert runtime._client is None
    assert runtime._process is None
    assert runtime._log_tasks == []
    assert runtime._temp_dir is None


async def test_openclaw_command_timeout_kills_and_reaps_process(
    monkeypatch: pytest.MonkeyPatch,
):
    process = MagicMock(spec=asyncio.subprocess.Process)
    process.returncode = None

    async def communicate() -> tuple[bytes, bytes]:
        if process.returncode is None:
            await asyncio.Event().wait()
        return b"", b""

    process.communicate = AsyncMock(side_effect=communicate)
    process.kill.side_effect = lambda: setattr(process, "returncode", -9)
    create_subprocess = AsyncMock(return_value=process)
    monkeypatch.setattr(adapter.asyncio, "create_subprocess_exec", create_subprocess)

    with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
        await adapter._command_output(
            Path("openclaw"), "--version", env={}, timeout=0.01
        )

    assert caught.value.code == "openclaw_command_timeout"
    assert caught.value.metadata == {"command": "--version", "timeout_seconds": 0.01}
    process.kill.assert_called_once_with()
    assert process.communicate.await_count == 2


async def test_openclaw_command_cancellation_kills_and_reaps_process(
    monkeypatch: pytest.MonkeyPatch,
):
    process = MagicMock(spec=asyncio.subprocess.Process)
    process.returncode = None
    started = asyncio.Event()

    async def communicate() -> tuple[bytes, bytes]:
        if process.returncode is None:
            started.set()
            await asyncio.Event().wait()
        return b"", b""

    process.communicate = AsyncMock(side_effect=communicate)
    process.kill.side_effect = lambda: setattr(process, "returncode", -9)
    create_subprocess = AsyncMock(return_value=process)
    monkeypatch.setattr(adapter.asyncio, "create_subprocess_exec", create_subprocess)
    task = asyncio.create_task(
        adapter._command_output(Path("openclaw"), "--version", env={}, timeout=30)
    )
    await started.wait()

    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    process.kill.assert_called_once_with()
    assert process.communicate.await_count == 2


@pytest.mark.parametrize(
    "authentication",
    [
        pytest.param(
            {
                "type": "oauth2",
                "client_id": "fabric-client",
            },
            id="oauth2",
        ),
        pytest.param(
            {
                "type": "service_account",
                "client_id": "fabric-client",
                "client_secret_env": "FABRIC_MCP_CLIENT_SECRET",
                "token_url": "https://auth.example.test/token",
            },
            id="service-account",
        ),
    ],
)
def test_openclaw_rejects_mcp_authentication(authentication: dict[str, object]):
    config = AgentConfig.from_mapping(
        {
            "models": {"default": {"provider": "test", "model": "fabric-echo"}},
            "mcp": {
                "servers": {
                    "remote": {
                        "transport": "streamable-http",
                        "url": "https://mcp.example.test/mcp",
                        "authentication": authentication,
                    }
                }
            },
        }
    )

    with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
        adapter._mcp_config(config)

    assert caught.value.code == "openclaw_unsupported_mcp_authentication"
    assert caught.value.metadata == {"field": "mcp.servers.remote.authentication"}


def test_openclaw_resolves_relative_command_without_path_fallback(tmp_path: Path):
    command = tmp_path / "tools" / "openclaw"
    command.parent.mkdir()
    command.write_text("#!/bin/sh\n", encoding="utf-8")
    command.chmod(0o755)

    assert (
        adapter._resolve_command({"openclaw_command": "tools/openclaw"}, tmp_path)
        == command.resolve()
    )
    with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
        adapter._resolve_command({"openclaw_command": "missing/openclaw"}, tmp_path)

    assert caught.value.code == "openclaw_command_not_found"


@pytest.mark.skipif(os.name == "nt", reason="POSIX signal forwarding")
def test_openclaw_signal_handler_forwards_to_gateway_process_group(
    monkeypatch: pytest.MonkeyPatch,
):
    runtime = adapter.OpenClawRuntime()
    mock_process = MagicMock(spec=asyncio.subprocess.Process)
    mock_process.pid = 1234
    mock_process.returncode = None
    runtime._process = mock_process
    forwarded: list[tuple[int, int]] = []
    monkeypatch.setattr(
        adapter.os,
        "killpg",
        lambda process_group, signum: forwarded.append((process_group, signum)),
    )

    runtime._install_signal_handlers()
    try:
        with pytest.raises(SystemExit) as caught:
            runtime._signal_handler(signal.SIGTERM, None)
    finally:
        runtime._restore_signal_handlers()

    assert caught.value.code == 128 + signal.SIGTERM
    assert forwarded == [(1234, signal.SIGTERM)]


def test_windows_job_assigns_process_and_enables_kill_on_close(
    monkeypatch: pytest.MonkeyPatch,
):
    mock_kernel32 = MagicMock()
    process_handle = ctypes.c_void_p(5678)
    mock_kernel32.CreateJobObjectW.return_value = ctypes.c_void_p(1234)
    mock_kernel32.SetInformationJobObject.return_value = True
    mock_kernel32.OpenProcess.return_value = process_handle
    mock_kernel32.AssignProcessToJobObject.return_value = True
    mock_kernel32.CloseHandle.return_value = True
    monkeypatch.setattr(_windows_job, "_kernel32", lambda: mock_kernel32)

    job = _windows_job.create_kill_on_close_job()
    _windows_job.assign_process(job, 42)
    _windows_job.close_job(job)

    assert job == 1234
    information = mock_kernel32.SetInformationJobObject.call_args.args[2]._obj
    assert information.basic_limit_information.limit_flags == 0x00002000
    mock_kernel32.OpenProcess.assert_called_once_with(0x0101, False, 42)
    mock_kernel32.AssignProcessToJobObject.assert_called_once_with(job, process_handle)


@pytest.mark.skipif(
    sys.platform != "linux" or shutil.which("setpriv") is None,
    reason="Linux setpriv parent-death signal",
)
def test_linux_parent_death_stops_openclaw_gateway(mock_openclaw: Path, tmp_path: Path):
    capture = tmp_path / "config.json"
    request_capture = tmp_path / "request.json"
    pid_path = tmp_path / "gateway.pid"
    stopped_path = tmp_path / "gateway.stopped"
    environment = os.environ.copy()
    environment.update(
        {
            "FAKE_OPENCLAW_CAPTURE": str(capture),
            "FAKE_OPENCLAW_REQUEST": str(request_capture),
            "FAKE_OPENCLAW_PID": str(pid_path),
            "FAKE_OPENCLAW_STOPPED": str(stopped_path),
        }
    )
    context = _context(tmp_path)
    request = {
        "operation": "start",
        "payload": {
            "config": _config(mock_openclaw).to_mapping(),
            "runtime_context": context.to_mapping(),
            "base_dir": str(tmp_path),
        },
    }
    process = subprocess.Popen(
        [sys.executable, "-m", "nemo_fabric_adapters.openclaw.adapter"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
    )
    gateway_pid: int | None = None
    try:
        assert process.stdin is not None
        assert process.stdout is not None
        process.stdin.write(json.dumps(request) + "\n")
        process.stdin.flush()
        response = json.loads(process.stdout.readline())
        assert response["outcome"]["status"] == "succeeded"
        gateway_pid = int(pid_path.read_text(encoding="utf-8"))

        process.kill()
        process.wait(timeout=5)
        deadline = time.monotonic() + 5
        while not stopped_path.exists() and time.monotonic() < deadline:
            time.sleep(0.05)

        assert stopped_path.read_text(encoding="utf-8") == "stopped"
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        if gateway_pid is not None and not stopped_path.exists():
            with suppress(ProcessLookupError):
                os.killpg(os.getpgid(gateway_pid), signal.SIGKILL)


@pytest.mark.skipif(
    sys.platform in {"darwin", "win32"}, reason="Workes locally, fails in CI"
)
async def test_openclaw_plan_doctor_and_run_without_credentials(
    mock_openclaw: Path, tmp_path: Path
):
    os.environ["FAKE_OPENCLAW_CAPTURE"] = str(tmp_path / "config.json")
    os.environ["FAKE_OPENCLAW_REQUEST"] = str(tmp_path / "request.json")
    config = FabricConfig.from_mapping(
        {
            "metadata": {"name": "openclaw-test"},
            "harness": {
                "adapter_id": "nvidia.fabric.openclaw",
                "resolution": "preinstalled",
                "settings": {"openclaw_command": str(mock_openclaw)},
            },
            "models": {
                "default": {
                    "provider": "test",
                    "model": "fabric-echo",
                    "base_url": "https://models.example.test/v1",
                }
            },
            "tools": {"enabled": ["browser"], "blocked": ["exec"]},
            "mcp": {
                "servers": {
                    "docs": {
                        "transport": "streamable-http",
                        "url": "https://mcp.example.test/mcp",
                    }
                }
            },
            "environment": {"provider": "local", "workspace": "."},
        }
    )
    fabric = Fabric()

    plan = fabric.plan(config, base_dir=tmp_path)
    report = await fabric.doctor(config, base_dir=tmp_path)
    result = await fabric.run(config, input="Hello.", base_dir=tmp_path)

    assert plan.adapter.adapter_id == "nvidia.fabric.openclaw"
    assert report.status == "pass"
    assert result.status == "succeeded"
    assert result.output == {"response": "OpenClaw response"}


@pytest.mark.parametrize(
    ("authentication", "capability"),
    [
        pytest.param(
            {"type": "oauth2"},
            "mcp.auth.oauth2",
            id="oauth2",
        ),
        pytest.param(
            {
                "type": "service_account",
                "client_id": "fabric-client",
                "client_secret_env": "FABRIC_MCP_CLIENT_SECRET",
                "token_url": "https://auth.example.test/token",
            },
            "mcp.auth.service_account",
            id="service-account",
        ),
    ],
)
def test_openclaw_plan_rejects_mcp_authentication(
    tmp_path: Path,
    authentication: dict[str, object],
    capability: str,
):
    config = FabricConfig.from_mapping(
        {
            "metadata": {"name": "openclaw-service-account"},
            "harness": {
                "adapter_id": "nvidia.fabric.openclaw",
                "resolution": "preinstalled",
            },
            "models": {"default": {"provider": "test", "model": "fabric-echo"}},
            "mcp": {
                "servers": {
                    "docs": {
                        "transport": "streamable-http",
                        "url": "https://mcp.example.test/mcp",
                        "authentication": authentication,
                    }
                }
            },
            "environment": {"provider": "local", "workspace": "."},
        }
    )

    with pytest.raises(FabricConfigError) as caught:
        Fabric().plan(config, base_dir=tmp_path)

    assert "mcp.servers.docs.authentication" in str(caught.value)
    assert capability in str(caught.value)


def test_openclaw_descriptor_and_module_entrypoint(repo_root: Path):
    descriptor = json.loads(
        (repo_root / "adapters/python/openclaw/openclaw.fabric-adapter.json").read_text(
            encoding="utf-8"
        )
    )
    result = subprocess.run(
        [sys.executable, "-m", "nemo_fabric_adapters.openclaw.adapter"],
        input="",
        text=True,
        capture_output=True,
        check=False,
    )

    assert descriptor["adapter_id"] == "nvidia.fabric.openclaw"
    assert descriptor["requirements"]["binaries"] == ["openclaw"]
    assert "tools.enabled" in descriptor["config"]["accepts"]
    assert "tools.blocked" in descriptor["config"]["accepts"]
    assert "mcp.auth.oauth2" not in descriptor["config"]["accepts"]
    assert "mcp.auth.service_account" not in descriptor["config"]["accepts"]
    assert descriptor["capabilities"]["streaming"] is False
    assert "telemetry" not in descriptor
    assert result.returncode == 0, result.stderr
