# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lifecycle and configuration tests for the OpenClaw adapter."""

from __future__ import annotations

import asyncio
import ctypes
import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from contextlib import suppress
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from nemo_fabric import Fabric
from nemo_fabric import FabricConfig
from nemo_fabric_adapter_contract.models import AgentConfig
from nemo_fabric_adapter_contract.models import AgentRunRequest
from nemo_fabric_adapter_contract.models import RuntimeContext
from nemo_fabric_adapters.openclaw import adapter
from nemo_fabric_adapters.openclaw import _windows_job


def _context(workspace: Path, *, relay: bool = False) -> RuntimeContext:
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
    if relay:
        payload["telemetry"] = {"relay_enabled": True}
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


@pytest.fixture(name="fake_openclaw")
def fake_openclaw_fixture(tmp_path: Path) -> Path:
    command = tmp_path / "openclaw"
    command.write_text(
        "#!"
        + sys.executable
        + "\n"
        + textwrap.dedent(
            """
            import json
            import os
            import signal
            import sys
            from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

            args = sys.argv[1:]
            if args == ["--version"]:
                print("OpenClaw 2099.1.0")
                raise SystemExit(0)
            if args[:3] == ["config", "validate", "--json"]:
                with open(os.environ["OPENCLAW_CONFIG_PATH"], encoding="utf-8") as stream:
                    value = json.load(stream)
                with open(os.environ["FAKE_OPENCLAW_CAPTURE"], "w", encoding="utf-8") as stream:
                    json.dump(value, stream)
                print(json.dumps({"valid": True}))
                raise SystemExit(0)
            if args[:3] == ["plugins", "list", "--json"]:
                print(json.dumps({"plugins": []}))
                raise SystemExit(0)
            if args[:3] == ["gateway", "call", "nemoRelay.status"]:
                print(json.dumps({"ok": True}))
                raise SystemExit(0)
            if args[:2] != ["gateway", "run"]:
                raise SystemExit(2)

            port = int(args[args.index("--port") + 1])
            token = os.environ["OPENCLAW_GATEWAY_TOKEN"]
            if pid_path := os.environ.get("FAKE_OPENCLAW_PID"):
                with open(pid_path, "w", encoding="utf-8") as stream:
                    stream.write(str(os.getpid()))

            def stop_gateway(*_unused):
                if stopped_path := os.environ.get("FAKE_OPENCLAW_STOPPED"):
                    with open(stopped_path, "w", encoding="utf-8") as stream:
                        stream.write("stopped")
                raise SystemExit(0)

            class Handler(BaseHTTPRequestHandler):
                def log_message(self, *unused):
                    pass

                def do_GET(self):
                    if self.path != "/readyz":
                        self.send_response(404)
                        self.end_headers()
                        return
                    self.send_response(200)
                    self.end_headers()

                def do_POST(self):
                    if self.headers.get("Authorization") != f"Bearer {token}":
                        self.send_response(401)
                        self.end_headers()
                        return
                    length = int(self.headers.get("Content-Length", "0"))
                    request = json.loads(self.rfile.read(length))
                    with open(os.environ["FAKE_OPENCLAW_REQUEST"], "w", encoding="utf-8") as stream:
                        json.dump(request, stream)
                    body = json.dumps({
                        "choices": [{"message": {"content": "OpenClaw response"}}],
                        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
                    }).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

            server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
            signal.signal(signal.SIGTERM, stop_gateway)
            server.serve_forever()
            """
        ),
        encoding="utf-8",
    )
    command.chmod(0o755)
    return command


async def test_openclaw_runtime_generates_config_invokes_and_cleans_up(
    fake_openclaw: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    capture = tmp_path / "config.json"
    request_capture = tmp_path / "request.json"
    monkeypatch.setenv("FAKE_OPENCLAW_CAPTURE", str(capture))
    monkeypatch.setenv("FAKE_OPENCLAW_REQUEST", str(request_capture))
    context = _context(tmp_path)
    runtime = adapter.OpenClawRuntime()

    await runtime.start(
        {
            "config": _config(fake_openclaw),
            "runtime_context": context.to_mapping(),
            "base_dir": str(tmp_path),
        }
    )
    state_root = Path(runtime._temp_dir.name)
    gateway_token = runtime._token
    assert gateway_token is not None
    try:
        result = await runtime.invoke(AgentRunRequest(input="Hello."), context)
        assert runtime._command == fake_openclaw.resolve()
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
        "user": "openclaw-runtime",
    }
    assert generated["gateway"]["bind"] == "loopback"
    assert generated["gateway"]["auth"]["token"] == {
        "source": "env",
        "provider": "default",
        "id": "OPENCLAW_GATEWAY_TOKEN",
    }
    assert gateway_token not in capture.read_text(encoding="utf-8")
    assert generated["gateway"]["http"]["endpoints"]["chatCompletions"] == {
        "enabled": True
    }
    assert generated["agents"]["defaults"]["workspace"] == str(tmp_path)
    assert generated["agents"]["defaults"]["model"] == {"primary": "test/fabric-echo"}
    assert generated["agents"]["defaults"]["models"]["test/fabric-echo"] == {
        "params": {"temperature": 0.2, "topP": 0.8, "maxTokens": 64}
    }
    assert generated["models"]["providers"]["test"]["baseUrl"] == (
        "https://models.example.test/v1"
    )
    assert generated["mcp"]["servers"]["local"]["command"] == "python"
    assert generated["mcp"]["servers"]["remote"]["transport"] == ("streamable-http")
    assert not state_root.exists()


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


@pytest.mark.skipif(sys.platform != "linux", reason="Linux parent-death signal")
def test_linux_parent_death_stops_openclaw_gateway(fake_openclaw: Path, tmp_path: Path):
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
            "config": _config(fake_openclaw).to_mapping(),
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


async def test_openclaw_plan_doctor_and_run_without_credentials(
    fake_openclaw: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("FAKE_OPENCLAW_CAPTURE", str(tmp_path / "config.json"))
    monkeypatch.setenv("FAKE_OPENCLAW_REQUEST", str(tmp_path / "request.json"))
    config = FabricConfig.from_mapping(
        {
            "metadata": {"name": "openclaw-test"},
            "harness": {
                "adapter_id": "nvidia.fabric.openclaw",
                "resolution": "preinstalled",
                "settings": {"openclaw_command": str(fake_openclaw)},
            },
            "models": {
                "default": {
                    "provider": "test",
                    "model": "fabric-echo",
                    "base_url": "https://models.example.test/v1",
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


async def test_openclaw_requires_relay_plugin_when_relay_is_requested(
    fake_openclaw: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    context = _context(tmp_path, relay=True)
    monkeypatch.setattr(
        adapter.common_utils,
        "load_relay_plugin_config",
        lambda _payload: {
            "version": 1,
            "components": [
                {
                    "kind": "observability",
                    "enabled": True,
                    "config": {"version": 3, "atif": {"enabled": True}},
                }
            ],
        },
    )
    runtime = adapter.OpenClawRuntime()

    with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
        await runtime.start(
            {
                "config": _config(fake_openclaw),
                "runtime_context": context.to_mapping(),
                "base_dir": str(tmp_path),
            }
        )

    assert caught.value.code == "openclaw_relay_plugin_missing"


@pytest.mark.parametrize(
    ("observability", "code"),
    [
        pytest.param(
            {"atof": {"enabled": True}},
            "openclaw_relay_streaming_unsupported",
            id="relay-streaming",
        ),
        pytest.param(
            {"opentelemetry": {"enabled": True, "endpoints": [{}]}},
            "openclaw_relay_otel_unsupported",
            id="relay-otel",
        ),
    ],
)
def test_openclaw_rejects_deferred_relay_modes(
    observability: dict[str, object], code: str
):
    with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
        adapter._relay_plugin_config(
            {
                "components": [
                    {
                        "kind": "observability",
                        "enabled": True,
                        "config": observability,
                    }
                ]
            }
        )

    assert caught.value.code == code


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
    assert descriptor["capabilities"]["streaming"] is False
    assert descriptor["telemetry"]["providers"]["relay"] == {
        "outputs": ["atif"],
        "integration_modes": ["hooks"],
    }
    assert result.returncode == 0, result.stderr
