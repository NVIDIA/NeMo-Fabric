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
import time
from contextlib import suppress
from pathlib import Path
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
                        "authentication": {
                            "type": "oauth2",
                            "scopes": ["docs.read", "docs.write"],
                            "redirect_uri": "http://127.0.0.1/oauth/callback",
                        },
                    },
                }
            },
        }
    )


@pytest.fixture(name="mock_openclaw")
def mock_openclaw_fixture(repo_root: Path) -> Path:
    return repo_root / "tests/_utils/mock_openclaw.py"


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
    assert generated["agents"]["defaults"]["model"] == {"primary": "test/fabric-echo"}
    assert generated["agents"]["defaults"]["models"]["test/fabric-echo"] == {
        "params": {"temperature": 0.2, "topP": 0.8, "maxTokens": 64}
    }
    assert generated["models"]["providers"]["test"]["baseUrl"] == (
        "https://models.example.test/v1"
    )
    assert generated["tools"] == {
        "allow": ["browser", "web_search"],
        "deny": ["exec"],
    }
    assert generated["mcp"]["servers"]["local"]["command"] == "python"
    assert generated["mcp"]["servers"]["remote"]["transport"] == ("streamable-http")
    assert generated["mcp"]["servers"]["remote"]["auth"] == "oauth"
    assert generated["mcp"]["servers"]["remote"]["oauth"] == {
        "scope": "docs.read docs.write",
        "redirectUrl": "http://127.0.0.1/oauth/callback",
    }
    assert not state_root.exists()


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


@pytest.mark.parametrize(
    ("authentication", "field"),
    [
        pytest.param({"client_id": "fabric-client"}, "client_id", id="client-id"),
        pytest.param(
            {
                "client_id": "fabric-client",
                "client_secret_env": "FABRIC_MCP_CLIENT_SECRET",
            },
            "client_secret_env",
            id="client-secret",
        ),
        pytest.param({"client_name": "Fabric"}, "client_name", id="client-name"),
        pytest.param(
            {
                "client_id": "fabric-client",
                "enable_dynamic_registration": False,
            },
            "enable_dynamic_registration",
            id="dynamic-registration",
        ),
        pytest.param(
            {"token_endpoint_auth_method": "none"},
            "token_endpoint_auth_method",
            id="token-endpoint-auth-method",
        ),
        pytest.param(
            {"authorization_timeout_seconds": 30},
            "authorization_timeout_seconds",
            id="authorization-timeout",
        ),
    ],
)
def test_openclaw_rejects_unmapped_mcp_oauth_fields(
    authentication: dict[str, object], field: str
):
    config = AgentConfig.from_mapping(
        {
            "models": {"default": {"provider": "test", "model": "fabric-echo"}},
            "mcp": {
                "servers": {
                    "remote": {
                        "transport": "streamable-http",
                        "url": "https://mcp.example.test/mcp",
                        "authentication": {"type": "oauth2", **authentication},
                    }
                }
            },
        }
    )

    with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
        adapter._mcp_config(config)

    assert caught.value.code == "openclaw_unsupported_mcp_authentication"
    assert caught.value.metadata == {
        "field": f"mcp.servers.remote.authentication.{field}"
    }


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
                        "authentication": {
                            "type": "oauth2",
                            "scopes": ["docs.read"],
                        },
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


def test_openclaw_plan_rejects_mcp_service_account(tmp_path: Path):
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
                        "authentication": {
                            "type": "service_account",
                            "client_id": "fabric-client",
                            "client_secret_env": "FABRIC_MCP_CLIENT_SECRET",
                            "token_url": "https://auth.example.test/token",
                        },
                    }
                }
            },
            "environment": {"provider": "local", "workspace": "."},
        }
    )

    with pytest.raises(FabricConfigError) as caught:
        Fabric().plan(config, base_dir=tmp_path)

    assert "mcp.servers.docs.authentication" in str(caught.value)
    assert "mcp.auth.service_account" in str(caught.value)


async def test_openclaw_requires_relay_plugin_when_relay_is_requested(
    mock_openclaw: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
                "config": _config(mock_openclaw),
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
    assert "tools.enabled" in descriptor["config"]["accepts"]
    assert "tools.blocked" in descriptor["config"]["accepts"]
    assert "mcp.auth.oauth2" in descriptor["config"]["accepts"]
    assert descriptor["capabilities"]["streaming"] is False
    assert descriptor["telemetry"]["providers"]["relay"] == {
        "outputs": ["atif"],
        "integration_modes": ["hooks"],
    }
    assert result.returncode == 0, result.stderr
