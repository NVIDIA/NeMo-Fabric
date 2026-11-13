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
import socket
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
import httpx
from nemo_fabric import Fabric
from nemo_fabric import FabricConfig
from nemo_fabric import RuntimeStatus
from nemo_fabric.errors import FabricConfigError
from nemo_fabric.errors import FabricRuntimeError
from nemo_fabric_adapter_contract.models import AgentConfig
from nemo_fabric_adapter_contract.models import AgentRunRequest
from nemo_fabric_adapter_contract.models import RuntimeContext
from nemo_fabric_adapters.openclaw import adapter
from nemo_fabric_adapters.openclaw import _windows_job


def _context(
    workspace: Path,
    runtime_id: str = "openclaw-runtime",
    deadline_millis: int | None = None,
) -> RuntimeContext:
    payload = {
        "runtime_id": runtime_id,
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
    if deadline_millis is not None:
        payload["deadline_millis"] = deadline_millis
    return RuntimeContext.from_mapping(payload)


def _config(command: Path, *, with_skills: bool = False) -> AgentConfig:
    settings: dict[str, object] = {"openclaw_command": str(command)}
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
            "skills": {"paths": ["skills"]} if with_skills else None,
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
                },
            },
        }
    )


def _dashboard_config(command: Path, dashboard: dict[str, object]) -> AgentConfig:
    config = _config(command)
    assert config.harness is not None
    config.harness.settings["interfaces"] = {"dashboard": dashboard}
    return config


def _dashboard_token(base_dir: Path) -> Path:
    return (
        base_dir
        / "artifacts/.fabric/openclaw/runtimes/openclaw-runtime/interfaces/dashboard.token"
    )


async def _start(
    runtime: adapter.OpenClawRuntime,
    tmp_path: Path,
    config: AgentConfig,
    **payload: object,
) -> None:
    await runtime.start(
        {
            "config": config,
            "runtime_context": _context(tmp_path).to_mapping(),
            "base_dir": str(tmp_path),
            **payload,
        }
    )


@pytest.fixture(name="mock_openclaw")
def mock_openclaw_fixture(repo_root: Path) -> Path:
    return repo_root / "tests/_utils/mock_openclaw.py"


@pytest.fixture(name="dashboard_port")
def dashboard_port_fixture() -> int:
    return adapter._select_port({})


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


def test_openclaw_retries_os_assigned_port_without_full_derived_range(
    monkeypatch: pytest.MonkeyPatch,
):
    high_server = MagicMock(spec=adapter.socket.socket)
    high_server.__enter__.return_value = high_server
    high_server.getsockname.return_value = ("127.0.0.1", 65_500)
    valid_server = MagicMock(spec=adapter.socket.socket)
    valid_server.__enter__.return_value = valid_server
    valid_server.getsockname.return_value = ("127.0.0.1", 20_000)
    mock_port_available = MagicMock(return_value=True)
    monkeypatch.setattr(
        adapter.socket,
        "create_server",
        MagicMock(side_effect=[high_server, valid_server]),
    )
    monkeypatch.setattr(adapter, "_port_available", mock_port_available)

    assert adapter._select_port({}) == 20_000
    mock_port_available.assert_called_once_with(20_002)


def test_openclaw_selects_base_inside_configured_port_range(
    monkeypatch: pytest.MonkeyPatch,
):
    mock_port_available = MagicMock(return_value=True)
    monkeypatch.setattr(adapter, "_port_available", mock_port_available)
    monkeypatch.setattr(adapter.secrets, "randbelow", MagicMock(return_value=7))

    assert (
        adapter._select_port({"port_range": {"start": 20_000, "end": 20_120}}) == 20_007
    )
    assert [item.args[0] for item in mock_port_available.call_args_list] == [
        20_007,
        20_009,
    ]
    assert 20_007 + adapter.MAX_DERIVED_PORT_OFFSET <= 20_120


@pytest.mark.parametrize(
    "settings",
    [
        {"port_range": {"start": 20_000}},
        {"port_range": {"start": 20_000, "end": 20_109}},
        {"port_range": {"start": True, "end": 20_110}},
        {"port_range": {"start": 65_425, "end": 65_536}},
    ],
)
def test_openclaw_rejects_invalid_port_range(settings: dict[str, object]):
    with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
        adapter._select_port(settings)

    assert caught.value.code == "openclaw_invalid_configuration"
    assert caught.value.metadata == {"field": "harness.settings.port_range"}


@pytest.mark.parametrize(
    "settings",
    [
        pytest.param(
            {"interfaces": {"terminal": {"port": 20_000}}}, id="terminal-has-no-port"
        ),
        pytest.param(
            {"interfaces": {"dashboard": {"port": 20_000, "token": "x"}}},
            id="unknown-field",
        ),
        pytest.param({"interfaces": {"dashboard": {}}}, id="missing-port"),
        pytest.param(
            {"interfaces": {"dashboard": {"port": 65_426}}}, id="port-too-high"
        ),
        pytest.param(
            {
                "interfaces": {"dashboard": {"port": 20_000}},
                "port_range": {"start": 20_000, "end": 20_200},
            },
            id="port-range-with-dashboard",
        ),
        pytest.param({"startup_timeout_seconds": 76}, id="startup-timeout-too-long"),
        pytest.param({"agent_id": "my agent"}, id="agent-id-spaces"),
        pytest.param({"agent_id": "a" * 65}, id="agent-id-too-long"),
        pytest.param({"agent_id": "-agent"}, id="agent-id-prefix"),
    ],
)
def test_openclaw_plan_rejects_invalid_settings(
    tmp_path: Path, settings: dict[str, object]
):
    config = FabricConfig.from_mapping(
        {
            "metadata": {"name": "openclaw-interfaces"},
            "harness": {
                "adapter_id": "nvidia.fabric.openclaw",
                "resolution": "preinstalled",
                "settings": settings,
            },
            "models": {"default": {"provider": "test", "model": "fabric-echo"}},
            "environment": {"provider": "local", "workspace": "."},
        }
    )

    with pytest.raises(FabricConfigError):
        Fabric().plan(config, base_dir=tmp_path)


@pytest.mark.parametrize("raw", ["2026.9.34 extra", "2026.9.34\nuntrusted suffix"])
async def test_gateway_records_the_same_version_it_validates(monkeypatch, raw):
    from nemo_fabric_adapters.common import provenance

    mock_record = MagicMock()
    monkeypatch.setattr(provenance, "record_version", mock_record)

    def respond(request):
        data = (
            {"data": [{"id": "openclaw/default"}]}
            if request.url.path == "/v1/models"
            else {"version": raw}
        )
        return httpx.Response(200, json=data)

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        validated = await adapter.OpenClawRuntime()._probe_gateway(
            client, "http://gateway.test", "default"
        )
    assert validated == "2026.9.34"
    assert mock_record.call_args.args[1] == validated


def test_openclaw_accepts_supported_version():
    assert (
        adapter._validate_openclaw_version("OpenClaw 2026.9.34 (3a9d69d)")
        == "2026.9.34"
    )


@pytest.mark.parametrize("version", ["2026.9.4", "2026.9.5", "2026.10.7"])
def test_openclaw_rejects_unsupported_version(version: str):
    with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
        adapter._validate_openclaw_version(f"OpenClaw {version} (abcdef0)")

    assert caught.value.code == "openclaw_unsupported_version"
    assert caught.value.metadata == {
        "detected_version": version,
        "supported_versions": ["2026.9.34"],
    }


@pytest.mark.parametrize("output", ["not-openclaw 2026.9.4", "OpenClaw development"])
def test_openclaw_rejects_unrecognized_version_output(output: str):
    with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
        adapter._validate_openclaw_version(output)

    assert caught.value.code == "openclaw_version_check_failed"


_SETPRIV_RUNS = 'shift 3\nexec "$@"\n'
_POSIX_SCRIPT = pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX shell scripts stand in for setpriv"
)


@pytest.mark.parametrize(
    ("platform", "script", "usable"),
    [
        pytest.param(
            "darwin", _SETPRIV_RUNS, False, id="non-linux", marks=_POSIX_SCRIPT
        ),
        pytest.param("linux", None, False, id="linux-without-setpriv"),
        pytest.param(
            "linux", _SETPRIV_RUNS, True, id="setpriv-runs", marks=_POSIX_SCRIPT
        ),
        # An OpenShell sandbox forbids applying capabilities, so setpriv exits
        # before running its command.
        pytest.param(
            "linux",
            "echo 'setpriv: activate capabilities: Operation not permitted' >&2\n"
            "exit 1\n",
            False,
            id="setpriv-refused",
            marks=_POSIX_SCRIPT,
        ),
    ],
)
async def test_openclaw_gateway_uses_setpriv_on_linux_only_when_it_can_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    platform: str,
    script: str | None,
    usable: bool,
):
    setpriv = tmp_path / "setpriv"
    if script is not None:
        setpriv.write_text("#!/bin/sh\n" + script, encoding="utf-8")
        setpriv.chmod(0o755)
    monkeypatch.setattr(adapter.sys, "platform", platform)
    monkeypatch.setattr(
        adapter.shutil,
        "which",
        MagicMock(return_value=str(setpriv) if script is not None else None),
    )

    prefix = await adapter._parent_death_prefix()

    assert prefix == ([str(setpriv), "--pdeathsig", "SIGTERM", "--"] if usable else [])


async def test_openclaw_gateway_skips_setpriv_when_the_probe_cannot_start(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(adapter.sys, "platform", "linux")
    monkeypatch.setattr(
        adapter.shutil, "which", MagicMock(return_value="/usr/bin/setpriv")
    )
    monkeypatch.setattr(
        adapter.asyncio,
        "create_subprocess_exec",
        AsyncMock(side_effect=OSError("permission denied")),
    )

    assert await adapter._parent_death_prefix() == []


async def test_openclaw_gateway_probe_timeout_kills_and_reaps_process(
    monkeypatch: pytest.MonkeyPatch,
):
    mock_process = MagicMock(spec=asyncio.subprocess.Process)
    mock_process.returncode = None
    mock_process.wait = AsyncMock(side_effect=asyncio.Event().wait)
    mock_process.communicate = AsyncMock(return_value=(b"", b""))
    monkeypatch.setattr(adapter.sys, "platform", "linux")
    monkeypatch.setattr(
        adapter.shutil, "which", MagicMock(return_value="/usr/bin/setpriv")
    )
    monkeypatch.setattr(
        adapter.asyncio,
        "create_subprocess_exec",
        AsyncMock(return_value=mock_process),
    )

    assert await adapter._parent_death_prefix(timeout=0.01) == []

    mock_process.kill.assert_called_once_with()
    mock_process.communicate.assert_awaited_once_with()


async def test_openclaw_gateway_probe_cancellation_kills_and_reaps_process(
    monkeypatch: pytest.MonkeyPatch,
):
    mock_process = MagicMock(spec=asyncio.subprocess.Process)
    mock_process.returncode = None
    started = asyncio.Event()

    async def wait() -> int:
        started.set()
        await asyncio.Event().wait()
        return 0

    mock_process.wait = AsyncMock(side_effect=wait)
    mock_process.communicate = AsyncMock(return_value=(b"", b""))
    monkeypatch.setattr(adapter.sys, "platform", "linux")
    monkeypatch.setattr(
        adapter.shutil, "which", MagicMock(return_value="/usr/bin/setpriv")
    )
    monkeypatch.setattr(
        adapter.asyncio,
        "create_subprocess_exec",
        AsyncMock(return_value=mock_process),
    )
    task = asyncio.create_task(adapter._parent_death_prefix())
    await started.wait()

    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    mock_process.kill.assert_called_once_with()
    mock_process.communicate.assert_awaited_once_with()


@pytest.mark.parametrize(
    "prefix",
    [
        pytest.param(
            ["/usr/bin/setpriv", "--pdeathsig", "SIGTERM", "--"], id="setpriv"
        ),
        # Where setpriv cannot run, the Gateway command runs unprefixed.
        pytest.param([], id="unprefixed"),
    ],
)
def test_openclaw_gateway_command_runs_the_gateway_after_its_prefix(prefix: list[str]):
    openclaw = Path("/usr/bin/openclaw")

    command = adapter._gateway_command(openclaw, port=20_000, prefix=prefix)

    assert command == [
        *prefix,
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


@pytest.mark.skipif(
    sys.platform in {"darwin", "win32"}, reason="Workes locally, fails in CI"
)
async def test_openclaw_runtime_generates_config_invokes_and_cleans_up(
    mock_openclaw: Path, tmp_path: Path
):
    skills = tmp_path / "skills"
    skills.mkdir()
    (skills / "SKILL.md").write_text("# Test skill\n", encoding="utf-8")
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
            "config": _config(mock_openclaw, with_skills=True),
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
    assert result.usage.total_tokens is None
    assert result.usage.extensions[adapter.OPENCLAW_USAGE_EVIDENCE] == {
        "observed": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
        "gaps": ["scope_unverified"],
    }
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
    assert generated["gateway"]["controlUi"] == {"enabled": False}
    assert not (tmp_path / "artifacts").exists()
    # The sandbox would deny OpenClaw's model catalog lookup anyway.
    assert generated["models"]["catalogRefresh"] == {"enabled": False}
    assert not state_root.exists()


@pytest.mark.parametrize(
    ("invalid_path", "expected_code"),
    [
        ("missing", "openclaw_skill_not_found"),
        ("file", "openclaw_skill_invalid"),
        ("missing_manifest", "openclaw_skill_invalid"),
    ],
)
def test_openclaw_rejects_invalid_skill_paths(
    mock_openclaw: Path,
    tmp_path: Path,
    invalid_path: str,
    expected_code: str,
):
    skill_path = tmp_path / "skills"
    if invalid_path == "file":
        skill_path.write_text("not a directory", encoding="utf-8")
    elif invalid_path == "missing_manifest":
        skill_path.mkdir()

    with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
        adapter._openclaw_config(
            _config(mock_openclaw, with_skills=True),
            _context(tmp_path),
            base_dir=tmp_path,
            port=20_000,
            token_env="OPENCLAW_GATEWAY_TOKEN",
        )

    assert caught.value.code == expected_code
    assert caught.value.metadata == {"field": "skills.paths[0]"}


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


def test_openclaw_uses_provider_local_model_id(tmp_path: Path):
    config = AgentConfig.from_mapping(
        {
            "models": {
                "default": {
                    "provider": "nvidia",
                    "model": "nvidia/nemotron-test",
                    "base_url": "https://integrate.api.nvidia.com/v1",
                }
            }
        }
    )

    generated = adapter._openclaw_config(
        config,
        _context(tmp_path),
        base_dir=tmp_path,
        port=20_000,
        token_env="OPENCLAW_GATEWAY_TOKEN",
    )

    assert generated["agents"]["defaults"]["model"]["primary"] == (
        "nvidia/nemotron-test"
    )
    assert generated["models"]["providers"]["nvidia"]["models"] == [
        {"id": "nemotron-test", "name": "nemotron-test"}
    ]


def test_openclaw_prepared_service_passes_through_native_channel_config(
    mock_openclaw: Path, tmp_path: Path
):
    config = _config(mock_openclaw)
    assert config.harness is not None
    config.harness.settings.update(
        {
            "agent_id": "reviewer",
            "channel_config": {
                "channels": {
                    "telegram": {
                        "defaultAccount": "support",
                        "accounts": {
                            "support": {
                                "botToken": {
                                    "source": "env",
                                    "provider": "default",
                                    "id": "TELEGRAM_SUPPORT_BOT_TOKEN",
                                },
                                "apiRoot": "http://127.0.0.1:19090",
                                "dmPolicy": "open",
                                "allowFrom": ["*"],
                            },
                            "alerts": {
                                "botToken": {
                                    "source": "env",
                                    "provider": "default",
                                    "id": "TELEGRAM_ALERTS_BOT_TOKEN",
                                },
                                "dmPolicy": "allowlist",
                                "allowFrom": ["tg:123456789"],
                            },
                        },
                    }
                },
                "bindings": [
                    {
                        "agentId": "reviewer",
                        "match": {"channel": "telegram", "accountId": "*"},
                    }
                ],
            },
        }
    )

    generated = adapter._openclaw_config(
        config,
        _context(tmp_path),
        base_dir=tmp_path,
        port=20_000,
        token_env="OPENCLAW_GATEWAY_TOKEN",
        service_mode=True,
    )

    model = generated["agents"]["defaults"]["models"]["test/fabric-echo"]
    assert model["agentRuntime"] == {"id": "openclaw"}
    assert generated["agents"]["entries"] == {"reviewer": {}}
    assert (
        generated["channels"] == config.harness.settings["channel_config"]["channels"]
    )
    assert generated["bindings"] == [
        {
            "agentId": "reviewer",
            "match": {"channel": "telegram", "accountId": "*"},
        }
    ]


def test_openclaw_explicitly_declares_default_agent_for_channel_bindings(
    mock_openclaw: Path, tmp_path: Path
):
    config = _config(mock_openclaw)
    assert config.harness is not None
    config.harness.settings["channel_config"] = {
        "channels": {"telegram": {"enabled": True}},
        "bindings": [
            {
                "agentId": "default",
                "match": {"channel": "telegram", "accountId": "default"},
            }
        ],
    }

    generated = adapter._openclaw_config(
        config,
        _context(tmp_path),
        base_dir=tmp_path,
        port=20_000,
        token_env="OPENCLAW_GATEWAY_TOKEN",
        service_mode=True,
    )

    assert generated["agents"]["entries"] == {"default": {}}


@pytest.mark.parametrize(
    "api_root",
    [
        "http://telegram.example.com",
        "http://localhost:8081",
        "ftp://127.0.0.1:8081",
        "https://user:password@telegram.example.com",
    ],
)
def test_openclaw_rejects_insecure_telegram_api_roots(
    mock_openclaw: Path, tmp_path: Path, api_root: str
):
    config = _config(mock_openclaw)
    assert config.harness is not None
    config.harness.settings["channel_config"] = {
        "channels": {"telegram": {"apiRoot": api_root}},
        "bindings": [
            {
                "agentId": "default",
                "match": {"channel": "telegram", "accountId": "default"},
            }
        ],
    }

    with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
        adapter._openclaw_config(
            config,
            _context(tmp_path),
            base_dir=tmp_path,
            port=20_000,
            token_env="OPENCLAW_GATEWAY_TOKEN",
            service_mode=True,
        )

    assert caught.value.code == "openclaw_invalid_configuration"
    assert caught.value.metadata["field"] == (
        "harness.settings.channel_config.channels.telegram.apiRoot"
    )


def test_openclaw_managed_runtime_rejects_channel_config(
    mock_openclaw: Path, tmp_path: Path
):
    config = _config(mock_openclaw)
    assert config.harness is not None
    config.harness.settings["channel_config"] = {
        "channels": {"telegram": {"enabled": True}},
        "bindings": [
            {
                "agentId": "default",
                "match": {"channel": "telegram", "accountId": "*"},
            }
        ],
    }

    with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
        adapter._openclaw_config(
            config,
            _context(tmp_path),
            base_dir=tmp_path,
            port=20_000,
            token_env="OPENCLAW_GATEWAY_TOKEN",
        )

    assert caught.value.code == "openclaw_channels_require_service"


def test_openclaw_channel_binding_must_target_configured_agent(
    mock_openclaw: Path, tmp_path: Path
):
    config = _config(mock_openclaw)
    assert config.harness is not None
    config.harness.settings["channel_config"] = {
        "channels": {"telegram": {"enabled": True}},
        "bindings": [
            {
                "agentId": "another-agent",
                "match": {"channel": "telegram", "accountId": "*"},
            }
        ],
    }

    with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
        adapter._openclaw_config(
            config,
            _context(tmp_path),
            base_dir=tmp_path,
            port=20_000,
            token_env="OPENCLAW_GATEWAY_TOKEN",
            service_mode=True,
        )

    assert caught.value.code == "openclaw_invalid_configuration"
    assert caught.value.metadata["field"] == (
        "harness.settings.channel_config.bindings[0].agentId"
    )


async def test_openclaw_channel_secret_refs_require_environment_variables(
    mock_openclaw: Path, tmp_path: Path
):
    os.environ.pop("TELEGRAM_SUPPORT_BOT_TOKEN", None)
    config = _config(mock_openclaw)
    assert config.harness is not None
    config.harness.settings["channel_config"] = {
        "channels": {
            "telegram": {
                "accounts": {
                    "support": {
                        "botToken": {
                            "source": "env",
                            "provider": "default",
                            "id": "TELEGRAM_SUPPORT_BOT_TOKEN",
                        }
                    }
                }
            }
        },
        "bindings": [
            {
                "agentId": "default",
                "match": {"channel": "telegram", "accountId": "support"},
            }
        ],
    }

    with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
        await adapter.OpenClawRuntime().start(
            {
                "config": config,
                "runtime_context": _context(tmp_path).to_mapping(),
                "base_dir": str(tmp_path),
                "service": {
                    "operation": "prepare",
                    "connection_path": str(tmp_path / "connection.json"),
                },
            }
        )

    assert caught.value.code == "openclaw_missing_channel_secret"
    assert caught.value.metadata == {
        "environment_variables": ["TELEGRAM_SUPPORT_BOT_TOKEN"]
    }


def test_openclaw_attach_rejects_deployment_owned_configuration(
    mock_openclaw: Path,
):
    config = _config(mock_openclaw, with_skills=True)
    config.models["default"].settings["supports_usage_in_streaming"] = True
    assert config.harness is not None
    config.harness.settings["channel_config"] = {
        "channels": {"telegram": {"enabled": True}},
        "bindings": [
            {
                "agentId": "default",
                "match": {"channel": "telegram", "accountId": "*"},
            }
        ],
    }

    with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
        adapter._validate_attach_config(config)

    assert caught.value.code == "openclaw_attach_configuration_unverifiable"
    assert set(caught.value.metadata["fields"]) == {
        "models.base_url",
        "models.settings.supports_usage_in_streaming",
        "tools",
        "mcp",
        "skills",
        "harness.settings.openclaw_command",
        "harness.settings.channel_config",
    }


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://127.0.0.1:18789",
        "http://[::1]:18789",
        "https://openclaw.example.com",
    ],
)
def test_openclaw_gateway_endpoint_accepts_secure_or_loopback_urls(endpoint: str):
    assert adapter._gateway_endpoint(endpoint) == endpoint


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://192.0.2.1:18789",
        "http://localhost:18789",
        "http://openclaw.example.com",
    ],
)
def test_openclaw_gateway_endpoint_rejects_nonliteral_or_remote_http(endpoint: str):
    with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
        adapter._gateway_endpoint(endpoint)

    assert caught.value.code == "openclaw_invalid_service_reference"


def test_openclaw_service_connection_without_fchmod(tmp_path: Path, monkeypatch):
    """Older Windows Python versions can persist an attached service safely."""
    monkeypatch.delattr(adapter.os, "fchmod", raising=False)
    connection_path = tmp_path / "private" / "service-connection.json"

    adapter._write_service_connection(
        connection_path,
        gateway_url="http://127.0.0.1:18789",
        token="TEST_ONLY_GATEWAY_TOKEN",
        agent_id="default",
    )

    assert adapter._read_service_connection(connection_path) == (
        "http://127.0.0.1:18789",
        "TEST_ONLY_GATEWAY_TOKEN",
        "default",
    )
    assert list(connection_path.parent.iterdir()) == [connection_path]
    if os.name == "posix":
        assert connection_path.stat().st_mode & 0o777 == 0o600


@pytest.mark.skipif(sys.platform == "win32", reason="Requires POSIX file modes")
def test_openclaw_service_connection_atomically_replaces_symlink(tmp_path: Path):
    private = tmp_path / "private"
    private.mkdir(mode=0o777)
    victim = tmp_path / "victim.json"
    victim.write_text("unchanged", encoding="utf-8")
    connection_path = private / "service-connection.json"
    connection_path.symlink_to(victim)

    adapter._write_service_connection(
        connection_path,
        gateway_url="http://127.0.0.1:18789",
        token="secret",
        agent_id="default",
    )

    assert not connection_path.is_symlink()
    assert victim.read_text(encoding="utf-8") == "unchanged"
    assert private.stat().st_mode & 0o777 == 0o700
    assert connection_path.stat().st_mode & 0o777 == 0o600


@pytest.mark.skipif(
    sys.platform in {"darwin", "win32"}, reason="Requires POSIX process supervision"
)
async def test_openclaw_prepared_service_supports_multiple_runtime_sessions(
    mock_openclaw: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    capture = tmp_path / "config.json"
    request_capture = tmp_path / "request.json"
    requests_capture = tmp_path / "requests.jsonl"
    stopped_capture = tmp_path / "stopped.txt"
    connection_path = tmp_path / "private" / "service-connection.json"
    os.environ.update(
        {
            "FAKE_OPENCLAW_CAPTURE": str(capture),
            "FAKE_OPENCLAW_REQUEST": str(request_capture),
            "FAKE_OPENCLAW_REQUESTS": str(requests_capture),
            "FAKE_OPENCLAW_STOPPED": str(stopped_capture),
        }
    )
    service_context = _context(tmp_path, "service-1")
    service = adapter.OpenClawRuntime()
    info = await service.start(
        {
            "config": _config(mock_openclaw),
            "runtime_context": service_context.to_mapping(),
            "base_dir": str(tmp_path),
            "service": {
                "operation": "prepare",
                "connection_path": str(connection_path),
            },
        }
    )
    assert info is not None
    assert info["adapter_id"] == "nvidia.fabric.openclaw"
    assert info["service_type"] == "openclaw_gateway"
    assert info["metadata"] == {"openclaw_version": "2026.9.34"}
    assert "gateway_token" not in json.dumps(info)
    assert connection_path.parent.stat().st_mode & 0o777 == 0o700
    assert connection_path.stat().st_mode & 0o777 == 0o600

    runtimes: list[adapter.OpenClawRuntime] = []
    contexts: list[RuntimeContext] = []
    try:
        for runtime_id in ("runtime-a", "runtime-b"):
            runtime = adapter.OpenClawRuntime()
            context = _context(tmp_path, runtime_id)
            await runtime.start(
                {
                    "config": _config(mock_openclaw),
                    "runtime_context": context.to_mapping(),
                    "base_dir": str(tmp_path),
                    "service": {
                        "operation": "connect",
                        "service_id": "service-1",
                        "connection_path": str(connection_path),
                    },
                }
            )
            runtimes.append(runtime)
            contexts.append(context)
        results = await asyncio.gather(
            *(
                runtime.invoke(
                    AgentRunRequest(input=f"hello from {context.runtime_id}"),
                    context,
                )
                for runtime, context in zip(
                    runtimes,
                    contexts,
                    strict=True,
                )
            )
        )
        assert all(result.status == "succeeded" for result in results)

        private_connection = json.loads(connection_path.read_text(encoding="utf-8"))
        monkeypatch.setenv("ATTACHED_OPENCLAW_GATEWAY_TOKEN", "incorrect-token")
        rejected = adapter.OpenClawRuntime()
        with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
            await rejected.start(
                {
                    "config": AgentConfig.from_mapping(
                        {
                            "models": {
                                "default": {
                                    "provider": "test",
                                    "model": "fabric-echo",
                                }
                            }
                        }
                    ),
                    "runtime_context": _context(
                        tmp_path, "rejected-attachment"
                    ).to_mapping(),
                    "base_dir": str(tmp_path),
                    "service": {
                        "operation": "attach",
                        "connection_path": str(tmp_path / "rejected-connection.json"),
                        "reference": {
                            "adapter_id": "nvidia.fabric.openclaw",
                            "service_type": "openclaw_gateway",
                            "connection": {
                                "gateway_url": info["connection"]["gateway_url"],
                                "gateway_token_env": "ATTACHED_OPENCLAW_GATEWAY_TOKEN",
                            },
                        },
                    },
                }
            )
        assert caught.value.code == "openclaw_service_authentication_failed"

        monkeypatch.setenv(
            "ATTACHED_OPENCLAW_GATEWAY_TOKEN",
            private_connection["gateway_token"],
        )
        attached = adapter.OpenClawRuntime()
        attach_context = _context(tmp_path, "attached-service")
        attach_config = AgentConfig.from_mapping(
            {
                "models": {"default": {"provider": "test", "model": "fabric-echo"}},
            }
        )
        attached_info = await attached.start(
            {
                "config": attach_config,
                "runtime_context": attach_context.to_mapping(),
                "base_dir": str(tmp_path),
                "service": {
                    "operation": "attach",
                    "connection_path": str(tmp_path / "attached-connection.json"),
                    "reference": {
                        "adapter_id": "nvidia.fabric.openclaw",
                        "service_type": "openclaw_gateway",
                        "connection": {
                            "gateway_url": info["connection"]["gateway_url"],
                            "gateway_token_env": "ATTACHED_OPENCLAW_GATEWAY_TOKEN",
                        },
                    },
                },
            }
        )
        assert attached_info is not None
        assert attached_info["metadata"] == {"openclaw_version": "2026.9.34"}
        await attached.stop()
        assert not stopped_capture.exists()

        requests = [
            json.loads(line)
            for line in requests_capture.read_text(encoding="utf-8").splitlines()
        ]
        assert {request["user"] for request in requests} == {
            "runtime-a",
            "runtime-b",
        }
        assert not stopped_capture.exists()
    finally:
        for runtime in runtimes:
            await runtime.stop()
        await service.stop()

    assert stopped_capture.read_text(encoding="utf-8") == "stopped"


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


async def test_openclaw_invocation_deadline_bounds_the_gateway_request(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    context = _context(tmp_path, deadline_millis=int(time.time() * 1000) + 50)
    runtime = adapter.OpenClawRuntime()
    mock_process = MagicMock(spec=asyncio.subprocess.Process)
    mock_process.returncode = None
    runtime._client = MagicMock()
    runtime._config = _config(tmp_path / "openclaw")
    runtime._context = context
    runtime._port = 12345
    runtime._process = mock_process

    request_started = asyncio.Event()
    request_cancelled = asyncio.Event()

    async def wait_for_gateway(*_args, **_kwargs):
        request_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            request_cancelled.set()
            raise

    monkeypatch.setattr(
        adapter, "_invoke_gateway", AsyncMock(side_effect=wait_for_gateway)
    )

    result = await asyncio.wait_for(
        runtime.invoke(AgentRunRequest(input="Hello."), context), timeout=5
    )

    assert request_started.is_set()
    assert request_cancelled.is_set()
    assert result.status == adapter.contract.AgentRunStatus.FAILED
    assert result.error is not None
    assert result.error.code == "timeout"


async def test_openclaw_read_timeout_applies_only_when_configured():
    runtime = adapter.OpenClawRuntime()
    unset = runtime._client_for({}, "token")
    configured = runtime._client_for({"read_timeout_seconds": 30}, "token")
    try:
        assert unset.timeout.read is None
        assert configured.timeout.read == 30
    finally:
        await unset.aclose()
        await configured.aclose()


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
    mock_process.wait.assert_awaited()
    mock_close_job.assert_called_once_with(1234)
    assert log_task.cancelled()
    mock_temp_dir.cleanup.assert_called_once_with()
    assert runtime._client is None
    assert runtime._process is None
    assert runtime._windows_job is None
    assert runtime._log_tasks == []
    assert runtime._temp_dir is None


@pytest.mark.skipif(
    sys.platform in {"darwin", "win32"},
    reason="Starts the mock Gateway, which CI runs only on Linux",
)
async def test_openclaw_dashboard_serves_control_ui_with_a_token_file(
    mock_openclaw: Path, tmp_path: Path, dashboard_port: int
):
    capture = tmp_path / "config.json"
    os.environ["FAKE_OPENCLAW_CAPTURE"] = str(capture)
    token_path = _dashboard_token(tmp_path)
    runtime = adapter.OpenClawRuntime()

    await _start(
        runtime, tmp_path, _dashboard_config(mock_openclaw, {"port": dashboard_port})
    )
    try:
        gateway_token = runtime._token
        stored_token = token_path.read_text(encoding="utf-8")
        token_mode = token_path.stat().st_mode & 0o777
        directory_mode = token_path.parent.stat().st_mode & 0o777
    finally:
        await runtime.stop()

    generated = json.loads(capture.read_text(encoding="utf-8"))
    assert generated["gateway"]["port"] == dashboard_port
    assert generated["gateway"]["controlUi"] == {
        "enabled": True,
        "allowedOrigins": [
            f"http://127.0.0.1:{dashboard_port}",
            f"http://localhost:{dashboard_port}",
        ],
    }
    assert stored_token == gateway_token
    assert token_mode == 0o600
    assert directory_mode == 0o700
    assert not token_path.exists()


@pytest.mark.skipif(
    sys.platform in {"darwin", "win32"},
    reason="Starts the mock Gateway, which CI runs only on Linux",
)
async def test_openclaw_dashboard_disabled_serves_no_control_ui(
    mock_openclaw: Path, tmp_path: Path, dashboard_port: int
):
    capture = tmp_path / "config.json"
    os.environ["FAKE_OPENCLAW_CAPTURE"] = str(capture)
    runtime = adapter.OpenClawRuntime()

    await _start(
        runtime,
        tmp_path,
        _dashboard_config(mock_openclaw, {"port": dashboard_port, "enabled": False}),
    )
    try:
        gateway_port = runtime._port
        token_exists = _dashboard_token(tmp_path).exists()
    finally:
        await runtime.stop()

    generated = json.loads(capture.read_text(encoding="utf-8"))
    assert generated["gateway"]["controlUi"] == {"enabled": False}
    assert gateway_port != dashboard_port
    assert not token_exists


@pytest.mark.skipif(
    sys.platform in {"darwin", "win32"},
    reason="Starts the mock Gateway, which CI runs only on Linux",
)
async def test_openclaw_dashboard_removes_the_token_when_start_fails_after_writing_it(
    mock_openclaw: Path,
    tmp_path: Path,
    dashboard_port: int,
    monkeypatch: pytest.MonkeyPatch,
):
    os.environ["FAKE_OPENCLAW_CAPTURE"] = str(tmp_path / "config.json")
    token_path = _dashboard_token(tmp_path)
    token_written: list[bool] = []

    # Start reads the HTTP read timeout only after it writes the token.
    def fail_after_token_write(_settings):
        token_written.append(token_path.exists())
        raise RuntimeError("start failed after writing the token")

    monkeypatch.setattr(adapter, "_read_timeout", fail_after_token_write)
    runtime = adapter.OpenClawRuntime()

    with pytest.raises(RuntimeError, match="after writing the token"):
        await _start(
            runtime,
            tmp_path,
            _dashboard_config(mock_openclaw, {"port": dashboard_port}),
        )

    assert token_written == [True]
    assert not token_path.exists()
    assert runtime._process is None


@pytest.mark.skipif(
    sys.platform in {"darwin", "win32"},
    reason="Starts the mock Gateway, which CI runs only on Linux",
)
async def test_openclaw_dashboard_stop_retries_a_token_it_could_not_remove(
    mock_openclaw: Path,
    tmp_path: Path,
    dashboard_port: int,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    os.environ["FAKE_OPENCLAW_CAPTURE"] = str(tmp_path / "config.json")
    token_path = _dashboard_token(tmp_path)
    runtime = adapter.OpenClawRuntime()
    await _start(
        runtime, tmp_path, _dashboard_config(mock_openclaw, {"port": dashboard_port})
    )
    unlink = Path.unlink

    def fail_token_unlink(path: Path, missing_ok: bool = False) -> None:
        if path == token_path:
            raise PermissionError("token is locked")
        unlink(path, missing_ok=missing_ok)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", fail_token_unlink)
        await runtime.stop()

    assert runtime._process is None
    assert token_path.exists()
    assert "could not remove its dashboard token" in caplog.text

    await runtime.stop()

    assert not token_path.exists()


@pytest.mark.skipif(
    sys.platform == "win32", reason="Runs the mock openclaw script directly"
)
@pytest.mark.parametrize("offset", [0, 2])
async def test_openclaw_dashboard_port_in_use_fails_start(
    mock_openclaw: Path, tmp_path: Path, dashboard_port: int, offset: int
):
    runtime = adapter.OpenClawRuntime()

    with socket.create_server(("127.0.0.1", dashboard_port + offset)):
        with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
            await _start(
                runtime,
                tmp_path,
                _dashboard_config(mock_openclaw, {"port": dashboard_port}),
            )

    assert caught.value.code == "openclaw_interface_port_unavailable"
    assert caught.value.metadata == {
        "field": "harness.settings.interfaces.dashboard.port",
        "port": dashboard_port + offset,
    }
    assert runtime._process is None
    assert not (tmp_path / "artifacts").exists()


@pytest.mark.parametrize("operation", ["prepare", "attach", "connect"])
async def test_openclaw_services_reject_interfaces(
    mock_openclaw: Path, tmp_path: Path, operation: str
):
    runtime = adapter.OpenClawRuntime()

    with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
        await _start(
            runtime,
            tmp_path,
            _dashboard_config(mock_openclaw, {"port": 20_000}),
            service={"operation": operation},
        )

    assert caught.value.code == "openclaw_interfaces_require_managed_runtime"
    assert runtime._process is None


@pytest.mark.skipif(
    sys.platform in {"darwin", "win32"},
    reason="Starts the mock Gateway, which CI runs only on Linux",
)
async def test_openclaw_stop_ends_the_gateway_where_group_signals_are_denied(
    mock_openclaw: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    os.environ["FAKE_OPENCLAW_CAPTURE"] = str(tmp_path / "config.json")
    runtime = adapter.OpenClawRuntime()
    await runtime.start(
        {
            "config": _config(mock_openclaw),
            "runtime_context": _context(tmp_path).to_mapping(),
            "base_dir": str(tmp_path),
        }
    )
    gateway = runtime._process

    # OpenShell's sandbox denies signals to a process group with EPERM.
    def deny_group_signals(_pgid: int, _signum: int) -> None:
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(os, "killpg", deny_group_signals)
    try:
        await runtime.stop()

        assert gateway.returncode is not None
    finally:
        if gateway.returncode is None:
            gateway.kill()
            await gateway.wait()


@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups")
async def test_openclaw_stop_kills_a_gateway_that_ignores_sigterm_where_group_signals_are_denied(
    monkeypatch: pytest.MonkeyPatch,
):
    runtime = adapter.OpenClawRuntime()
    runtime._shutdown_timeout = 0.01
    mock_process = MagicMock(spec=asyncio.subprocess.Process)
    mock_process.pid = 1234
    mock_process.returncode = None
    wait_calls = 0

    async def wait_for_process():
        nonlocal wait_calls
        wait_calls += 1
        if wait_calls == 1:
            await asyncio.Event().wait()

    mock_process.wait = AsyncMock(side_effect=wait_for_process)
    runtime._process = mock_process
    signals: list[tuple[int, int]] = []

    def deny_group_signals(_pgid: int, _signum: int) -> None:
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(adapter.os, "killpg", deny_group_signals)
    monkeypatch.setattr(
        adapter.os, "kill", lambda pid, signum: signals.append((pid, signum))
    )

    await runtime.stop()

    assert signals == [(1234, signal.SIGTERM), (1234, signal.SIGKILL)]


@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups")
@pytest.mark.parametrize("group_signals_denied", [False, True])
async def test_openclaw_stop_is_bounded_with_descendant_holding_gateway_pipes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    group_signals_denied: bool,
):
    marker = tmp_path / "descendant.ready"
    descendant_code = """
import signal
import sys
import time
from pathlib import Path

signal.signal(signal.SIGTERM, signal.SIG_IGN)
Path(sys.argv[1]).touch()
time.sleep(60)
"""
    gateway_code = """
import signal
import subprocess
import sys
import time

if sys.argv[3] == "ignore-term":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
subprocess.Popen([sys.executable, "-c", sys.argv[1], sys.argv[2]])
time.sleep(60)
"""
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        gateway_code,
        descendant_code,
        str(marker),
        "ignore-term" if group_signals_denied else "exit-on-term",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    runtime = adapter.OpenClawRuntime()
    runtime._process = process
    runtime._shutdown_timeout = 0.1
    original_killpg = os.killpg

    def deny_group_signals(_pgid: int, _signum: int) -> None:
        raise PermissionError(1, "Operation not permitted")

    try:
        async with asyncio.timeout(5):
            while not marker.exists():
                await asyncio.sleep(0.01)

        if group_signals_denied:
            monkeypatch.setattr(os, "killpg", deny_group_signals)

        await asyncio.wait_for(runtime.stop(), timeout=2)

        assert process.returncode is not None
        assert process.stdout is not None
        assert process.stderr is not None
        if not group_signals_denied:
            # EOF proves the descendant closed the inherited pipes, whether
            # Process.wait returns at leader exit or waits for pipe closure.
            async with asyncio.timeout(1):
                await asyncio.gather(process.stdout.read(), process.stderr.read())
    finally:
        with suppress(ProcessLookupError):
            original_killpg(process.pid, signal.SIGKILL)
        await process.wait()
        if process.stdout is not None and process.stderr is not None:
            await asyncio.gather(process.stdout.read(), process.stderr.read())


@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups")
@pytest.mark.parametrize("group_signals_denied", [False, True])
async def test_openclaw_stop_does_not_kill_a_gateway_that_exited_during_cancellation(
    monkeypatch: pytest.MonkeyPatch, group_signals_denied: bool
):
    runtime = adapter.OpenClawRuntime()
    mock_process = MagicMock(spec=asyncio.subprocess.Process)
    mock_process.pid = 1234
    mock_process.returncode = None

    async def exit_then_cancel():
        mock_process.returncode = 0
        raise asyncio.CancelledError

    mock_process.wait = AsyncMock(side_effect=exit_then_cancel)
    runtime._process = mock_process
    group_signals: list[tuple[int, int]] = []
    pid_signals: list[tuple[int, int]] = []

    def signal_group(pid: int, signum: int) -> None:
        group_signals.append((pid, signum))
        if group_signals_denied:
            raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(adapter.os, "killpg", signal_group)
    monkeypatch.setattr(
        adapter.os, "kill", lambda pid, signum: pid_signals.append((pid, signum))
    )

    with pytest.raises(asyncio.CancelledError):
        await runtime.stop()

    assert group_signals == [(1234, signal.SIGTERM), (1234, signal.SIGKILL)]
    # Once the Gateway is reaped, only the process group may be signaled.
    assert pid_signals == ([(1234, signal.SIGTERM)] if group_signals_denied else [])


@pytest.mark.skipif(
    sys.platform == "win32", reason="Runs the mock openclaw script directly"
)
async def test_openclaw_startup_timeout_bounds_all_of_start(
    mock_openclaw: Path,
    tmp_path: Path,
    dashboard_port: int,
    monkeypatch: pytest.MonkeyPatch,
):
    # Each command finishes within the timeout, but the two together don't.
    monkeypatch.setenv("FAKE_OPENCLAW_COMMAND_DELAY", "0.6")
    monkeypatch.setenv("FAKE_OPENCLAW_CAPTURE", str(tmp_path / "config.json"))
    config = _dashboard_config(mock_openclaw, {"port": dashboard_port})
    assert config.harness is not None
    config.harness.settings["startup_timeout_seconds"] = 1
    runtime = adapter.OpenClawRuntime()

    with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
        await runtime.start(
            {
                "config": config,
                "runtime_context": _context(tmp_path).to_mapping(),
                "base_dir": str(tmp_path),
            }
        )

    assert caught.value.code == "openclaw_command_timeout"
    assert runtime._process is None
    assert not (tmp_path / "artifacts").exists()


async def test_openclaw_startup_deadline_bounds_a_pending_readiness_request():
    async def never_respond(reader, _writer):
        await reader.read()

    server = await asyncio.start_server(never_respond, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    runtime = adapter.OpenClawRuntime()
    mock_process = MagicMock(spec=asyncio.subprocess.Process)
    mock_process.returncode = None
    runtime._process = mock_process
    loop = asyncio.get_running_loop()
    started = loop.time()
    try:
        with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
            await runtime._wait_started(port, "token", 0.2)
    finally:
        server.close()

    assert caught.value.code == "openclaw_gateway_start_timeout"
    # Each readiness request otherwise waits up to 1 second for a response.
    assert loop.time() - started < 0.8


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

    # stop() awaits only mocks and in-process tasks here, and no timer has to
    # fire, so each step finishes within a few event-loop turns. Counting turns
    # instead of seconds keeps a garbage-collector pause on a slow runner from
    # failing the test.
    async def run_event_loop_until(condition):
        for _ in range(100):
            if condition():
                return True
            await asyncio.sleep(0)
        return condition()

    stop_task = asyncio.create_task(runtime.stop())
    try:
        assert await run_event_loop_until(entered.is_set)
        stop_task.cancel()
        assert await run_event_loop_until(stop_task.done)
        assert stop_task.cancelled()
    finally:
        release.set()
        stop_task.cancel()
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
    mock_process = MagicMock(spec=asyncio.subprocess.Process)
    mock_process.returncode = None

    async def communicate() -> tuple[bytes, bytes]:
        if mock_process.returncode is None:
            await asyncio.Event().wait()
        return b"", b""

    mock_process.communicate = AsyncMock(side_effect=communicate)
    mock_process.kill.side_effect = lambda: setattr(mock_process, "returncode", -9)
    create_subprocess = AsyncMock(return_value=mock_process)
    monkeypatch.setattr(adapter.asyncio, "create_subprocess_exec", create_subprocess)

    with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
        await adapter._command_output(
            Path("openclaw"), "--version", env={}, timeout=0.01
        )

    assert caught.value.code == "openclaw_command_timeout"
    assert caught.value.metadata == {"command": "--version", "timeout_seconds": 0.01}
    mock_process.kill.assert_called_once_with()
    assert mock_process.communicate.await_count == 2


async def test_openclaw_command_failure_uses_stdout_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
):
    process = MagicMock(spec=asyncio.subprocess.Process)
    process.returncode = 1
    process.communicate = AsyncMock(
        return_value=(b'{"issues":[{"message":"invalid binding"}]}', b"")
    )
    monkeypatch.setattr(
        adapter.asyncio,
        "create_subprocess_exec",
        AsyncMock(return_value=process),
    )

    with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
        await adapter._command_output(
            Path("openclaw"), "config", "validate", "--json", env={}, timeout=30
        )

    assert caught.value.code == "openclaw_command_failed"
    assert caught.value.metadata["detail"] == (
        '{"issues":[{"message":"invalid binding"}]}'
    )


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


@pytest.mark.skipif(os.name == "nt", reason="POSIX signal forwarding")
def test_openclaw_signal_handler_forwards_to_gateway_where_group_signals_are_denied(
    monkeypatch: pytest.MonkeyPatch,
):
    runtime = adapter.OpenClawRuntime()
    mock_process = MagicMock(spec=asyncio.subprocess.Process)
    mock_process.pid = 1234
    mock_process.returncode = None
    runtime._process = mock_process
    forwarded: list[tuple[int, int]] = []

    def deny_group_signals(_pgid: int, _signum: int) -> None:
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(adapter.os, "killpg", deny_group_signals)
    monkeypatch.setattr(
        adapter.os, "kill", lambda pid, signum: forwarded.append((pid, signum))
    )

    runtime._install_signal_handlers()
    try:
        with pytest.raises(SystemExit):
            runtime._signal_handler(signal.SIGTERM, None)
    finally:
        runtime._restore_signal_handlers()

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
    sys.platform != "linux" or not asyncio.run(adapter._parent_death_prefix()),
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
        # File creation precedes the gateway flushing its shutdown marker.
        while time.monotonic() < deadline:
            if (
                stopped_path.exists()
                and stopped_path.read_text(encoding="utf-8") == "stopped"
            ):
                break
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


@pytest.mark.skipif(
    sys.platform in {"darwin", "win32"}, reason="Requires POSIX process supervision"
)
async def test_openclaw_fabric_service_supports_multiple_runtimes(
    mock_openclaw: Path, tmp_path: Path
):
    os.environ.update(
        {
            "FAKE_OPENCLAW_CAPTURE": str(tmp_path / "config.json"),
            "FAKE_OPENCLAW_REQUEST": str(tmp_path / "request.json"),
            "FAKE_OPENCLAW_REQUESTS": str(tmp_path / "requests.jsonl"),
            "FAKE_OPENCLAW_STOPPED": str(tmp_path / "stopped.txt"),
        }
    )
    config = FabricConfig.from_mapping(
        {
            "metadata": {"name": "openclaw-service-test"},
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
            "environment": {"provider": "local", "workspace": "."},
        }
    )
    fabric = Fabric()
    service = await fabric.prepare_service(config, base_dir=tmp_path)
    first = await fabric.start_runtime(config, base_dir=tmp_path, service=service)
    second = await fabric.start_runtime(config, base_dir=tmp_path, service=service)
    try:
        first_result, second_result = await asyncio.gather(
            first.invoke(input="first"), second.invoke(input="second")
        )
        assert first_result.status == "succeeded"
        assert second_result.status == "succeeded"
        assert first.handle.service_id == service.service_id
        assert second.handle.service_id == service.service_id
        assert first.runtime_id != second.runtime_id
        with pytest.raises(FabricRuntimeError, match="active runtimes"):
            await service.release()
    finally:
        await first.stop()
        await second.stop()
        await service.release()

    requests = [
        json.loads(line)
        for line in (tmp_path / "requests.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert {request["user"] for request in requests} == {
        first.runtime_id,
        second.runtime_id,
    }
    assert (tmp_path / "stopped.txt").read_text(encoding="utf-8") == "stopped"


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
    assert descriptor["requirements"] == {}
    assert "agent_runtime" not in descriptor["settings_schema"]["properties"]
    assert "telegram" not in descriptor["settings_schema"]["properties"]
    assert "channel_config" in descriptor["settings_schema"]["properties"]
    assert "tools.enabled" in descriptor["config"]["accepts"]
    assert "tools.blocked" in descriptor["config"]["accepts"]
    assert "mcp.auth.oauth2" not in descriptor["config"]["accepts"]
    assert "mcp.auth.service_account" not in descriptor["config"]["accepts"]
    assert descriptor["capabilities"]["streaming"] is False
    assert "telemetry" not in descriptor
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("supports_usage", [True, False, None])
def test_openclaw_config_maps_streaming_usage_capability(
    tmp_path: Path, supports_usage: bool | None
):
    config = _config(Path("openclaw"))
    if supports_usage is not None:
        config.models["default"].settings["supports_usage_in_streaming"] = (
            supports_usage
        )
    generated = adapter._openclaw_config(
        config, _context(tmp_path), base_dir=tmp_path, port=20000, token_env="TOKEN"
    )
    model = generated["models"]["providers"]["test"]["models"][0]
    if supports_usage is None:
        assert "compat" not in model
    else:
        assert model["compat"] == {"supportsUsageInStreaming": supports_usage}


@pytest.mark.parametrize("value", [None, "true", 1])
def test_openclaw_config_rejects_invalid_streaming_usage_capability(
    tmp_path: Path, value
):
    config = _config(Path("openclaw"))
    config.models["default"].settings["supports_usage_in_streaming"] = value
    with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
        adapter._openclaw_config(
            config, _context(tmp_path), base_dir=tmp_path, port=20000, token_env="TOKEN"
        )
    assert caught.value.code == "openclaw_invalid_configuration"


def test_openclaw_streaming_usage_capability_requires_custom_endpoint(tmp_path: Path):
    config = _config(Path("openclaw"))
    config.models["default"].base_url = None
    config.models["default"].settings["supports_usage_in_streaming"] = True
    with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
        adapter._openclaw_config(
            config, _context(tmp_path), base_dir=tmp_path, port=20000, token_env="TOKEN"
        )

    assert caught.value.code == "openclaw_invalid_configuration"


@pytest.mark.parametrize(
    ("counters", "observed", "gaps"),
    [
        (None, {}, ["scope_unverified", "missing_usage"]),
        ({}, {}, ["scope_unverified", "missing_usage"]),
        (
            {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            ["scope_unverified", "zero_unverified"],
        ),
        (
            {"input_tokens": 42, "output_tokens": 8, "total_tokens": 50},
            {"input_tokens": 42, "output_tokens": 8, "total_tokens": 50},
            ["scope_unverified"],
        ),
    ],
)
def test_openclaw_gateway_usage_remains_observational(counters, observed, gaps):
    from nemo_fabric_adapter_contract.models import AgentUsage

    usage = adapter._gateway_usage_evidence(
        AgentUsage(**counters) if counters is not None else None
    )
    assert usage.input_tokens is None
    assert usage.output_tokens is None
    assert usage.total_tokens is None
    assert usage.cost_usd is None
    assert usage.extensions == {
        adapter.OPENCLAW_USAGE_EVIDENCE: {"observed": observed, "gaps": gaps}
    }


async def test_openclaw_gateway_stream_keeps_usage_observational():
    import httpx

    chunks = [
        {"choices": [{"delta": {"content": "done"}}]},
        {
            "choices": [],
            "usage": {"prompt_tokens": 42, "completion_tokens": 8, "total_tokens": 50},
        },
    ]
    body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
    body += "data: [DONE]\n\n"
    response = httpx.Response(
        200, text=body, headers={"content-type": "text/event-stream"}
    )
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: response)
    ) as client:
        text, usage = await adapter._invoke_gateway(
            client,
            "http://gateway.example.test/v1/chat/completions",
            messages=[{"role": "user", "content": "finish"}],
            temperature=None,
            top_p=None,
            max_tokens=None,
            user="runtime-test",
        )
    assert text == "done"
    assert usage.input_tokens is None
    assert usage.output_tokens is None
    assert usage.extensions[adapter.OPENCLAW_USAGE_EVIDENCE] == {
        "observed": {"input_tokens": 42, "output_tokens": 8, "total_tokens": 50},
        "gaps": ["scope_unverified"],
    }


@pytest.mark.skipif(os.name != "posix", reason="The native terminal requires POSIX")
@pytest.mark.parametrize(
    "selection, shared",
    [
        ({}, False),
        ({"terminal": {"enabled": False}}, False),
        ({"terminal": {}}, True),
    ],
)
@pytest.mark.parametrize("agent_id", ["default", "Reviewer"])
async def test_native_configuration_owns_prompt_once(
    mock_openclaw, tmp_path, selection, shared, agent_id
):
    capture = tmp_path / "config.json"
    request_capture = tmp_path / "request.json"
    os.environ["FAKE_OPENCLAW_CAPTURE"] = str(capture)
    os.environ["FAKE_OPENCLAW_REQUEST"] = str(request_capture)
    config = _config(mock_openclaw)
    config.harness.settings["interfaces"] = selection
    config.harness.settings["agent_id"] = agent_id
    runtime = adapter.OpenClawRuntime()
    await _start(runtime, tmp_path, config)
    try:
        result = await runtime.invoke(
            AgentRunRequest(input="Check shared configuration."), _context(tmp_path)
        )
        assert result.status == "succeeded"
        request = json.loads(request_capture.read_text())
        assert [message["role"] for message in request["messages"]] == (
            ["user"] if shared else ["system", "user"]
        )
        generated = json.loads(capture.read_text())
        if shared:
            entry = generated["plugins"]["entries"]["fabric-shared-config"]
            assert entry["config"] == {
                "agentId": agent_id,
                "systemInstruction": "Be concise.",
            }
            directory = _dashboard_token(tmp_path).parent
            assert (directory / "terminal.running").exists()
            assert (directory / "gateway.json").exists()
            assert (await runtime.check("ready"))["activity"] == "unknown"
        else:
            assert "plugins" not in generated
            assert not _dashboard_token(tmp_path).parent.exists()
        for level in ("live", "active", "ready"):
            report = await runtime.check(level)
            assert report["outcome"] == "passed", report
    finally:
        await runtime.stop()
    assert not (_dashboard_token(tmp_path).parent / "terminal.running").exists()


@pytest.mark.parametrize(
    "agent_id", ["my agent", "-agent", "_agent", "a" * 65, "é", " reviewer "]
)
async def test_openclaw_rejects_noncanonical_agent_ids(
    mock_openclaw, tmp_path, agent_id
):
    config = _config(mock_openclaw)
    config.harness.settings["agent_id"] = agent_id
    runtime = adapter.OpenClawRuntime()
    with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
        await _start(runtime, tmp_path, config)
    assert caught.value.code == "openclaw_invalid_configuration"
    assert caught.value.metadata == {"field": "harness.settings.agent_id"}
    assert runtime._process is None


@pytest.mark.skipif(os.name != "posix", reason="The native terminal requires POSIX")
@pytest.mark.parametrize("mode", ["missing", "inactive", "wrong-agent"])
async def test_shared_instruction_probe_fails_before_publishing_interfaces(
    mock_openclaw, tmp_path, mode
):
    os.environ["FAKE_OPENCLAW_SHARED_CONFIG"] = mode
    os.environ["FAKE_OPENCLAW_CAPTURE"] = str(tmp_path / "config.json")
    config = _config(mock_openclaw)
    config.harness.settings["interfaces"] = {"terminal": {}}
    runtime = adapter.OpenClawRuntime()
    with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
        await _start(runtime, tmp_path, config)
    assert caught.value.code == "openclaw_shared_configuration_unavailable"
    assert runtime._process is None
    assert not (_dashboard_token(tmp_path).parent / "gateway.json").exists()
    assert not (_dashboard_token(tmp_path).parent / "terminal.running").exists()


@pytest.mark.skipif(
    not os.environ.get("FABRIC_OPENCLAW_TEST_COMMAND"),
    reason="Set FABRIC_OPENCLAW_TEST_COMMAND for actual plugin failure qualification",
)
@pytest.mark.parametrize(
    "failure",
    ["missing", "renamed", "conversation-denied", "prompt-denied", "wrong-agent"],
)
async def test_native_shared_hook_failures_reject_startup(
    tmp_path, api_server, monkeypatch, failure
):
    plugin = tmp_path / "plugin"
    shutil.copytree(Path(adapter.__file__).with_name("native_config"), plugin)
    entrypoint = plugin / "index.mjs"
    if failure == "missing":
        entrypoint.write_text(
            'throw new Error("plugin load failure for qualification");'
        )
    elif failure == "renamed":
        entrypoint.write_text(
            entrypoint.read_text().replace(
                'api.on("before_prompt_build"', 'api.on("agent_end"'
            )
        )
    original = adapter._openclaw_config

    def generated_config(*args, **kwargs):
        generated = original(*args, **kwargs)
        generated["plugins"]["load"]["paths"] = [str(plugin)]
        shared = generated["plugins"]["entries"]["fabric-shared-config"]
        if failure == "conversation-denied":
            shared["hooks"]["allowConversationAccess"] = False
        elif failure == "prompt-denied":
            shared["hooks"]["allowPromptInjection"] = False
        elif failure == "wrong-agent":
            shared["config"]["agentId"] = "other"
        return generated

    monkeypatch.setattr(adapter, "_openclaw_config", generated_config)
    config = _config(Path(os.environ["FABRIC_OPENCLAW_TEST_COMMAND"]))
    config.harness.settings["interfaces"] = {"terminal": {}}
    config.models["default"].base_url = f"{api_server}/v1"
    runtime = adapter.OpenClawRuntime()
    with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
        await _start(runtime, tmp_path, config)
    assert caught.value.code == "openclaw_shared_configuration_unavailable"
    assert runtime._process is None
    assert not (_dashboard_token(tmp_path).parent / "gateway.json").exists()
    async with httpx.AsyncClient(trust_env=False) as client:
        assert (await client.get(f"{api_server}/_requests")).json() == []


@pytest.mark.parametrize("agent_id", ["Reviewer", "agent_1-A", "a" * 64])
def test_openclaw_preserves_valid_agent_id_spelling(agent_id):
    assert adapter._agent_id({"agent_id": agent_id}) == agent_id


def _mcp_diagnostic(name: str, message: str) -> dict[str, object]:
    return {
        "id": f"mcp-server-diagnostic:{name}",
        "severity": "warning",
        "message": f'MCP server "{name}": {message}',
        "servers": [name],
    }


def _mcp_status_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, inventories: list[object]
) -> tuple[adapter.OpenClawRuntime, AsyncMock]:
    """Return a started-looking runtime whose Gateway returns ``inventories`` in turn."""

    context = _context(tmp_path, runtime_id="Runtime-1")
    runtime = adapter.OpenClawRuntime()
    mock_process = MagicMock(spec=asyncio.subprocess.Process)
    mock_process.returncode = None
    runtime._client = MagicMock()
    runtime._config = _config(tmp_path / "openclaw")
    runtime._context = context
    runtime._port = 12345
    runtime._process = mock_process
    runtime._command = tmp_path / "openclaw"
    runtime._command_env = {"OPENCLAW_GATEWAY_TOKEN": "secret-token"}
    runtime._agent_id = "Reviewer"
    runtime._mcp_transports = {"local": "stdio", "remote": "streamable-http"}
    monkeypatch.setattr(
        adapter,
        "_invoke_gateway",
        AsyncMock(return_value=("OpenClaw response", adapter.contract.AgentUsage())),
    )
    status = AsyncMock(side_effect=[json.dumps(value) for value in inventories])
    monkeypatch.setattr(adapter, "_command_output", status)
    return runtime, status


def test_openclaw_unavailable_mcp_servers_use_native_diagnostics_without_messages():
    """OpenClaw unavailable MCP servers use native diagnostics without messages."""
    inventory = {
        "groups": [],
        "notices": [
            _mcp_diagnostic(
                "remote", "fetch failed | ECONNREFUSED https://user:pw@example.test"
            ),
            _mcp_diagnostic(
                "local",
                'MCP server "local" timed out: did not complete initialize within 30s',
            ),
            _mcp_diagnostic("plugin-owned", "fetch failed"),
            {"id": "mcp-not-yet-connected", "servers": ["local"], "message": "x"},
            {"id": "unsupported-tool-schema:local__x", "message": "x"},
            "not a notice",
        ],
    }

    unavailable = adapter._unavailable_mcp_servers(
        inventory, {"local": "stdio", "remote": "streamable-http"}
    )

    assert unavailable == [
        {"name": "remote", "transport": "streamable-http", "reason": "failed"},
        {"name": "local", "transport": "stdio", "reason": "timed_out"},
    ]
    assert "ECONNREFUSED" not in json.dumps(unavailable)
    assert adapter._unavailable_mcp_servers({"ok": False}, {"local": "stdio"}) == []
    assert adapter._unavailable_mcp_servers(None, {"local": "stdio"}) == []


async def test_openclaw_results_report_current_unavailable_mcp_servers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
):
    """OpenClaw results report current unavailable MCP servers."""
    runtime, status = _mcp_status_runtime(
        tmp_path,
        monkeypatch,
        [
            {"notices": [_mcp_diagnostic("remote", "fetch failed")]},
            {
                "notices": [
                    _mcp_diagnostic("remote", "fetch failed"),
                    _mcp_diagnostic("local", "spawn ENOENT"),
                ]
            },
            {"notices": []},
        ],
    )
    context = runtime._context
    assert context is not None

    first = await runtime.invoke(AgentRunRequest(input="One."), context)
    second = await runtime.invoke(AgentRunRequest(input="Two."), context)
    recovered = await runtime.invoke(AgentRunRequest(input="Three."), context)

    assert first.output == {
        "response": "OpenClaw response",
        "mcp": {
            "unavailable_servers": [
                {"name": "remote", "transport": "streamable-http", "reason": "failed"}
            ]
        },
    }
    assert second.output["mcp"] == {
        "unavailable_servers": [
            {"name": "local", "transport": "stdio", "reason": "failed"},
            {"name": "remote", "transport": "streamable-http", "reason": "failed"},
        ]
    }
    assert recovered.output == {"response": "OpenClaw response"}
    warnings = capsys.readouterr().err.splitlines()
    assert warnings == [
        "NeMo Fabric OpenClaw: MCP server 'remote' (streamable-http) is unavailable: failed",
        "NeMo Fabric OpenClaw: MCP server 'local' (stdio) is unavailable: failed",
    ]
    args = status.await_args_list[0].args
    assert args[1:7] == (
        "gateway",
        "call",
        "tools.effective",
        "--json",
        "--port",
        "12345",
    )
    assert json.loads(args[8]) == {"sessionKey": "agent:reviewer:openai-user:runtime-1"}
    assert "secret-token" not in json.dumps([str(arg) for arg in args])
    assert status.await_args_list[0].kwargs["env"] == runtime._command_env


async def test_openclaw_http_failures_report_unavailable_mcp_servers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """OpenClaw HTTP failures report unavailable MCP servers."""
    runtime, _ = _mcp_status_runtime(
        tmp_path,
        monkeypatch,
        [{"notices": [_mcp_diagnostic("local", "spawn ENOENT")]}],
    )
    request = adapter.httpx.Request(
        "POST", "http://127.0.0.1:12345/v1/chat/completions"
    )
    monkeypatch.setattr(
        adapter,
        "_invoke_gateway",
        AsyncMock(
            side_effect=adapter.httpx.HTTPStatusError(
                "bad", request=request, response=adapter.httpx.Response(502)
            )
        ),
    )
    assert runtime._context is not None

    result = await runtime.invoke(AgentRunRequest(input="Hello."), runtime._context)

    assert result.status == adapter.contract.AgentRunStatus.FAILED
    assert result.output == {
        "mcp": {
            "unavailable_servers": [
                {"name": "local", "transport": "stdio", "reason": "failed"}
            ]
        }
    }


async def test_openclaw_mcp_status_failure_keeps_the_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    """OpenClaw MCP status failure keeps the result."""
    runtime, status = _mcp_status_runtime(tmp_path, monkeypatch, [])
    status.side_effect = adapter.lifecycle.LifecycleError(
        "openclaw_command_failed", "OpenClaw command failed"
    )
    assert runtime._context is not None

    first = await runtime.invoke(AgentRunRequest(input="One."), runtime._context)
    status.side_effect = None
    status.return_value = "not json"
    second = await runtime.invoke(AgentRunRequest(input="Two."), runtime._context)

    assert first.output == second.output == {"response": "OpenClaw response"}
    assert caplog.text.count("could not report MCP server availability") == 1


async def test_openclaw_skips_mcp_status_without_configured_servers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """OpenClaw skips MCP status without configured servers."""
    runtime, status = _mcp_status_runtime(tmp_path, monkeypatch, [])
    runtime._mcp_transports = {}
    assert runtime._context is not None

    result = await runtime.invoke(AgentRunRequest(input="Hello."), runtime._context)

    assert result.output == {"response": "OpenClaw response"}
    status.assert_not_awaited()


async def test_openclaw_mcp_status_cancellation_propagates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """OpenClaw MCP status cancellation propagates."""
    runtime, status = _mcp_status_runtime(tmp_path, monkeypatch, [])
    started = asyncio.Event()

    async def wait_forever(*_args, **_kwargs):
        started.set()
        await asyncio.Event().wait()

    status.side_effect = wait_forever
    assert runtime._context is not None
    task = asyncio.create_task(
        runtime.invoke(AgentRunRequest(input="Hello."), runtime._context)
    )
    await asyncio.wait_for(started.wait(), timeout=5)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.skipif(os.name == "nt", reason="process groups are POSIX-only")
async def test_status_command_timeout_stops_children_holding_the_pipes(tmp_path):
    """A stalled status command whose child keeps the pipes open stops promptly."""
    child_pid_file = tmp_path / "child.pid"
    script = tmp_path / "stall.py"
    script.write_text(
        "import pathlib, subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"pathlib.Path({str(child_pid_file)!r}).write_text(str(child.pid))\n"
        "time.sleep(60)\n"
    )
    command = tmp_path / "openclaw"
    command.write_text(f"#!/bin/sh\nexec {sys.executable} {script}\n")
    command.chmod(0o755)

    started = time.monotonic()
    with pytest.raises(adapter.lifecycle.LifecycleError) as error:
        await adapter._command_output(
            command,
            "gateway",
            "call",
            env=dict(os.environ),
            timeout=1,
            process_group=True,
        )

    assert error.value.code == "openclaw_command_timeout"
    assert time.monotonic() - started < 1 + adapter.COMMAND_REAP_SECONDS + 2
    child_pid = int(child_pid_file.read_text())
    stat = Path(f"/proc/{child_pid}/stat")
    for _ in range(50):
        try:
            os.kill(child_pid, 0)
            # On Linux the state follows the parenthesized command name; a
            # zombie ("Z") has exited. Without /proc, os.kill is the check.
            if stat.read_text().rsplit(")", 1)[1].split()[0] == "Z":
                break
        except ProcessLookupError:
            break
        except (FileNotFoundError, IndexError):
            if sys.platform.startswith("linux"):
                break  # the process exited while being checked
        time.sleep(0.05)
    else:
        os.kill(child_pid, signal.SIGKILL)
        pytest.fail("the status command's child kept running")


def _native_controls() -> dict[str, object]:
    return {
        "agents": {
            "defaults": {
                "heartbeat": {"every": "0m"},
                "compaction": {"memoryFlush": {"enabled": False}},
            }
        },
        "tools": {
            "exec": {"notifyOnExit": False},
            "toolSearch": False,
            "codeMode": False,
        },
        "plugins": {
            "entries": {"memory-core": {"config": {"dreaming": {"enabled": False}}}}
        },
    }


GENERATED_AGENT_DEFAULTS = {
    "workspace",
    "skipBootstrap",
    "model",
    "models",
    "contextInjection",
}


def _generated(config: AgentConfig, tmp_path: Path, **options: object):
    return adapter._openclaw_config(
        config,
        _context(tmp_path),
        base_dir=tmp_path,
        port=20_000,
        token_env="OPENCLAW_GATEWAY_TOKEN",
        **options,
    )


def _assert_native_controls_merged(generated: dict[str, object]):
    defaults = generated["agents"]["defaults"]
    assert set(defaults) == GENERATED_AGENT_DEFAULTS | {"heartbeat", "compaction"}
    assert defaults["heartbeat"] == {"every": "0m"}
    assert defaults["compaction"] == {"memoryFlush": {"enabled": False}}
    assert defaults["model"] == {"primary": "test/fabric-echo"}
    assert defaults["contextInjection"] == "never"
    assert generated["tools"] == {
        "allow": ["browser", "web_search"],
        "deny": ["exec"],
        "exec": {"notifyOnExit": False},
        "toolSearch": False,
        "codeMode": False,
    }
    assert generated["plugins"]["entries"]["memory-core"] == {
        "config": {"dreaming": {"enabled": False}}
    }


@pytest.mark.parametrize(
    "settings",
    [
        pytest.param({}, id="absent"),
        pytest.param({"agents": {}, "tools": {}, "plugins": {}}, id="empty-sections"),
        pytest.param(
            {
                "agents": {"defaults": {"heartbeat": {}, "compaction": {}}},
                "tools": {"exec": {}},
                "plugins": {"entries": {"memory-core": {"config": {}}}},
            },
            id="empty-leaves",
        ),
    ],
)
def test_openclaw_omitted_native_controls_keep_generated_config(
    mock_openclaw: Path, tmp_path: Path, settings: dict[str, object]
):
    baseline = _generated(_config(mock_openclaw), tmp_path)
    config = _config(mock_openclaw)
    config.harness.settings.update(settings)

    generated = _generated(config, tmp_path)

    assert generated == baseline
    assert set(generated["agents"]["defaults"]) == GENERATED_AGENT_DEFAULTS
    assert generated["tools"] == {"allow": ["browser", "web_search"], "deny": ["exec"]}
    assert "plugins" not in generated


@pytest.mark.parametrize(
    ("settings", "path", "value"),
    [
        pytest.param(
            {"agents": {"defaults": {"heartbeat": {"every": "1h30m"}}}},
            ("agents", "defaults", "heartbeat", "every"),
            "1h30m",
            id="heartbeat",
        ),
        pytest.param(
            {
                "agents": {
                    "defaults": {"compaction": {"memoryFlush": {"enabled": True}}}
                }
            },
            ("agents", "defaults", "compaction", "memoryFlush", "enabled"),
            True,
            id="memory-flush",
        ),
        pytest.param(
            {"tools": {"exec": {"notifyOnExit": True}}},
            ("tools", "exec", "notifyOnExit"),
            True,
            id="notify-on-exit",
        ),
        pytest.param(
            {"tools": {"toolSearch": True}},
            ("tools", "toolSearch"),
            True,
            id="tool-search",
        ),
        pytest.param(
            {"tools": {"codeMode": True}},
            ("tools", "codeMode"),
            True,
            id="code-mode",
        ),
        pytest.param(
            {
                "plugins": {
                    "entries": {
                        "memory-core": {"config": {"dreaming": {"enabled": True}}}
                    }
                }
            },
            ("plugins", "entries", "memory-core", "config", "dreaming", "enabled"),
            True,
            id="dreaming",
        ),
    ],
)
def test_openclaw_writes_only_authored_native_controls(
    mock_openclaw: Path,
    tmp_path: Path,
    settings: dict[str, object],
    path: tuple[str, ...],
    value: object,
):
    # The generated configuration plus exactly the authored leaf.
    expected = _generated(_config(mock_openclaw), tmp_path)
    parent = expected
    for key in path[:-1]:
        parent = parent.setdefault(key, {})
    parent[path[-1]] = value
    config = _config(mock_openclaw)
    config.harness.settings.update(settings)

    assert _generated(config, tmp_path) == expected


@pytest.mark.parametrize("service_mode", [False, True])
@pytest.mark.parametrize("shared_plugin", [False, True])
def test_openclaw_native_controls_merge_with_generated_config(
    mock_openclaw: Path, tmp_path: Path, service_mode: bool, shared_plugin: bool
):
    config = _config(mock_openclaw)
    config.harness.settings.update(_native_controls())
    if shared_plugin:
        config.harness.settings["interfaces"] = {"terminal": {}}

    generated = _generated(config, tmp_path, service_mode=service_mode)

    _assert_native_controls_merged(generated)
    if shared_plugin:
        assert set(generated["plugins"]["entries"]) == {
            "fabric-shared-config",
            "memory-core",
        }
        assert generated["plugins"]["load"]["paths"]
    else:
        assert generated["plugins"] == {
            "entries": {"memory-core": {"config": {"dreaming": {"enabled": False}}}}
        }


@pytest.mark.parametrize(
    ("settings", "field"),
    [
        pytest.param(
            {"tools": {"allow": ["exec"]}}, "harness.settings.tools.allow", id="tools"
        ),
        pytest.param(
            {"agents": {"defaults": {"workspace": "/tmp"}}},
            "harness.settings.agents.defaults.workspace",
            id="agent-defaults",
        ),
    ],
)
def test_openclaw_native_controls_never_replace_generated_keys(
    mock_openclaw: Path, tmp_path: Path, settings: dict[str, object], field: str
):
    # The descriptor rejects these keys; the writer also refuses to overwrite.
    config = _config(mock_openclaw)
    config.harness.settings.update(settings)

    with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
        _generated(config, tmp_path)

    assert caught.value.code == "openclaw_invalid_configuration"
    assert caught.value.metadata == {"field": field}


@pytest.mark.skipif(
    sys.platform == "win32", reason="Runs the mock openclaw script directly"
)
@pytest.mark.parametrize(
    "gateway",
    [
        "managed",
        pytest.param(
            "interfaces",
            marks=pytest.mark.skipif(
                os.name != "posix", reason="The native terminal requires POSIX"
            ),
        ),
        "prepared",
    ],
)
async def test_owned_gateways_receive_native_controls_before_startup(
    mock_openclaw: Path, tmp_path: Path, dashboard_port: int, gateway: str
):
    capture = tmp_path / "config.json"
    os.environ["FAKE_OPENCLAW_CAPTURE"] = str(capture)
    os.environ["FAKE_OPENCLAW_REQUEST"] = str(tmp_path / "request.json")
    config = _config(mock_openclaw)
    config.harness.settings.update(_native_controls())
    payload: dict[str, object] = {}
    if gateway == "interfaces":
        config.harness.settings["interfaces"] = {
            "dashboard": {"port": dashboard_port},
            "terminal": {},
        }
    connection_path = tmp_path / "private" / "service-connection.json"
    if gateway == "prepared":
        payload["service"] = {
            "operation": "prepare",
            "connection_path": str(connection_path),
        }
    runtime = adapter.OpenClawRuntime()
    await _start(runtime, tmp_path, config, **payload)
    try:
        # OpenClaw validates the generated file before the Gateway starts.
        generated = json.loads(capture.read_text(encoding="utf-8"))
        _assert_native_controls_merged(generated)
        if gateway == "interfaces":
            directory = _dashboard_token(tmp_path).parent
            assert (directory / "terminal.running").exists()
            assert json.loads((directory / "gateway.json").read_text())["http_url"] == (
                f"http://127.0.0.1:{dashboard_port}"
            )
            assert generated["gateway"]["controlUi"]["enabled"] is True
            assert "fabric-shared-config" in generated["plugins"]["entries"]
        if gateway == "prepared":
            session = adapter.OpenClawRuntime()
            context = _context(tmp_path, "runtime-a")
            await session.start(
                {
                    "config": config,
                    "runtime_context": context.to_mapping(),
                    "base_dir": str(tmp_path),
                    "service": {
                        "operation": "connect",
                        "service_id": "service-1",
                        "connection_path": str(connection_path),
                    },
                }
            )
            try:
                result = await session.invoke(
                    AgentRunRequest(input="Use the prepared Gateway."), context
                )
            finally:
                await session.stop()
            assert result.status == "succeeded"
    finally:
        await runtime.stop()


@pytest.mark.parametrize("section", ["agents", "tools", "plugins"])
async def test_openclaw_attach_rejects_native_controls(tmp_path: Path, section: str):
    os.environ["OPENCLAW_GATEWAY_TOKEN"] = "caller-owned-token"
    config = AgentConfig.from_mapping(
        {
            "harness": {"settings": {section: _native_controls()[section]}},
            "models": {"default": {"provider": "test", "model": "fabric-echo"}},
        }
    )
    runtime = adapter.OpenClawRuntime()

    with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
        await _start(
            runtime,
            tmp_path,
            config,
            service={
                "operation": "attach",
                "connection_path": str(tmp_path / "service-connection.json"),
                "reference": {
                    "adapter_id": "nvidia.fabric.openclaw",
                    "service_type": "openclaw_gateway",
                    "connection": {
                        "gateway_url": "http://127.0.0.1:9",
                        "gateway_token_env": "OPENCLAW_GATEWAY_TOKEN",
                    },
                },
            },
        )

    assert caught.value.code == "openclaw_attach_configuration_unverifiable"
    assert caught.value.metadata == {"fields": [f"harness.settings.{section}"]}
    assert not (tmp_path / "service-connection.json").exists()


@pytest.mark.skipif(os.name == "nt", reason="Mock Gateway is a POSIX executable")
@pytest.mark.parametrize("agent_id", ["default", "Reviewer", "agent_1-A"])
async def test_openclaw_retains_native_state_and_explicitly_resumes_the_selected_session(
    tmp_path: Path, mock_openclaw: Path, agent_id
):
    from nemo_fabric_adapter_contract.models import SessionRef

    home = tmp_path / "retained"
    home.mkdir()
    os.environ["FAKE_OPENCLAW_CAPTURE"] = str(tmp_path / "config.json")
    os.environ["FAKE_OPENCLAW_REQUEST"] = str(tmp_path / "request.json")
    os.environ["FAKE_OPENCLAW_MEMORY_REPLY"] = "1"
    config = _config(mock_openclaw)
    config.harness.settings["agent_id"] = agent_id

    async def start(runtime_id, reference=None):
        context = _context(tmp_path, runtime_id)
        context.state_dir = str(home)
        context.session_ref = reference
        runtime = adapter.OpenClawRuntime()
        try:
            output = await runtime.start(
                {
                    "config": config,
                    "runtime_context": context.to_mapping(),
                    "base_dir": str(tmp_path),
                }
            )
        except BaseException:
            await runtime.stop()
            raise
        return runtime, context, output

    first, context, output = await start("first-runtime")
    try:
        assert isinstance(output, dict)
        reference = SessionRef(
            schema_version="fabric.session_ref/v1alpha1",
            adapter_id="nvidia.fabric.openclaw",
            session_id=output["session_id"],
        )
        await first.invoke(
            AgentRunRequest(input="The private code is meadow-7483."), context
        )
        (home / "memory").mkdir()
        (home / "memory/fact.txt").write_text("retained memory")
        (home / "skills").mkdir()
        (home / "skills/installed.md").write_text("retained skill")
    finally:
        await first.stop()

    assert (home / "memory/fact.txt").read_text() == "retained memory"
    assert (home / "skills/installed.md").read_text() == "retained skill"
    fresh, context, fresh_output = await start("newer-runtime")
    try:
        assert fresh_output["session_id"] != reference.session_id
        result = await fresh.invoke(
            AgentRunRequest(input="A separate conversation."), context
        )
        assert "meadow-7483" not in result.output["response"]
    finally:
        await fresh.stop()

    resumed, context, resumed_output = await start("replacement-runtime", reference)
    try:
        assert resumed_output["session_id"] == reference.session_id
        result = await resumed.invoke(
            AgentRunRequest(input="Recall the private code."), context
        )
        assert "meadow-7483" in result.output["response"]
        sent = json.loads((tmp_path / "request.json").read_text())
        assert "meadow-7483" not in json.dumps(sent)
    finally:
        await resumed.stop()


@pytest.mark.skipif(os.name == "nt", reason="Mock Gateway is a POSIX executable")
async def test_openclaw_explicit_resume_rejects_a_missing_native_session(
    tmp_path: Path, mock_openclaw: Path
):
    from nemo_fabric_adapter_contract.models import SessionRef

    context = _context(tmp_path)
    context.state_dir = str(tmp_path / "retained")
    context.session_ref = SessionRef(
        schema_version="fabric.session_ref/v1alpha1",
        adapter_id="nvidia.fabric.openclaw",
        session_id="2b734539-7fa7-49d3-b07b-b6f06f83c90d",
    )
    os.environ["FAKE_OPENCLAW_CAPTURE"] = str(tmp_path / "config.json")
    os.environ["FAKE_OPENCLAW_REQUEST"] = str(tmp_path / "request.json")
    runtime = adapter.OpenClawRuntime()
    try:
        with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
            await runtime.start(
                {
                    "config": _config(mock_openclaw),
                    "runtime_context": context.to_mapping(),
                    "base_dir": str(tmp_path),
                }
            )
        assert caught.value.code == "session_not_found"
        assert not (tmp_path / "request.json").exists()
    finally:
        await runtime.stop()


@pytest.mark.skipif(os.name == "nt", reason="Mock Gateway is a POSIX executable")
async def test_openclaw_retained_service_resumes_after_connected_runtime_stops(
    tmp_path: Path, mock_openclaw: Path
):
    from nemo_fabric_adapter_contract.models import SessionRef

    os.environ["FAKE_OPENCLAW_CAPTURE"] = str(tmp_path / "config.json")
    os.environ["FAKE_OPENCLAW_REQUEST"] = str(tmp_path / "request.json")
    os.environ["FAKE_OPENCLAW_MEMORY_REPLY"] = "1"
    home = tmp_path / "retained"
    config = _config(mock_openclaw)
    context = _context(tmp_path, "service-host")
    context.state_dir = str(home)
    connection = tmp_path / "private/service.json"
    owner = adapter.OpenClawRuntime()
    await owner.start(
        {
            "config": config,
            "runtime_context": context.to_mapping(),
            "base_dir": str(tmp_path),
            "service": {"operation": "prepare", "connection_path": str(connection)},
        }
    )
    process = owner._process
    try:
        for number in range(2):
            context = _context(tmp_path, f"connected-{number}")
            context.state_dir = str(home)
            if number:
                context.session_ref = reference
            runtime = adapter.OpenClawRuntime()
            try:
                output = await runtime.start(
                    {
                        "config": config,
                        "runtime_context": context.to_mapping(),
                        "base_dir": str(tmp_path),
                        "service": {
                            "operation": "connect",
                            "connection_path": str(connection),
                        },
                    }
                )
                assert isinstance(output, dict)
                if not number:
                    reference = SessionRef(
                        schema_version="fabric.session_ref/v1alpha1",
                        adapter_id="nvidia.fabric.openclaw",
                        session_id=output["session_id"],
                    )
                else:
                    assert output["session_id"] == reference.session_id
                result = await runtime.invoke(
                    AgentRunRequest(
                        input="Remember copper-9182."
                        if not number
                        else "Recall the saved code."
                    ),
                    context,
                )
                assert "copper-9182" in result.output["response"]
            finally:
                await runtime.stop()
            assert process.returncode is None
    finally:
        await owner.stop()
    assert home.is_dir()


async def test_openclaw_retained_attach_fails_before_accessing_caller_owned_service(
    tmp_path: Path,
):
    context = _context(tmp_path)
    context.state_dir = str(tmp_path / "retained")
    runtime = adapter.OpenClawRuntime()
    config = AgentConfig.from_mapping(
        {"models": {"default": {"provider": "test", "model": "echo"}}}
    )
    with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
        await runtime.start(
            {
                "config": config,
                "runtime_context": context.to_mapping(),
                "base_dir": str(tmp_path),
                "service": {
                    "operation": "attach",
                    "connection_path": str(tmp_path / "caller.json"),
                    "reference": {},
                },
            }
        )
    assert caught.value.code == "session_resume_unsupported"
    assert not (tmp_path / "caller.json").exists()


@pytest.mark.parametrize("platform", ["posix", "nt"])
async def test_openclaw_failed_gateway_cleanup_retains_process_for_retry(
    monkeypatch, platform
):
    mock_os = MagicMock(spec=os, wraps=os)
    mock_os.name = platform
    monkeypatch.setattr(adapter, "os", mock_os)
    runtime = adapter.OpenClawRuntime()
    process = MagicMock(spec=asyncio.subprocess.Process)
    process.pid = 12345
    process.returncode = None
    process.wait = AsyncMock(return_value=0)
    process.terminate.side_effect = PermissionError("denied")
    runtime._process = process
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            adapter.common_utils,
            "signal_process_group",
            MagicMock(side_effect=PermissionError("denied")),
        )
        with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
            await runtime.stop()
    assert caught.value.code == "cleanup_unconfirmed"
    assert runtime._process is process
    if platform == "nt":
        process.terminate.assert_called_once_with()
    else:
        process.terminate.assert_not_called()
    process.returncode = 0
    await runtime.stop()
    assert runtime._process is None


@pytest.mark.skipif(os.name == "nt", reason="Uses POSIX inherited writer locks")
async def test_openclaw_gateway_holds_the_state_lock_after_controller_releases_it(
    tmp_path: Path, mock_openclaw: Path
):
    import fcntl

    os.environ["FAKE_OPENCLAW_CAPTURE"] = str(tmp_path / "config.json")
    os.environ["FAKE_OPENCLAW_REQUEST"] = str(tmp_path / "request.json")
    lock = (tmp_path / "state.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.environ["FABRIC_STATE_LOCK_FD"] = str(lock.fileno())
    runtime = adapter.OpenClawRuntime()
    context = _context(tmp_path)
    context.state_dir = str(tmp_path / "retained")
    try:
        await runtime.start(
            {
                "config": _config(mock_openclaw),
                "runtime_context": context.to_mapping(),
                "base_dir": str(tmp_path),
            }
        )
        lock.close()
        with (tmp_path / "state.lock").open("w") as competing:
            with pytest.raises(BlockingIOError):
                fcntl.flock(competing, fcntl.LOCK_EX | fcntl.LOCK_NB)
        await runtime.stop()
        with (tmp_path / "state.lock").open("w") as replacement:
            fcntl.flock(replacement, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        lock.close()
        await runtime.stop()


@pytest.mark.skipif(os.name == "nt", reason="Mock Gateway is a POSIX executable")
async def test_openclaw_uncertain_turn_warns_and_waits_for_explicit_acknowledgement(
    tmp_path: Path, mock_openclaw: Path
):
    from nemo_fabric_adapter_contract.models import SessionRef

    home = tmp_path / "retained"
    os.environ["FAKE_OPENCLAW_CAPTURE"] = str(tmp_path / "config.json")
    os.environ["FAKE_OPENCLAW_REQUEST"] = str(tmp_path / "request.json")
    context = _context(tmp_path, "first-runtime")
    context.state_dir = str(home)
    config = _config(mock_openclaw)
    first = adapter.OpenClawRuntime()
    output = await first.start(
        {
            "config": config,
            "runtime_context": context.to_mapping(),
            "base_dir": str(tmp_path),
        }
    )
    try:
        await first.invoke(AgentRunRequest(input="Create a ticket."), context)
    finally:
        await first.stop()
    reference = SessionRef(
        schema_version="fabric.session_ref/v1alpha1",
        adapter_id="nvidia.fabric.openclaw",
        session_id=output["session_id"],
    )
    path = home / "mock-native-sessions.json"
    native = json.loads(path.read_text())
    entry = next(iter(native.values()))
    entry["messages"].pop()
    entry["abortedLastRun"] = True
    path.write_text(json.dumps(native))
    marker = home / ".fabric-turns" / f"{reference.session_id}.json"
    marker.write_text(
        json.dumps(
            {
                "state": "running",
                "message_count": 0,
                "request_id": "interrupted-request",
            }
        )
    )
    previous_request = (tmp_path / "request.json").read_bytes()
    context = _context(tmp_path, "replacement")
    context.state_dir = str(home)
    context.session_ref = reference
    resumed = adapter.OpenClawRuntime()
    try:
        output = await resumed.start(
            {
                "config": config,
                "runtime_context": context.to_mapping(),
                "base_dir": str(tmp_path),
            }
        )
        assert output["session_id"] == reference.session_id
        assert output["recovery_warnings"][0]["code"] == "session_recovery_required"
        assert (
            output["recovery_warnings"][0]["metadata"]["request_id"]
            == "interrupted-request"
        )
        blocked = await resumed.invoke(AgentRunRequest(input="Continue."), context)
        assert blocked.error.code == "session_recovery_required"
        assert (tmp_path / "request.json").read_bytes() == previous_request
        result = await resumed.invoke(
            AgentRunRequest(
                input="The ticket exists. Continue using ticket 123.",
                acknowledge_recovery=True,
            ),
            context,
        )
        assert result.status == "succeeded"
    finally:
        await resumed.stop()


@pytest.mark.skipif(os.name == "nt", reason="Mock Gateway is a POSIX executable")
async def test_openclaw_retained_start_removes_stale_dashboard_token_when_disabled(
    tmp_path: Path, mock_openclaw: Path
):
    home = tmp_path / "retained"
    token = home / "interfaces/dashboard.token"
    token.parent.mkdir(parents=True)
    token.write_text("obsolete-test-token")
    os.environ["FAKE_OPENCLAW_CAPTURE"] = str(tmp_path / "config.json")
    os.environ["FAKE_OPENCLAW_REQUEST"] = str(tmp_path / "request.json")
    context = _context(tmp_path)
    context.state_dir = str(home)
    runtime = adapter.OpenClawRuntime()
    try:
        await runtime.start(
            {
                "config": _config(mock_openclaw),
                "runtime_context": context.to_mapping(),
                "base_dir": str(tmp_path),
            }
        )
        assert not token.exists()
    finally:
        await runtime.stop()


@pytest.mark.skipif(os.name == "nt", reason="Mock Gateway is a POSIX executable")
async def test_openclaw_does_not_claim_the_old_identity_after_a_native_session_reset(
    tmp_path: Path, mock_openclaw: Path
):
    home = tmp_path / "retained"
    os.environ["FAKE_OPENCLAW_CAPTURE"] = str(tmp_path / "config.json")
    os.environ["FAKE_OPENCLAW_REQUEST"] = str(tmp_path / "request.json")
    context = _context(tmp_path)
    context.state_dir = str(home)
    runtime = adapter.OpenClawRuntime()
    try:
        await runtime.start(
            {
                "config": _config(mock_openclaw),
                "runtime_context": context.to_mapping(),
                "base_dir": str(tmp_path),
            }
        )
        native_path = home / "mock-native-sessions.json"
        values = json.loads(native_path.read_text())
        entry = next(iter(values.values()))
        entry["sessionId"] = "4fbc1ff2-6b43-4618-b2b3-29d7b2a7ed38"
        native_path.write_text(json.dumps(values))
        # A second Gateway process sees the reset state after replacement.
        runtime._session_call = AsyncMock(
            side_effect=lambda method, _params: (
                {"session": entry}
                if method == "sessions.describe"
                else {"messages": []}
            )
        )
        runtime._native_session.call = runtime._session_call
        with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
            await runtime.invoke(
                AgentRunRequest(input="Use the original conversation."), context
            )
        assert caught.value.code == "session_identity_changed"
        assert not (tmp_path / "request.json").exists()
    finally:
        await runtime.stop()


@pytest.mark.skipif(os.name == "nt", reason="Uses POSIX descriptor inheritance")
@pytest.mark.parametrize("descriptor", ["1", "-1", "invalid", "99999999"])
async def test_openclaw_rejects_an_invalid_inherited_state_lock(
    tmp_path: Path, mock_openclaw: Path, descriptor: str
):
    os.environ["FAKE_OPENCLAW_CAPTURE"] = str(tmp_path / "config.json")
    context = _context(tmp_path)
    context.environment.env["FABRIC_STATE_LOCK_FD"] = descriptor
    runtime = adapter.OpenClawRuntime()
    try:
        with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
            await runtime.start(
                {
                    "config": _config(mock_openclaw),
                    "runtime_context": context.to_mapping(),
                    "base_dir": str(tmp_path),
                }
            )
        assert caught.value.code == "session_lock_invalid"
        assert runtime._process is None
    finally:
        await runtime.stop()


@pytest.mark.skipif(os.name == "nt", reason="Mock Gateway is a POSIX executable")
async def test_openclaw_retained_dashboard_token_rotates_and_remains_private(
    tmp_path: Path, mock_openclaw: Path, dashboard_port: int
):
    import stat

    os.environ["FAKE_OPENCLAW_CAPTURE"] = str(tmp_path / "config.json")
    context = _context(tmp_path)
    home = tmp_path / "retained"
    context.state_dir = str(home)
    token_path = home / "interfaces/dashboard.token"
    config = _dashboard_config(mock_openclaw, {"port": dashboard_port})
    tokens = []
    for _ in range(2):
        runtime = adapter.OpenClawRuntime()
        try:
            await runtime.start(
                {
                    "config": config,
                    "runtime_context": context.to_mapping(),
                    "base_dir": str(tmp_path),
                }
            )
            tokens.append(token_path.read_text())
            assert tokens[-1] == runtime._token
            if os.name == "posix":
                assert stat.S_IMODE(home.stat().st_mode) == 0o700
                assert stat.S_IMODE(token_path.stat().st_mode) == 0o600
                assert (
                    stat.S_IMODE(runtime._native_session.marker.stat().st_mode) == 0o600
                )
        finally:
            await runtime.stop()
        assert not token_path.exists()
    assert tokens[0] != tokens[1]


@pytest.mark.skipif(os.name == "nt", reason="Mock Gateway is a POSIX executable")
@pytest.mark.parametrize("failure", ["timeout", "http429", "http500", "unconfirmed"])
async def test_retained_invocation_failure_preserves_identity_and_recovery_gate(
    tmp_path: Path, mock_openclaw: Path, failure: str
):
    os.environ["FAKE_OPENCLAW_CAPTURE"] = str(tmp_path / "config.json")
    os.environ["FAKE_OPENCLAW_REQUEST"] = str(tmp_path / "request.json")
    context = _context(tmp_path)
    context.state_dir = str(tmp_path / "retained")
    runtime = adapter.OpenClawRuntime()
    try:
        started = await runtime.start(
            {
                "config": _config(mock_openclaw),
                "runtime_context": context.to_mapping(),
                "base_dir": str(tmp_path),
            }
        )
        await runtime.invoke(
            AgentRunRequest(input="An earlier completed turn."), context
        )
        (tmp_path / "request.json").unlink()
        original = runtime._invoke_monitored_gateway
        if failure == "timeout":
            error = TimeoutError()
        elif failure != "unconfirmed":
            response = httpx.Response(
                int(failure[4:]),
                request=httpx.Request("POST", "http://127.0.0.1/v1/chat/completions"),
            )
            error = httpx.HTTPStatusError(
                "Native invocation failed", request=response.request, response=response
            )
        runtime._invoke_monitored_gateway = (
            AsyncMock(return_value=("A response without durable completion.", None))
            if failure == "unconfirmed"
            else AsyncMock(side_effect=error)
        )
        failed = await runtime.invoke(
            AgentRunRequest(input="Create a ticket."), context
        )
        assert failed.session_id == started["session_id"]
        if failure == "unconfirmed":
            assert failed.status == "succeeded"
        else:
            assert failed.error.code == (
                "timeout" if failure == "timeout" else "openclaw_http_error"
            )
        marker = runtime._native_session.marker.read_bytes()
        blocked = await runtime.invoke(
            AgentRunRequest(input="Retry the ticket."), context
        )
        assert blocked.error.code == "session_recovery_required"
        assert blocked.session_id == started["session_id"]
        assert runtime._native_session.marker.read_bytes() == marker
        runtime._invoke_monitored_gateway.assert_awaited_once()
        assert not (tmp_path / "request.json").exists()
        runtime._invoke_monitored_gateway = original
        continued = await runtime.invoke(
            AgentRunRequest(
                input="I inspected the ticket service. Continue.",
                acknowledge_recovery=True,
            ),
            context,
        )
        assert continued.status == "succeeded"
        assert continued.session_id == started["session_id"]
    finally:
        await runtime.stop()


async def test_retained_mcp_status_targets_the_selected_conversation(
    tmp_path, monkeypatch
):
    runtime, status = _mcp_status_runtime(tmp_path, monkeypatch, [{"groups": []}])
    runtime._native_session = MagicMock(spec=adapter.sessions.NativeSession)
    runtime._native_session.key = "agent:reviewer:openai-user:retained-conversation"
    await runtime._mcp_output("Runtime-1", asyncio.get_running_loop().time() + 10)
    arguments = status.await_args.args
    params = json.loads(arguments[arguments.index("--params") + 1])
    assert params["sessionKey"] == runtime._native_session.key


@pytest.mark.skipif(sys.platform == "win32", reason="Runs the mock CLI directly")
async def test_openclaw_public_stop_waits_for_gateway_ownership_release(
    mock_openclaw: Path, tmp_path: Path
):
    stopped_path = tmp_path / "stopped.txt"
    pid_path = tmp_path / "gateway.pid"
    environment = {
        "FAKE_OPENCLAW_CAPTURE": str(tmp_path / "config.json"),
        "FAKE_OPENCLAW_REQUEST": str(tmp_path / "request.json"),
        "FAKE_OPENCLAW_PID": str(pid_path),
        "FAKE_OPENCLAW_STOPPED": str(stopped_path),
        # Cross both former deadlines (5 s adapter, 10 s native host). The
        # child writes its release marker only after handling SIGTERM.
        "FAKE_OPENCLAW_SHUTDOWN_DELAY": "12",
    }
    os.environ.update(environment)
    os.environ["ADAPTER_PYTHON"] = sys.executable
    config = FabricConfig.from_mapping(
        {
            "metadata": {"name": "openclaw-stop-test"},
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
            "environment": {"env_names": list(environment)},
        }
    )
    runtime = await Fabric().start_runtime(config, base_dir=tmp_path)
    try:
        result = await runtime.invoke(input="Complete a turn before shutdown.")
        assert result.status == "succeeded"
        gateway_pid = int(pid_path.read_text(encoding="utf-8"))
    finally:
        await runtime.stop()

    assert runtime.status is RuntimeStatus.STOPPED
    assert stopped_path.read_text(encoding="utf-8") == "stopped"
    with pytest.raises(ProcessLookupError):
        os.kill(gateway_pid, 0)


@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups")
async def test_openclaw_stop_allows_gateway_to_release_ownership(
    monkeypatch: pytest.MonkeyPatch,
):
    runtime = adapter.OpenClawRuntime()
    mock_process = MagicMock(spec=asyncio.subprocess.Process)
    mock_process.pid = 1234
    mock_process.returncode = None
    ownership_released = False
    wait_for = asyncio.wait_for

    # Compress the observed seven-second native cleanup and the production
    # grace period equally; this regression must not take seconds to run.
    async def scaled_wait_for(awaitable, timeout):
        return await wait_for(awaitable, timeout=timeout / 100)

    async def finish_shutdown():
        nonlocal ownership_released
        if mock_process.returncode is not None:
            return mock_process.returncode
        await asyncio.sleep(0.07)
        ownership_released = True
        mock_process.returncode = 0
        return 0

    mock_process.wait = AsyncMock(side_effect=finish_shutdown)
    runtime._process = mock_process
    signals = []

    def signal_group(pid, signum, *, fallback_to_pid=True):
        signals.append((pid, signum, fallback_to_pid))

    monkeypatch.setattr(adapter.asyncio, "wait_for", scaled_wait_for)
    monkeypatch.setattr(adapter.common_utils, "signal_process_group", signal_group)

    await runtime.stop()

    assert ownership_released, "Gateway must release ownership before escalation"
    assert mock_process.returncode == 0
    assert (1234, signal.SIGKILL, True) not in signals
