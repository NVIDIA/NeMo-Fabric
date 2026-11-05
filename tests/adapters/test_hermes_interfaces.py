# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Hermes's API server and dashboard served by a managed runtime."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import socket
import sqlite3
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
import requests
from nemo_fabric import Fabric
from nemo_fabric import FabricConfig
from nemo_fabric.errors import FabricConfigError
from nemo_fabric_adapter_contract.models import AgentConfig
from nemo_fabric_adapter_contract.models import RuntimeContext

pytestmark = pytest.mark.usefixtures("requires_hermes_agent")

if importlib.util.find_spec("run_agent") is not None:
    from nemo_fabric_adapters.hermes import adapter
    from nemo_fabric_adapters.hermes import configuration
    from nemo_fabric_adapters.hermes import interfaces

MODEL_KEY = "fabric-test-model-key"


def _free_port() -> int:
    with socket.create_server(("127.0.0.1", 0)) as server:
        return server.getsockname()[1]


def _agent_config(
    settings: dict[str, object], *, base_url: str | None = None
) -> AgentConfig:
    model: dict[str, object] = {
        "provider": "openai",
        "model": "fabric-echo",
        "api_key_env": "FABRIC_TEST_MODEL_KEY",
    }
    if base_url is not None:
        model["base_url"] = base_url
    return AgentConfig.from_mapping(
        {"harness": {"settings": settings}, "models": {"default": model}}
    )


def _payload(
    tmp_path: Path, settings: dict[str, object], *, base_url: str | None = None
) -> dict[str, object]:
    os.environ["FABRIC_TEST_MODEL_KEY"] = MODEL_KEY
    context = RuntimeContext.from_mapping(
        {
            "runtime_id": "runtime-interfaces",
            "invocation_id": "invocation-1",
            "request_id": "request-1",
            "environment": {
                "environment_id": "environment-1",
                "provider": "test",
                "control_location": "in_env_control",
                "ownership": "caller_owned",
                "workspace": str(tmp_path),
            },
            "artifacts": {},
        }
    )
    return {
        "base_dir": str(tmp_path),
        "runtime_context": context.to_mapping(),
        "config": _agent_config(settings, base_url=base_url),
    }


def _token_path(tmp_path: Path) -> Path:
    return (
        tmp_path
        / "artifacts/.fabric/hermes/runtimes/runtime-interfaces/interfaces/api.token"
    )


@pytest.fixture(name="web_dist")
def web_dist_fixture(tmp_path: Path) -> Path:
    dist = tmp_path / "web_dist"
    dist.mkdir()
    (dist / "index.html").write_text(
        "<html><head></head><body>fabric-test-dashboard</body></html>",
        encoding="utf-8",
    )
    os.environ["HERMES_WEB_DIST"] = str(dist)
    return dist


def _interface(name: str, **fields: object) -> dict[str, object]:
    return {"interfaces": {name: fields}}


# Settings that the descriptor's schema rejects during planning.
INVALID_SETTINGS = [
    pytest.param(_interface("terminal"), id="unsupported"),
    pytest.param(_interface("api", port=8642, key="x"), id="unknown-field"),
    pytest.param(_interface("api"), id="missing-port"),
    pytest.param(_interface("dashboard", port=65_536), id="port-too-high"),
    pytest.param({"startup_timeout_seconds": 75.5}, id="timeout-too-long"),
]


@pytest.mark.parametrize("settings", INVALID_SETTINGS)
def test_hermes_plan_rejects_invalid_settings(
    tmp_path: Path, settings: dict[str, object]
):
    config = FabricConfig.from_mapping(
        {
            "metadata": {"name": "hermes-interfaces"},
            "harness": {
                "adapter_id": "nvidia.fabric.hermes",
                "resolution": "preinstalled",
                "settings": settings,
            },
            "models": {"default": {"provider": "nvidia", "model": "test-model"}},
            "environment": {"provider": "local", "workspace": "."},
        }
    )

    with pytest.raises(FabricConfigError):
        Fabric().plan(config, base_dir=tmp_path)


def test_hermes_interfaces_read_the_model_endpoint_and_key_from_config():
    config = configuration.build_hermes_config(
        _agent_config({}, base_url="http://127.0.0.1:9/v1"),
        workspace=".",
        interfaces={"api": 8642},
    )

    assert config["model"] == {
        "provider": "custom",
        "default": "fabric-echo",
        "base_url": "http://127.0.0.1:9/v1",
        "api_key": "${FABRIC_TEST_MODEL_KEY}",
    }
    assert config["agent"]["cron_drain_timeout"] == 3
    assert config["agent"]["restart_drain_timeout"] == 0


async def test_hermes_gateway_agents_receive_fabric_behavior(
    tmp_path: Path, api_server: str
):
    port = _free_port()
    payload = _payload(
        tmp_path, _interface("api", port=port), base_url=f"{api_server}/v1"
    )
    config = payload["config"].to_mapping()
    config["models"]["default"].update(temperature=0.0, top_p=0.41, max_tokens=128)
    config["instructions"] = {
        "system": {"mode": "replace", "content": "FABRIC_PARITY_MARKER"}
    }
    config["tools"] = {"enabled": ["file"], "blocked": ["terminal"]}
    payload["config"] = AgentConfig.from_mapping(config)
    runtime = adapter.HermesRuntime()
    try:
        await runtime.start(payload)
        config_path = _token_path(tmp_path).with_name("agent-config.json")
        assert config_path.stat().st_mode & 0o777 == 0o600
        token = _token_path(tmp_path).read_text()
        for number in range(2):
            reply = await asyncio.to_thread(
                requests.post,
                f"http://127.0.0.1:{port}/v1/chat/completions",
                headers={"Authorization": f"Bearer {token}"},
                json={
                    "model": "hermes-agent",
                    "messages": [
                        {"role": "user", "content": f"fresh conversation {number}"}
                    ],
                },
                timeout=60,
            )
            assert reply.status_code == 200, reply.text
        captured = await asyncio.to_thread(
            requests.get, f"{api_server}/_requests", timeout=5
        )
        # Native session titling may make separate tool-free auxiliary calls.
        requests_seen = [request for request in captured.json() if request.get("tools")]
        assert len(requests_seen) == 2
        for request in requests_seen:
            assert "FABRIC_PARITY_MARKER" in json.dumps(request["messages"])
            assert request["temperature"] == 0.0
            assert request["top_p"] == 0.41
            assert request["max_tokens"] == 128
            assert {tool["function"]["name"] for tool in request["tools"]} == {
                "read_file",
                "write_file",
                "patch",
                "search_files",
            }
    finally:
        await runtime.stop()
    assert not config_path.exists()


async def test_api_turns_export_atif_without_resetting_native_conversation(
    tmp_path: Path, api_server: str
):
    port = _free_port()
    payload = _payload(
        tmp_path, _interface("api", port=port), base_url=f"{api_server}/v1"
    )
    payload["runtime_context"]["telemetry"] = {
        "relay_enabled": True,
        "metadata": {"telemetry_providers": ["relay"]},
    }
    relay_dir = tmp_path / "relay"
    wrapper = tmp_path / "relay.json"
    wrapper.write_text(
        json.dumps(
            {
                "relay": {
                    "config": {
                        "version": 4,
                        "atof": {
                            "enabled": True,
                            "sinks": [
                                {
                                    "type": "file",
                                    "output_directory": str(relay_dir / "atof"),
                                    "filename": "events.atof.jsonl",
                                }
                            ],
                        },
                        "atif": {
                            "enabled": True,
                            "output_directory": str(relay_dir / "atif"),
                        },
                    }
                }
            }
        )
    )
    os.environ["FABRIC_RELAY_CONFIG_PATH"] = str(wrapper)
    runtime = adapter.HermesRuntime()
    messages = [{"role": "user", "content": "first native turn"}]
    try:
        await runtime.start(payload)
        token = _token_path(tmp_path).read_text()
        for turn in range(2):
            reply = await asyncio.to_thread(
                requests.post,
                f"http://127.0.0.1:{port}/v1/chat/completions",
                headers={"Authorization": f"Bearer {token}"},
                json={"model": "hermes-agent", "messages": messages},
                timeout=60,
            )
            assert reply.status_code == 200, reply.text
            trajectories = list((relay_dir / "atif").rglob("*.json"))
            assert len(trajectories) == turn + 1
            assert all(json.loads(path.read_text())["steps"] for path in trajectories)
            messages.extend(
                [
                    reply.json()["choices"][0]["message"],
                    {"role": "user", "content": "second native turn"},
                ]
            )
        connection = sqlite3.connect(runtime._hermes_home / "state.db")
        try:
            sessions = connection.execute(
                "SELECT id FROM sessions WHERE source = 'api_server'"
            ).fetchall()
        finally:
            connection.close()
        assert len(sessions) == 1
    finally:
        await runtime.stop()


@pytest.mark.skipif(sys.platform != "linux", reason="Reads Linux procfs")
async def test_hermes_serves_api_and_dashboard_with_the_configured_model(
    tmp_path: Path, api_server: str, web_dist: Path
):
    api_port, dashboard_port = _free_port(), _free_port()
    runtime = adapter.HermesRuntime()
    await runtime.start(
        _payload(
            tmp_path,
            {
                "interfaces": {
                    "api": {"port": api_port},
                    "dashboard": {"port": dashboard_port},
                }
            },
            base_url=f"{api_server}/v1",
        )
    )
    token_path = _token_path(tmp_path)
    try:
        token = token_path.read_text(encoding="utf-8")
        token_mode = token_path.stat().st_mode & 0o777
        api = f"http://127.0.0.1:{api_port}"
        unauthorized = await asyncio.to_thread(
            requests.get, f"{api}/v1/models", timeout=10
        )
        reply = await asyncio.to_thread(
            requests.post,
            f"{api}/v1/chat/completions",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "model": "hermes-agent",
                "messages": [{"role": "user", "content": "ping"}],
            },
            timeout=60,
        )
        dashboard = await asyncio.to_thread(
            requests.get, f"http://127.0.0.1:{dashboard_port}/", timeout=10
        )
        dashboard_environment = next(
            Path(f"/proc/{child.process.pid}/environ").read_bytes()
            for child in runtime._interfaces._children
            if child.name == "dashboard"
        )
    finally:
        await runtime.stop()
    model_keys = requests.get(f"{api_server}/_chat_authorization_headers", timeout=5)

    assert token_mode == 0o600
    assert unauthorized.status_code == 401
    assert reply.status_code == 200, reply.text
    assert "latest=ping" in reply.json()["choices"][0]["message"]["content"]
    assert model_keys.json()
    assert set(model_keys.json()) == {f"Bearer {MODEL_KEY}"}
    assert "fabric-test-dashboard" in dashboard.text
    # On loopback, Hermes embeds its session token: the dashboard has no sign-in.
    assert "window.__HERMES_SESSION_TOKEN__" in dashboard.text
    assert b"API_SERVER_KEY=" not in dashboard_environment
    assert not token_path.exists()


async def test_hermes_stop_cuts_off_an_in_flight_api_request(
    tmp_path: Path, api_server: str
):
    requests.post(f"{api_server}/_scenario", json={"delay_seconds": 60}, timeout=5)
    api_port = _free_port()
    runtime = adapter.HermesRuntime()
    await runtime.start(
        _payload(
            tmp_path,
            {"interfaces": {"api": {"port": api_port}}},
            base_url=f"{api_server}/v1",
        )
    )
    token = _token_path(tmp_path).read_text(encoding="utf-8")
    in_flight = asyncio.create_task(
        asyncio.to_thread(
            requests.post,
            f"http://127.0.0.1:{api_port}/v1/chat/completions",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "model": "hermes-agent",
                "messages": [{"role": "user", "content": "slow"}],
            },
            timeout=90,
        )
    )
    try:
        deadline = time.monotonic() + 30
        while not requests.get(f"{api_server}/_requests", timeout=5).json():
            assert time.monotonic() < deadline, (
                "the API request never reached the model"
            )
            await asyncio.sleep(0.1)
    except BaseException:
        await runtime.stop()
        in_flight.cancel()
        raise

    started = time.monotonic()
    await runtime.stop()
    elapsed = time.monotonic() - started
    in_flight.cancel()

    assert elapsed < 8
    assert not _token_path(tmp_path).exists()


async def test_hermes_interfaces_fail_start_on_windows_before_any_child_starts(
    tmp_path: Path, api_server: str, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(interfaces, "POSIX", False)
    # start() stops whatever it started when it fails, so check the spawn
    # itself rather than the children left afterwards.
    spawn = AsyncMock()
    monkeypatch.setattr(interfaces.asyncio, "create_subprocess_exec", spawn)
    # Refusing before Hermes's setup keeps MCP discovery, which can take
    # minutes, from outlasting a caller's start deadline.
    write_config = MagicMock(wraps=configuration.write_hermes_config)
    monkeypatch.setattr(configuration, "write_hermes_config", write_config)
    runtime = adapter.HermesRuntime()

    with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
        await runtime.start(
            _payload(
                tmp_path,
                {"interfaces": {"api": {"port": _free_port()}}},
                base_url=f"{api_server}/v1",
            )
        )

    assert caught.value.code == "hermes_interfaces_unsupported_platform"
    spawn.assert_not_called()
    write_config.assert_not_called()
    assert not _token_path(tmp_path).exists()


async def test_hermes_interface_port_in_use_fails_start(
    tmp_path: Path, api_server: str
):
    runtime = adapter.HermesRuntime()
    with socket.create_server(("127.0.0.1", 0)) as server:
        port = server.getsockname()[1]
        with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
            await runtime.start(
                _payload(
                    tmp_path,
                    {"interfaces": {"api": {"port": port}}},
                    base_url=f"{api_server}/v1",
                )
            )

    assert caught.value.code == "hermes_interface_port_unavailable"
    assert caught.value.metadata == {
        "field": "harness.settings.interfaces.api.port",
        "port": port,
    }
    assert not _token_path(tmp_path).exists()


def test_hermes_disabled_interface_is_not_started():
    settings = {
        "interfaces": {
            "api": {"port": 8642, "enabled": False},
            "dashboard": {"port": 9119},
        }
    }

    assert interfaces.interface_ports(settings) == {"dashboard": 9119}


async def test_hermes_interface_startup_timeout_counts_from_the_start_of_start(
    tmp_path: Path, api_server: str, monkeypatch: pytest.MonkeyPatch
):
    write_hermes_config = configuration.write_hermes_config
    timeouts: list[float] = []

    def slow_write_hermes_config(*args, **kwargs):
        time.sleep(0.5)
        return write_hermes_config(*args, **kwargs)

    async def record_start(self, ports, *, timeout, **kwargs):
        timeouts.append(timeout)

    monkeypatch.setattr(configuration, "write_hermes_config", slow_write_hermes_config)
    monkeypatch.setattr(interfaces.InterfaceChildren, "start", record_start)
    runtime = adapter.HermesRuntime()
    await runtime.start(
        _payload(
            tmp_path,
            {"startup_timeout_seconds": 5, "interfaces": {"api": {"port": 8642}}},
            base_url=f"{api_server}/v1",
        )
    )
    await runtime.stop()

    assert len(timeouts) == 1
    assert timeouts[0] <= 4.5


async def test_hermes_interface_start_timeout_stops_the_started_child(
    tmp_path: Path, api_server: str, monkeypatch: pytest.MonkeyPatch
):
    create_subprocess_exec = asyncio.create_subprocess_exec
    processes: list[asyncio.subprocess.Process] = []

    async def record_process(*args, **kwargs):
        process = await create_subprocess_exec(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(interfaces.asyncio, "create_subprocess_exec", record_process)
    runtime = adapter.HermesRuntime()

    with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
        await runtime.start(
            _payload(
                tmp_path,
                {
                    "startup_timeout_seconds": 0.001,
                    "interfaces": {"api": {"port": _free_port()}},
                },
                base_url=f"{api_server}/v1",
            )
        )

    assert caught.value.code == "hermes_interface_start_timeout"
    assert len(processes) == 1
    assert processes[0].returncode is not None
    assert not runtime._interfaces._children
    assert not _token_path(tmp_path).exists()


@pytest.mark.skipif(sys.platform == "win32", reason="Uses POSIX process groups")
async def test_hermes_cancelled_interface_stop_still_kills_a_child_that_ignores_sigterm():
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "import signal, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "print('ready', flush=True)\n"
        "time.sleep(60)\n",
        stdout=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    try:
        assert await process.stdout.readline() == b"ready\n"
        children = interfaces.InterfaceChildren()
        children._children.append(interfaces._Child("api", 0, process))

        stopping = asyncio.create_task(children.stop())
        await asyncio.sleep(0.5)
        stopping.cancel()
        with pytest.raises(asyncio.CancelledError):
            await stopping

        assert await asyncio.wait_for(process.wait(), timeout=2) is not None
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()


async def test_hermes_dashboard_without_a_built_frontend_fails_start(
    tmp_path: Path, api_server: str
):
    os.environ["HERMES_WEB_DIST"] = str(tmp_path / "missing")
    runtime = adapter.HermesRuntime()

    with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
        await runtime.start(
            _payload(
                tmp_path,
                {"interfaces": {"dashboard": {"port": _free_port()}}},
                base_url=f"{api_server}/v1",
            )
        )

    assert caught.value.code == "hermes_dashboard_frontend_missing"
