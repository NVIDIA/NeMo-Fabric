# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Hermes's API server and dashboard served by a managed runtime."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import signal
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
from nemo_fabric import CheckRequest
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
    pytest.param({"sdk": {"enabled": False}}, id="no-execution-path"),
    pytest.param({"sdk": {"enabled": "false"}}, id="sdk-not-boolean"),
    pytest.param(
        {
            "sdk": {"enabled": False},
            "interfaces": {"terminal": {"port": 19119, "enabled": False}},
        },
        id="disabled-native-is-not-execution",
    ),
    pytest.param(_interface("unknown"), id="unsupported"),
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


async def test_managed_api_does_not_enable_inherited_messaging_platforms(
    tmp_path, api_server
):
    os.environ["TELEGRAM_BOT_TOKEN"] = "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghi"
    os.environ["WEBHOOK_ENABLED"] = "true"
    os.environ["HERMES_DISABLE_LAZY_INSTALLS"] = "1"
    port = _free_port()
    runtime = adapter.HermesRuntime()
    try:
        await runtime.start(
            _payload(
                tmp_path, _interface("api", port=port), base_url=f"{api_server}/v1"
            )
        )
        from gateway.config import Platform, load_gateway_config

        config = load_gateway_config()
        assert config.platforms[Platform.TELEGRAM].enabled is False
        assert config.platforms[Platform.WEBHOOK].enabled is False
        assert config.multiplex_profiles is False
        # Only the child receives API_SERVER_KEY; no other platform may connect.
        assert set(config.get_connected_platforms()) <= {Platform.API_SERVER}
        reply = await asyncio.to_thread(
            requests.get, f"http://127.0.0.1:{port}/health", timeout=5
        )
        assert reply.status_code == 200
        log = (runtime._hermes_home / "logs/fabric-interface-api.log").read_text()
        assert "attempting install" not in log
        assert os.environ["TELEGRAM_BOT_TOKEN"].startswith("123456789:")
    finally:
        await runtime.stop()


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
        captured = await asyncio.to_thread(
            requests.get, f"{api_server}/_requests", timeout=5
        )
        model_turns = [request for request in captured.json() if request.get("tools")]
        assert len(model_turns) == 2
        history = json.dumps(model_turns[-1]["messages"])
        assert "first native turn" in history
        assert "second native turn" in history
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


@pytest.mark.parametrize("kept", [False, True])
async def test_hermes_interfaces_fail_start_on_windows_before_any_child_starts(
    tmp_path: Path, api_server: str, monkeypatch: pytest.MonkeyPatch, kept: bool
):
    """Clear stale credentials and reject native interfaces before setup."""
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
    payload = _payload(
        tmp_path,
        {"interfaces": {"api": {"port": _free_port()}}},
        base_url=f"{api_server}/v1",
    )
    token_path = _token_path(tmp_path)
    if kept:
        state_dir = tmp_path / "artifacts/.fabric/state/nvidia.fabric.hermes/assistant"
        payload["runtime_context"]["state_dir"] = str(state_dir)
        token_path = state_dir / "interfaces/api.token"
    token_path.parent.mkdir(parents=True)
    token_path.write_text("token-from-a-crashed-runtime", encoding="utf-8")

    with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
        await runtime.start(payload)

    assert caught.value.code == "hermes_interfaces_unsupported_platform"
    spawn.assert_not_called()
    write_config.assert_not_called()
    assert not token_path.exists()


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


async def test_hermes_failed_interface_start_removes_a_token_left_in_kept_state(
    tmp_path: Path, api_server: str
):
    """Remove a retained API token even when an occupied port blocks startup."""
    state_dir = tmp_path / "artifacts/.fabric/state/nvidia.fabric.hermes/assistant"
    stale_token = state_dir / "interfaces" / "api.token"
    stale_token.parent.mkdir(parents=True)
    stale_token.write_text("token-from-a-crashed-runtime", encoding="utf-8")
    runtime = adapter.HermesRuntime()
    with socket.create_server(("127.0.0.1", 0)) as server:
        payload = _payload(
            tmp_path,
            {"interfaces": {"api": {"port": server.getsockname()[1]}}},
            base_url=f"{api_server}/v1",
        )
        payload["runtime_context"]["state_dir"] = str(state_dir)
        with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
            await runtime.start(payload)

    assert caught.value.code == "hermes_interface_port_unavailable"
    assert not stale_token.exists()


@pytest.mark.parametrize("kept", [False, True])
async def test_sdk_only_start_removes_stale_interface_files(
    tmp_path, api_server, monkeypatch, kept
):
    """Remove stale native attachment files while preserving unrelated data."""
    payload = _payload(tmp_path, {}, base_url=f"{api_server}/v1")
    home = _token_path(tmp_path).parent.parent
    if kept:
        home = tmp_path / "kept-state"
        payload["runtime_context"]["state_dir"] = str(home)
    interface_dir = home / "interfaces"
    interface_dir.mkdir(parents=True)
    stale_files = [
        interface_dir / name
        for name in ("api.token", "tui.json", "tui.token", "agent-config.json")
    ]
    for path in stale_files:
        path.write_text("stale interface data", encoding="utf-8")
    unrelated = interface_dir / "keep.txt"
    unrelated.write_text("retained data", encoding="utf-8")
    spawn = AsyncMock()
    monkeypatch.setattr(interfaces.asyncio, "create_subprocess_exec", spawn)

    runtime = adapter.HermesRuntime()
    try:
        await runtime.start(payload)
        assert all(not path.exists() for path in stale_files)
        assert unrelated.read_text(encoding="utf-8") == "retained data"
        spawn.assert_not_called()
    finally:
        await runtime.stop()


@pytest.mark.parametrize(
    "name", ["api.token", "tui.json", "tui.token", "agent-config.json"]
)
@pytest.mark.parametrize("failure", ["directory", "permission"])
async def test_start_reports_interface_cleanup_failures_before_hermes_setup(
    tmp_path, api_server, monkeypatch, name, failure
):
    """Return a stable error before setup when attachment cleanup fails."""
    payload = _payload(tmp_path, {}, base_url=f"{api_server}/v1")
    home = tmp_path / "kept-state"
    payload["runtime_context"]["state_dir"] = str(home)
    path = home / "interfaces" / name
    path.parent.mkdir(parents=True)
    if failure == "directory":
        path.mkdir()
    else:
        path.write_text("stale interface data", encoding="utf-8")
        unlink = Path.unlink

        def deny_unlink(candidate, *args, **kwargs):
            """Simulate a permission failure for the selected attachment."""
            if candidate == path:
                raise PermissionError("PRIVATE_UNLINK_DETAIL")
            return unlink(candidate, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", deny_unlink)
    write_config = MagicMock(wraps=configuration.write_hermes_config)
    monkeypatch.setattr(configuration, "write_hermes_config", write_config)
    spawn = AsyncMock()
    monkeypatch.setattr(interfaces.asyncio, "create_subprocess_exec", spawn)

    runtime = adapter.HermesRuntime()
    try:
        with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
            await runtime.start(payload)
    finally:
        await runtime.stop()

    assert caught.value.code == "hermes_interface_cleanup_failed"
    assert isinstance(caught.value.__cause__, OSError)
    assert "PRIVATE_UNLINK_DETAIL" not in str(caught.value)
    write_config.assert_not_called()
    spawn.assert_not_called()


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


@pytest.mark.parametrize("explicit_endpoint", [False, True])
async def test_native_openai_endpoint_does_not_use_ambient_fallback(
    tmp_path, api_server, monkeypatch, explicit_endpoint
):
    os.environ["OPENAI_BASE_URL"] = "http://ambient.invalid/v1"
    import run_agent

    # This test observes child environment construction, not provider login.
    monkeypatch.setattr(run_agent, "AIAgent", MagicMock())
    environments = []

    async def record_start(self, ports, *, environment, **kwargs):
        environments.append(environment)

    monkeypatch.setattr(interfaces.InterfaceChildren, "start", record_start)
    runtime = adapter.HermesRuntime()
    try:
        await runtime.start(
            _payload(
                tmp_path,
                {"interfaces": {"api": {"port": 8642}}},
                base_url=f"{api_server}/v1" if explicit_endpoint else None,
            )
        )
        assert environments[0]["OPENAI_API_KEY"] == MODEL_KEY
        if explicit_endpoint:
            assert runtime._hermes_config["model"]["base_url"] == f"{api_server}/v1"
        else:
            assert "OPENAI_BASE_URL" not in environments[0]
        assert os.environ["OPENAI_BASE_URL"] == "http://ambient.invalid/v1"
    finally:
        await runtime.stop()


@pytest.mark.parametrize("failed_attribute", ["_token_path", "_agent_config_path"])
async def test_private_file_removal_failure_still_stops_children_and_retries(
    tmp_path, monkeypatch, caplog, failed_attribute
):
    children = interfaces.InterfaceChildren()
    for attribute in ("_token_path", "_agent_config_path"):
        path = tmp_path / attribute
        path.write_text("private")
        setattr(children, attribute, path)
    failed_path = getattr(children, failed_attribute)
    process = MagicMock(pid=1234, returncode=None, wait=AsyncMock(return_value=0))
    children._children.append(interfaces._Child("api", 0, process))
    signal_group = MagicMock()
    monkeypatch.setattr(interfaces.common_utils, "signal_process_group", signal_group)
    unlink = Path.unlink

    def deny_private_file(path, *args, **kwargs):
        if path == failed_path:
            raise PermissionError("test removal denied")
        return unlink(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", deny_private_file)
        await children.stop()

    assert process.wait.await_count == 1
    assert signal_group.call_count == 2
    assert not children._children
    assert getattr(children, failed_attribute) == failed_path
    assert "Could not remove Hermes interface file" in caplog.text
    await children.stop()
    assert not failed_path.exists()
    assert children._token_path is None
    assert children._agent_config_path is None


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


@pytest.mark.parametrize("sdk_enabled", [True, False])
async def test_sdk_selection_preserves_eager_default(
    tmp_path, monkeypatch, sdk_enabled
):
    import run_agent
    import hermes_state

    agent = MagicMock()
    database = MagicMock()
    monkeypatch.setattr(run_agent, "AIAgent", agent)
    monkeypatch.setattr(hermes_state, "SessionDB", database)
    monkeypatch.setattr(interfaces.InterfaceChildren, "start", AsyncMock())
    settings = {"interfaces": {"api": {"port": 18642}}}
    if not sdk_enabled:
        settings["sdk"] = {"enabled": False}
    runtime = adapter.HermesRuntime()
    try:
        await runtime.start(_payload(tmp_path, settings))
        assert agent.call_count == int(sdk_enabled)
        assert database.call_count == int(sdk_enabled)
        if not sdk_enabled:
            result = await runtime.invoke(MagicMock(), MagicMock())
            assert result.status == "failed"
            assert result.error.code == "hermes_sdk_disabled"
            agent.assert_not_called()
    finally:
        await runtime.stop()


async def test_sdk_disabled_requires_an_enabled_native_path(tmp_path):
    runtime = adapter.HermesRuntime()
    with pytest.raises(adapter.lifecycle.LifecycleError) as error:
        await runtime.start(
            _payload(
                tmp_path,
                {
                    "sdk": {"enabled": False},
                    "interfaces": {"api": {"port": 18642, "enabled": False}},
                },
            )
        )
    assert error.value.code == "hermes_no_execution_path"


@pytest.mark.parametrize("level", ["live", "active", "ready"])
async def test_sdk_disabled_health_observes_native_backend(
    tmp_path, monkeypatch, level
):
    monkeypatch.setattr(interfaces.InterfaceChildren, "start", AsyncMock())
    probe = AsyncMock(return_value=None)
    monkeypatch.setattr(interfaces.InterfaceChildren, "check", probe)
    runtime = adapter.HermesRuntime()
    try:
        await runtime.start(
            _payload(
                tmp_path,
                {"sdk": {"enabled": False}, "interfaces": {"api": {"port": 18642}}},
            )
        )
        result = await runtime.check(level)
        assert result["outcome"] == "passed"
        assert result["activity"] == "unknown"
        assert runtime._agent is None
        probe.assert_awaited_once_with(level)
        probe.return_value = "agent_exited"
        assert (await runtime.check(level))["reason_code"] == "agent_exited"
    finally:
        await runtime.stop()


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize(
    "sdk_enabled,kill_api",
    [
        (False, False),
        (True, False),
        pytest.param(
            True,
            True,
            marks=pytest.mark.skipif(
                sys.platform != "linux", reason="owned child lookup uses /proc"
            ),
            id="sdk-survives-api-exit",
        ),
    ],
)
async def test_sdk_invocation_preserves_native_runtime_health(
    tmp_path, api_server, streaming, sdk_enabled, kill_api
):
    os.environ["ADAPTER_PYTHON"] = sys.executable
    os.environ["FABRIC_TEST_MODEL_KEY"] = MODEL_KEY
    port = _free_port()
    config = FabricConfig.from_mapping(
        {
            "metadata": {"name": "native-only-health"},
            "harness": {
                "adapter_id": "nvidia.fabric.hermes",
                "settings": {
                    "sdk": {"enabled": sdk_enabled},
                    "interfaces": {"api": {"port": port}},
                },
            },
            "models": {
                "default": {
                    "provider": "openai",
                    "model": "fabric-echo",
                    "base_url": f"{api_server}/v1",
                    "api_key_env": "FABRIC_TEST_MODEL_KEY",
                }
            },
            "tools": {"enabled": []},
            "environment": {
                "provider": "local",
                "workspace": str(tmp_path),
                "artifacts": str(tmp_path / "artifacts"),
            },
        }
    )
    if streaming:
        config.enable_relay(output_dir=str(tmp_path / "relay"))
    fabric = Fabric()
    environment = await fabric.prepare_environment(config, base_dir=tmp_path)
    try:
        async with await fabric.start_runtime_in(
            config, environment, base_dir=tmp_path, streaming=streaming
        ) as runtime:
            for _ in range(2):
                if streaming:
                    stream = runtime.invoke_stream(input="Check the SDK path.")
                    async for _ in stream:
                        pass
                    result = await stream.result()
                else:
                    result = await runtime.invoke(input="Check the SDK path.")
                if sdk_enabled:
                    assert result.status == "succeeded", result.to_mapping()
                    assert result.error is None
                else:
                    assert result.status == "failed"
                    assert result.error.code == "hermes_sdk_disabled"
                    assert result.error.retryable is False
                for level in ("live", "active", "ready"):
                    report = await fabric.check(
                        environment=environment,
                        request=CheckRequest(
                            scope="runtime",
                            runtime_id=runtime.runtime_id,
                            level=level,
                        ),
                    )
                    assert report.outcome == "passed", json.dumps(report.to_mapping())
                    assert report.runtime["activity"] == (
                        "idle" if sdk_enabled else "unknown"
                    )
            token = (
                tmp_path
                / "artifacts/.fabric/hermes/runtimes"
                / runtime.runtime_id
                / "interfaces/api.token"
            ).read_text()
            response = await asyncio.to_thread(
                requests.post,
                f"http://127.0.0.1:{port}/v1/chat/completions",
                headers={"Authorization": f"Bearer {token}"},
                json={
                    "model": "hermes-agent",
                    "messages": [{"role": "user", "content": "native still works"}],
                },
                timeout=30,
            )
            assert response.status_code == 200, response.text
            assert response.json()["choices"][0]["message"]["content"]
            if kill_api:
                host_pid = result.metadata["host_pid"]
                child_pids = (
                    Path(f"/proc/{host_pid}/task/{host_pid}/children")
                    .read_text()
                    .split()
                )
                api_children = []
                for pid in child_pids:
                    arguments = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
                    if b"nemo_fabric_adapters.hermes.native" in arguments:
                        assert any(
                            runtime.runtime_id.encode() in argument
                            for argument in arguments
                        )
                        api_children.append(int(pid))
                assert len(api_children) == 1
                os.kill(api_children[0], signal.SIGKILL)
                async with asyncio.timeout(5):
                    while not interfaces.local_interfaces.port_available(port):
                        await asyncio.sleep(0.01)
                for level in ("live", "active", "ready"):
                    report = await fabric.check(
                        environment=environment,
                        request=CheckRequest(
                            scope="runtime", runtime_id=runtime.runtime_id, level=level
                        ),
                    )
                    assert report.outcome == "passed", report.to_mapping()
                    assert report.runtime == {"state": "active", "activity": "idle"}
                result = await runtime.invoke(input="SDK after interface failure.")
                assert result.status == "succeeded", result.to_mapping()
                assert result.metadata["host_pid"] == host_pid
                captured = await asyncio.to_thread(
                    requests.get, f"{api_server}/_requests", timeout=5
                )
                assert any(
                    "Check the SDK path." in json.dumps(request["messages"])
                    and "SDK after interface failure."
                    in json.dumps(request["messages"])
                    for request in captured.json()
                )
    finally:
        await fabric.release_environment(environment)


@pytest.mark.parametrize(
    "settings, expected",
    [
        ({"terminal": {"port": 19119}}, {"terminal": 19119}),
        (
            {"terminal": {"port": 19119}, "dashboard": {"port": 19119}},
            {"dashboard": 19119},
        ),
        (
            {
                "terminal": {"port": 19119, "enabled": False},
                "dashboard": {"port": 19119},
            },
            {"dashboard": 19119},
        ),
    ],
)
def test_tui_clients_share_one_backend(settings, expected):
    assert interfaces.interface_ports({"interfaces": settings}) == expected


@pytest.mark.parametrize(
    "settings, code",
    [
        (
            {"terminal": {"port": 19119}, "dashboard": {"port": 19120}},
            "hermes_tui_port_conflict",
        ),
        (
            {"terminal": {"port": 19119}, "api": {"port": 19119}},
            "hermes_interface_port_conflict",
        ),
    ],
)
def test_tui_backend_port_conflicts_fail(settings, code):
    with pytest.raises(adapter.lifecycle.LifecycleError) as error:
        interfaces.interface_ports({"interfaces": settings})
    assert error.value.code == code


@pytest.mark.parametrize("dashboard_enabled", [True, False])
async def test_terminal_backend_authentication_and_cleanup(
    tmp_path, web_dist, dashboard_enabled
):
    from nemo_fabric_adapters.hermes import terminal

    port = _free_port()
    settings = {"sdk": {"enabled": False}, "interfaces": {"terminal": {"port": port}}}
    if dashboard_enabled:
        settings["interfaces"]["dashboard"] = {"port": port}
    else:
        os.environ["HERMES_WEB_DIST"] = str(tmp_path / "missing-web-assets")
    runtime = adapter.HermesRuntime()
    connection = _token_path(tmp_path).with_name("tui.json")
    try:
        await runtime.start(_payload(tmp_path, settings))
        assert runtime._agent is None
        assert len(runtime._interfaces._children) == 1
        assert connection.stat().st_mode & 0o777 == 0o600
        token = connection.with_name("tui.token").read_text()
        assert token not in connection.read_text()
        assert await interfaces.tui_ready(port, token)
        assert not await interfaces.tui_ready(port, "wrong-token")
        environment = await asyncio.to_thread(
            terminal.attachment_environment, connection
        )
        assert environment["HERMES_TUI_GATEWAY_URL"].startswith(
            f"ws://127.0.0.1:{port}/api/ws?"
        )
        assert (await runtime.check("ready"))["outcome"] == "passed"
        assert (await runtime.check("ready"))["activity"] == "unknown"
    finally:
        await runtime.stop()
    assert not connection.exists()
    assert not connection.with_name("tui.token").exists()
    assert interfaces.local_interfaces.port_available(port)


@pytest.mark.parametrize("dashboard_enabled", [False, True])
async def test_tui_turns_export_atif_while_client_stays_connected(
    tmp_path, api_server, web_dist, dashboard_enabled
):
    import aiohttp

    port = _free_port()
    settings = {"sdk": {"enabled": False}, "interfaces": {"terminal": {"port": port}}}
    if dashboard_enabled:
        settings["interfaces"]["dashboard"] = {"port": port}
    payload = _payload(tmp_path, settings, base_url=f"{api_server}/v1")
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
    try:
        await runtime.start(payload)
        token = _token_path(tmp_path).with_name("tui.token").read_text()
        async with aiohttp.ClientSession() as client:
            async with client.ws_connect(
                f"http://127.0.0.1:{port}/api/ws", params={"token": token}
            ) as ws:
                await ws.send_json(
                    {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "session.create",
                        "params": {"cwd": str(tmp_path)},
                    }
                )
                while True:
                    frame = await ws.receive_json(timeout=30)
                    if frame.get("id") == 1:
                        assert "result" in frame, frame
                        session_id = frame["result"]["session_id"]
                        break
                for turn in range(2):
                    await ws.send_json(
                        {
                            "jsonrpc": "2.0",
                            "id": turn + 2,
                            "method": "prompt.submit",
                            "params": {
                                "session_id": session_id,
                                "text": f"native TUI turn {turn}",
                            },
                        }
                    )
                    while True:
                        frame = await ws.receive_json(timeout=60)
                        if frame.get("id") == turn + 2:
                            assert "result" in frame, frame
                        if (
                            frame.get("method") == "event"
                            and frame["params"]["type"] == "message.complete"
                        ):
                            break
                    for _ in range(50):
                        trajectories = list((relay_dir / "atif").rglob("*.json"))
                        if len(trajectories) == turn + 1:
                            break
                        await asyncio.sleep(0.1)
                    assert len(trajectories) == turn + 1
                    assert all(
                        json.loads(path.read_text())["steps"] for path in trajectories
                    )
                    assert not ws.closed
        captured = await asyncio.to_thread(
            requests.get, f"{api_server}/_requests", timeout=5
        )
        model_turns = [request for request in captured.json() if request.get("tools")]
        history = json.dumps(model_turns[-1]["messages"])
        assert "native TUI turn 0" in history
        assert "native TUI turn 1" in history
        assert runtime._agent is None
    finally:
        await runtime.stop()


async def test_tui_readiness_times_out_after_websocket_upgrade():
    from aiohttp import web

    async def silent_backend(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for _ in ws:
            pass
        return ws

    app = web.Application()
    app.router.add_get("/api/ws", silent_backend)
    runner = web.AppRunner(app)
    await runner.setup()
    port = _free_port()
    try:
        await web.TCPSite(runner, "127.0.0.1", port).start()
        started = time.monotonic()
        assert not await asyncio.wait_for(
            interfaces.tui_ready(port, "test-token"), timeout=4
        )
        assert time.monotonic() - started < 3
    finally:
        await runner.cleanup()


async def test_four_paths_keep_conversations_separate_and_restart_cleanly(
    tmp_path, api_server, web_dist
):
    import aiohttp
    from nemo_fabric_adapter_contract.models import AgentRunRequest

    api_port, tui_port = _free_port(), _free_port()
    payload = _payload(
        tmp_path,
        {
            "interfaces": {
                "api": {"port": api_port},
                "dashboard": {"port": tui_port},
                "terminal": {"port": tui_port},
            }
        },
        base_url=f"{api_server}/v1",
    )
    runtime = adapter.HermesRuntime()
    labels = [
        "sdk-concurrent-marker",
        "api-concurrent-marker",
        "browser-concurrent-marker",
        "terminal-concurrent-marker",
    ]

    async def tui_turn(client, token, label):
        async with client.ws_connect(
            f"http://127.0.0.1:{tui_port}/api/ws", params={"token": token}
        ) as ws:
            await ws.send_json(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "session.create",
                    "params": {"cwd": str(tmp_path)},
                }
            )
            while True:
                frame = await ws.receive_json(timeout=30)
                if frame.get("id") == 1:
                    session_id = frame["result"]["session_id"]
                    break
            await ws.send_json(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "prompt.submit",
                    "params": {"session_id": session_id, "text": label},
                }
            )
            while True:
                frame = await ws.receive_json(timeout=60)
                if (
                    frame.get("method") == "event"
                    and frame["params"]["type"] == "message.complete"
                ):
                    return session_id

    try:
        await runtime.start(payload)
        token = _token_path(tmp_path).read_text()
        tui_token = _token_path(tmp_path).with_name("tui.token").read_text()
        assert len(runtime._interfaces._children) == 2
        async with aiohttp.ClientSession() as client:

            async def api_turn():
                async with client.post(
                    f"http://127.0.0.1:{api_port}/v1/chat/completions",
                    headers={"Authorization": f"Bearer {token}"},
                    json={
                        "model": "hermes-agent",
                        "messages": [{"role": "user", "content": labels[1]}],
                    },
                ) as response:
                    assert response.status == 200
                    await response.read()

            results = await asyncio.gather(
                runtime.invoke(
                    AgentRunRequest.from_mapping({"input": labels[0]}),
                    RuntimeContext.from_mapping(payload["runtime_context"]),
                ),
                api_turn(),
                tui_turn(client, tui_token, labels[2]),
                tui_turn(client, tui_token, labels[3]),
            )
            assert results[2] != results[3]
            async with client.get(f"{api_server}/_requests") as response:
                captured = await response.json()
        turns = [request for request in captured if request.get("tools")]
        assert len(turns) == 4
        for request in turns:
            history = json.dumps(request["messages"])
            assert sum(label in history for label in labels) == 1
        await runtime.stop()
        assert interfaces.local_interfaces.port_available(api_port)
        assert interfaces.local_interfaces.port_available(tui_port)
        await runtime.start(payload)
        assert (await runtime.check("ready"))["outcome"] == "passed"
    finally:
        await runtime.stop()


def _native_controls(tool_search: object = "off") -> dict[str, object]:
    return {
        "auxiliary": {
            "title_generation": {"enabled": False},
            "background_review": {"enabled": False},
        },
        "tools": {"tool_search": {"enabled": tool_search}},
    }


def _hermes_native_controls() -> dict[str, object]:
    """Read the controls through Hermes's own loaders for the current home."""
    from hermes_cli.config import load_config
    from tools.tool_search import load_config as load_tool_search_config

    auxiliary = load_config()["auxiliary"]
    return {
        "title_generation": auxiliary["title_generation"]["enabled"],
        "background_review": auxiliary["background_review"]["enabled"],
        "tool_search": load_tool_search_config().enabled,
    }


@pytest.mark.parametrize(
    ("settings", "expected"),
    [
        pytest.param(
            {},
            {
                "title_generation": True,
                "background_review": True,
                "tool_search": "auto",
            },
            id="omitted-keeps-hermes-defaults",
        ),
        *[
            pytest.param(
                _native_controls(tool_search),
                {
                    "title_generation": False,
                    "background_review": False,
                    "tool_search": expected,
                },
                id=f"tool-search-{tool_search}",
            )
            for tool_search, expected in (
                (True, "on"),
                (False, "off"),
                ("on", "on"),
                ("off", "off"),
                ("auto", "auto"),
            )
        ],
    ],
)
async def test_sdk_agent_reads_native_controls_at_initialization(
    tmp_path, monkeypatch, settings, expected
):
    import hermes_state
    import run_agent

    observed: dict[str, object] = {}

    def construct(*_args: object, **_kwargs: object) -> MagicMock:
        observed.update(_hermes_native_controls())
        return MagicMock()

    monkeypatch.setattr(run_agent, "AIAgent", MagicMock(side_effect=construct))
    monkeypatch.setattr(hermes_state, "SessionDB", MagicMock())
    runtime = adapter.HermesRuntime()
    try:
        await runtime.start(_payload(tmp_path, settings))
        assert os.environ["HERMES_HOME"] == str(runtime._hermes_home)
    finally:
        await runtime.stop()

    assert observed == expected


@pytest.mark.parametrize(
    ("interface", "command"),
    [("api", "gateway"), ("dashboard", "dashboard"), ("terminal", "serve")],
)
@pytest.mark.usefixtures("web_dist")
async def test_native_interfaces_read_native_controls_before_startup(
    tmp_path, monkeypatch, interface, command
):
    from nemo_fabric_adapters.hermes import terminal

    spawned: list[tuple[tuple[str, ...], str, dict[str, object]]] = []

    async def spawn(*arguments: str, env: dict[str, str], **_options: object):
        # A child process reads config.yaml from its HERMES_HOME at startup.
        os.environ["HERMES_HOME"] = env["HERMES_HOME"]
        spawned.append((arguments, env["HERMES_HOME"], _hermes_native_controls()))
        return MagicMock(returncode=0)

    monkeypatch.setattr(interfaces.asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(interfaces, "_wait_healthy", AsyncMock())
    monkeypatch.setattr(terminal, "tui_ready", AsyncMock(return_value=True))
    settings = {
        **_native_controls(False),
        "sdk": {"enabled": False},
        **_interface(interface, port=_free_port()),
    }
    runtime = adapter.HermesRuntime()
    try:
        await runtime.start(_payload(tmp_path, settings))
        home = str(runtime._hermes_home)
        if interface == "terminal":
            # The terminal renderer attaches to the backend that owns this home.
            connection = runtime._hermes_home / "interfaces" / "tui.json"
            attachment = await asyncio.to_thread(
                terminal.attachment_environment, connection
            )
            assert attachment["HERMES_HOME"] == home
    finally:
        await runtime.stop()

    [(arguments, child_home, observed)] = spawned
    assert arguments[1:3] == ("-m", "nemo_fabric_adapters.hermes.native")
    assert arguments[5] == command
    assert child_home == home
    assert observed == {
        "title_generation": False,
        "background_review": False,
        "tool_search": "off",
    }


@pytest.fixture(name="interface_processes")
def interface_processes_fixture(tmp_path: Path, monkeypatch):
    import json
    import signal
    import textwrap

    import psutil

    script = tmp_path / "interface_process.py"
    script.write_text(
        textwrap.dedent(
            """\
            import fcntl
            import json
            import os
            from pathlib import Path
            import signal
            import subprocess
            import sys
            import time
            import psutil

            home = Path(os.environ["HERMES_HOME"])
            mode = sys.argv[1]
            with (home / "processes.jsonl").open("a") as stream:
                stream.write(json.dumps({"pid": os.getpid(), "parent": os.getppid(), "mode": mode, "created": psutil.Process().create_time()}) + "\\n")
            stopping = False
            def stop(signum, frame):
                global stopping
                stopping = True
            signal.signal(signal.SIGTERM, stop)
            signal.signal(signal.SIGHUP, stop)

            def spawn(child):
                if child == "daemon":
                    from hermes_cli.gateway_runtime_start import spawn_unmanaged_gateway
                    original = subprocess.Popen
                    def detached(command, **kwargs):
                        if "hermes_cli.main" in command and "gateway" in command and "run" in command:
                            (home / "bootstrap-command.json").write_text(json.dumps(command))
                            return original([sys.executable, __file__, child], **kwargs)
                        return original(command, **kwargs)
                    subprocess.Popen = detached
                    try:
                        return spawn_unmanaged_gateway(
                            home, deadline=time.monotonic() + 5, idle_exit=True
                        )
                    finally:
                        subprocess.Popen = original
                return subprocess.Popen(
                    [sys.executable, __file__, child],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    close_fds=True,
                    start_new_session=True,
                )

            if mode == "api":
                lock = (home / "gateway.lock").open("a")
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    sys.exit(17)
                (home / "api-live").touch()
                (home / "api.ready").touch()
                while not stopping:
                    time.sleep(0.01)
                (home / "api-live").unlink(missing_ok=True)
            elif mode == "dashboard":
                chat = spawn("chat")
                while not (home / "chat.ready").exists():
                    time.sleep(0.01)
                (home / "dashboard.ready").touch()
                while not stopping:
                    time.sleep(0.01)
                # Native dashboard shutdown reaches retained terminals after other cleanup.
                time.sleep(0.25)
                if chat.poll() is None:
                    chat.terminate()
                    chat.wait(timeout=5)
            elif mode == "chat":
                if os.environ.get("RACE_IN_FLIGHT") == "1":
                    spawn("bootstrap")
                (home / "chat.ready").touch()
                while not stopping and (home / "api-live").exists():
                    time.sleep(0.01)
                if not stopping:
                    spawn("daemon")
            elif mode == "bootstrap":
                (home / "bootstrap.ready").touch()
                while not stopping and not (home / "release-bootstrap").exists():
                    time.sleep(0.01)
                if not stopping:
                    spawn("daemon")
                    os._exit(0)
            elif mode == "foreign":
                (home / "foreign.ready").touch()
                while not stopping:
                    time.sleep(0.01)
            elif mode == "daemon":
                lock = (home / "gateway.lock").open("a")
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    # An in-flight recovery may have decided to launch before stop began.
                    pass
                (home / "daemon.ready").touch()
                while not stopping:
                    time.sleep(0.01)
            """
        )
    )
    original_spawn = asyncio.create_subprocess_exec
    homes = {}

    async def spawn(*arguments, **keywords):
        name = "api" if arguments[3] == "gateway" else "dashboard"
        process = await original_spawn(sys.executable, str(script), name, **keywords)
        homes[process.pid] = Path(keywords["env"]["HERMES_HOME"])
        return process

    async def healthy(child, deadline):
        while asyncio.get_running_loop().time() < deadline:
            if child.process.returncode is not None:
                raise RuntimeError(f"{child.name} exited: {child.process.returncode}")
            if (homes[child.process.pid] / f"{child.name}.ready").exists():
                return
            await asyncio.sleep(0.01)
        raise TimeoutError(child.name)

    monkeypatch.setattr(interfaces.asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(interfaces, "_wait_healthy", healthy)
    monkeypatch.setattr(
        interfaces.local_interfaces, "port_available", lambda port: True
    )
    yield tmp_path
    for records in tmp_path.rglob("processes.jsonl"):
        for line in records.read_text().splitlines():
            try:
                record = json.loads(line)
                process = psutil.Process(record["pid"])
                if (
                    process.create_time() == record["created"]
                    and process.is_running()
                    and process.status() != psutil.STATUS_ZOMBIE
                ):
                    process.send_signal(signal.SIGKILL)
            except psutil.NoSuchProcess:
                pass


@pytest.mark.skipif(os.name != "posix", reason="Uses POSIX process groups")
@pytest.mark.parametrize("deny_groups", [False, True])
async def test_stop_closes_chat_before_gateway_and_allows_restart(
    interface_processes, web_dist, monkeypatch, deny_groups
):
    import fcntl

    if deny_groups:

        def denied_group(pid, signum):
            raise PermissionError("process group signals denied")

        monkeypatch.setattr(interfaces.common_utils.os, "killpg", denied_group)
    children = interfaces.InterfaceChildren()
    environment = dict(os.environ, HERMES_HOME=str(interface_processes))
    arguments = {
        "hermes_home": interface_processes,
        "token_path": interface_processes / "interfaces/api.token",
        "timeout": 5,
        "environment": environment,
    }
    await children.start({"api": 8642, "dashboard": 9119}, **arguments)
    await children.stop()

    with (interface_processes / "gateway.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (interface_processes / "daemon.ready").exists()
    for name in ("api", "dashboard", "chat"):
        (interface_processes / f"{name}.ready").unlink()
    await children.start({"api": 8642, "dashboard": 9119}, **arguments)
    await children.stop()


@pytest.mark.skipif(os.name != "posix", reason="Uses POSIX process groups")
async def test_stop_reaps_detached_recovery_that_already_lost_its_parent(
    interface_processes, web_dist
):
    import json

    import psutil

    children = interfaces.InterfaceChildren()
    await children.start(
        {"api": 8642, "dashboard": 9119},
        hermes_home=interface_processes,
        token_path=interface_processes / "interfaces/api.token",
        timeout=5,
        environment=dict(
            os.environ, HERMES_HOME=str(interface_processes), RACE_IN_FLIGHT="1"
        ),
    )
    (interface_processes / "release-bootstrap").touch()
    for _ in range(500):
        if (interface_processes / "daemon.ready").exists():
            break
        await asyncio.sleep(0.01)
    assert (interface_processes / "daemon.ready").exists()
    records = [
        json.loads(line)
        for line in (interface_processes / "processes.jsonl").read_text().splitlines()
    ]
    daemon = psutil.Process(
        next(row["pid"] for row in records if row["mode"] == "daemon")
    )
    await children.stop()
    assert not daemon.is_running() or daemon.status() == psutil.STATUS_ZOMBIE


@pytest.mark.skipif(os.name != "posix", reason="Uses POSIX process groups")
async def test_stop_attempts_direct_children_when_ownership_inspection_fails(
    interface_processes, web_dist, monkeypatch
):
    children = interfaces.InterfaceChildren()
    await children.start(
        {"api": 8642, "dashboard": 9119},
        hermes_home=interface_processes,
        token_path=interface_processes / "interfaces/api.token",
        timeout=5,
        environment=dict(os.environ, HERMES_HOME=str(interface_processes)),
    )
    processes = [child.process for child in children._children]

    def denied_inspection(_children):
        raise PermissionError("process inspection denied")

    monkeypatch.setattr(interfaces, "_owned_processes", denied_inspection)
    with pytest.raises((PermissionError, interfaces.lifecycle.LifecycleError)):
        await children.stop()
    for process in processes:
        assert await asyncio.wait_for(process.wait(), timeout=2) is not None
    assert children._children


@pytest.mark.skipif(os.name != "posix", reason="Uses POSIX process groups")
async def test_explicit_session_resume_after_chat_recovery_shutdown(
    interface_processes, web_dist, api_server
):
    import json

    from nemo_fabric_adapter_contract.models import AgentRunRequest

    settings = {"interfaces": {"api": {"port": 8642}, "dashboard": {"port": 9119}}}
    payload = _payload(interface_processes, settings, base_url=api_server + "/v1")
    home = interface_processes / "retained"
    payload["runtime_context"]["state_dir"] = str(home)
    config = payload["config"].to_mapping()
    config["tools"] = {"enabled": []}
    payload["config"] = AgentConfig.from_mapping(config)
    nonce = "retained-conversation-after-browser-reconnect"
    first = adapter.HermesRuntime()
    try:
        started = await first.start(payload)
        reference = {
            "schema_version": "fabric.session_ref/v1alpha1",
            "adapter_id": "nvidia.fabric.hermes",
            "session_id": started["session_id"],
        }
        result = await first.invoke(
            AgentRunRequest(input=nonce),
            RuntimeContext.from_mapping(payload["runtime_context"]),
        )
        assert result.status == "succeeded"
    finally:
        await first.stop()
    for name in ("api", "dashboard", "chat"):
        (home / f"{name}.ready").unlink()
    payload["runtime_context"]["runtime_id"] = "runtime-resumed"
    payload["runtime_context"]["session_ref"] = reference
    resumed = adapter.HermesRuntime()
    try:
        restored = await resumed.start(payload)
        assert restored["session_id"] == reference["session_id"]
        result = await resumed.invoke(
            AgentRunRequest(input="Use our earlier conversation."),
            RuntimeContext.from_mapping(payload["runtime_context"]),
        )
        assert result.status == "succeeded"
        captured = requests.get(api_server + "/_requests", timeout=5).json()
        assert nonce in json.dumps(captured[-1]["messages"])
    finally:
        await resumed.stop()


@pytest.mark.skipif(os.name != "posix", reason="Uses POSIX process groups")
async def test_stop_preserves_an_unrelated_process_in_the_same_home(
    interface_processes, web_dist
):
    import subprocess

    foreign = subprocess.Popen(
        [sys.executable, str(interface_processes / "interface_process.py"), "foreign"],
        env=dict(os.environ, HERMES_HOME=str(interface_processes)),
        start_new_session=True,
    )
    try:
        children = interfaces.InterfaceChildren()
        await children.start(
            {"dashboard": 9119},
            hermes_home=interface_processes,
            token_path=interface_processes / "interfaces/api.token",
            timeout=5,
            environment=dict(os.environ, HERMES_HOME=str(interface_processes)),
        )
        await children.stop()
        assert foreign.poll() is None
    finally:
        foreign.terminate()
        await asyncio.to_thread(foreign.wait, timeout=5)


@pytest.mark.skipif(os.name != "posix", reason="Uses POSIX process groups")
async def test_failed_cleanup_blocks_replacement_before_native_state_is_opened(
    interface_processes, web_dist, monkeypatch, api_server
):
    children = interfaces.InterfaceChildren()
    await children.start(
        {"api": 8642, "dashboard": 9119},
        hermes_home=interface_processes,
        token_path=interface_processes / "interfaces/api.token",
        timeout=5,
        environment=dict(os.environ, HERMES_HOME=str(interface_processes)),
    )
    inspect_owned = interfaces._owned_processes

    def denied_inspection(_children):
        raise PermissionError("process inspection denied")

    monkeypatch.setattr(interfaces, "_owned_processes", denied_inspection)
    with pytest.raises(interfaces.lifecycle.LifecycleError):
        await children.stop()
    for child in children._children:
        await asyncio.wait_for(child.process.wait(), timeout=2)
    monkeypatch.setattr(interfaces, "_owned_processes", inspect_owned)
    payload = _payload(interface_processes, {}, base_url=api_server + "/v1")
    payload["runtime_context"]["state_dir"] = str(interface_processes)
    replacement = adapter.HermesRuntime()
    try:
        with pytest.raises(interfaces.lifecycle.LifecycleError) as caught:
            await replacement.start(payload)
        assert caught.value.code == "hermes_interface_cleanup_unconfirmed"
        assert not (interface_processes / "config.yaml").exists()
        assert not (interface_processes / "state.db").exists()
    finally:
        await replacement.stop()


@pytest.mark.parametrize(
    ("same_user", "recent", "refused"),
    [(True, True, True), (False, True, False), (True, False, False)],
)
def test_ownership_inspection_refuses_an_unreadable_possible_descendant(
    monkeypatch, same_user, recent, refused
):
    from unittest.mock import MagicMock

    import psutil

    issued = time.time()
    current_user = psutil.Process().uids()
    process = MagicMock(spec=psutil.Process)
    process.info = {
        "status": psutil.STATUS_RUNNING,
        "create_time": issued if recent else issued - 60,
        "uids": current_user
        if same_user
        else current_user._replace(real=current_user.real + 1),
    }
    process.environ.side_effect = psutil.AccessDenied()
    monkeypatch.setattr(psutil, "pids", lambda: [42])
    monkeypatch.setattr(psutil, "process_iter", lambda *args, **kwargs: iter([process]))
    if refused:
        with pytest.raises(interfaces.lifecycle.LifecycleError) as caught:
            interfaces._owned_processes({"a" * 43: issued})
        assert caught.value.code == "hermes_interface_cleanup_unconfirmed"
    else:
        assert interfaces._owned_processes({"a" * 43: issued}) == []


@pytest.mark.skipif(os.name != "posix", reason="Uses POSIX symlinks")
@pytest.mark.parametrize("directory_link", [False, True])
def test_interface_ownership_rejects_symlinks(tmp_path, directory_link):
    outside = tmp_path / "outside"
    outside.mkdir()
    record = outside / "process-owners.json"
    contents = '{"private-marker": 1}'
    record.write_text(contents)
    home = tmp_path / "home"
    home.mkdir()
    if directory_link:
        (home / "interfaces").symlink_to(outside, target_is_directory=True)
    else:
        (home / "interfaces").mkdir()
        (home / "interfaces/process-owners.json").symlink_to(record)
    with pytest.raises(interfaces.lifecycle.LifecycleError) as caught:
        interfaces.require_released_home(home)
    assert caught.value.code == "hermes_interface_cleanup_unconfirmed"
    assert "private-marker" not in str(caught.value)
    assert record.read_text() == contents


@pytest.mark.parametrize(
    "contents",
    [
        "not-json",
        "[]",
        '{"short": 1}',
        "x" * 4097,
        pytest.param(
            '{"' + "a" * 43 + '":' + "9" * 1000 + "}", id="timestamp-overflow"
        ),
    ],
)
def test_interface_ownership_rejects_unreadable_records(tmp_path, contents):
    directory = tmp_path / "interfaces"
    directory.mkdir()
    record = directory / "process-owners.json"
    record.write_text(contents)
    with pytest.raises(interfaces.lifecycle.LifecycleError) as caught:
        interfaces.require_released_home(tmp_path)
    assert caught.value.code == "hermes_interface_cleanup_unconfirmed"
    assert record.read_text() == contents


@pytest.mark.skipif(os.name != "posix", reason="Uses POSIX processes")
async def test_interface_ownership_releases_a_stale_record_after_processes_exit(
    interface_processes, web_dist
):
    children = interfaces.InterfaceChildren()
    await children.start(
        {"api": 8642},
        hermes_home=interface_processes,
        token_path=interface_processes / "interfaces/api.token",
        timeout=5,
        environment=dict(os.environ, HERMES_HOME=str(interface_processes)),
    )
    record = interface_processes / "interfaces/process-owners.json"
    owners = record.read_text()
    assert record.stat().st_mode & 0o777 == 0o600
    await children.stop()
    assert not record.exists()
    record.write_text(owners)
    interfaces.require_released_home(interface_processes)
    assert not record.exists()


@pytest.mark.skipif(os.name != "posix", reason="Uses POSIX processes")
@pytest.mark.parametrize("replacement", [False, True])
async def test_fork_during_ownership_scan_cannot_confirm_cleanup(
    interface_processes, web_dist, monkeypatch, replacement
):
    import json

    import psutil

    children = interfaces.InterfaceChildren()
    await children.start(
        {"api": 8642, "dashboard": 9119},
        hermes_home=interface_processes,
        token_path=interface_processes / "interfaces/api.token",
        timeout=5,
        environment=dict(
            os.environ, HERMES_HOME=str(interface_processes), RACE_IN_FLIGHT="1"
        ),
    )
    records = interface_processes / "processes.jsonl"
    for _ in range(500):
        if (interface_processes / "bootstrap.ready").exists():
            break
        await asyncio.sleep(0.01)
    assert (interface_processes / "bootstrap.ready").exists()
    rows = [json.loads(line) for line in records.read_text().splitlines()]
    for row in rows:
        if row["mode"] != "bootstrap":
            process = psutil.Process(row["pid"])
            assert process.create_time() == row["created"]
            process.kill()
    for child in children._children:
        await asyncio.wait_for(child.process.wait(), timeout=2)
    original_iter = psutil.process_iter
    paused = False

    def fork_after_snapshot(*args, **kwargs):
        nonlocal paused
        if paused:
            return original_iter(*args, **kwargs)
        paused = True
        snapshot = list(original_iter(*args, **kwargs))
        (interface_processes / "release-bootstrap").touch()
        deadline = time.monotonic() + 5
        bootstrap = psutil.Process(
            next(row["pid"] for row in rows if row["mode"] == "bootstrap")
        )
        while time.monotonic() < deadline:
            gone = (
                not bootstrap.is_running() or bootstrap.status() == psutil.STATUS_ZOMBIE
            )
            if (interface_processes / "daemon.ready").exists() and gone:
                return iter(snapshot)
            time.sleep(0.01)
        raise AssertionError("bootstrap did not fork and exit")

    monkeypatch.setattr(psutil, "process_iter", fork_after_snapshot)
    try:
        if replacement:
            with pytest.raises(interfaces.lifecycle.LifecycleError) as caught:
                interfaces.require_released_home(interface_processes)
            assert caught.value.code == "hermes_interface_cleanup_unconfirmed"
        else:
            assert not await interfaces._wait_stopped(children._children, timeout=2)
        assert (interface_processes / "interfaces/process-owners.json").exists()
    finally:
        monkeypatch.setattr(psutil, "process_iter", original_iter)
        await children.stop()
