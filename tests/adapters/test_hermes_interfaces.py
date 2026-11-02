# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Hermes's API server and dashboard served by a managed runtime."""

from __future__ import annotations

import asyncio
import importlib.util
import os
import socket
import sys
import time
from pathlib import Path

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
    while not requests.get(f"{api_server}/_requests", timeout=5).json():
        await asyncio.sleep(0.1)

    started = time.monotonic()
    await runtime.stop()
    elapsed = time.monotonic() - started
    in_flight.cancel()

    assert elapsed < 8
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


async def test_hermes_failed_interface_start_removes_a_token_left_in_kept_state(
    tmp_path: Path, api_server: str
):
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
