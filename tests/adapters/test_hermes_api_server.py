# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hermes API server mode forwards invocations to Hermes' native Responses API."""

import asyncio
import json
import sys
import types
import urllib.error
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from nemo_fabric_adapter_contract.models import (
    AgentConfig,
    AgentRunRequest,
    RuntimeContext,
)

if sys.version_info >= (3, 14):
    pytest.skip("Hermes adapter supports Python 3.11–3.13", allow_module_level=True)
if sys.platform == "win32":
    pytest.skip("Hermes API server mode requires a POSIX host", allow_module_level=True)

import yaml
from nemo_fabric import Fabric, FabricConfig, FabricConfigError
from nemo_fabric_adapters.common import lifecycle
from nemo_fabric_adapters.common.credentials import interface_token
from nemo_fabric_adapters.hermes import api_server, telemetry

COMPLETED = {
    "status": "completed",
    "id": "turn-1",
    "output": [{"content": [{"type": "output_text", "text": "hello"}]}],
}


def _context(tmp_path, runtime_id="fixture"):
    return {
        "runtime_id": runtime_id,
        "invocation_id": "invoke",
        "request_id": "request",
        "artifacts": {},
        "environment": {
            "provider": "local",
            "environment_id": "local",
            "ownership": "caller_owned",
            "control_location": "external_control",
            "workspace": str(tmp_path),
        },
    }


def _config(settings):
    return AgentConfig.from_mapping(
        {
            "harness": {"settings": {"mode": "api_server", **settings}},
            "instructions": {"system": {"content": "Be brief."}},
            "models": {
                "default": {
                    "provider": "openai",
                    "model": "fixture",
                    "base_url": "https://fixture.invalid/v1",
                    "api_key_env": "FIXTURE_KEY",
                    "api": "openai-responses",
                }
            },
        }
    )


def _payload(config, tmp_path, runtime_id="fixture"):
    return {
        "config": config,
        "runtime_context": _context(tmp_path, runtime_id),
        "base_dir": str(tmp_path),
    }


@pytest.fixture(name="native")
def native_fixture(monkeypatch):
    """Replace the native Hermes process and HTTP API."""

    process = MagicMock(returncode=None, pid=1234)
    process.wait = AsyncMock()
    spawn = AsyncMock(return_value=process)
    monkeypatch.setattr(api_server.asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(api_server.os, "killpg", MagicMock())
    request = MagicMock(return_value={"data": [{"id": "primary"}]})
    monkeypatch.setattr(api_server, "api_request", request)
    monkeypatch.setenv("FIXTURE_KEY", "fixture-placeholder")
    return SimpleNamespace(process=process, spawn=spawn, request=request)


async def _start(tmp_path, config, **context):
    runtime = api_server.HermesApiServerRuntime()
    payload = _payload(config, tmp_path)
    payload["runtime_context"].update(context)
    await runtime.start(payload)
    return runtime


async def test_state_dir_keeps_the_credential_and_rewrites_config(tmp_path, native):
    state = tmp_path / "state"
    state.mkdir()
    (state / "api.log").touch(mode=0o644)  # Retained from an earlier runtime.
    web = {"state_dir": str(state), "native_config": {"web": {"backend": "tavily"}}}
    runtime = await _start(tmp_path, _config(web))
    written = (state / "config.yaml").read_text()
    assert yaml.safe_load(written)["providers"]["fabric"]["key_env"] == "FIXTURE_KEY"
    assert "fixture-placeholder" not in written, "the key stays in the environment"
    token = interface_token(state)
    with pytest.raises(lifecycle.LifecycleError, match="in use"):
        await _start(tmp_path, _config(web))
    await runtime.stop()
    for name in ["config.yaml", "interface-token", "api.log"]:
        assert (state / name).stat().st_mode & 0o777 == 0o600, name

    runtime = await _start(tmp_path, _config({"state_dir": str(state)}))
    assert "web" not in yaml.safe_load((state / "config.yaml").read_text())
    assert interface_token(state) == token, "the credential survives restarts"
    await runtime.stop()


async def test_without_state_dir_native_state_belongs_to_the_runtime(tmp_path, native):
    runtime = await _start(tmp_path, _config({}), runtime_id="runtime-1")
    assert runtime.home == tmp_path / "artifacts/.fabric/hermes/runtimes/runtime-1"
    await runtime.stop()


async def test_hermes_gets_only_the_configured_endpoint_and_relay_plugin(
    tmp_path, native, monkeypatch
):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://inherited.invalid/v1")
    relay = tmp_path / "relay.toml"
    monkeypatch.setattr(
        telemetry, "write_hermes_relay_plugin_config", lambda _: (relay, {})
    )
    config = _config({}).to_mapping()
    del config["models"]["default"]["base_url"], config["models"]["default"]["api"]
    runtime = await _start(
        tmp_path, AgentConfig.from_mapping(config), telemetry={"relay_enabled": True}
    )
    env = native.spawn.call_args.kwargs["env"]
    assert "OPENAI_BASE_URL" not in env
    assert env["HERMES_NEMO_RELAY_PLUGINS_TOML"] == str(relay)
    written = yaml.safe_load((runtime.home / "config.yaml").read_text())
    assert written["plugins"]["enabled"] == ["observability/nemo_relay"]
    await runtime.stop()


async def test_invocations_forward_instructions_and_chain_responses(tmp_path, native):
    runtime = await _start(tmp_path, _config({}))
    context = RuntimeContext.from_mapping(_context(tmp_path))
    native.request.return_value = COMPLETED
    result = await runtime.invoke(AgentRunRequest(input="hello"), context)
    assert result.output["response"] == "hello"
    call = native.request.call_args.kwargs
    assert call["body"]["instructions"] == "Be brief."
    assert "previous_response_id" not in call["body"]
    assert call["timeout"] is None, "runtime.timeout_seconds bounds the turn"
    await runtime.invoke(AgentRunRequest(input="again"), context)
    assert native.request.call_args.kwargs["body"]["previous_response_id"] == "turn-1"
    await runtime.stop()


async def test_definite_failures_keep_the_runtime_and_uncertain_ones_stop_it(
    tmp_path, native
):
    runtime = await _start(tmp_path, _config({}))
    context = RuntimeContext.from_mapping(_context(tmp_path))

    async def invoke(outcome):
        native.request.side_effect = outcome if isinstance(outcome, Exception) else None
        native.request.return_value = outcome
        return await runtime.invoke(AgentRunRequest(input="hello"), context)

    http_error = urllib.error.HTTPError("http://127.0.0.1", 500, "error", {}, None)
    for definite in [{"status": "failed", "id": "turn-0"}, http_error]:
        assert (await invoke(definite)).status == "failed"
        assert (await invoke(COMPLETED)).status == "succeeded", "Hermes keeps running"
    assert (await invoke(OSError("unknown outcome"))).status == "failed"
    with pytest.raises(lifecycle.LifecycleError, match="no replay"):
        await invoke(COMPLETED)


async def test_stop_reaps_hermes_even_when_cancelled(tmp_path, native, monkeypatch):
    runtime = await _start(tmp_path, _config({}))
    log = runtime._log
    signals = []
    waits = []

    async def wait():
        waits.append(True)
        if len(waits) == 1:
            await asyncio.sleep(3600)  # SIGTERM is ignored.
        native.process.returncode = -9

    def killpg(pid, sig):
        signals.append(sig)
        if sig == api_server.signal.SIGKILL:
            raise ProcessLookupError  # The group exited meanwhile.

    native.process.wait = wait
    monkeypatch.setattr(api_server.os, "killpg", killpg)
    stopping = asyncio.create_task(runtime.stop())
    await asyncio.sleep(0.05)
    stopping.cancel()
    with pytest.raises(asyncio.CancelledError):
        await stopping
    assert signals == [api_server.signal.SIGTERM, api_server.signal.SIGKILL]
    assert native.process.returncode == -9, "cancellation does not abandon the child"
    assert log.closed and runtime._state_lock is None and runtime.process is None


async def test_native_server_stops_when_its_adapter_exits(tmp_path, monkeypatch):
    disconnected = MagicMock()

    class NativeApi:
        def __init__(self, config):
            pass

        async def connect(self):
            return True

        async def disconnect(self):
            disconnected()

    gateway_config = types.ModuleType("gateway.config")
    gateway_config.PlatformConfig = lambda **fields: fields
    gateway_api = types.ModuleType("gateway.platforms.api_server")
    gateway_api.APIServerAdapter = NativeApi
    monkeypatch.setitem(sys.modules, "gateway.config", gateway_config)
    monkeypatch.setitem(sys.modules, "gateway.platforms.api_server", gateway_api)
    interface_token(tmp_path, create=True)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv(
        "FABRIC_HERMES_INTERFACES", json.dumps(api_server.interface_settings({}))
    )
    parents = iter([4242, 4242])  # The adapter exits after the server starts.
    monkeypatch.setattr(api_server.os, "getppid", lambda: next(parents, 1))
    await asyncio.wait_for(api_server._serve_native_interfaces(), 5)
    disconnected.assert_called_once()


def test_the_dashboard_is_opt_in_and_partial_settings_get_defaults():
    assert api_server.interface_settings({})["dashboard"]["enabled"] is False
    dashboard = api_server.interface_settings(
        {"interfaces": {"dashboard": {"port": 9000}}}
    )
    assert dashboard["dashboard"] == {
        "enabled": True,
        "port": 9000,
        "internalPort": 19119,
        "tui": {"enabled": True},
    }


def test_planning_rejects_api_server_settings_in_sdk_mode(tmp_path):
    def plan(settings):
        config = FabricConfig.from_mapping(
            {
                "metadata": {"name": "hermes-modes"},
                "harness": {"adapter_id": "nvidia.fabric.hermes", "settings": settings},
                "models": {"default": {"provider": "openai", "model": "fixture"}},
            }
        )
        return Fabric().plan(config, base_dir=tmp_path)

    plan({"mode": "api_server", "interfaces": {"api": {"port": 8643}}})
    with pytest.raises(FabricConfigError, match="interfaces"):
        plan({"interfaces": {"api": {"port": 8643}}})
    with pytest.raises(FabricConfigError, match="state_dir"):
        plan({"mode": "sdk", "state_dir": str(tmp_path)})
