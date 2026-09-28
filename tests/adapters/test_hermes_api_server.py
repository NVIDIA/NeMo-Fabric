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
from nemo_fabric_adapters.hermes import api_server, configuration, telemetry

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


async def test_retained_state_is_written_once_and_checked_for_drift(tmp_path, native):
    state = tmp_path / "state"
    config = _config(
        {"state_dir": str(state), "native_config": {"web": {"backend": "tavily"}}}
    )
    runtime = api_server.HermesApiServerRuntime()
    await runtime.start(_payload(config, tmp_path))
    written = (state / "config.yaml").read_text()
    native_config = yaml.safe_load(written)
    assert native_config["web"] == {"backend": "tavily"}
    assert native_config["model"]["api_mode"] == "codex_responses"
    assert native_config["model"]["provider"] == "custom:fabric"
    assert native_config["providers"]["fabric"]["key_env"] == "FIXTURE_KEY"
    assert "fixture-placeholder" not in written
    token_path = state / "interface-token"
    assert token_path.stat().st_mode & 0o777 == 0o600
    token = token_path.read_text()
    with pytest.raises(lifecycle.LifecycleError, match="in use"):
        await api_server.HermesApiServerRuntime().start(_payload(config, tmp_path))
    await runtime.stop()

    restarted = api_server.HermesApiServerRuntime()
    await restarted.start(_payload(config, tmp_path))
    assert token_path.read_text() == token, "the credential survives restarts"
    await restarted.stop()

    changed = _config(
        {"state_dir": str(state), "native_config": {"web": {"backend": "brave"}}}
    )
    with pytest.raises(lifecycle.LifecycleError, match="conflicts"):
        await api_server.HermesApiServerRuntime().start(_payload(changed, tmp_path))
    dropped = _config({"state_dir": str(state)})
    with pytest.raises(lifecycle.LifecycleError, match="conflicts"):
        await api_server.HermesApiServerRuntime().start(_payload(dropped, tmp_path))


async def test_a_retained_log_is_made_owner_only(tmp_path, native):
    state = tmp_path / "state"
    state.mkdir()
    (state / "api.log").touch(mode=0o644)
    runtime = api_server.HermesApiServerRuntime()
    await runtime.start(_payload(_config({"state_dir": str(state)}), tmp_path))
    assert (state / "api.log").stat().st_mode & 0o777 == 0o600
    await runtime.stop()


async def test_without_state_dir_native_state_is_scoped_to_the_runtime(
    tmp_path, native
):
    runtime = api_server.HermesApiServerRuntime()
    await runtime.start(_payload(_config({}), tmp_path, runtime_id="runtime-1"))
    assert runtime.home == tmp_path / "artifacts/.fabric/hermes/runtimes/runtime-1"
    assert (runtime.home / "config.yaml").exists()
    await runtime.stop()


async def test_invocations_forward_instructions_and_chain_responses(
    tmp_path, native
):
    runtime = api_server.HermesApiServerRuntime()
    await runtime.start(_payload(_config({}), tmp_path))
    context = RuntimeContext.from_mapping(_context(tmp_path))
    native.request.return_value = COMPLETED
    result = await runtime.invoke(AgentRunRequest(input="hello"), context)
    assert result.status == "succeeded"
    assert result.output["response"] == "hello"
    first = native.request.call_args.kwargs["body"]
    assert first["instructions"] == "Be brief."
    assert native.request.call_args.kwargs["timeout"] is None, (
        "runtime.timeout_seconds, enforced by NeMo Fabric, bounds the turn"
    )
    assert "previous_response_id" not in first
    await runtime.invoke(AgentRunRequest(input="again"), context)
    assert native.request.call_args.kwargs["body"]["previous_response_id"] == "turn-1"
    await runtime.stop()


async def test_definite_failures_keep_the_runtime_and_uncertain_ones_stop_it(
    tmp_path, native
):
    runtime = api_server.HermesApiServerRuntime()
    await runtime.start(_payload(_config({}), tmp_path))
    context = RuntimeContext.from_mapping(_context(tmp_path))
    native.request.return_value = {"status": "failed", "id": "turn-0"}
    result = await runtime.invoke(AgentRunRequest(input="hello"), context)
    assert result.status == "failed"
    native.request.return_value = COMPLETED
    result = await runtime.invoke(AgentRunRequest(input="hello"), context)
    assert result.status == "succeeded", "a reported failure does not stop Hermes"
    native.request.side_effect = urllib.error.HTTPError(
        "http://127.0.0.1/v1/responses", 500, "Internal Server Error", {}, None
    )
    result = await runtime.invoke(AgentRunRequest(input="hello"), context)
    assert result.status == "failed"
    assert result.error.code == "hermes_invocation_failed"
    native.request.side_effect = None
    result = await runtime.invoke(AgentRunRequest(input="hello"), context)
    assert result.status == "succeeded", "an HTTP error response does not stop Hermes"

    native.request.side_effect = OSError("uncertain network result")
    result = await runtime.invoke(AgentRunRequest(input="again"), context)
    assert result.status == "failed"
    with pytest.raises(lifecycle.LifecycleError, match="no replay"):
        await runtime.invoke(AgentRunRequest(input="again"), context)


def test_interface_defaults_apply_to_partially_declared_interfaces():
    assert api_server.interface_settings({})["dashboard"]["enabled"] is False
    dashboard = api_server.interface_settings(
        {"interfaces": {"dashboard": {"port": 9000}}}
    )["dashboard"]
    assert dashboard["enabled"] is True
    assert dashboard["port"] == 9000
    assert dashboard["tui"] == {"enabled": True}


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


async def test_relay_telemetry_enables_the_hermes_relay_plugin(
    tmp_path, native, monkeypatch
):
    relay_config = tmp_path / "relay.toml"
    monkeypatch.setattr(
        telemetry, "write_hermes_relay_plugin_config", lambda payload: (relay_config, {})
    )
    payload = _payload(_config({}), tmp_path)
    payload["runtime_context"]["telemetry"] = {"relay_enabled": True}
    runtime = api_server.HermesApiServerRuntime()
    await runtime.start(payload)
    written = yaml.safe_load((runtime.home / "config.yaml").read_text())
    assert written["plugins"]["enabled"] == ["observability/nemo_relay"]
    env = native.spawn.call_args.kwargs["env"]
    assert env["HERMES_NEMO_RELAY_PLUGINS_TOML"] == str(relay_config)
    await runtime.stop()


def test_plugin_settings_compose_with_enabled_plugins(tmp_path):
    config = _config(
        {
            "plugins_enabled": ["web/tavily"],
            "native_config": {"plugins": {"entries": {"web/tavily": {"depth": 2}}}},
        }
    )
    native = configuration.build_hermes_config(
        config, workspace=str(tmp_path), relay_enabled=True
    )
    assert native["plugins"] == {
        "entries": {"web/tavily": {"depth": 2}},
        "enabled": ["web/tavily", "observability/nemo_relay"],
    }


def test_retained_state_checks_every_section_native_config_can_set(repo_root):
    descriptor = json.loads(
        (repo_root / "adapters/python/hermes/hermes.fabric-adapter.json").read_text()
    )
    native_config = descriptor["settings_schema"]["properties"]["native_config"]
    assert set(native_config["properties"]) <= configuration.CONFIGURED_SECTIONS


async def test_native_server_stops_when_its_adapter_exits(tmp_path, monkeypatch):
    disconnected = MagicMock()

    class NativeApi:
        def __init__(self, config):
            self.config = config

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
    parents = iter([4242, 4242])
    monkeypatch.setattr(api_server.os, "getppid", lambda: next(parents, 1))
    await asyncio.wait_for(api_server._serve_native_interfaces(), 5)
    disconnected.assert_called_once()


async def test_an_inherited_model_endpoint_does_not_reach_hermes(
    tmp_path, native, monkeypatch
):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://inherited.invalid/v1")
    config = AgentConfig.from_mapping(
        {
            "harness": {"settings": {"mode": "api_server"}},
            "models": {
                "default": {
                    "provider": "openai",
                    "model": "fixture",
                    "api_key_env": "FIXTURE_KEY",
                }
            },
        }
    )
    runtime = api_server.HermesApiServerRuntime()
    await runtime.start(_payload(config, tmp_path))
    assert "OPENAI_BASE_URL" not in native.spawn.call_args.kwargs["env"]
    await runtime.stop()


async def test_stop_finishes_shutdown_when_cancelled_or_the_process_is_gone(
    tmp_path, native, monkeypatch
):
    runtime = api_server.HermesApiServerRuntime()
    await runtime.start(_payload(_config({}), tmp_path))
    log = runtime._log
    reaped = asyncio.Event()
    waits = 0

    async def wait():
        nonlocal waits
        waits += 1
        if waits == 1:
            await asyncio.sleep(3600)
        native.process.returncode = -9
        reaped.set()

    native.process.wait = wait
    signals = []

    def killpg(pid, sig):
        signals.append(sig)
        if sig == api_server.signal.SIGKILL:
            raise ProcessLookupError

    monkeypatch.setattr(api_server.os, "killpg", killpg)
    stopping = asyncio.create_task(runtime.stop())
    await asyncio.sleep(0.05)
    stopping.cancel()
    with pytest.raises(asyncio.CancelledError):
        await stopping
    assert reaped.is_set(), "cancellation does not abandon the child"
    assert signals == [api_server.signal.SIGTERM, api_server.signal.SIGKILL]
    assert log.closed and runtime._log is None and runtime._state_lock is None
    assert runtime.process is None

