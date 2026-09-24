# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run Hermes' native API server, and optionally its dashboard, for one runtime.

In ``api_server`` mode the adapter starts Hermes' OpenAI-compatible Responses
API in a supervised child process and forwards each invocation to it, instead
of embedding the Hermes SDK. The same child can also serve the Hermes dashboard
and browser TUI. Native state lives in ``harness.settings.state_dir`` when
retained across runtimes, or in a runtime-scoped directory otherwise. The mode
requires a POSIX host.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import urllib.request
from pathlib import Path
from typing import Any

import yaml
from nemo_fabric_adapter_contract.models import AgentConfig
from nemo_fabric_adapter_contract.models import AgentRunError
from nemo_fabric_adapter_contract.models import AgentRunResult
from nemo_fabric_adapter_contract.models import AgentRunStatus
from nemo_fabric_adapter_contract.models import RuntimeContext
from nemo_fabric_adapters.common import instructions as common_instructions
from nemo_fabric_adapters.common import lifecycle
from nemo_fabric_adapters.common.credentials import interface_token
from nemo_fabric_adapters.hermes import configuration

MODEL_NAME = "primary"
RESPONSE_LIMIT = 4 * 1024 * 1024
STARTUP_TIMEOUT_SECONDS = 75
INVOKE_TIMEOUT_SECONDS = 280
SHUTDOWN_TIMEOUT_SECONDS = 10
MAX_DASHBOARD_CLIENTS = 128
DASHBOARD_HOME = Path("profiles") / "dashboard-home"


def interface_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """Resolve declared native interfaces; the dashboard is opt-in."""

    interfaces = settings.get("interfaces", {})
    dashboard = interfaces.get("dashboard")
    return {
        "apiPort": interfaces.get("api", {}).get("port", 8642),
        "dashboard": {
            "enabled": dashboard is not None and dashboard.get("enabled", True),
            "port": (dashboard or {}).get("port", 18789),
            "internalPort": (dashboard or {}).get("internalPort", 19119),
            "tui": (dashboard or {}).get("tui", {"enabled": True}),
        },
    }


def configuration_matches(home: Path, native: dict[str, Any]) -> bool:
    """Whether every NeMo Fabric-owned top-level section is unchanged on disk."""

    actual = yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))
    return isinstance(actual, dict) and all(
        actual.get(key) == value for key, value in native.items()
    )


def _initialize(home: Path, native: dict[str, Any]) -> None:
    """Write native configuration and its credential once, or verify them."""

    home.mkdir(parents=True, mode=0o700, exist_ok=True)
    path = home / "config.yaml"
    if path.exists():
        interface_token(home)
        if not configuration_matches(home, native):
            raise lifecycle.LifecycleError(
                "hermes_configuration_conflict",
                "Retained Hermes configuration conflicts with NeMo Fabric-owned settings",
                metadata={"state_dir": str(home)},
            )
        return
    interface_token(home, create=True)
    with open(
        path, "x", encoding="utf-8", opener=lambda name, flags: os.open(name, flags, 0o600)
    ) as output:
        yaml.safe_dump(native, output, sort_keys=False)
        output.flush()
        os.fsync(output.fileno())


def api_request(
    home: Path,
    port: int,
    path: str,
    *,
    body: dict[str, Any] | None = None,
    timeout: float = 3,
) -> dict[str, Any]:
    """Call the loopback Hermes API with the retained bearer credential."""

    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=None if body is None else json.dumps(body).encode(),
        headers={
            "Authorization": "Bearer " + interface_token(home),
            "Content-Type": "application/json",
        },
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        raw = response.read(RESPONSE_LIMIT + 1)
    if len(raw) > RESPONSE_LIMIT:
        raise ValueError("Hermes response exceeds the size limit")
    return json.loads(raw)


def _healthy(home: Path, interfaces: dict[str, Any], native: dict[str, Any]) -> bool:
    try:
        if not configuration_matches(home, native):
            return False
        models = api_request(home, interfaces["apiPort"], "/v1/models")
        if models["data"][0]["id"] != MODEL_NAME:
            return False
        dashboard = interfaces["dashboard"]
        if not dashboard["enabled"]:
            return True
        dashboard_home = home / DASHBOARD_HOME
        if not configuration_matches(dashboard_home, native):
            return False
        request = urllib.request.Request(
            f"http://127.0.0.1:{dashboard['port']}/api/sessions?limit=1",
            headers={"X-Hermes-Session-Token": interface_token(dashboard_home)},
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=3) as response:
            return response.status == 200
    except (OSError, ValueError, KeyError, IndexError, RuntimeError):
        return False


def _response_text(result: dict[str, Any]) -> str:
    return "\n".join(
        content["text"]
        for item in result.get("output", [])
        for content in item.get("content", [])
        if content.get("type") == "output_text"
    )


class HermesApiServerRuntime:
    """One supervised Hermes API server process owned by a NeMo Fabric runtime."""

    def __init__(self) -> None:
        self.home: Path | None = None
        self.process: asyncio.subprocess.Process | None = None
        self._log: Any = None
        self._state_lock: Any = None
        self._failed = False
        self._lock = asyncio.Lock()
        self._previous_response: str | None = None

    async def start(self, payload: dict[str, Any]) -> None:
        config: AgentConfig = payload["config"]
        context = RuntimeContext.from_mapping(payload["runtime_context"])
        base_dir = payload["base_dir"]
        settings = configuration._settings(config)
        instruction = common_instructions.system_instruction(
            config, adapter="Hermes", supported_modes={"replace"}
        )
        self._instructions = instruction.content if instruction else None
        self._runtime_id = context.runtime_id
        self._interfaces = interface_settings(settings)
        workspace = context.environment.workspace or base_dir
        state_dir = settings.get("state_dir")
        self.home = (
            Path(base_dir, state_dir).resolve()
            if state_dir
            else configuration.runtime_home(context, base_dir)
        )
        self._native = configuration.build_hermes_config(config, workspace=workspace)
        model = configuration._selected_model(config)
        api_key_env = configuration._api_key_env(model)
        if not os.environ.get(api_key_env):
            raise lifecycle.LifecycleError(
                "hermes_missing_api_key",
                f"Hermes API key environment variable {api_key_env} is not set",
            )
        import fcntl

        self.home.mkdir(parents=True, mode=0o700, exist_ok=True)
        self._state_lock = (self.home / "adapter.lock").open("a")
        try:
            try:
                fcntl.flock(self._state_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                raise lifecycle.LifecycleError(
                    "hermes_state_in_use",
                    "Hermes native state is in use by another runtime",
                    metadata={"state_dir": str(self.home)},
                ) from None
            _initialize(self.home, self._native)
            if self._interfaces["dashboard"]["enabled"]:
                _initialize(self.home / DASHBOARD_HOME, self._native)
            env = dict(
                os.environ,
                HERMES_HOME=str(self.home),
                OPENAI_API_KEY=os.environ[api_key_env],
                HERMES_DISABLE_LAZY_INSTALLS="1",
                FABRIC_HERMES_INTERFACES=json.dumps(self._interfaces),
            )
            if model.base_url:
                env["OPENAI_BASE_URL"] = model.base_url
            self._configure_relay(payload, config, env)
            self._log = (self.home / "api.log").open("ab")
            self.process = await asyncio.create_subprocess_exec(
                sys.executable,
                __file__,
                "server",
                env=env,
                cwd=workspace,
                stdout=self._log,
                stderr=self._log,
                start_new_session=True,
            )
            await self._wait_healthy()
        except BaseException:
            await self.stop()
            raise

    def _configure_relay(
        self, payload: dict[str, Any], config: AgentConfig, env: dict[str, str]
    ) -> None:
        telemetry = payload["runtime_context"].get("telemetry") or {}
        if not telemetry.get("relay_enabled"):
            return
        from nemo_fabric_adapters.hermes.telemetry import HERMES_RELAY_ENV_NAMES
        from nemo_fabric_adapters.hermes.telemetry import (
            write_hermes_relay_plugin_config,
        )

        path, _ = write_hermes_relay_plugin_config(
            {**payload, "config": config.to_mapping()}
        )
        for name in HERMES_RELAY_ENV_NAMES:
            env.pop(name, None)
        env["HERMES_NEMO_RELAY_PLUGINS_TOML"] = str(path)

    async def _wait_healthy(self) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + STARTUP_TIMEOUT_SECONDS
        while loop.time() < deadline:
            if self.process.returncode is not None:
                raise lifecycle.LifecycleError(
                    "hermes_api_server_exited",
                    "Hermes API server exited during startup; inspect api.log",
                    metadata={"log": str(self.home / "api.log")},
                )
            if await asyncio.to_thread(
                _healthy, self.home, self._interfaces, self._native
            ):
                return
            await asyncio.sleep(0.1)
        raise lifecycle.LifecycleError(
            "hermes_api_server_not_ready",
            "Hermes API server did not become ready; inspect api.log",
            metadata={"log": str(self.home / "api.log")},
        )

    async def invoke(self, request: Any, context: RuntimeContext) -> AgentRunResult:
        async with self._lock:
            if (
                self._failed
                or context.runtime_id != self._runtime_id
                or self.process is None
                or self.process.returncode is not None
            ):
                raise lifecycle.LifecycleError(
                    "hermes_runtime_unavailable",
                    "Hermes API server runtime is unavailable; no replay",
                )
            if not isinstance(request.input, str):
                raise lifecycle.LifecycleError(
                    "hermes_unsupported_input", "Hermes requires a text prompt"
                )
            if not configuration_matches(self.home, self._native):
                raise lifecycle.LifecycleError(
                    "hermes_configuration_drift",
                    "Hermes configuration changed outside NeMo Fabric",
                )
            body: dict[str, Any] = {
                "model": MODEL_NAME,
                "input": request.input,
                "store": True,
            }
            if self._instructions is not None:
                body["instructions"] = self._instructions
            if self._previous_response:
                body["previous_response_id"] = self._previous_response
            try:
                result = await asyncio.to_thread(
                    api_request,
                    self.home,
                    self._interfaces["apiPort"],
                    "/v1/responses",
                    body=body,
                    timeout=INVOKE_TIMEOUT_SECONDS,
                )
            except BaseException as error:
                # The turn may still be running natively; stop Hermes rather
                # than accept an overlapping or replayed request.
                self._failed = True
                await self.stop()
                if not isinstance(error, Exception):
                    raise
                return self._failed_result(
                    "hermes_invocation_uncertain",
                    "Hermes invocation outcome is unknown; no replay; inspect api.log",
                )
            if result.get("status") != "completed" or not result.get("id"):
                return self._failed_result(
                    "hermes_invocation_failed",
                    "Hermes reported an unsuccessful response",
                )
            self._previous_response = result["id"]
            return AgentRunResult(
                status=AgentRunStatus.SUCCEEDED,
                output={
                    "harness": "hermes",
                    "response": _response_text(result),
                    "response_id": result["id"],
                },
            )

    @staticmethod
    def _failed_result(code: str, message: str) -> AgentRunResult:
        return AgentRunResult(
            status=AgentRunStatus.FAILED,
            output={"harness": "hermes"},
            error=AgentRunError(code=code, message=message),
        )

    async def stop(self) -> None:
        if self.process is not None and self.process.returncode is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
                await asyncio.wait_for(self.process.wait(), SHUTDOWN_TIMEOUT_SECONDS)
            except TimeoutError:
                os.killpg(self.process.pid, signal.SIGKILL)
                await self.process.wait()
            except ProcessLookupError:
                await self.process.wait()
        if self._log is not None:
            self._log.close()
            self._log = None
        if self._state_lock is not None:
            self._state_lock.close()
            self._state_lock = None


async def _serve_native_interfaces() -> None:
    """Child process: serve the Hermes API and, when declared, its dashboard."""

    from gateway.config import PlatformConfig
    from gateway.platforms.api_server import APIServerAdapter

    home = Path(os.environ["HERMES_HOME"])
    interfaces = json.loads(os.environ["FABRIC_HERMES_INTERFACES"])
    dashboard = interfaces["dashboard"]
    adapter = APIServerAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "host": "127.0.0.1",
                "port": interfaces["apiPort"],
                "key": interface_token(home),
                "model_name": MODEL_NAME,
            },
        )
    )
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
    child = None
    forwarder = None
    clients: set[asyncio.Task[None]] = set()

    async def relay(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        clients.add(task)
        target_writer = None

        async def copy(source, destination) -> None:
            while data := await source.read(65536):
                destination.write(data)
                await destination.drain()

        try:
            if len(clients) > MAX_DASHBOARD_CLIENTS:
                return
            target_reader, target_writer = await asyncio.open_connection(
                "127.0.0.1", dashboard["internalPort"]
            )
            upstream = asyncio.create_task(copy(reader, target_writer))
            downstream = asyncio.create_task(copy(target_reader, writer))
            try:
                await asyncio.wait(
                    [upstream, downstream], return_when=asyncio.FIRST_COMPLETED
                )
            finally:
                upstream.cancel()
                downstream.cancel()
                await asyncio.gather(upstream, downstream, return_exceptions=True)
        except OSError:
            pass
        finally:
            writer.close()
            if target_writer is not None:
                target_writer.close()
            clients.discard(task)

    try:
        if not await adapter.connect():
            raise RuntimeError("Hermes native API startup failed")
        if dashboard["enabled"]:
            dashboard_home = home / DASHBOARD_HOME
            env = dict(
                os.environ,
                HERMES_HOME=str(dashboard_home),
                HERMES_DASHBOARD_SESSION_TOKEN=interface_token(dashboard_home),
            )
            child = await asyncio.create_subprocess_exec(
                sys.executable, __file__, "dashboard", env=env
            )
            forwarder = await asyncio.start_server(
                relay, "127.0.0.1", dashboard["port"]
            )
        while not stop.is_set():
            if child is not None and child.returncode is not None:
                raise RuntimeError("Hermes dashboard exited")
            try:
                await asyncio.wait_for(stop.wait(), 0.2)
            except TimeoutError:
                pass
    finally:
        if forwarder is not None:
            forwarder.close()
            await forwarder.wait_closed()
        pending = list(clients)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        if child is not None and child.returncode is None:
            child.terminate()
            try:
                await asyncio.wait_for(child.wait(), 5)
            except TimeoutError:
                child.kill()
                await child.wait()
        await adapter.disconnect()


def _serve_dashboard() -> None:
    """Grandchild process: serve the Hermes dashboard on its internal port."""

    from hermes_cli import web_server

    dashboard = json.loads(os.environ["FABRIC_HERMES_INTERFACES"])["dashboard"]
    # Pinned Hermes gates browser chat and its WebSocket endpoints on this flag.
    web_server._DASHBOARD_EMBEDDED_CHAT_ENABLED = dashboard["tui"]["enabled"]
    web_server.start_server(
        host="127.0.0.1", port=dashboard["internalPort"], open_browser=False
    )


if __name__ == "__main__":
    if sys.argv[1:] == ["server"]:
        asyncio.run(_serve_native_interfaces())
    elif sys.argv[1:] == ["dashboard"]:
        _serve_dashboard()
