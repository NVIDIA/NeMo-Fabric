# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optional Hermes native API and dashboard, owned by the Hermes adapter."""

import asyncio
import fcntl
import json
import os
import signal
import sys
import urllib.request
from pathlib import Path
from nemo_fabric_adapters.common.credentials import token
from nemo_fabric_adapters.hermes.configuration import (
    build_hermes_config,
    _selected_model,
    _api_key_env,
)

ROOT = Path(os.environ.get("HERMES_HOME", "/sandbox/.hermes"))
PORT = 8642


def configuration_matches(native, home=None):
    import yaml

    actual = yaml.safe_load(((home or ROOT) / "config.yaml").read_text())
    return isinstance(actual, dict) and all(
        actual.get(key) == value for key, value in native.items()
    )


def initialize(native, home=None):
    home = home or ROOT
    home.mkdir(parents=True, mode=0o700, exist_ok=True)
    path = home / "config.yaml"
    token(home, create=not path.exists())
    if path.exists():
        if not configuration_matches(native, home):
            raise RuntimeError(
                "native Hermes configuration conflicts with Fabric-owned settings"
            )
        return
    with open(path, "x", opener=lambda p, flags: os.open(p, flags, 0o600)) as output:
        json.dump(native, output)
        output.flush()
        os.fsync(output.fileno())


def interface_settings(settings):
    interfaces = (settings or {}).get("interfaces", {})
    dashboard = interfaces.get("dashboard", {"enabled": True})
    return {
        "apiPort": interfaces.get("api", {}).get("port", 8642),
        "dashboard": {
            "enabled": dashboard["enabled"],
            "port": dashboard.get("port", 18789),
            "internalPort": dashboard.get("internalPort", 19119),
            "tui": dashboard.get("tui", {"enabled": True}),
        },
    }


def api_request(path, body=None, timeout=3, port=PORT):
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}" + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={
            "Authorization": "Bearer " + token(ROOT),
            "Content-Type": "application/json",
        },
    )
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(
        request, timeout=timeout
    ) as response:
        raw = response.read(4 * 1024 * 1024 + 1)
        if len(raw) > 4 * 1024 * 1024:
            raise RuntimeError("Hermes response exceeds limit")
        return json.loads(raw)


def healthy(settings, native):
    interfaces = interface_settings(settings)
    try:
        if not configuration_matches(native):
            return False
        if api_request("/v1/models", port=interfaces["apiPort"])["data"][0]["id"] != "primary":
            return False
        dashboard = interfaces["dashboard"]
        if dashboard["enabled"]:
            home = ROOT / "profiles/dashboard-home"
            if not configuration_matches(native, home):
                return False
            request = urllib.request.Request(f"http://127.0.0.1:{dashboard['port']}/api/sessions?limit=1", headers={"X-Hermes-Session-Token": token(home)})
            with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=3) as response:
                return response.status == 200
        return True
    except (OSError, ValueError, KeyError, IndexError, RuntimeError):
        return False


async def native_server():
    from gateway.config import PlatformConfig
    from gateway.platforms.api_server import APIServerAdapter

    settings = json.loads(os.environ["FABRIC_HERMES_INTERFACES"])
    dashboard = settings["dashboard"]
    adapter = APIServerAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "host": "127.0.0.1",
                "port": settings["apiPort"],
                "key": token(ROOT),
                "model_name": "primary",
            },
        )
    )
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
    child = None
    forwarder = None
    clients = set()

    async def relay(reader, writer):
        task = asyncio.current_task()
        clients.add(task)
        target_writer = None

        async def copy(source, destination):
            while data := await source.read(65536):
                destination.write(data)
                await destination.drain()

        try:
            if len(clients) > 128:
                return
            target_reader, target_writer = await asyncio.open_connection(
                "127.0.0.1", dashboard["internalPort"]
            )
            a = asyncio.create_task(copy(reader, target_writer))
            b = asyncio.create_task(copy(target_reader, writer))
            try:
                await asyncio.wait([a, b], return_when=asyncio.FIRST_COMPLETED)
            finally:
                a.cancel()
                b.cancel()
                await asyncio.gather(a, b, return_exceptions=True)
        except OSError:
            pass
        finally:
            writer.close()
            if target_writer:
                target_writer.close()
            clients.discard(task)

    try:
        if not await adapter.connect():
            raise RuntimeError("Hermes native API startup failed")
        if dashboard["enabled"]:
            home = ROOT / "profiles/dashboard-home"
            env = dict(
                os.environ,
                HERMES_HOME=str(home),
                HERMES_DASHBOARD_SESSION_TOKEN=token(home),
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
        if forwarder:
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


def native_dashboard():
    from hermes_cli import web_server

    settings = json.loads(os.environ["FABRIC_HERMES_INTERFACES"])["dashboard"]
    # Pinned Hermes exposes this shared gate for browser chat and WebSocket endpoints.
    web_server._DASHBOARD_EMBEDDED_CHAT_ENABLED = settings["tui"]["enabled"]
    web_server.start_server(
        host="127.0.0.1", port=settings["internalPort"], open_browser=False
    )


class HermesServiceRuntime:
    def __init__(self):
        self.process = None
        self.log = None
        self.state_lock = None
        self.failed = False
        self.lock = asyncio.Lock()
        self.previous_response = None

    async def start(self, payload):
        global ROOT
        config = payload["config"]
        self.settings = config.harness.settings if config.harness else {}
        self.runtime_id = payload["runtime_context"]["runtime_id"]
        workspace = (
            payload["runtime_context"]["environment"].get("workspace")
            or payload["base_dir"]
        )
        ROOT = Path(
            self.settings.get("home", str(Path(payload["base_dir"]) / ".hermes"))
        )
        self.native = build_hermes_config(config, workspace=workspace)
        model = _selected_model(config)
        key = os.environ[_api_key_env(model)]
        ROOT.mkdir(parents=True, mode=0o700, exist_ok=True)
        self.state_lock = (ROOT / "adapter.lock").open("a")
        try:
            fcntl.flock(self.state_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            initialize(self.native)
            interfaces = interface_settings(self.settings)
            if interfaces["dashboard"]["enabled"]:
                initialize(self.native, ROOT / "profiles/dashboard-home")
            self.log = (ROOT / "api.log").open("ab")
            env = dict(
                os.environ,
                HERMES_HOME=str(ROOT),
                OPENAI_API_KEY=key,
                HERMES_DISABLE_LAZY_INSTALLS="1",
                FABRIC_HERMES_INTERFACES=json.dumps(interfaces),
            )
            if model.base_url:
                env["OPENAI_BASE_URL"] = model.base_url
            telemetry = payload["runtime_context"].get("telemetry") or {}
            if telemetry.get("relay_enabled"):
                from nemo_fabric_adapters.hermes.telemetry import (
                    HERMES_RELAY_ENV_NAMES,
                    write_hermes_relay_plugin_config,
                )

                path, _ = write_hermes_relay_plugin_config(
                    {**payload, "config": config.to_mapping()}
                )
                for name in HERMES_RELAY_ENV_NAMES:
                    env.pop(name, None)
                env["HERMES_NEMO_RELAY_PLUGINS_TOML"] = str(path)
            self.process = await asyncio.create_subprocess_exec(
                sys.executable,
                __file__,
                "server",
                env=env,
                cwd=workspace,
                stdout=self.log,
                stderr=self.log,
                start_new_session=True,
            )
            deadline = asyncio.get_running_loop().time() + 75
            while asyncio.get_running_loop().time() < deadline:
                if self.process.returncode is not None:
                    raise RuntimeError(
                        "Hermes API exited during startup; inspect api.log"
                    )
                if await asyncio.to_thread(healthy, self.settings, self.native):
                    return
                await asyncio.sleep(0.1)
            raise RuntimeError("Hermes API readiness timed out")
        except BaseException:
            await self.stop()
            raise

    async def invoke(self, request, context):
        from nemo_fabric_adapter_contract.models import (
            AgentRunError,
            AgentRunResult,
            AgentRunStatus,
        )

        async with self.lock:
            if (
                self.failed
                or context.runtime_id != self.runtime_id
                or self.process.returncode is not None
            ):
                raise RuntimeError("Hermes runtime unavailable; no replay")
            if not isinstance(request.input, str):
                raise ValueError("Hermes requires a text prompt")
            if not configuration_matches(self.native):
                raise RuntimeError("Hermes configuration drifted")
            body = {
                "model": "primary",
                "input": request.input,
                "store": True,
            }
            if self.previous_response:
                body["previous_response_id"] = self.previous_response
            try:
                result = await asyncio.to_thread(
                    api_request,
                    "/v1/responses",
                    body,
                    280,
                    interface_settings(self.settings)["apiPort"],
                )
                if result.get("status") != "completed" or not result.get("id"):
                    raise RuntimeError("Hermes returned an unsuccessful response")
                self.previous_response = result["id"]
                text = "\n".join(
                    c["text"]
                    for item in result.get("output", [])
                    for c in item.get("content", [])
                    if c.get("type") == "output_text"
                )
                return AgentRunResult(
                    status=AgentRunStatus.SUCCEEDED,
                    output={
                        "harness": "hermes",
                        "response": text,
                        "response_id": result["id"],
                    },
                )
            except BaseException as error:
                self.failed = True
                await self.stop()
                if not isinstance(error, Exception):
                    raise
                return AgentRunResult(
                    status=AgentRunStatus.FAILED,
                    output={"harness": "hermes"},
                    error=AgentRunError(
                        code="hermes_invocation_failed",
                        message="Hermes invocation failed; no replay; inspect api.log",
                    ),
                )

    async def stop(self):
        if self.process is not None and self.process.returncode is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
                await asyncio.wait_for(self.process.wait(), 10)
            except TimeoutError:
                os.killpg(self.process.pid, signal.SIGKILL)
                await self.process.wait()
            except ProcessLookupError:
                await self.process.wait()
        if self.log:
            self.log.close()
        if self.state_lock:
            self.state_lock.close()
            self.state_lock = None


if __name__ == "__main__":
    if sys.argv[1:] == ["server"]:
        asyncio.run(native_server())
    elif sys.argv[1:] == ["dashboard"]:
        native_dashboard()
