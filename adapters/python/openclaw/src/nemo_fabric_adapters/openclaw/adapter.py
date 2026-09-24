# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Persistent OpenClaw gateway driven by Fabric's public AgentConfig."""

import asyncio
import copy
import fcntl
import json
import os
import re
import signal
from pathlib import Path

from nemo_fabric_adapter_contract.models import (
    AgentConfig,
    AgentRunError,
    AgentRunResult,
    AgentRunStatus,
)
from nemo_fabric_adapters.common import lifecycle
from nemo_fabric_adapters.common.credentials import token


def native_configuration(config, workspace):
    settings = config.harness.settings if config.harness else {}
    result = copy.deepcopy(settings.get("native_config", {}))
    if "models" in result:
        raise ValueError("native_config.models conflicts with Fabric models")
    if not config.models or "default" not in config.models:
        raise ValueError("OpenClaw requires models.default")
    name = settings.get("agent_name", "main")
    if not re.fullmatch(r"[a-z][a-z0-9-]{0,39}", name):
        raise ValueError("invalid OpenClaw agent_name")
    agents = result.setdefault("agents", {})
    if "entries" in agents or "ownership" in agents:
        raise ValueError("native_config.agents entries and ownership are Fabric-owned")
    defaults = agents.setdefault("defaults", {})
    for field in ("model", "workspace", "timeoutSeconds"):
        if field in defaults:
            raise ValueError(
                f"native_config.agents.defaults.{field} conflicts with Fabric config"
            )
    defaults.update(
        {
            "workspace": str(workspace),
            "timeoutSeconds": settings.get("timeout_seconds", 260),
        }
    )
    defaults.setdefault("sandbox", {"mode": "off"})
    providers = {}
    choices = {}
    for role, model in config.models.items():
        api = model.extensions.get("api", model.settings.get("api"))
        if api not in ("openai-completions", "openai-responses", "anthropic-messages"):
            raise ValueError("OpenClaw requires each model.settings.api")
        if not model.base_url:
            raise ValueError("OpenClaw requires each model.base_url")
        key = "fabric_" + role
        providers[key] = {
            "baseUrl": model.base_url,
            "api": api,
            "apiKey": "${" + model.api_key_env + "}" if model.api_key_env else "unused",
            "timeoutSeconds": defaults["timeoutSeconds"],
            "models": [
                {
                    "name": model.model,
                    "input": ["text"],
                    "contextWindow": 32768,
                    "maxTokens": model.max_tokens or 4096,
                    **model.settings.get("model_metadata", {}),
                    "id": model.model,
                }
            ],
        }
        choices[key + "/" + model.model] = {"alias": role}
    effort = config.models["default"].settings.get("reasoning_effort")
    if effort is not None:
        defaults["thinkingDefault"] = effort
    primary = "fabric_default/" + config.models["default"].model
    entry = {
        "workspace": str(workspace),
        "model": {"primary": primary},
        "models": choices,
        "modelPolicy": {"allow": list(choices)},
    }
    if config.tools is not None:
        entry["tools"] = {}
        if config.tools.enabled is not None:
            entry["tools"]["allow"] = config.tools.enabled
        if config.tools.blocked:
            entry["tools"]["deny"] = config.tools.blocked
    agents.update({"entries": {name: entry}, "ownership": "explicit"})
    result["models"] = {"mode": "replace", "providers": providers}
    gateway = result.setdefault("gateway", {})
    for key, value in {
        "mode": "local",
        "bind": "loopback",
        "port": 18789,
        "auth": {"mode": "none"},
        "controlUi": {"enabled": False},
    }.items():
        gateway.setdefault(key, value)
    if gateway["controlUi"].get("enabled"):
        if gateway["auth"] == {"mode": "none"}:
            gateway["auth"] = {"mode": "token", "token": "${FABRIC_INTERFACE_TOKEN}"}
        gateway["controlUi"].setdefault(
            "allowedOrigins",
            [
                f"http://127.0.0.1:{gateway['port']}",
                f"http://localhost:{gateway['port']}",
            ],
        )
    if gateway["bind"] != "loopback" and gateway["auth"].get("mode") == "none":
        raise ValueError("non-loopback OpenClaw gateway requires authentication")
    result.setdefault("cron", {"enabled": False})
    result.setdefault("update", {"checkOnStart": False, "auto": {"enabled": False}})
    return result


def configuration_matches(actual, expected):
    # Exact equality of declared sections rejects added tools, routes, and agents;
    # unrelated native bookkeeping may be retained outside the declared sections.
    return isinstance(actual, dict) and all(
        actual.get(key) == value for key, value in expected.items()
    )


def normalize_messages(messages):
    normalized = []
    for message in messages:
        role = message.get("role")
        content = message.get("content", [])
        blocks = content if isinstance(content, list) else []
        text = (
            content
            if isinstance(content, str)
            else "\n".join(
                b["text"]
                for b in blocks
                if b.get("type") == "text" and isinstance(b.get("text"), str)
            )
        )
        item = {"role": "tool" if role == "toolResult" else role, "content": text}
        if role == "assistant":
            item["tool_calls"] = [
                {
                    "id": b["id"],
                    "type": "function",
                    "function": {
                        "name": b["name"],
                        "arguments": json.dumps(b.get("arguments", {})),
                    },
                }
                for b in blocks
                if b.get("type") == "toolCall"
            ]
        if role == "toolResult":
            item["tool_call_id"] = message.get("toolCallId", "")
            item["is_error"] = message.get("isError", False)
        normalized.append(item)
    return normalized


class OpenClawRuntime:
    def __init__(self):
        self.process = None
        self.log = None
        self.failed = False
        self.lock = asyncio.Lock()
        self.state_lock = None

    async def start(self, payload):
        config = payload["config"]
        context = payload["runtime_context"]
        settings = config.harness.settings if config.harness else {}
        self.runtime_id = context["runtime_id"]
        self.name = settings.get("agent_name", "main")
        workspace = context["environment"].get("workspace") or payload["base_dir"]
        self.native = native_configuration(config, workspace)
        self.home = Path(
            settings.get("home", str(Path(payload["base_dir"]) / ".openclaw"))
        )
        self.cli = settings.get("cli", "/app/openclaw.mjs")
        self.home.mkdir(parents=True, mode=0o700, exist_ok=True)
        self.state_lock = open(self.home / "adapter.lock", "a")
        try:
            fcntl.flock(self.state_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.state_lock.close()
            self.state_lock = None
            raise RuntimeError("retained channel state is already in use") from None
        try:
            path = self.home / "openclaw.json"
            if path.exists() and not configuration_matches(
                json.loads(path.read_text()), self.native
            ):
                raise RuntimeError(
                    "native configuration conflicts with Fabric-owned settings"
                )
            if not path.exists():
                with open(
                    path, "x", opener=lambda p, flags: os.open(p, flags, 0o600)
                ) as output:
                    json.dump(self.native, output)
                    output.flush()
                    os.fsync(output.fileno())
            self.env = dict(
                os.environ,
                OPENCLAW_HOME=str(self.home.parent),
                OPENCLAW_STATE_DIR=str(self.home),
                OPENCLAW_CONFIG_PATH=str(path),
            )
            if (
                self.native["gateway"]["auth"].get("token")
                == "${FABRIC_INTERFACE_TOKEN}"
            ):
                self.env["FABRIC_INTERFACE_TOKEN"] = token(self.home, create=True)
                self.env["OPENCLAW_GATEWAY_TOKEN"] = self.env["FABRIC_INTERFACE_TOKEN"]
            self.log = (self.home / "gateway.log").open("ab")
            self.process = await asyncio.create_subprocess_exec(
                "node",
                self.cli,
                "gateway",
                env=self.env,
                cwd=workspace,
                stdout=self.log,
                stderr=self.log,
                start_new_session=True,
            )
            deadline = asyncio.get_running_loop().time() + 75
            while asyncio.get_running_loop().time() < deadline:
                if self.process.returncode is not None:
                    raise RuntimeError("OpenClaw gateway exited during startup")
                try:
                    await self.rpc("health", {}, timeout=3)
                    return
                except (OSError, ValueError, RuntimeError, TimeoutError):
                    await asyncio.sleep(0.1)
            raise RuntimeError("OpenClaw gateway did not become healthy")
        except BaseException:
            await self.stop()
            raise

    async def rpc(self, method, params, timeout=280):
        process = await asyncio.create_subprocess_exec(
            "node",
            self.cli,
            "gateway",
            "call",
            method,
            "--params",
            json.dumps(params),
            "--expect-final",
            "--json",
            "--timeout",
            str(timeout * 1000),
            env=self.env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout + 5)
        except BaseException:
            if process.returncode is None:
                os.killpg(process.pid, signal.SIGKILL)
            await process.wait()
            raise
        if process.returncode:
            raise RuntimeError(
                f"OpenClaw RPC {method} failed: {stderr.decode(errors='replace')[-2000:]}"
            )
        return json.loads(stdout)

    async def invoke(self, request, context):
        async with self.lock:
            return await self._invoke(request, context)

    async def _invoke(self, request, context):
        if (
            self.failed
            or context.runtime_id != self.runtime_id
            or self.process.returncode is not None
        ):
            raise lifecycle.LifecycleError(
                "openclaw_runtime_unavailable",
                "OpenClaw runtime is unavailable; no replay",
            )
        entries = self.native["agents"]["entries"]
        name = next(iter(entries))
        message = request.input
        if isinstance(message, dict) and set(message) == {"agent", "message"}:
            name, message = message["agent"], message["message"]
        if (
            not isinstance(message, str)
            or not isinstance(name, str)
            or name not in entries
        ):
            raise ValueError("expected text or a declared agent and text message")
        if not configuration_matches(
            json.loads((self.home / "openclaw.json").read_text()), self.native
        ):
            raise lifecycle.LifecycleError(
                "openclaw_configuration_drift", "deployment-owned configuration drifted"
            )
        session_key = f"agent:{name}:fabric-{self.runtime_id}"
        try:
            seconds = self.native["agents"]["defaults"]["timeoutSeconds"]
            result = await self.rpc(
                "agent",
                {
                    "agentId": name,
                    "sessionKey": session_key,
                    "message": message,
                    "idempotencyKey": context.invocation_id,
                    "deliver": False,
                    "timeout": seconds,
                },
                timeout=seconds + 20,
            )
            if result.get("status") != "ok":
                raise RuntimeError("OpenClaw returned a non-successful terminal result")
            native = result.get("result", {})
            if (
                native.get("meta", {}).get("aborted")
                or native.get("meta", {}).get("error")
                or any(p.get("isError") for p in native.get("payloads", []))
            ):
                raise RuntimeError("OpenClaw agent turn aborted or failed")
            response = "\n".join(
                p["text"]
                for p in native.get("payloads", [])
                if isinstance(p.get("text"), str)
            )
            history = await self.rpc(
                "chat.history", {"sessionKey": session_key, "limit": 200}, timeout=15
            )
            return AgentRunResult(
                status=AgentRunStatus.SUCCEEDED,
                output={
                    "harness": "openclaw",
                    "response": response,
                    "session_key": session_key,
                    "gateway_pid": self.process.pid,
                    "messages": normalize_messages(history.get("messages", [])),
                    "native_result": result,
                },
            )
        except BaseException as error:
            # An RPC failure can leave the gateway still working on the turn.
            # Quarantine and stop it, rather than accepting overlapping requests.
            self.failed = True
            await self.stop()
            if not isinstance(error, Exception):
                raise
            return AgentRunResult(
                status=AgentRunStatus.FAILED,
                output={"harness": "openclaw", "session_key": session_key},
                error=AgentRunError(
                    code="openclaw_invocation_failed", message=str(error)
                ),
            )

    async def stop(self):
        await self.stop_gateway()
        if self.state_lock is not None:
            self.state_lock.close()
            self.state_lock = None

    async def stop_gateway(self):
        if self.process is not None and self.process.returncode is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
                await asyncio.wait_for(self.process.wait(), 10)
            except TimeoutError:
                os.killpg(self.process.pid, signal.SIGKILL)
                await self.process.wait()
            except ProcessLookupError:
                await self.process.wait()
        if self.log is not None:
            self.log.close()


if __name__ == "__main__":
    lifecycle.serve(OpenClawRuntime, config_loader=AgentConfig.from_mapping)
