#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Goose CLI adapter for NVIDIA NeMo Fabric."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import signal
import sys
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Any

from nemo_fabric_adapter_contract import models as contract
from nemo_fabric_adapters.common import instructions as common_instructions
from nemo_fabric_adapters.common import lifecycle
from nemo_fabric_adapters.common import utils as common_utils

HARNESS = "Goose"
SESSION_NAME = "nemo-fabric"
# Goose's own text when --max-turns ends a run.
MAX_TURNS_NOTICE = (
    "I've reached the maximum number of actions I can do without user input"
)
# Goose logs an extension that fails to start to stderr and continues without it.
FAILED_EXTENSION = re.compile(r"Failed to start extension '([^']+)'")
MCP_TIMEOUT_SECONDS = 300
STOP_GRACE_SECONDS = 5.0
# Parent settings that would override the normalized model or system instruction.
OVERRIDDEN_ENV = (
    "OPENAI_BASE_URL",
    "OPENAI_HOST",
    "OPENAI_BASE_PATH",
    "GOOSE_TEMPERATURE",
    "GOOSE_SYSTEM_PROMPT_FILE_PATH",
)
USAGE_FIELDS = (
    "input_tokens",
    "cache_read_input_tokens",
    "output_tokens",
    "total_tokens",
)


def _selected_model(config: contract.AgentConfig) -> contract.AgentModelConfig:
    model = config.models.get("default")
    if model is None and len(config.models) == 1:
        model = next(iter(config.models.values()))
    if model is None:
        raise lifecycle.LifecycleError(
            "goose_model_required",
            "Goose requires a default model or exactly one configured model.",
        )
    if model.provider != "openai" and model.base_url is None:
        # Goose's OpenAI provider serves every model, so another provider needs its endpoint.
        raise lifecycle.LifecycleError(
            "goose_base_url_required",
            f"Goose requires base_url for the OpenAI-compatible provider {model.provider!r}.",
        )
    return model


def _resolve_command(settings: dict[str, Any], base_dir: Path) -> Path:
    configured = settings.get("goose_command", "goose")
    if not isinstance(configured, str) or not configured.strip():
        raise lifecycle.LifecycleError(
            "goose_invalid_configuration",
            "Goose goose_command must be a non-empty string.",
            metadata={"field": "harness.settings.goose_command"},
        )
    value = configured.strip()
    command = Path(value)
    # A bare name is looked up on PATH. A relative path with a directory,
    # including ./goose, resolves from the base directory.
    if not command.is_absolute() and os.path.dirname(value):
        command = base_dir / command
    found = shutil.which(str(command))
    if found is None:
        raise lifecycle.LifecycleError(
            "goose_command_not_found",
            f"Goose command {configured!r} was not found or is not executable.",
            metadata={"field": "harness.settings.goose_command"},
        )
    return Path(found).resolve()


def _workspace(context: contract.RuntimeContext, base_dir: str) -> Path:
    path = Path(context.environment.workspace or base_dir)
    if not path.is_absolute():
        path = Path(base_dir) / path
    path = path.resolve()
    if not path.is_dir():
        raise lifecycle.LifecycleError(
            "goose_workspace_invalid",
            f"The Goose workspace is not a directory: {path}",
        )
    return path


def _mcp_extensions(config: contract.AgentConfig) -> dict[str, dict[str, Any]]:
    """Map MCP servers to Goose extensions, which Goose loads beside its defaults."""

    extensions: dict[str, dict[str, Any]] = {}
    servers = config.mcp.servers if config.mcp is not None else {}
    for name, server in servers.items():
        if (
            server.authentication is not None
            or server.allowed_tools is not None
            or server.blocked_tools
        ):
            raise lifecycle.LifecycleError(
                "goose_mcp_configuration_unsupported",
                f"Goose does not support MCP authentication or tool policy for server {name!r}.",
            )
        extension: dict[str, Any] = {
            "enabled": True,
            "name": name,
            "description": "",
            "timeout": MCP_TIMEOUT_SECONDS,
        }
        transport = server.transport.strip().lower()
        if transport == "stdio" and not server.custom_headers:
            extension.update(
                type="stdio",
                cmd=os.path.expandvars(server.url),
                args=server.args,
                envs=server.env,
            )
        elif (
            transport in {"http", "streamable-http", "streamable_http"}
            and not server.args
        ):
            try:
                headers = common_utils.expand_http_headers(name, server.custom_headers)
            except (TypeError, ValueError) as error:
                raise lifecycle.LifecycleError(
                    "goose_mcp_header_invalid",
                    f"MCP headers are invalid for server {name!r}.",
                ) from error
            extension.update(
                type="streamable_http",
                uri=os.path.expandvars(server.url),
                headers=headers,
            )
        else:
            raise lifecycle.LifecycleError(
                "goose_mcp_configuration_unsupported",
                f"Goose supports stdio and streamable HTTP MCP servers; server {name!r} "
                f"uses {server.transport!r} with fields Goose cannot apply.",
            )
        extensions[name] = extension
    return extensions


def _events(stdout: bytes) -> list[dict[str, Any]]:
    events = []
    for line in stdout.decode(errors="replace").splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue  # Goose prints a banner before its JSON events.
        if isinstance(event, dict):
            events.append(event)
    return events


def _text(message: dict[str, Any]) -> str:
    return "".join(
        block.get("text") or ""
        for block in message.get("content") or []
        if isinstance(block, dict) and block.get("type") == "text"
    )


def _final_message(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the last assistant message, joining the chunks Goose streams under one ID."""

    final: dict[str, Any] | None = None
    for event in events:
        message = event.get("message")
        if event.get("type") != "message" or not isinstance(message, dict):
            continue
        if message.get("role") != "assistant":
            final = None
        elif final is not None and message.get("id") == final["id"]:
            final["text"] += _text(message)
        else:
            final = {
                "id": message.get("id"),
                "text": _text(message),
                # Model output carries inference metadata. Goose's own notices,
                # such as the turn limit and model request errors, do not.
                "from_model": "inference" in (message.get("metadata") or {}),
            }
    return final


def _kill_sessions(sessions: list[int]) -> None:
    """Kill the tool commands that exited Goose runs left in their sessions.

    Goose starts each shell command in its own process group, so on Linux this
    kills every process in each run's session. Other platforms reach only the
    run's process group. The kernel keeps a session ID allocated while any
    process belongs to the session, so a live process with that ID means the
    session ended and the ID was reused.
    """

    sessions = [session for session in sessions if not _pid_exists(session)]
    if sys.platform != "linux":
        for session in sessions:
            with suppress(ProcessLookupError, PermissionError):
                os.killpg(session, signal.SIGKILL)
        return
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            # The fields after the command name start with state, ppid, pgrp, session.
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[3]) in sessions:
                os.kill(int(entry.name), signal.SIGKILL)
        except (OSError, IndexError, ValueError):
            continue


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    return True


async def _terminate(process: asyncio.subprocess.Process) -> None:
    """Stop Goose, reap it, then kill the tool commands it started."""

    with suppress(ProcessLookupError):
        process.terminate()
    try:
        await asyncio.wait_for(process.wait(), STOP_GRACE_SECONDS)
    except TimeoutError:
        with suppress(ProcessLookupError):
            process.kill()
        await process.wait()
    if os.name == "posix":
        _kill_sessions([process.pid])


def _failed(
    code: str, message: str, usage: contract.AgentUsage | None = None
) -> contract.AgentRunResult:
    return contract.AgentRunResult(
        status=contract.AgentRunStatus.FAILED,
        output=None,
        error=contract.AgentRunError(code=code, message=message),
        usage=usage,
    )


class GooseRuntime:
    def __init__(self) -> None:
        self._command: Path | None = None
        self._root: tempfile.TemporaryDirectory[str] | None = None
        self._workspace: Path | None = None
        self._args: list[str] = []
        self._env: dict[str, str] = {}
        self._resume = False
        # Each run's session ID, which tool commands it left running still carry.
        self._sessions: list[int] = []
        # Goose reports cumulative session totals. None means the last run lost them.
        self._usage_totals: dict[str, int] | None = dict.fromkeys(USAGE_FIELDS, 0)

    async def start(self, payload: dict[str, Any]) -> None:
        config: contract.AgentConfig = payload["config"]
        context = contract.RuntimeContext.from_mapping(payload.get("runtime_context"))
        base_dir = common_utils.base_dir(payload)
        instruction = common_instructions.system_instruction(
            config, adapter=HARNESS, supported_modes={"replace", "append"}
        )
        model = _selected_model(config)
        settings = config.harness.settings if config.harness is not None else {}
        command = _resolve_command(settings, Path(base_dir))
        workspace = _workspace(context, base_dir)
        extensions = _mcp_extensions(config)
        env = {
            name: value
            for name, value in os.environ.items()
            if name not in OVERRIDDEN_ENV
        }
        if model.api_key_env is not None:
            api_key = os.environ.get(model.api_key_env)
            if not api_key:
                raise lifecycle.LifecycleError(
                    "goose_credential_missing",
                    f"The model credential environment variable {model.api_key_env!r} is not set.",
                )
            env["OPENAI_API_KEY"] = api_key
        if model.base_url is not None:
            env["OPENAI_BASE_URL"] = model.base_url
        if model.temperature is not None:
            env["GOOSE_TEMPERATURE"] = str(model.temperature)

        args = ["run", "--name", SESSION_NAME, "--output-format", "stream-json"]
        if config.runtime is not None and config.runtime.max_turns is not None:
            args += ["--max-turns", str(config.runtime.max_turns)]
        root = tempfile.TemporaryDirectory(prefix="nemo-fabric-goose-")
        try:
            # Goose keeps its config, sessions, and logs below this per-runtime root.
            config_dir = Path(root.name) / "config"
            config_dir.mkdir()
            # Goose reads config.yaml as YAML, and JSON is valid YAML.
            (config_dir / "config.yaml").write_text(
                json.dumps({"extensions": extensions}), encoding="utf-8"
            )
            if instruction is not None and instruction.mode == "replace":
                prompt_path = Path(root.name) / "system.md"
                prompt_path.write_text(instruction.content, encoding="utf-8")
                env["GOOSE_SYSTEM_PROMPT_FILE_PATH"] = str(prompt_path)
            elif instruction is not None:
                # The = form keeps an instruction that starts with "-" from parsing as an option.
                args.append(f"--system={instruction.content}")
        except BaseException:
            root.cleanup()
            raise
        env.update(
            {
                "GOOSE_PATH_ROOT": root.name,
                "GOOSE_PROVIDER": "openai",
                "GOOSE_MODEL": model.model,
                # Headless runs cannot answer tool approval prompts.
                "GOOSE_MODE": "auto",
                "GOOSE_DISABLE_KEYRING": "1",
                "GOOSE_TELEMETRY_OFF": "1",
            }
        )
        self._command = command
        self._root = root
        self._workspace = workspace
        self._args = [*args, "--instructions", "-"]
        self._env = env

    async def invoke(
        self,
        request: contract.AgentRunRequest,
        context: contract.RuntimeContext,
    ) -> contract.AgentRunResult:
        if self._command is None:
            raise lifecycle.LifecycleError(
                "goose_not_started", "The Goose runtime is not started."
            )
        deadline = lifecycle.invocation_deadline(context)
        if not self._resume and self._sessions:
            self._resume = await self._session_exists(deadline)
        process = await asyncio.create_subprocess_exec(
            str(self._command),
            *self._args,
            *(["--resume"] if self._resume else []),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self._workspace,
            env=self._env,
            start_new_session=os.name == "posix",
        )
        self._sessions.append(process.pid)
        prompt = common_utils.normalize_user_input(request.input).encode()
        try:
            async with asyncio.timeout_at(deadline):
                stdout, stderr = await process.communicate(prompt)
        except TimeoutError:
            await _terminate(process)
            self._usage_totals = None
            return _failed(
                "goose_timeout", "Goose did not finish before the invocation deadline."
            )
        except BaseException:
            await _terminate(process)
            raise

        events = _events(stdout)
        self._resume = self._resume or process.returncode == 0
        usage = self._usage(
            next((e for e in reversed(events) if e.get("type") == "complete"), None)
        )
        if process.returncode != 0:
            error = next(
                (e for e in reversed(events) if e.get("type") == "error"), None
            )
            detail = (
                str(error.get("error"))
                if error is not None
                else stderr.decode(errors="replace").strip()[-2000:]
            )
            return _failed(
                "goose_exit_failure",
                f"Goose exited with code {process.returncode}: {detail}",
                usage,
            )
        failed_extensions = sorted(
            set(FAILED_EXTENSION.findall(stderr.decode(errors="replace")))
        )
        if failed_extensions:
            return _failed(
                "goose_extension_failed",
                f"Goose could not start extensions: {', '.join(failed_extensions)}",
                usage,
            )
        final = _final_message(events)
        if final is None or not final["text"]:
            return _failed(
                "goose_no_response",
                "Goose finished without an assistant response.",
                usage,
            )
        if not final["from_model"]:
            code = (
                "goose_max_turns_reached"
                if final["text"].startswith(MAX_TURNS_NOTICE)
                else "goose_model_error"
            )
            return _failed(code, final["text"], usage)
        return contract.AgentRunResult(
            status=contract.AgentRunStatus.SUCCEEDED,
            output={"response": final["text"]},
            usage=usage,
        )

    async def _session_exists(self, deadline: float) -> bool:
        """Return whether a failed earlier run created this runtime's session.

        Goose creates the named session before its first model call, so a run
        can fail or time out after creating it without reporting any event.
        """

        process = await asyncio.create_subprocess_exec(
            str(self._command),
            "session",
            "list",
            "--format",
            "json",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            cwd=self._workspace,
            env=self._env,
        )
        try:
            async with asyncio.timeout_at(deadline):
                stdout, _ = await process.communicate()
        except BaseException as error:
            with suppress(ProcessLookupError):
                process.kill()
            await process.wait()
            if isinstance(error, TimeoutError):
                return False
            raise
        try:
            sessions = json.loads(stdout)
        except ValueError:
            return False
        return isinstance(sessions, list) and any(
            isinstance(session, dict) and session.get("name") == SESSION_NAME
            for session in sessions
        )

    def _usage(self, complete: dict[str, Any] | None) -> contract.AgentUsage | None:
        """Return this run's usage as the change in Goose's session totals."""

        totals = None
        if complete is not None and all(
            isinstance(complete.get(name), int) for name in USAGE_FIELDS
        ):
            totals = {name: complete[name] for name in USAGE_FIELDS}
        baseline, self._usage_totals = self._usage_totals, totals
        if totals is None or baseline is None:
            return None
        delta = {name: totals[name] - baseline[name] for name in USAGE_FIELDS}
        if any(value < 0 for value in delta.values()):
            return None
        return contract.AgentUsage(
            input_tokens=delta["input_tokens"],
            cached_input_tokens=delta["cache_read_input_tokens"],
            # Goose counts OpenAI prompt tokens, which include cached tokens.
            input_tokens_include_cache=True,
            output_tokens=delta["output_tokens"],
            total_tokens=delta["total_tokens"],
        )

    async def stop(self) -> None:
        root, sessions = self._root, self._sessions
        self.__init__()
        if os.name == "posix":
            _kill_sessions(sessions)
        if root is not None:
            root.cleanup()


def main() -> None:
    lifecycle.serve(GooseRuntime, config_loader=contract.AgentConfig.from_mapping)


# Selected software reported by the shared lifecycle host; no dependency scan.
PROVENANCE_PACKAGES = (
    (
        "adapter",
        "nemo-fabric-adapters-goose",
        "nemo_fabric_adapters.goose.adapter",
    ),
)

if __name__ == "__main__":
    main()
