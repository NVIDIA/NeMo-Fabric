# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Invocation-scoped Hermes execution through the public NeMo Relay CLI."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import os
import re
import shutil
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nemo_fabric_adapters.common.utils as common_utils
from nemo_fabric_adapters.common.relay_gateway import RelayCliContract


HERMES_MINIMUM_VERSION = (0, 18, 2)
HERMES_MAXIMUM_VERSION = (0, 19, 0)
HERMES_VERSION_TIMEOUT_SECONDS = 5.0
PROCESS_STOP_TIMEOUT_SECONDS = 10.0
MAX_CAPTURE_BYTES = 4 * 1024 * 1024
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
INHERITED_ENV_NAMES = {
    "APPDATA",
    "COMSPEC",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "LOCALAPPDATA",
    "NO_PROXY",
    "PATH",
    "PATHEXT",
    "SHELL",
    "SSL_CERT_DIR",
    "SSL_CERT_FILE",
    "SYSTEMROOT",
    "TEMP",
    "TMP",
    "TMPDIR",
    "USER",
    "USERPROFILE",
    "XDG_CACHE_HOME",
    "XDG_DATA_HOME",
    "http_proxy",
    "https_proxy",
    "no_proxy",
}


class HermesRelayError(RuntimeError):
    """Hermes/Relay CLI lifecycle or contract failure."""


@dataclass(frozen=True)
class HermesRelayLaunch:
    """Stable runtime inputs used to execute invocation-scoped Relay runs."""

    relay_executable: Path
    relay_contract: RelayCliContract
    hermes_executable: Path
    hermes_config_path: Path
    hermes_home: Path
    cwd: Path
    env: dict[str, str]
    base_url: str
    model: str
    runtime_id: str
    settings: dict[str, Any]
    model_config: dict[str, Any]
    plugin_config: dict[str, Any]


@dataclass(frozen=True)
class HermesRelayResult:
    """Bounded output and evidence from one Relay-owned Hermes process."""

    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    stdout_path: Path
    stderr_path: Path
    config_path: Path
    plugin_config_path: Path
    plugin_config: dict[str, Any]
    truncated: bool


def resolve_executable(root: Path, value: str | Path, *, name: str) -> Path:
    """Resolve a configured executable without invoking a shell."""

    command = Path(value)
    if len(command.parts) == 1:
        resolved = shutil.which(str(command))
    else:
        candidate = command if command.is_absolute() else root / command
        resolved = shutil.which(str(candidate.resolve()))
    if resolved is None:
        raise HermesRelayError(f"{name} executable was not found")
    return Path(resolved).resolve()


def hermes_cli_version(executable: Path) -> tuple[int, int, int]:
    """Validate the Hermes CLI features consumed by the gateway path."""

    try:
        completed = subprocess.run(
            [str(executable), "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=HERMES_VERSION_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise HermesRelayError("Hermes CLI version could not be determined") from error
    # Hermes renders the package version as ``v0.18.2`` before a separate
    # calendar build version. A word-boundary regex skips the package version
    # because both ``v`` and ``0`` are word characters.
    match = re.search(r"(?<!\d)(\d+)\.(\d+)\.(\d+)(?!\d)", completed.stdout)
    if completed.returncode != 0 or match is None:
        raise HermesRelayError("Hermes CLI version could not be determined")
    version = tuple(int(value) for value in match.groups())
    if not HERMES_MINIMUM_VERSION <= version < HERMES_MAXIMUM_VERSION:
        rendered = ".".join(str(value) for value in version)
        raise HermesRelayError(
            f"unsupported Hermes CLI version {rendered}; "
            "Fabric requires >=0.18.2,<0.19.0 for Relay execution"
        )
    return version


def validate_openai_upstream(
    settings: dict[str, Any], model_config: dict[str, Any], base_url: str | None
) -> str:
    """Reject provider configurations the transparent OpenAI route cannot preserve."""

    provider = str(
        settings.get("provider") or model_config.get("provider") or ""
    ).lower()
    if provider == "anthropic":
        raise HermesRelayError(
            "Hermes Relay execution requires an OpenAI-compatible provider endpoint"
        )
    if not base_url:
        raise HermesRelayError(
            "Hermes Relay execution requires an OpenAI-compatible base URL"
        )
    return base_url


def child_environment(
    settings: dict[str, Any],
    model_config: dict[str, Any],
    plugin_config: dict[str, Any],
    hermes_home: Path,
) -> dict[str, str]:
    """Build a least-privilege child environment without mutating Fabric.

    Relay and Hermes receive portable process variables, explicitly configured
    values, the selected model credential, and environment variables named by
    the Relay plugin configuration. Unrelated host credentials are not copied.
    """

    configured = settings.get("env") or {}
    if not isinstance(configured, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in configured.items()
    ):
        raise HermesRelayError("harness.settings.env must contain strings")
    inherited = common_utils.virtualenv_subprocess_env()
    names = set(INHERITED_ENV_NAMES)
    names.update(name for name in inherited if name.startswith("LC_"))
    names.update(_referenced_environment_names(plugin_config))
    api_key_env = str(
        settings.get("api_key_env")
        or model_config.get("api_key_env")
        or "NVIDIA_API_KEY"
    )
    names.add(api_key_env)
    env = {name: inherited[name] for name in names if name in inherited}
    for name in ("PATH", "VIRTUAL_ENV"):
        if name in inherited:
            env[name] = inherited[name]
    env.update(configured)
    if api_key_env in inherited:
        # Hermes is intentionally forced through its OpenAI-compatible custom
        # provider so Relay can own the gateway. Keep the source credential
        # available for plugin references and map it only in the child.
        env["OPENAI_API_KEY"] = inherited[api_key_env]
    env.update(
        {
            "HOME": str(hermes_home),
            "HERMES_HOME": str(hermes_home),
            "HERMES_YOLO_MODE": "1",
            "HERMES_ACCEPT_HOOKS": "1",
            "HERMES_SESSION_SOURCE": "fabric",
            "TERMINAL_ENV": str(settings.get("terminal_backend", "local")),
            "TERMINAL_TIMEOUT": str(settings.get("terminal_timeout", 60)),
            "XDG_CONFIG_HOME": str(hermes_home / ".config"),
        }
    )
    return env


def _referenced_environment_names(value: Any) -> set[str]:
    """Collect credential variable names from normalized Relay configuration."""

    names: set[str] = set()
    if isinstance(value, list):
        for item in value:
            names.update(_referenced_environment_names(item))
        return names
    if not isinstance(value, dict):
        return names
    for key, item in value.items():
        if key == "header_env" and isinstance(item, dict):
            names.update(
                name for name in item.values() if isinstance(name, str) and name
            )
        elif (key.endswith("_env") or key.endswith("_var")) and isinstance(item, str):
            if item:
                names.add(item)
        names.update(_referenced_environment_names(item))
    return names


def build_hermes_args(launch: HermesRelayLaunch, prompt: str) -> list[str]:
    """Build the Hermes portion passed after ``nemo-relay run --``."""

    args = [
        *common_utils.normalize_list(
            launch.settings.get("hermes_args") or launch.settings.get("command_args")
        ),
        "chat",
        "-Q",
        "--query",
        prompt,
        "--continue",
        launch.runtime_id,
        "--model",
        launch.model,
        "--provider",
        "custom",
    ]
    toolsets = common_utils.normalize_list(launch.settings.get("enabled_toolsets"))
    if toolsets:
        args.extend(["--toolsets", ",".join(toolsets)])
    return args


def redact_command(command: list[str]) -> list[str]:
    """Remove prompt contents from command evidence."""

    redacted: list[str] = []
    redact_next = False
    for argument in command:
        if redact_next:
            redacted.append("<prompt>")
            redact_next = False
        else:
            redacted.append(argument)
        if argument in {"--query", "--oneshot", "-z"}:
            redact_next = True
    return redacted


def quiet_response(stdout: str) -> str:
    """Parse Hermes 0.18.x's documented quiet response channel."""

    lines: list[str] = []
    for line in stdout.splitlines():
        plain = ANSI_ESCAPE.sub("", line).strip()
        # Hermes 0.18.2 emits this optional-security diagnostic on stdout
        # before it enters quiet mode. It is not part of the model response.
        if plain.startswith("⚠ tirith security scanner enabled but not available"):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


class HermesRelayRunner:
    """Own at most one Relay/Hermes process for a persistent Fabric runtime."""

    def __init__(self, launch: HermesRelayLaunch) -> None:
        self._launch = launch
        self._process: asyncio.subprocess.Process | None = None
        self._sequence = 0
        self._invoke_lock = asyncio.Lock()

    @property
    def relay_version(self) -> tuple[int, int, int]:
        return self._launch.relay_contract.version

    async def invoke(self, prompt: str, invocation_id: str) -> HermesRelayResult:
        async with self._invoke_lock:
            self._sequence += 1
            invocation_dir = (
                self._launch.hermes_home
                / "relay"
                / "invocations"
                / f"{_safe_id(invocation_id)}-{self._sequence:04d}"
            )
            config_path, plugin_config_path, plugin_config = _write_configs(
                self._launch, invocation_dir
            )
            command = [
                str(self._launch.relay_executable),
                "run",
                "--config",
                str(config_path),
                "--agent",
                "hermes",
                "--",
                *build_hermes_args(self._launch, prompt),
            ]
            stdout_path = invocation_dir / "stdout.log"
            stderr_path = invocation_dir / "stderr.log"
            try:
                preexec_fn = (
                    _linux_parent_death_signal
                    if sys.platform.startswith("linux")
                    else None
                )
                process = await asyncio.create_subprocess_exec(
                    *command,
                    cwd=self._launch.cwd,
                    env=self._launch.env,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    start_new_session=True,
                    preexec_fn=preexec_fn,
                )
            except OSError as error:
                raise HermesRelayError(
                    f"NeMo Relay could not start; see {stderr_path}"
                ) from error
            self._process = process
            stdout_task = asyncio.create_task(_drain(process.stdout))
            stderr_task = asyncio.create_task(_drain(process.stderr))
            try:
                returncode = await process.wait()
            except asyncio.CancelledError:
                try:
                    await _stop_process(process)
                finally:
                    await asyncio.gather(
                        stdout_task, stderr_task, return_exceptions=True
                    )
                raise
            finally:
                if self._process is process:
                    self._process = None
            stdout_bytes, stdout_truncated = await stdout_task
            stderr_bytes, stderr_truncated = await stderr_task
            stdout_path.write_bytes(stdout_bytes)
            stderr_path.write_bytes(stderr_bytes)
            return HermesRelayResult(
                command=redact_command(command),
                returncode=returncode,
                stdout=stdout_bytes.decode("utf-8", errors="replace"),
                stderr=stderr_bytes.decode("utf-8", errors="replace"),
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                config_path=config_path,
                plugin_config_path=plugin_config_path,
                plugin_config=plugin_config,
                truncated=stdout_truncated or stderr_truncated,
            )

    async def stop(self) -> None:
        process = self._process
        if process is not None:
            await _stop_process(process)


def _write_configs(
    launch: HermesRelayLaunch, invocation_dir: Path
) -> tuple[Path, Path, dict[str, Any]]:
    try:
        import tomli_w
    except ImportError as error:
        raise HermesRelayError("tomli_w is required for Relay execution") from error

    invocation_dir.mkdir(parents=True, exist_ok=False)
    config_path = invocation_dir / "config.toml"
    plugin_config_path = invocation_dir / "plugins.toml"
    plugin_config = copy.deepcopy(launch.plugin_config)
    _scope_artifact_directories(plugin_config, invocation_dir / "artifacts")
    config_path.write_text(
        tomli_w.dumps(
            {
                "agents": {
                    "hermes": {
                        "command": str(launch.hermes_executable),
                        "hooks_path": str(launch.hermes_config_path),
                    }
                },
                "upstream": {"openai_base_url": launch.base_url},
            }
        ),
        encoding="utf-8",
    )
    plugin_config_path.write_text(
        tomli_w.dumps(
            common_utils.relay_cli_plugin_config(
                plugin_config,
                observability_version=launch.relay_contract.observability_version,
            )
        ),
        encoding="utf-8",
    )
    return config_path, plugin_config_path, plugin_config


def _scope_artifact_directories(
    plugin_config: dict[str, Any], artifact_dir: Path
) -> None:
    """Keep overwrite-mode exporters distinct across runtime invocations."""

    for component in plugin_config.get("components", []):
        if not isinstance(component, dict) or component.get("kind") != "observability":
            continue
        config = component.get("config") or {}
        atof = config.get("atof")
        if isinstance(atof, dict):
            for sink in atof.get("sinks") or []:
                if isinstance(sink, dict) and sink.get("type") == "file":
                    sink["output_directory"] = str(artifact_dir / "atof")
                    (artifact_dir / "atof").mkdir(parents=True, exist_ok=True)
        atif = config.get("atif")
        if isinstance(atif, dict) and atif.get("enabled"):
            atif["output_directory"] = str(artifact_dir / "atif")
            (artifact_dir / "atif").mkdir(parents=True, exist_ok=True)


async def _drain(
    stream: asyncio.StreamReader | None,
) -> tuple[bytes, bool]:
    if stream is None:
        return b"", False
    captured = bytearray()
    tail = bytearray()
    truncated = False
    marker = b"\n...[output truncated]...\n"
    available = max(0, MAX_CAPTURE_BYTES - len(marker))
    head_limit = available // 2
    tail_limit = available - head_limit
    while chunk := await stream.read(64 * 1024):
        if not truncated and len(captured) + len(chunk) <= MAX_CAPTURE_BYTES:
            captured.extend(chunk)
            continue
        if not truncated:
            combined = captured + chunk
            captured = combined[:head_limit]
            if tail_limit:
                tail = combined[-tail_limit:]
            truncated = True
            continue
        if tail_limit:
            tail.extend(chunk)
            if len(tail) > tail_limit:
                del tail[:-tail_limit]
    if not truncated:
        return bytes(captured), False
    return bytes(captured) + marker + bytes(tail), True


async def _stop_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGINT)
        else:
            process.terminate()
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(process.wait(), timeout=PROCESS_STOP_TIMEOUT_SECONDS)
        return
    except TimeoutError:
        pass
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(process.wait(), timeout=PROCESS_STOP_TIMEOUT_SECONDS)
        return
    except TimeoutError:
        pass
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except ProcessLookupError:
        return
    await asyncio.wait_for(process.wait(), timeout=PROCESS_STOP_TIMEOUT_SECONDS)


def _safe_id(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip(".-")[:48]
    digest = hashlib.sha256(value.encode()).hexdigest()[:8]
    return f"{normalized or 'invocation'}-{digest}"


def _linux_parent_death_signal() -> None:
    """Ask Linux to stop Relay if Fabric's Python host is killed on timeout."""

    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGTERM) != 0:
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_PDEATHSIG) failed")
    if os.getppid() == 1:
        os.kill(os.getpid(), signal.SIGTERM)
