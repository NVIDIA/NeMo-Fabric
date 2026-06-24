# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python client for NeMo Fabric.

The SDK uses the native Rust binding when the package is installed with
maturin. It falls back to the Fabric CLI when the native extension is not
available or when a CLI command is configured explicitly.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import importlib
import json
import os
import shlex
import subprocess
import uuid
from collections.abc import AsyncIterator, Mapping
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    _native = importlib.import_module("nemo_fabric._native")
except ImportError:
    _native = None


class FabricCliError(RuntimeError):
    """Raised when the Fabric CLI exits unsuccessfully."""

    def __init__(self, command: Sequence[str], returncode: int, stdout: str, stderr: str) -> None:
        super().__init__(f"Fabric CLI failed with exit code {returncode}: {' '.join(command)}")
        self.command = tuple(command)
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class FabricNativeUnavailableError(RuntimeError):
    """Raised when an SDK method requires the native extension."""


@dataclass(frozen=True)
class FabricClient:
    """Python entrypoint for Fabric config, planning, diagnostics, and runs."""

    command: tuple[str, ...] | None = None
    cwd: Path | None = None

    async def __aenter__(self) -> "FabricClient":
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None

    def validate(self, path: str | Path) -> str:
        """Validate a Fabric agent directory or config file."""

        native = self._native_module()
        if native is not None:
            return native.validate(str(path))
        return self._call_text(["validate", str(path)])

    def inspect(
        self, path: str | Path, *, profile: str | Sequence[str] | None = None
    ) -> dict[str, Any]:
        """Resolve and return the effective Fabric config."""

        native = self._native_module()
        native_profile = _native_profile_arg(profile)
        if native is not None:
            return json.loads(native.inspect(str(path), native_profile))
        args = ["inspect", str(path)]
        args.extend(_profile_args(profile))
        return self._call_json(args)

    def plan(
        self, path: str | Path, *, profile: str | Sequence[str] | None = None
    ) -> dict[str, Any]:
        """Resolve an agent/profile into a run plan."""

        native = self._native_module()
        native_profile = _native_profile_arg(profile)
        if native is not None:
            return json.loads(native.plan(str(path), native_profile))
        args = ["plan", str(path)]
        args.extend(_profile_args(profile))
        return self._call_json(args)

    def plan_config(
        self,
        config: Mapping[str, Any] | Any,
        *,
        profile_configs: Sequence[Mapping[str, Any] | Any] | None = None,
        base_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        """Resolve an in-memory typed config into a run plan."""

        native = self._require_native_module("plan_config")
        return json.loads(
            native.plan_config(
                _config_json(config),
                _profiles_json(profile_configs),
                None if base_dir is None else str(base_dir),
            )
        )

    async def doctor(
        self, path: str | Path, *, profile: str | Sequence[str] | None = None
    ) -> dict[str, Any]:
        """Diagnose a run plan without installing or running the harness."""

        native = self._native_module()
        native_profile = _native_profile_arg(profile)
        if native is not None:
            return await _call_blocking(
                lambda: json.loads(native.doctor(str(path), native_profile))
            )
        args = ["doctor", str(path)]
        args.extend(_profile_args(profile))
        return await self._call_json_async(args)

    async def doctor_config(
        self,
        config: Mapping[str, Any] | Any,
        *,
        profile_configs: Sequence[Mapping[str, Any] | Any] | None = None,
        base_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        """Diagnose an in-memory typed config without running the harness."""

        native = self._require_native_module("doctor_config")
        return await _call_blocking(
            lambda: json.loads(
                native.doctor_config(
                    _config_json(config),
                    _profiles_json(profile_configs),
                    None if base_dir is None else str(base_dir),
                )
            )
        )

    async def run(
        self,
        path: str | Path,
        *,
        profile: str | Sequence[str] | None = None,
        input_text: str = "",
        input_file: str | Path | None = None,
        request: dict[str, Any] | None = None,
        request_file: str | Path | None = None,
    ) -> dict[str, Any]:
        """Run an agent/profile through the selected Fabric adapter."""

        native = self._native_module()
        native_profile = _native_profile_arg(profile)
        if native is not None:
            request_payload = _run_request_payload(
                input_text=input_text,
                input_file=input_file,
                request=request,
                request_file=request_file,
            )
            plan = json.loads(native.plan(str(path), native_profile))
            return await _run_native_lifecycle(native, plan, request_payload)
        args = ["run", str(path)]
        args.extend(_profile_args(profile))
        if request_file is not None:
            args.extend(["--request-file", str(request_file)])
        elif request is not None:
            args.extend(["--request-json", json.dumps(request)])
        elif input_file is not None:
            args.extend(["--input-file", str(input_file)])
        else:
            args.extend(["--input", input_text])
        return await self._call_json_async(args)

    async def run_config(
        self,
        config: Mapping[str, Any] | Any,
        *,
        profile_configs: Sequence[Mapping[str, Any] | Any] | None = None,
        base_dir: str | Path | None = None,
        input_text: str = "",
        input_file: str | Path | None = None,
        request: dict[str, Any] | None = None,
        request_file: str | Path | None = None,
    ) -> dict[str, Any]:
        """Run an in-memory typed config through the selected Fabric adapter."""

        native = self._require_native_module("run_config")
        request_payload = _run_request_payload(
            input_text=input_text,
            input_file=input_file,
            request=request,
            request_file=request_file,
        )
        plan = json.loads(
            native.plan_config(
                _config_json(config),
                _profiles_json(profile_configs),
                None if base_dir is None else str(base_dir),
            )
        )
        return await _run_native_lifecycle(native, plan, request_payload)

    async def start(
        self,
        path: str | Path,
        *,
        profile: str | Sequence[str] | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> "Session":
        """Open a multi-turn session over an agent/profile runtime.

        Args:
            path: Agent package directory or config file to resolve.
            profile: Profile name, or several applied in order, layered onto the
                base config.
            overrides: Config overrides applied to every turn in the session; a
                turn's own ``overrides`` merge over these.

        Returns:
            An active :class:`Session` bound to the resolved plan.

        Raises:
            FabricNativeUnavailableError: The native extension is unavailable
                (sessions are not supported over the CLI fallback).
        """

        native = self._require_native_module("start")
        plan = self.plan(path, profile=profile)
        runtime = await _call_blocking(
            lambda: json.loads(native.start_runtime(json.dumps(plan)))
        )
        return Session(client=self, plan=plan, runtime=runtime, overrides=overrides)

    async def start_config(
        self,
        config: Mapping[str, Any] | Any,
        *,
        profile_configs: Sequence[Mapping[str, Any] | Any] | None = None,
        base_dir: str | Path | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> "Session":
        """Open a multi-turn session over an in-memory typed config.

        Args:
            config: Typed Fabric config as a mapping or a Pydantic-like object
                (``model_dump()``/``dict()``); no agent directory required.
            profile_configs: Profile configs layered onto the base config, in order.
            base_dir: Resolution root for relative paths and package-local
                adapters. ``None`` resolves against the process working directory.
            overrides: Config overrides applied to every turn; a turn's own
                ``overrides`` merge over these.

        Returns:
            An active :class:`Session` bound to the resolved plan.

        Raises:
            FabricNativeUnavailableError: The native extension is unavailable.
        """

        native = self._require_native_module("start_config")
        plan = self.plan_config(
            config, profile_configs=profile_configs, base_dir=base_dir
        )
        runtime = await _call_blocking(
            lambda: json.loads(native.start_runtime(json.dumps(plan)))
        )
        return Session(client=self, plan=plan, runtime=runtime, overrides=overrides)

    def _command(self) -> tuple[str, ...]:
        if self.command is not None:
            return self.command
        env_command = os.environ.get("FABRIC_CLI")
        if env_command:
            return tuple(shlex.split(env_command))
        return ("fabric",)

    def _call_text(self, args: Iterable[str]) -> str:
        completed = self._run(args)
        return completed.stdout.strip()

    def _call_json(self, args: Iterable[str]) -> dict[str, Any]:
        completed = self._run(args)
        return json.loads(completed.stdout)

    async def _call_json_async(self, args: Iterable[str]) -> dict[str, Any]:
        completed = await self._run_async(args)
        return json.loads(completed.stdout)

    def _run(self, args: Iterable[str]) -> subprocess.CompletedProcess[str]:
        command = [*self._command(), *args]
        completed = subprocess.run(
            command,
            cwd=self.cwd,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise FabricCliError(command, completed.returncode, completed.stdout, completed.stderr)
        return completed

    async def _run_async(self, args: Iterable[str]) -> subprocess.CompletedProcess[str]:
        command = [*self._command(), *args]
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=None if self.cwd is None else str(self.cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, stderr_bytes = await process.communicate()
        stdout = stdout_bytes.decode()
        stderr = stderr_bytes.decode()
        if process.returncode != 0:
            raise FabricCliError(command, process.returncode or 1, stdout, stderr)
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)

    def _native_module(self) -> Any | None:
        if self.command is not None:
            return None
        if os.environ.get("FABRIC_CLI"):
            return None
        return _native

    def _require_native_module(self, method: str) -> Any:
        native = self._native_module()
        if native is None:
            raise FabricNativeUnavailableError(
                f"{method} requires the nemo_fabric native extension; "
                "the CLI fallback only supports file-based agent configs"
            )
        return native


class SessionStatus(str, Enum):
    """Lifecycle state of a :class:`Session`."""

    ACTIVE = "active"
    STOPPED = "stopped"
    CANCELLED = "cancelled"


class Session:
    """A multi-turn session over a Fabric runtime.

    Created by :meth:`FabricClient.start` / :meth:`FabricClient.start_config`.
    Each :meth:`invoke` runs one turn through the same core ``RuntimeHandle``.
    Harness state is owned by the selected adapter/runtime, not replayed from a
    Python-side transcript.
    """

    def __init__(
        self,
        *,
        client: "FabricClient",
        plan: dict[str, Any],
        runtime: dict[str, Any],
        overrides: dict[str, Any] | None = None,
    ) -> None:
        self._client = client
        self._plan = plan
        self._runtime = runtime
        self._overrides = overrides
        self._messages: list[Any] = []
        self._invocations: list[dict[str, Any]] = []
        self._status = SessionStatus.ACTIVE
        self._current_task: asyncio.Task[Any] | None = None

    @property
    def status(self) -> SessionStatus:
        return self._status

    @property
    def messages(self) -> list[Any]:
        """Read-only deep copy of the accumulated transcript."""

        return deepcopy(self._messages)

    @property
    def invocations(self) -> list[dict[str, Any]]:
        """Per-turn ``{request_id, runtime_id, invocation_id}`` correlation data."""

        return list(self._invocations)

    @property
    def runtime(self) -> dict[str, Any]:
        """Read-only deep copy of the active ``RuntimeHandle``."""

        return deepcopy(self._runtime)

    @property
    def runtime_id(self) -> str:
        """Canonical Fabric runtime id for this session."""

        return str(self._runtime["runtime_id"])

    @property
    def info(self) -> dict[str, Any]:
        """Summary handle for the active Fabric runtime and selected adapter."""

        return {
            "runtime_id": self._runtime.get("runtime_id"),
            "agent_name": self._plan.get("agent_name"),
            "profile": self._plan.get("profile"),
            "harness_type": _harness_type(self._plan),
            "adapter_kind": _adapter_kind(self._plan),
        }

    async def invoke(
        self,
        input_text: str | None = None,
        *,
        request: dict[str, Any] | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run one turn on the session runtime.

        Args:
            input_text: Text input for the turn. Ignored when ``request`` is given.
            request: A full ``RunRequest`` mapping for the turn, as an alternative
                to ``input_text``.
            overrides: Per-turn config overrides, merged over the session-level
                overrides passed to :meth:`FabricClient.start`.

        Returns:
            The turn's normalized ``RunResult`` mapping. ``messages`` is updated
            only when the adapter returns a ``messages`` list in its output.

        Raises:
            RuntimeError: The session is not active (already stopped or cancelled).
        """

        if self._status is not SessionStatus.ACTIVE:
            raise RuntimeError(f"cannot invoke a {self._status.value} session")
        if self._current_task is not None:
            raise RuntimeError(
                "session is already running a turn; turns are ordered (one at a time)"
            )
        # Claim the turn before any await so callers cannot concurrently invoke
        # the same runtime handle.
        self._current_task = asyncio.current_task()
        try:
            request_payload = _run_request_payload(
                input_text=input_text or "",
                input_file=None,
                request=request,
                request_file=None,
            )
            # Merge overrides as session < request < per-turn; request-level
            # overrides must not bypass the documented session/turn merge.
            merged_overrides = _merge_overrides(self._overrides, request_payload.get("overrides"))
            merged_overrides = _merge_overrides(merged_overrides, overrides)
            if merged_overrides is not None:
                request_payload["overrides"] = merged_overrides
            native = self._client._require_native_module("invoke")
            result = await _call_blocking(
                lambda: json.loads(
                    native.invoke_runtime(
                        json.dumps(self._plan),
                        json.dumps(self._runtime),
                        json.dumps(request_payload),
                    )
                )
            )
            self._absorb(result)
            return result
        finally:
            self._current_task = None

    async def stream(
        self,
        input_text: str | None = None,
        *,
        request: dict[str, Any] | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Run one turn and yield its events, then the final ``RunResult``.

        Buffered: the turn runs to completion via :meth:`invoke`, then the
        normalized ``events`` are yielded in order, followed by the terminal
        ``RunResult`` (the last item). The async-iterator shape is
        forward-compatible with live token streaming if a harness exposes one.

        Args:
            input_text: Text input for the turn. Ignored when ``request`` is given.
            request: A full ``RunRequest`` mapping for the turn.
            overrides: Per-turn config overrides, merged over the session-level
                overrides.

        Yields:
            Each ``fabric-event`` mapping for the turn, in order, then the final
            ``RunResult`` mapping as the terminal item.

        Raises:
            RuntimeError: The session is not active (already stopped or cancelled).
        """

        result = await self.invoke(input_text, request=request, overrides=overrides)
        for event in result.get("events") or []:
            yield event
        yield result

    async def cancel(self) -> None:
        """Cancel the in-flight turn and close the session. Idempotent.

        Cooperative: cancels the awaiting :meth:`invoke` / :meth:`stream`
        coroutine and marks the session ``CANCELLED``. Already-dispatched
        blocking native calls may run to completion and their result is discarded.
        """

        if self._status is not SessionStatus.ACTIVE:
            return
        task = self._current_task
        if task is not None and not task.done() and task is not asyncio.current_task():
            task.cancel()
        try:
            await self._stop_runtime()
        finally:
            self._status = SessionStatus.CANCELLED

    async def stop(self) -> None:
        """Finalize the session. Idempotent."""

        if self._status is SessionStatus.ACTIVE:
            await self._stop_runtime()
            self._status = SessionStatus.STOPPED

    async def _stop_runtime(self) -> None:
        native = self._client._require_native_module("stop")
        await _call_blocking(
            lambda: json.loads(
                native.stop_runtime(json.dumps(self._plan), json.dumps(self._runtime))
            )
        )

    def _absorb(self, result: Any) -> None:
        """Record the turn's handles and advance the transcript from its ``RunResult``."""

        if not isinstance(result, dict):
            return
        self._invocations.append(
            {
                "request_id": result.get("request_id"),
                "runtime_id": result.get("runtime_id"),
                "invocation_id": result.get("invocation_id"),
            }
        )
        output = result.get("output")
        if not isinstance(output, dict):
            return
        messages = output.get("messages")
        if isinstance(messages, list) and messages:
            self._messages = deepcopy(messages)

    async def __aenter__(self) -> "Session":
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        await self.stop()


def _merge_overrides(
    base: dict[str, Any] | None, extra: dict[str, Any] | None
) -> dict[str, Any] | None:
    merged: dict[str, Any] = {}
    if isinstance(base, dict):
        merged.update(base)
    if isinstance(extra, dict):
        merged.update(extra)
    return merged or None


def _profile_args(profile: str | Sequence[str] | None) -> list[str]:
    if profile is None:
        return []
    if isinstance(profile, str):
        return ["--profile", profile]
    args: list[str] = []
    for value in profile:
        args.extend(["--profile", value])
    return args


def _native_profile_arg(profile: str | Sequence[str] | None) -> str | list[str] | None:
    if profile is None or isinstance(profile, str):
        return profile
    profiles = list(profile)
    if not profiles:
        return None
    return profiles


def _config_json(config: Mapping[str, Any] | Any) -> str:
    return json.dumps(_json_compatible(config))


def _profiles_json(profiles: Sequence[Mapping[str, Any] | Any] | None) -> str | None:
    if profiles is None:
        return None
    return json.dumps([_json_compatible(profile) for profile in profiles])


def _json_compatible(value: Mapping[str, Any] | Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    if hasattr(value, "dict"):
        return value.dict(exclude_none=True)
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(
        "config values must be mappings or Pydantic-like objects with model_dump()/dict()"
    )


def _run_request_payload(
    *,
    input_text: str,
    input_file: str | Path | None,
    request: dict[str, Any] | None,
    request_file: str | Path | None,
) -> dict[str, Any]:
    if request_file is not None:
        with Path(request_file).open(encoding="utf-8") as stream:
            payload = json.load(stream)
    elif request is not None:
        payload = json.loads(json.dumps(request))
    elif input_file is not None:
        payload = {"input": Path(input_file).read_text(encoding="utf-8")}
    else:
        payload = {"input": input_text}
    if not isinstance(payload, dict):
        raise TypeError("request payload must be a JSON object")
    payload.setdefault("request_id", f"request-{uuid.uuid4().hex}")
    payload.setdefault("context", {})
    if not isinstance(payload["context"], dict):
        raise TypeError("request context must be a JSON object")
    return payload


async def _run_native_lifecycle(
    native: Any,
    plan: dict[str, Any],
    request: dict[str, Any],
) -> dict[str, Any]:
    def _run() -> dict[str, Any]:
        plan_json = json.dumps(plan)
        runtime = json.loads(native.start_runtime(plan_json))
        runtime_json = json.dumps(runtime)
        result: dict[str, Any] | None = None
        try:
            result = json.loads(
                native.invoke_runtime(plan_json, runtime_json, json.dumps(request))
            )
            return result
        finally:
            stop_events = json.loads(native.stop_runtime(plan_json, runtime_json))
            if isinstance(result, dict) and isinstance(stop_events, list):
                result.setdefault("events", []).extend(stop_events)

    return await _call_blocking(_run)


async def _call_blocking(func: Any) -> Any:
    loop = asyncio.get_running_loop()
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="fabric-sdk",
    ) as executor:
        return await loop.run_in_executor(executor, func)


def _adapter_kind(plan: dict[str, Any]) -> str:
    descriptor = ((plan.get("adapter_descriptor") or {}).get("descriptor") or {})
    return descriptor.get("adapter_kind", "process")


def _harness_type(plan: dict[str, Any]) -> str:
    descriptor = ((plan.get("adapter_descriptor") or {}).get("descriptor") or {})
    return descriptor.get("adapter_id", "unknown")
