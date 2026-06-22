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
import inspect
import importlib
import json
import os
import shlex
import subprocess
import sys
import time
import uuid
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
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
    """Raised when a typed-config SDK method needs the native extension."""


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
            inline_entrypoint = _inline_adapter_entrypoint(plan)
            if inline_entrypoint is not None:
                return await _run_inline_adapter(plan, request_payload, inline_entrypoint)
            return await _call_blocking(
                lambda: json.loads(
                    native.run(
                        str(path),
                        native_profile,
                        input_text,
                        None if input_file is None else str(input_file),
                        None if request is None else json.dumps(request),
                        None if request_file is None else str(request_file),
                    )
                )
            )
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
        inline_entrypoint = _inline_adapter_entrypoint(plan)
        if inline_entrypoint is not None:
            return await _run_inline_adapter(plan, request_payload, inline_entrypoint)
        return await _call_blocking(
            lambda: json.loads(
                native.run_config(
                    _config_json(config),
                    _profiles_json(profile_configs),
                    None if base_dir is None else str(base_dir),
                    input_text,
                    None if input_file is None else str(input_file),
                    None if request is None else json.dumps(request),
                    None if request_file is None else str(request_file),
                )
            )
        )

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


def _inline_adapter_entrypoint(plan: dict[str, Any]) -> tuple[str, str] | None:
    descriptor = ((plan.get("adapter_descriptor") or {}).get("descriptor") or {})
    if descriptor.get("adapter_kind") != "python":
        return None
    runner = descriptor.get("runner") or {}
    module = runner.get("module")
    callable_name = runner.get("callable")
    if not module or not callable_name:
        return None
    return str(module), str(callable_name)


async def _run_inline_adapter(
    plan: dict[str, Any],
    request: dict[str, Any],
    entrypoint: tuple[str, str],
) -> dict[str, Any]:
    runtime_id = _new_id("runtime")
    invocation_id = _new_id("invocation")
    environment = _environment_handle(plan)
    artifacts = _artifact_manifest(plan)
    relay_runtime = _prepare_relay_runtime_config(
        plan, runtime_id, invocation_id, request, artifacts
    )
    payload = _fabric_adapter_payload(
        plan,
        runtime_id,
        invocation_id,
        environment,
        artifacts,
        relay_runtime,
        request,
    )
    module_name, callable_name = entrypoint
    events = [
        _event(
            "runtime_start",
            f"started runtime {runtime_id}",
            {
                "runtime_id": runtime_id,
                "environment_id": environment["environment_id"],
                "environment_provider": environment["provider"],
            },
        ),
        _event(
            "invocation_start",
            f"starting inline python adapter for {_harness_type(plan)}",
            {
                "runtime_id": runtime_id,
                "invocation_id": invocation_id,
                "module": module_name,
                "callable": callable_name,
            },
        ),
    ]
    metadata = {
        "adapter_runner": "python_inline",
        "module": module_name,
        "callable": callable_name,
        "environment_provider": environment["provider"],
    }
    try:
        output = await _call_inline_adapter(entrypoint, plan, payload, relay_runtime["env"])
        status = "failed" if isinstance(output, dict) and output.get("failed") else "succeeded"
        error = _output_error(output, metadata) if status == "failed" else None
    except Exception as exc:  # noqa: BLE001 - normalize adapter failures for consumers.
        output = {}
        status = "failed"
        error = {
            "stage": "invoke",
            "code": "python_inline_adapter_error",
            "message": str(exc),
            "retryable": False,
            "metadata": {
                **metadata,
                "exception_type": type(exc).__name__,
            },
        }
    _collect_inline_adapter_artifacts(output, artifacts)
    events.extend(
        [
            _event(
                "invocation_end",
                f"inline python adapter completed with status {status}",
                {
                    "runtime_id": runtime_id,
                    "invocation_id": invocation_id,
                },
            ),
            _event(
                "runtime_stop",
                f"stopped runtime {runtime_id}",
                {"runtime_id": runtime_id},
            ),
        ]
    )
    result = {
        "agent_name": plan["agent_name"],
        "profile": plan.get("profile"),
        "harness_type": _harness_type(plan),
        "adapter_kind": _adapter_kind(plan),
        "adapter_id": _adapter_id(plan),
        "runtime_id": runtime_id,
        "invocation_id": invocation_id,
        "request_id": request["request_id"],
        "status": status,
        "output": output,
        "artifacts": artifacts,
        "telemetry": _telemetry_ref(plan, relay_runtime),
        "events": events,
        "metadata": metadata,
    }
    if error is not None:
        result["error"] = error
    return result


async def _call_inline_adapter(
    entrypoint: tuple[str, str],
    plan: dict[str, Any],
    payload: dict[str, Any],
    relay_env: dict[str, str],
) -> Any:
    module_name, callable_name = entrypoint
    adapter_root = ((plan.get("adapter_descriptor") or {}).get("root"))
    added_paths = _prepend_adapter_paths(adapter_root)
    try:
        module = importlib.import_module(module_name)
        func = _resolve_attr(module, callable_name)
        with _patched_environ(relay_env):
            if inspect.iscoroutinefunction(func):
                return await func(payload)
            return await _call_blocking(lambda: func(payload))
    finally:
        _restore_sys_path(added_paths)


async def _call_blocking(func: Any) -> Any:
    loop = asyncio.get_running_loop()
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="fabric-sdk",
    ) as executor:
        return await loop.run_in_executor(executor, func)


def _fabric_adapter_payload(
    plan: dict[str, Any],
    runtime_id: str,
    invocation_id: str,
    environment: dict[str, Any],
    artifacts: dict[str, Any],
    relay_runtime: dict[str, Any],
    request: dict[str, Any],
) -> dict[str, Any]:
    effective_config = _effective_config(plan)
    return {
        "effective_config": effective_config,
        "runtime_context": {
            "runtime_id": runtime_id,
            "invocation_id": invocation_id,
            "request_id": request["request_id"],
            "environment": environment,
            "artifacts": artifacts,
            "telemetry": _runtime_telemetry_context(plan, relay_runtime),
        },
        "request": request,
        "capability_plan": plan.get("capability_plan") or {},
        "telemetry_plan": plan.get("telemetry_plan"),
    }


def _effective_config(plan: dict[str, Any]) -> dict[str, Any]:
    if isinstance(plan.get("effective_config"), dict):
        effective = json.loads(json.dumps(plan["effective_config"]))
    else:
        effective = {
            "agent_name": plan.get("agent_name"),
            "profile": plan.get("profile"),
            "profiles": plan.get("profiles") or [],
            "agent_root": plan.get("agent_root"),
            "config_path": plan.get("config_path"),
            "config_root": plan.get("config_root"),
            "config": plan.get("config") or {},
        }
    effective["agent_root"] = _absolute_plan_path(effective.get("agent_root"))
    effective["config_path"] = _absolute_plan_path(effective.get("config_path"))
    effective["config_root"] = _absolute_plan_path(effective.get("config_root"))
    return effective


def _runtime_telemetry_context(
    plan: dict[str, Any], relay_runtime: dict[str, Any]
) -> dict[str, Any] | None:
    telemetry = plan.get("telemetry_plan")
    if telemetry is None:
        return None
    metadata: dict[str, Any] = {}
    for source, target in (
        ("relay_mode", "relay_mode"),
        ("relay_project", "relay_project"),
        ("relay_output_dir", "relay_output_dir"),
        ("adapter_outputs", "adapter_outputs"),
    ):
        if source in telemetry and telemetry[source] is not None:
            metadata[target] = telemetry[source]
    return {
        "relay_enabled": bool(telemetry.get("relay_enabled")),
        "config_path": relay_runtime.get("config_path"),
        "env": relay_runtime.get("env") or {},
        "metadata": metadata,
    }


def _environment_handle(plan: dict[str, Any]) -> dict[str, Any]:
    environment_plan = plan.get("environment_plan") or {}
    config = plan.get("config") or {}
    runtime = config.get("runtime") or {}
    return {
        "environment_id": _new_id("environment"),
        "provider": environment_plan.get("provider", "local"),
        "control_location": environment_plan.get("control_location", "external_control"),
        "workspace": _absolute_plan_path(environment_plan.get("workspace") or plan.get("agent_root")),
        "artifacts": environment_plan.get("artifacts") or _runtime_artifact_path(plan, runtime),
        "ownership": environment_plan.get("ownership", "caller_owned"),
        "connection": environment_plan.get("connection", {}),
        "metadata": {
            **(environment_plan.get("settings") or {}),
            **(environment_plan.get("metadata") or {}),
        },
    }


def _artifact_manifest(plan: dict[str, Any]) -> dict[str, Any]:
    root = _runtime_artifact_path(plan, ((plan.get("config") or {}).get("runtime") or {}))
    if root is not None:
        Path(root).mkdir(parents=True, exist_ok=True)
    return {
        "root": root,
        "artifacts": [],
    }


def _runtime_artifact_path(plan: dict[str, Any], runtime: dict[str, Any]) -> str | None:
    artifacts = runtime.get("artifacts")
    if artifacts:
        return str(_resolve_plan_path(plan.get("config_root"), artifacts))
    environment_artifacts = (plan.get("environment_plan") or {}).get("artifacts")
    if environment_artifacts:
        return str(environment_artifacts)
    return None


def _prepare_relay_runtime_config(
    plan: dict[str, Any],
    runtime_id: str,
    invocation_id: str,
    request: dict[str, Any],
    artifacts: dict[str, Any],
) -> dict[str, Any]:
    telemetry = plan.get("telemetry_plan")
    if not telemetry or not telemetry.get("relay_enabled"):
        return {"config_path": None, "env": {}}
    root = artifacts.get("root")
    if not root:
        return {"config_path": None, "env": {}}
    relay_config = {
        "schema_version": "fabric.relay/v1alpha1",
        "relay": {
            "enabled": True,
            "mode": telemetry.get("relay_mode") or "sdk",
            "project": telemetry.get("relay_project"),
            "output_dir": telemetry.get("relay_output_dir"),
            "config": telemetry.get("relay_config") or {},
        },
        "fabric": {
            "agent_name": plan.get("agent_name"),
            "profile": plan.get("profile"),
            "harness_type": _harness_type(plan),
            "adapter_id": _adapter_id(plan),
            "runtime_id": runtime_id,
            "invocation_id": invocation_id,
            "request_id": request["request_id"],
            "adapter_outputs": telemetry.get("adapter_outputs") or [],
        },
    }
    path = Path(root) / "relay-config.json"
    path.write_text(json.dumps(relay_config, indent=2, sort_keys=True), encoding="utf-8")
    _add_artifact(artifacts, "relay_config", "telemetry_config", path, "application/json")
    return {
        "config_path": str(path.resolve()),
        "env": {
            "FABRIC_RELAY_ENABLED": "true",
            "FABRIC_RELAY_MODE": telemetry.get("relay_mode") or "sdk",
            "FABRIC_RELAY_CONFIG_PATH": str(path.resolve()),
        },
    }


def _collect_inline_adapter_artifacts(output: Any, artifacts: dict[str, Any]) -> None:
    if not isinstance(output, dict):
        return
    for index, artifact in enumerate(output.get("relay_artifacts") or []):
        path = artifact.get("path")
        if not path:
            continue
        _add_artifact(
            artifacts,
            f"relay_{artifact.get('kind', 'artifact')}_{index}",
            artifact.get("kind", "telemetry"),
            Path(path),
            "application/json",
        )


def _add_artifact(
    manifest: dict[str, Any],
    name: str,
    kind: str,
    path: Path,
    media_type: str | None = None,
) -> None:
    manifest.setdefault("artifacts", []).append(
        {
            "name": name,
            "kind": kind,
            "path": str(path),
            "media_type": media_type,
        }
    )


def _telemetry_ref(plan: dict[str, Any], relay_runtime: dict[str, Any]) -> dict[str, Any] | None:
    telemetry = plan.get("telemetry_plan")
    if telemetry is None:
        return None
    metadata: dict[str, Any] = {}
    for source, target in (
        ("relay_mode", "relay_mode"),
        ("relay_project", "relay_project"),
        ("relay_output_dir", "relay_output_dir"),
        ("relay_config", "relay_config"),
        ("adapter_outputs", "adapter_outputs"),
    ):
        if source in telemetry and telemetry[source] is not None:
            metadata[target] = telemetry[source]
    if relay_runtime.get("config_path"):
        metadata["relay_config_path"] = relay_runtime["config_path"]
    return {
        "relay_enabled": bool(telemetry.get("relay_enabled")),
        "metadata": metadata,
    }


def _output_error(output: Any, metadata: dict[str, Any]) -> dict[str, Any]:
    message = "inline python adapter returned failed status"
    if isinstance(output, dict) and output.get("error"):
        message = str(output["error"])
    return {
        "stage": "invoke",
        "code": "python_inline_adapter_failed",
        "message": message,
        "retryable": False,
        "metadata": metadata,
    }


def _event(kind: str, message: str, metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_id": _new_id("event"),
        "timestamp_millis": int(time.time() * 1000),
        "kind": kind,
        "message": message,
        "metadata": metadata,
    }


def _adapter_id(plan: dict[str, Any]) -> str | None:
    descriptor = ((plan.get("adapter_descriptor") or {}).get("descriptor") or {})
    harness = (plan.get("config") or {}).get("harness") or {}
    return descriptor.get("adapter_id") or harness.get("adapter_id")


def _adapter_kind(plan: dict[str, Any]) -> str:
    descriptor = ((plan.get("adapter_descriptor") or {}).get("descriptor") or {})
    return descriptor.get("adapter_kind", "process")


def _harness_type(plan: dict[str, Any]) -> str:
    descriptor = ((plan.get("adapter_descriptor") or {}).get("descriptor") or {})
    return descriptor.get("adapter_id", "unknown")


def _new_id(prefix: str) -> str:
    return f"{prefix}-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"


def _absolute_plan_path(path: Any) -> str | None:
    if path is None:
        return None
    return str(Path(path).resolve())


def _resolve_plan_path(root: Any, path: Any) -> Path:
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    if root is None:
        return path_obj
    return Path(root) / path_obj


def _prepend_adapter_paths(adapter_root: Any) -> list[str]:
    if not adapter_root:
        return []
    root = Path(adapter_root)
    candidates = [root / "src", root / "python"]
    added: list[str] = []
    for candidate in candidates:
        if candidate.is_dir():
            value = str(candidate)
            sys.path.insert(0, value)
            added.append(value)
    return added


def _restore_sys_path(added_paths: list[str]) -> None:
    for value in added_paths:
        try:
            sys.path.remove(value)
        except ValueError:
            pass


def _resolve_attr(module: Any, dotted_name: str) -> Any:
    value = module
    for part in dotted_name.split("."):
        value = getattr(value, part)
    return value


@contextmanager
def _patched_environ(updates: Mapping[str, str]):
    previous: dict[str, str | None] = {}
    for key, value in updates.items():
        previous[key] = os.environ.get(key)
        os.environ[key] = value
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
