"""Python client for NeMo Fabric.

The SDK uses the native Rust binding when the package is installed with
maturin. It falls back to the Fabric CLI when the native extension is not
available or when a CLI command is configured explicitly.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import shlex
import subprocess
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

    def inspect(self, path: str | Path) -> dict[str, Any]:
        """Load and print the normalized Fabric document."""

        native = self._native_module()
        if native is not None:
            return json.loads(native.inspect(str(path)))
        return self._call_json(["inspect", str(path)])

    def plan(self, path: str | Path, *, profile: str | Sequence[str] | None = None) -> dict[str, Any]:
        """Resolve an agent/profile into a run plan."""

        native = self._native_module()
        native_profile = _native_profile_arg(profile)
        if native is not None:
            return json.loads(native.plan(str(path), native_profile))
        args = ["plan", str(path)]
        args.extend(_profile_args(profile))
        return self._call_json(args)

    async def doctor(
        self, path: str | Path, *, profile: str | Sequence[str] | None = None
    ) -> dict[str, Any]:
        """Diagnose a run plan without installing or running the harness."""

        native = self._native_module()
        native_profile = _native_profile_arg(profile)
        if native is not None:
            return await asyncio.to_thread(
                lambda: json.loads(native.doctor(str(path), native_profile))
            )
        args = ["doctor", str(path)]
        args.extend(_profile_args(profile))
        return await self._call_json_async(args)

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
            return await asyncio.to_thread(
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
