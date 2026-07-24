# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native Python client for resolving and running NVIDIA NeMo Fabric agents."""

from __future__ import annotations

import asyncio
import importlib
import json
import os
from collections.abc import Mapping
from typing import Any
from nemo_fabric.errors import (
    FabricConfigError,
    FabricError,
    FabricNativeUnavailableError,
    FabricRuntimeError,
)
from nemo_fabric.models import FabricConfig, RunRequest
from nemo_fabric.runtime import (
    Runtime,
    _call_blocking,
    _json_mapping,
    _run_native_lifecycle,
    _run_request_payload,
)
from nemo_fabric.types import (
    DoctorReport,
    RunPlan,
    RunResult,
)

try:
    _native = importlib.import_module("nemo_fabric._native")
except ImportError:
    _native = None


class Fabric:
    """Primary Python entrypoint for NeMo Fabric.

    Every lifecycle method accepts a complete, typed ``FabricConfig`` plus an
    optional ``base_dir`` used to resolve relative paths. Compose variants in
    Python before calling the SDK. The ``doctor()``, ``plan()``, and ``run()``
    results are typed, read-only mapping models. ``start_runtime()`` returns an
    active ``Runtime`` handle.

    ``Fabric`` uses the native Rust extension. SDK calls raise
    ``FabricNativeUnavailableError`` when the native extension is not
    installed.

    See the Getting Started overview for runnable single-invocation,
    typed-config, and multi-turn examples.
    """

    def plan(
        self,
        config: FabricConfig,
        *,
        base_dir: str | os.PathLike[str] | None = None,
    ) -> RunPlan:
        """Resolve a complete typed configuration into an immutable execution plan.

        Planning resolves the selected adapter and reports optional runtime
        capabilities such as streaming, updates, and cancellation. Planning
        does not start the runtime.

        Args:
            config: Complete typed ``FabricConfig``. Raw mappings are not
                accepted.
            base_dir: Base directory for resolving relative paths.

        Returns:
            A ``RunPlan`` containing the canonical config, path context,
            adapter, and declared runtime capabilities.

        Raises:
            FabricConfigError: If the config or adapter resolution is invalid.
            FabricNativeUnavailableError: If the native extension is not
                installed.
        """

        native = self._require_native_module("plan")
        try:
            raw = native.plan_config(
                _config_json(config),
                _base_dir_arg(base_dir),
            )
            return RunPlan.from_mapping(json.loads(raw))
        except FabricError:
            raise
        except Exception as error:
            raise FabricConfigError(str(error)) from error

    async def doctor(
        self,
        config: FabricConfig,
        *,
        base_dir: str | os.PathLike[str] | None = None,
    ) -> DoctorReport:
        """Diagnose a planned agent without starting its runtime.

        Doctor checks the resolved adapter, capability mappings, and declared
        environment requirements using the native Fabric core.

        Args:
            config: Complete typed ``FabricConfig``.
            base_dir: Base directory for resolving relative paths.

        Returns:
            A ``DoctorReport`` with aggregate status and ordered checks.

        Raises:
            FabricConfigError: If inputs or native diagnostic output are
                invalid.
            FabricNativeUnavailableError: If the native extension is not
                installed.
        """

        native = self._require_native_module("doctor")

        def diagnose() -> DoctorReport:
            raw = native.doctor_config(
                _config_json(config),
                _base_dir_arg(base_dir),
            )
            return DoctorReport.from_mapping(json.loads(raw))

        try:
            return await _call_blocking(diagnose)
        except FabricError:
            raise
        except Exception as error:
            raise FabricConfigError(str(error)) from error

    async def run(
        self,
        config: FabricConfig,
        *,
        base_dir: str | os.PathLike[str] | None = None,
        input: Any = None,
        request: RunRequest | None = None,
    ) -> RunResult:
        """Execute one complete start, invoke, and stop lifecycle.

        ``input`` and ``request`` are mutually exclusive. Omitting both produces
        an empty text input. Use ``RunRequest`` when the invocation needs a
        caller-owned request ID, context, or overrides.
        Fabric attempts to stop a started runtime even when invocation fails.

        Args:
            config: Complete typed ``FabricConfig``.
            base_dir: Base directory for resolving relative paths.
            input: JSON-compatible invocation input.
            request: Complete validated ``RunRequest``.

        Returns:
            The normalized ``RunResult``, including output, artifacts,
            telemetry references, lifecycle events, and structured error data.

        Raises:
            FabricConfigError: If input and request are combined, request data is not
                JSON-compatible, or config resolution fails.
            FabricNativeUnavailableError: If the native extension is not
                installed.
            FabricRuntimeError: If the native runtime lifecycle fails before a
                normalized result can be returned.
        """

        plan = await _call_blocking(lambda: self.plan(config, base_dir=base_dir))
        request_payload = _run_request_payload(
            input=input,
            request=request,
        )
        native = self._require_native_module("run")
        return RunResult.from_mapping(
            await _run_native_lifecycle(native, plan.to_mapping(), request_payload)
        )

    async def start_runtime(
        self,
        config: FabricConfig,
        *,
        base_dir: str | os.PathLike[str] | None = None,
        overrides: Mapping[str, Any] | None = None,
    ) -> Runtime:
        """Start a stateful runtime for one or more ordered invocations.

        Each call starts a new logical runtime. Runtime-scoped overrides are
        recursively merged below invocation-scoped overrides.

        Args:
            config: Complete typed ``FabricConfig``.
            base_dir: Base directory for resolving relative paths.
            overrides: JSON-compatible overrides applied to every invocation
                in the runtime unless superseded by invocation overrides.

        Returns:
            An active ``Runtime``. Use it as an asynchronous context
            manager to guarantee runtime shutdown.

        Raises:
            FabricConfigError: If inputs or overrides are invalid.
            FabricNativeUnavailableError: If the native extension is not
                installed.
            FabricRuntimeError: If runtime startup fails.
        """

        runtime_overrides = _json_mapping(overrides, "runtime overrides")
        plan = await _call_blocking(lambda: self.plan(config, base_dir=base_dir))
        native = self._require_native_module("start_runtime")
        started_runtime: dict[str, Any] | None = None

        def start() -> dict[str, Any]:
            nonlocal started_runtime
            started_runtime = json.loads(
                native.start_runtime(json.dumps(plan.to_mapping()))
            )
            return started_runtime

        try:
            runtime = await _call_blocking(start)
        except asyncio.CancelledError:
            if started_runtime is not None:
                try:
                    await _call_blocking(
                        lambda: json.loads(
                            native.stop_runtime(
                                json.dumps(plan.to_mapping()),
                                json.dumps(started_runtime),
                            )
                        )
                    )
                except Exception:
                    pass
            raise
        except FabricError:
            raise
        except Exception as error:
            raise FabricRuntimeError(str(error), stage="start") from error
        return Runtime(
            client=self,
            plan=plan,
            runtime=runtime,
            overrides=runtime_overrides,
        )

    def _native_module(self) -> Any | None:
        return _native

    def _require_native_module(self, method: str) -> Any:
        native = self._native_module()
        if native is None:
            raise FabricNativeUnavailableError(
                f"{method} requires the nemo_fabric native extension",
                stage=method,
                code="native_unavailable",
            )
        return native


def _config_json(config: FabricConfig) -> str:
    if not isinstance(config, FabricConfig):
        if isinstance(config, Mapping):
            raise FabricConfigError(
                "config mappings are not accepted directly; "
                "use FabricConfig.from_mapping(...) first"
            )
        raise FabricConfigError("config must be a FabricConfig")
    return json.dumps(config.to_mapping())


def _base_dir_arg(base_dir: str | os.PathLike[str] | None) -> str | None:
    return None if base_dir is None else os.fspath(base_dir)
