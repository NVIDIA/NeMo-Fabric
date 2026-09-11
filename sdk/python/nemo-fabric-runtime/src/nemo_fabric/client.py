# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native Python client for resolving and running NVIDIA NeMo Fabric agents."""

from __future__ import annotations

import asyncio
import importlib
import json
import os
from collections.abc import Mapping
from contextlib import AsyncExitStack
from typing import Any

from nemo_fabric._collector_client import _AtofCollectorClient
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
from nemo_fabric.streaming import (
    _configured_stream_sink,
    _relay_enabled,
    _with_stream_sink,
)
from nemo_fabric.types import (
    DoctorReport,
    EnvironmentHandle,
    EnvironmentReference,
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
    active local ``Runtime`` handle. Explicit environment users call either
    ``prepare_environment()`` or ``attach_environment()``, then
    ``start_runtime_in()`` and ``release_environment()`` separately.

    ``Fabric`` uses the native Rust extension. SDK calls raise
    ``FabricNativeUnavailableError`` when the native extension is not
    installed.

    See the Getting Started overview for runnable single-invocation,
    typed-config, and multi-turn examples.
    """

    def __init__(self) -> None:
        pass

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
        environment requirements using the native NeMo Fabric core.

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
        NeMo Fabric attempts to stop a started runtime even when invocation fails.

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
        streaming: bool = False,
        launch_collector: bool | None = None,
    ) -> Runtime:
        """Start a stateful runtime for one or more ordered invocations.

        Each call starts a new logical runtime. Runtime-scoped overrides are
        recursively merged below invocation-scoped overrides. With NVIDIA NeMo
        Relay enabled, ``streaming=True`` uses collector-backed streaming.
        By default, streaming starts an embedded collector. Set
        ``launch_collector=False`` to use an externally managed collector.

        Args:
            config: Complete typed ``FabricConfig``.
            base_dir: Base directory for resolving relative paths.
            overrides: JSON-compatible overrides applied to every invocation
                in the runtime unless superseded by invocation overrides.
            streaming: Whether to enable collector-backed NeMo Relay ATOF
                streaming for ``Runtime.invoke_stream()``.
            launch_collector: Whether to launch an embedded collector. ``None``
                defaults to ``True`` when streaming is enabled. ``False`` uses
                an externally managed collector. This argument cannot be set
                unless ``streaming=True``.

        Returns:
            An active ``Runtime``. Use it as an asynchronous context
            manager to guarantee runtime shutdown.

        Raises:
            FabricConfigError: If inputs or overrides are invalid, streaming is
                requested without NeMo Relay enabled, ``launch_collector`` is
                set without streaming, or an external collector has no sink.
            FabricNativeUnavailableError: If the native extension is not
                installed.
            FabricRuntimeError: If runtime startup fails.
        """

        return await self._start_runtime(
            config,
            base_dir=base_dir,
            overrides=overrides,
            streaming=streaming,
        )

    async def prepare_environment(
        self,
        config: FabricConfig,
        *,
        base_dir: str | os.PathLike[str] | None = None,
    ) -> EnvironmentHandle:
        """Prepare an execution environment.

        The returned handle is independent of any runtime session. The caller
        owns the lifecycle decision and must eventually pass it to
        ``release_environment()``.

        Args:
            config: Complete typed ``FabricConfig`` describing the environment.
            base_dir: Base directory for resolving relative paths.

        Returns:
            A typed, immutable ``EnvironmentHandle``.

        Raises:
            FabricConfigError: If config resolution or the returned handle is invalid.
            FabricNativeUnavailableError: If the native extension is not installed.
            FabricRuntimeError: If environment preparation fails.
        """

        plan = await _call_blocking(lambda: self.plan(config, base_dir=base_dir))
        native = self._require_native_module("prepare_environment")
        try:
            raw = await _call_blocking(
                lambda: native.prepare_environment(json.dumps(plan.to_mapping()))
            )
            return EnvironmentHandle.from_mapping(json.loads(raw))
        except FabricError:
            raise
        except Exception as error:
            raise FabricRuntimeError(str(error), stage="environment_prepare") from error

    async def attach_environment(
        self,
        config: FabricConfig,
        reference: EnvironmentReference,
        *,
        base_dir: str | os.PathLike[str] | None = None,
    ) -> EnvironmentHandle:
        """Verify and attach to an existing caller-owned environment.

        Attachment does not create the provider resource and does not grant
        Fabric deletion authority. The returned handle can be passed to
        ``start_runtime_in()`` and later to ``release_environment()`` to detach.

        Args:
            config: Complete typed ``FabricConfig`` with caller-owned environment settings.
            reference: Provider-specific identity of the existing resource.
            base_dir: Base directory for resolving relative paths.

        Returns:
            A verified, immutable ``EnvironmentHandle``.

        Raises:
            FabricConfigError: If config, reference, or returned handle is invalid.
            FabricNativeUnavailableError: If the native extension is not installed.
            FabricRuntimeError: If environment verification or attachment fails.
        """

        reference_json = _environment_reference_json(reference)
        plan = await _call_blocking(lambda: self.plan(config, base_dir=base_dir))
        native = self._require_native_module("attach_environment")
        try:
            raw = await _call_blocking(
                lambda: native.attach_environment(
                    json.dumps(plan.to_mapping()), reference_json
                )
            )
            return EnvironmentHandle.from_mapping(json.loads(raw))
        except FabricError:
            raise
        except Exception as error:
            raise FabricRuntimeError(str(error), stage="environment_attach") from error

    async def start_runtime_in(
        self,
        config: FabricConfig,
        environment: EnvironmentHandle,
        *,
        base_dir: str | os.PathLike[str] | None = None,
        overrides: Mapping[str, Any] | None = None,
        streaming: bool = False,
    ) -> Runtime:
        """Start one stateful runtime in an explicitly prepared or attached environment.

        Starting or stopping the runtime does not release ``environment``. This
        lets consumers run sequential sessions, or coordinate concurrent
        sessions, without hiding environment ownership inside a session API.

        Args:
            config: Complete typed ``FabricConfig`` matching the environment.
            environment: Handle returned by ``prepare_environment()`` or
                ``attach_environment()``.
            base_dir: Base directory for resolving relative paths.
            overrides: JSON-compatible runtime-scoped invocation overrides.
            streaming: Whether to provision NeMo Relay ATOF streaming.

        Returns:
            An active ``Runtime`` bound to ``environment``.

        Raises:
            FabricConfigError: If inputs are invalid or the handle does not match the plan.
            FabricNativeUnavailableError: If the native extension is not installed.
            FabricRuntimeError: If runtime startup fails.
        """

        _environment_json(environment)
        return await self._start_runtime(
            config,
            environment=environment,
            base_dir=base_dir,
            overrides=overrides,
            streaming=streaming,
        )

    async def release_environment(self, environment: EnvironmentHandle) -> None:
        """Release or detach a prepared environment through its provider.

        Local and externally owned environments detach without deletion.
        Provider-managed, Fabric-owned environments may be deleted according
        to their normalized ownership contract.

        Args:
            environment: Handle returned by ``prepare_environment()`` or
                ``attach_environment()``.

        Raises:
            FabricConfigError: If ``environment`` is not a typed handle.
            FabricNativeUnavailableError: If the native extension is not installed.
            FabricRuntimeError: If release or detach fails.
        """

        environment_json = _environment_json(environment)
        native = self._require_native_module("release_environment")
        try:
            await _call_blocking(lambda: native.release_environment(environment_json))
        except FabricError:
            raise
        except Exception as error:
            raise FabricRuntimeError(str(error), stage="environment_release") from error

    async def _start_runtime(
        self,
        config: FabricConfig,
        *,
        environment: EnvironmentHandle | None = None,
        base_dir: str | os.PathLike[str] | None = None,
        overrides: Mapping[str, Any] | None = None,
        streaming: bool = False,
    ) -> Runtime:
        runtime_overrides = _json_mapping(overrides, "runtime overrides")
        collector: AsyncExitStack | None = None
        collector_client: _AtofCollectorClient | None = None
        runtime_config = config

        async def close_streaming_resources() -> None:
            try:
                if collector_client is not None:
                    await collector_client.aclose()
            finally:
                if collector is not None:
                    await collector.aclose()

        if launch_collector is not None and not streaming:
            raise FabricConfigError("launch_collector requires streaming=True")
        if streaming and not _relay_enabled(config):
            raise FabricConfigError("streaming requires Relay telemetry to be enabled")
        if streaming:
            try:
                if launch_collector is not False:
                    try:
                        from nemo_fabric_collector import serve_collector
                    except ImportError as error:
                        raise FabricConfigError(
                            "local adapter streaming requires the collector; "
                            "install nemo-fabric[streaming]"
                        ) from error
                    collector = AsyncExitStack()
                    collector_base_url = await collector.enter_async_context(
                        serve_collector(host="127.0.0.1", port=0, standalone=True)
                    )
                    runtime_config = _with_stream_sink(config, collector_base_url)
                    stream_sink = _configured_stream_sink(runtime_config)
                    if stream_sink is None:
                        raise RuntimeError("failed to configure the ATOF collector")
                else:
                    stream_sink = _configured_stream_sink(config)
                    if stream_sink is None:
                        raise FabricConfigError(
                            "external collector streaming requires a configured "
                            "nemo-fabric-stream collector sink"
                        )
                collector_client = _AtofCollectorClient.from_sink(stream_sink)
                if runtime_config is config:
                    runtime_config = config.model_copy(deep=True)
                runtime_stream_sink = _configured_stream_sink(runtime_config)
                if runtime_stream_sink is not None:
                    runtime_stream_sink.url = (
                        f"{collector_client.base_url}/v1/atof"
                    )
            except asyncio.CancelledError:
                await close_streaming_resources()
                raise
            except FabricError:
                await close_streaming_resources()
                raise
            except Exception as error:
                await close_streaming_resources()
                raise FabricRuntimeError(
                    str(error),
                    stage="start",
                    code="collector_start_failed",
                ) from error

        try:
            plan = await _call_blocking(
                lambda: self.plan(runtime_config, base_dir=base_dir)
            )
            method = "start_runtime_in" if environment is not None else "start_runtime"
            native = self._require_native_module(method)
        except BaseException:
            await close_streaming_resources()
            raise
        started_runtime: dict[str, Any] | None = None

        def start() -> dict[str, Any]:
            nonlocal started_runtime
            plan_json = json.dumps(plan.to_mapping())
            if environment is None:
                raw = native.start_runtime(plan_json)
            else:
                raw = native.start_runtime_in(plan_json, _environment_json(environment))
            started_runtime = json.loads(raw)
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
            await close_streaming_resources()
            raise
        except FabricError:
            await close_streaming_resources()
            raise
        except Exception as error:
            await close_streaming_resources()
            raise FabricRuntimeError(str(error), stage="start") from error
        return Runtime(
            client=self,
            plan=plan,
            runtime=runtime,
            overrides=runtime_overrides,
            collector=collector,
            collector_client=collector_client,
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


def _environment_json(environment: EnvironmentHandle) -> str:
    if not isinstance(environment, EnvironmentHandle):
        if isinstance(environment, Mapping):
            raise FabricConfigError(
                "environment mappings are not accepted directly; "
                "use EnvironmentHandle.from_mapping(...) first"
            )
        raise FabricConfigError("environment must be an EnvironmentHandle")
    return json.dumps(environment.to_mapping())


def _environment_reference_json(reference: EnvironmentReference) -> str:
    if not isinstance(reference, EnvironmentReference):
        if isinstance(reference, Mapping):
            raise FabricConfigError(
                "reference mappings are not accepted directly; "
                "use EnvironmentReference.from_mapping(...) first"
            )
        raise FabricConfigError("reference must be an EnvironmentReference")
    return json.dumps(reference.to_mapping())


def _base_dir_arg(base_dir: str | os.PathLike[str] | None) -> str | None:
    return None if base_dir is None else os.fspath(base_dir)
