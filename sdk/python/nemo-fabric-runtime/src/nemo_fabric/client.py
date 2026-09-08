# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native Python client for resolving and running NVIDIA NeMo Fabric agents."""

from __future__ import annotations

import asyncio
import importlib
import json
import os
from collections.abc import Awaitable, Callable, Mapping
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

    def __init__(self) -> None:
        self._collector_lock = asyncio.Lock()
        self._collector_stack: AsyncExitStack | None = None
        self._collector_base_url: str | None = None
        self._collector_users = 0

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

        runtime_overrides = _json_mapping(overrides, "runtime overrides")
        collector_client: _AtofCollectorClient | None = None
        release_collector: Callable[[], Awaitable[None]] | None = None
        runtime_config = config

        async def close_streaming_resources() -> None:
            try:
                if collector_client is not None:
                    await collector_client.aclose()
            finally:
                if release_collector is not None:
                    await release_collector()

        if launch_collector is not None and not streaming:
            raise FabricConfigError("launch_collector requires streaming=True")
        if streaming and not _relay_enabled(config):
            raise FabricConfigError("streaming requires Relay telemetry to be enabled")
        if streaming:
            try:
                if launch_collector is not False:
                    collector_base_url, release_collector = (
                        await self._acquire_collector()
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
            native = self._require_native_module("start_runtime")
        except BaseException:
            await close_streaming_resources()
            raise
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
            collector_client=collector_client,
            release_collector=release_collector,
        )

    async def _acquire_collector(
        self,
    ) -> tuple[str, Callable[[], Awaitable[None]]]:
        async with self._collector_lock:
            if self._collector_stack is None:
                try:
                    from nemo_fabric_collector import serve_collector
                except ImportError as error:
                    raise FabricConfigError(
                        "local adapter streaming requires the collector; "
                        "install nemo-fabric[collector]"
                    ) from error
                stack = AsyncExitStack()
                try:
                    base_url = await stack.enter_async_context(
                        serve_collector(host="127.0.0.1", port=0)
                    )
                except BaseException:
                    await stack.aclose()
                    raise
                self._collector_stack = stack
                self._collector_base_url = base_url
            base_url = self._collector_base_url
            if base_url is None:
                raise RuntimeError("ATOF collector did not provide a base URL")
            self._collector_users += 1

        released = False

        async def release() -> None:
            nonlocal released
            if released:
                return
            released = True
            await self._release_collector()

        return base_url, release

    async def _release_collector(self) -> None:
        stack: AsyncExitStack | None = None
        async with self._collector_lock:
            if self._collector_users == 0:
                return
            self._collector_users -= 1
            if self._collector_users == 0:
                stack = self._collector_stack
                self._collector_stack = None
                self._collector_base_url = None
        if stack is not None:
            await stack.aclose()

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
