# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native Python client for resolving and running NeMo Fabric agents."""

from __future__ import annotations

import importlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, overload

from nemo_fabric._config_sources import (
    AgentSource,
    PathProfiles,
    PathSource,
    config_json,
    config_profiles,
    is_config_source,
    path_arg,
    path_profiles,
    profiles_json,
    validate_base_dir,
)
from nemo_fabric.errors import (
    FabricCapabilityError,
    FabricConfigError,
    FabricError,
    FabricNativeUnavailableError,
    FabricRuntimeError,
)
from nemo_fabric.session import (
    Session,
    _call_blocking,
    _json_mapping,
    _require_session_runtime,
    _run_native_lifecycle,
    _run_request_payload,
)
from nemo_fabric.types import (
    DoctorReport,
    EffectiveConfig,
    FabricConfig,
    FabricProfileConfig,
    RunPlan,
    RunRequest,
    RunResult,
)

try:
    _native = importlib.import_module("nemo_fabric._native")
except ImportError:
    _native = None


class Fabric:
    """Primary Python entrypoint for NeMo Fabric.

    The client accepts either a path-backed agent package or a typed
    ``FabricConfig``. Path-backed sources select profiles by name; typed
    sources accept ordered ``FabricProfileConfig`` values and may use
    ``base_dir`` to resolve relative paths. All inspection and execution APIs
    return typed, read-only mapping models.

    ``Fabric`` is native-only. The ``fabric`` CLI is a separate public
    surface over the same Rust core; SDK calls raise
    ``FabricNativeUnavailableError`` when the native extension is not
    installed.

    The client is also an asynchronous context manager. Leaving the context
    does not stop independently created sessions; use each ``Session`` as
    an asynchronous context manager or call ``Session.stop()`` explicitly.

    See the Getting Started overview for runnable one-shot, typed-config, and
    multi-turn examples.
    """

    async def __aenter__(self) -> "Fabric":
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None

    @overload
    def resolve(
        self,
        agent: PathSource,
        *,
        profiles: PathProfiles | None = None,
        base_dir: None = None,
    ) -> EffectiveConfig: ...

    @overload
    def resolve(
        self,
        agent: FabricConfig,
        *,
        profiles: Sequence[FabricProfileConfig] | None = None,
        base_dir: PathSource | None = None,
    ) -> EffectiveConfig: ...

    def resolve(
        self,
        agent: AgentSource,
        *,
        profiles: PathProfiles | Sequence[FabricProfileConfig] | None = None,
        base_dir: PathSource | None = None,
    ) -> EffectiveConfig:
        """Resolve an agent source and its ordered profile overlays.

        Resolution validates and normalizes configuration but does not resolve
        an adapter or compute runtime capabilities. Use ``plan()`` when those
        execution details are required.

        Args:
            agent: Agent-package directory or config-file path, or a typed
                ``FabricConfig``. Raw mappings are not accepted; convert
                them with ``FabricConfig.from_mapping()``.
            profiles: One profile name or an ordered sequence of names for a
                path-backed source. For a typed source, an ordered sequence of
                ``FabricProfileConfig`` values.
            base_dir: Base directory for resolving relative paths in a typed
                config. Valid only when ``agent`` is a ``FabricConfig``.

        Returns:
            The normalized ``EffectiveConfig`` snapshot.

        Raises:
            FabricConfigError: If the source, profile stack, or resolved config
                is invalid.
            FabricNativeUnavailableError: If the native extension is not
                installed.
        """

        native = self._require_native_module("resolve")
        try:
            if is_config_source(agent):
                typed_profiles = config_profiles(profiles)  # type: ignore[arg-type]
                raw = native.resolve_config(
                    config_json(agent),
                    profiles_json(typed_profiles),
                    validate_base_dir(agent, base_dir),
                )
            else:
                validate_base_dir(agent, base_dir)
                raw = native.inspect(
                    path_arg(agent), path_profiles(profiles)  # type: ignore[arg-type]
                )
            return EffectiveConfig.from_mapping(json.loads(raw))
        except FabricError:
            raise
        except Exception as error:
            raise FabricConfigError(str(error)) from error

    @overload
    def plan(
        self,
        agent: PathSource,
        *,
        profiles: PathProfiles | None = None,
        base_dir: None = None,
    ) -> RunPlan: ...

    @overload
    def plan(
        self,
        agent: FabricConfig,
        *,
        profiles: Sequence[FabricProfileConfig] | None = None,
        base_dir: PathSource | None = None,
    ) -> RunPlan: ...

    def plan(
        self,
        agent: AgentSource,
        *,
        profiles: PathProfiles | Sequence[FabricProfileConfig] | None = None,
        base_dir: PathSource | None = None,
    ) -> RunPlan:
        """Resolve an agent source into an immutable execution plan.

        Planning applies profiles, resolves the selected adapter, and reports
        the runtime capabilities that gate session, service, streaming, update,
        cancellation, and concurrency APIs. It does not start the runtime.

        Args:
            agent: Agent-package directory or config-file path, or a typed
                ``FabricConfig``. Raw mappings are not accepted.
            profiles: One profile name or an ordered sequence of names for a
                path-backed source. For a typed source, an ordered sequence of
                ``FabricProfileConfig`` values.
            base_dir: Base directory for resolving relative paths in a typed
                config. Valid only when ``agent`` is a ``FabricConfig``.

        Returns:
            A ``RunPlan`` containing the effective config, adapter, and
            declared runtime capabilities.

        Raises:
            FabricConfigError: If the source, profile stack, config, or adapter
                resolution is invalid.
            FabricNativeUnavailableError: If the native extension is not
                installed.
        """

        native = self._require_native_module("plan")
        try:
            if is_config_source(agent):
                typed_profiles = config_profiles(profiles)  # type: ignore[arg-type]
                raw = native.plan_config(
                    config_json(agent),
                    profiles_json(typed_profiles),
                    validate_base_dir(agent, base_dir),
                )
            else:
                validate_base_dir(agent, base_dir)
                raw = native.plan(
                    path_arg(agent), path_profiles(profiles)  # type: ignore[arg-type]
                )
            return RunPlan.from_mapping(json.loads(raw))
        except FabricError:
            raise
        except Exception as error:
            raise FabricConfigError(str(error)) from error

    @overload
    async def doctor(
        self,
        agent: PathSource,
        *,
        profiles: PathProfiles | None = None,
        base_dir: None = None,
    ) -> DoctorReport: ...

    @overload
    async def doctor(
        self,
        agent: FabricConfig,
        *,
        profiles: Sequence[FabricProfileConfig] | None = None,
        base_dir: PathSource | None = None,
    ) -> DoctorReport: ...

    async def doctor(
        self,
        agent: AgentSource,
        *,
        profiles: PathProfiles | Sequence[FabricProfileConfig] | None = None,
        base_dir: PathSource | None = None,
    ) -> DoctorReport:
        """Diagnose a planned agent without starting its runtime.

        Doctor checks the resolved adapter, capability mappings, and declared
        environment requirements using the native Fabric core.

        Args:
            agent: Agent-package directory or config-file path, or a typed
                ``FabricConfig``.
            profiles: One profile name or an ordered sequence of names for a
                path-backed source. For a typed source, an ordered sequence of
                ``FabricProfileConfig`` values.
            base_dir: Base directory for resolving relative paths in a typed
                config. Valid only when ``agent`` is a ``FabricConfig``.

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
            if is_config_source(agent):
                typed_profiles = config_profiles(profiles)  # type: ignore[arg-type]
                raw = native.doctor_config(
                    config_json(agent),
                    profiles_json(typed_profiles),
                    validate_base_dir(agent, base_dir),
                )
            else:
                validate_base_dir(agent, base_dir)
                raw = native.doctor(
                    path_arg(agent), path_profiles(profiles)  # type: ignore[arg-type]
                )
            return DoctorReport.from_mapping(json.loads(raw))

        try:
            return await _call_blocking(diagnose)
        except FabricError:
            raise
        except Exception as error:
            raise FabricConfigError(str(error)) from error

    @overload
    async def run(
        self,
        agent: PathSource,
        *,
        profiles: PathProfiles | None = None,
        base_dir: None = None,
        input: Any = None,
        input_file: str | Path | None = None,
        request: RunRequest | Mapping[str, Any] | None = None,
        request_file: str | Path | None = None,
        request_id: str | None = None,
        context: Mapping[str, Any] | None = None,
        overrides: Mapping[str, Any] | None = None,
    ) -> RunResult: ...

    @overload
    async def run(
        self,
        agent: FabricConfig,
        *,
        profiles: Sequence[FabricProfileConfig] | None = None,
        base_dir: PathSource | None = None,
        input: Any = None,
        input_file: str | Path | None = None,
        request: RunRequest | Mapping[str, Any] | None = None,
        request_file: str | Path | None = None,
        request_id: str | None = None,
        context: Mapping[str, Any] | None = None,
        overrides: Mapping[str, Any] | None = None,
    ) -> RunResult: ...

    async def run(
        self,
        agent: AgentSource,
        *,
        profiles: PathProfiles | Sequence[FabricProfileConfig] | None = None,
        base_dir: PathSource | None = None,
        input: Any = None,
        input_file: str | Path | None = None,
        request: RunRequest | Mapping[str, Any] | None = None,
        request_file: str | Path | None = None,
        request_id: str | None = None,
        context: Mapping[str, Any] | None = None,
        overrides: Mapping[str, Any] | None = None,
    ) -> RunResult:
        """Execute one complete start, invoke, and stop lifecycle.

        Exactly zero or one of ``input``, ``input_file``, ``request``, and
        ``request_file`` may be supplied. Omitting all four produces an empty
        text input. A complete ``request`` or ``request_file`` cannot be mixed
        with separate ``request_id``, ``context``, or ``overrides`` fields.
        Fabric attempts to stop a started runtime even when invocation fails.

        Args:
            agent: Agent-package directory or config-file path, or a typed
                ``FabricConfig``.
            profiles: One profile name or an ordered sequence of names for a
                path-backed source. For a typed source, an ordered sequence of
                ``FabricProfileConfig`` values.
            base_dir: Base directory for resolving relative paths in a typed
                config. Valid only when ``agent`` is a ``FabricConfig``.
            input: JSON-compatible invocation input.
            input_file: UTF-8 file whose contents become the invocation input.
            request: Complete ``RunRequest`` or compatible mapping.
            request_file: UTF-8 JSON file containing a complete request.
            request_id: Caller-owned request identifier. Fabric generates one
                when omitted.
            context: Caller-owned, JSON-compatible request metadata.
            overrides: JSON-compatible invocation-scoped config overrides.

        Returns:
            The normalized ``RunResult``, including output, artifacts,
            telemetry references, lifecycle events, and structured error data.

        Raises:
            FabricConfigError: If sources are combined, request data is not
                JSON-compatible, or config resolution fails.
            FabricNativeUnavailableError: If the native extension is not
                installed.
            FabricRuntimeError: If the native runtime lifecycle fails before a
                normalized result can be returned.
        """

        plan = await _call_blocking(
            lambda: self.plan(  # type: ignore[arg-type]
                agent, profiles=profiles, base_dir=base_dir
            )
        )
        request_payload = _run_request_payload(
            input=input,
            input_file=input_file,
            request=request,
            request_file=request_file,
            request_id=request_id,
            context=context,
            overrides=overrides,
        )
        native = self._require_native_module("run")
        return RunResult.from_mapping(
            await _run_native_lifecycle(native, plan.to_mapping(), request_payload)
        )

    @overload
    async def start_session(
        self,
        agent: PathSource,
        *,
        profiles: PathProfiles | None = None,
        base_dir: None = None,
        session_id: str | None = None,
        overrides: Mapping[str, Any] | None = None,
    ) -> Session: ...

    @overload
    async def start_session(
        self,
        agent: FabricConfig,
        *,
        profiles: Sequence[FabricProfileConfig] | None = None,
        base_dir: PathSource | None = None,
        session_id: str | None = None,
        overrides: Mapping[str, Any] | None = None,
    ) -> Session: ...

    async def start_session(
        self,
        agent: AgentSource,
        *,
        profiles: PathProfiles | Sequence[FabricProfileConfig] | None = None,
        base_dir: PathSource | None = None,
        session_id: str | None = None,
        overrides: Mapping[str, Any] | None = None,
    ) -> Session:
        """Start a stateful, multi-turn session runtime.

        The resolved plan must declare the session capability. Each call starts
        a new runtime. ``session_id`` is the stable conversation identifier; if
        omitted, the new runtime identifier is used. Session-scoped overrides
        are recursively merged below invocation-scoped overrides.

        Args:
            agent: Agent-package directory or config-file path, or a typed
                ``FabricConfig``.
            profiles: One profile name or an ordered sequence of names for a
                path-backed source. For a typed source, an ordered sequence of
                ``FabricProfileConfig`` values.
            base_dir: Base directory for resolving relative paths in a typed
                config. Valid only when ``agent`` is a ``FabricConfig``.
            session_id: Stable caller-owned conversation identifier. Defaults
                to the generated runtime identifier.
            overrides: JSON-compatible overrides applied to every invocation
                in the session unless superseded by invocation overrides.

        Returns:
            An active ``Session``. Use it as an asynchronous context
            manager to guarantee runtime shutdown.

        Raises:
            FabricConfigError: If inputs or overrides are invalid.
            FabricNativeUnavailableError: If the native extension is not
                installed.
            FabricCapabilityError: If the resolved runtime does not support
                sessions.
            FabricRuntimeError: If runtime startup fails.
        """

        session_overrides = _json_mapping(overrides, "session overrides")
        plan = await _call_blocking(
            lambda: self.plan(  # type: ignore[arg-type]
                agent, profiles=profiles, base_dir=base_dir
            )
        )
        _require_session_runtime(plan, "start_session")
        native = self._require_native_module("start_session")
        try:
            runtime = await _call_blocking(
                lambda: json.loads(native.start_runtime(json.dumps(plan.to_mapping())))
            )
        except FabricError:
            raise
        except Exception as error:
            raise FabricRuntimeError(str(error), stage="start") from error
        return Session(
            client=self,
            plan=plan,
            runtime=runtime,
            overrides=session_overrides,
            session_id=session_id,
        )

    @overload
    async def start_service(
        self,
        agent: PathSource,
        *,
        profiles: PathProfiles | None = None,
        base_dir: None = None,
        service_id: str | None = None,
        overrides: Mapping[str, Any] | None = None,
    ) -> Any: ...

    @overload
    async def start_service(
        self,
        agent: FabricConfig,
        *,
        profiles: Sequence[FabricProfileConfig] | None = None,
        base_dir: PathSource | None = None,
        service_id: str | None = None,
        overrides: Mapping[str, Any] | None = None,
    ) -> Any: ...

    async def start_service(
        self,
        agent: AgentSource,
        *,
        profiles: PathProfiles | Sequence[FabricProfileConfig] | None = None,
        base_dir: PathSource | None = None,
        service_id: str | None = None,
        overrides: Mapping[str, Any] | None = None,
    ) -> Any:
        """Validate a service request and report the unsupported operation.

        Service handles are part of the reserved SDK contract, but the current
        Fabric runtime does not implement service creation. This method validates
        inputs and resolves the plan before raising
        ``FabricCapabilityError`` with code ``service_not_supported``.

        Args:
            agent: Agent-package directory or config-file path, or a typed
                ``FabricConfig``.
            profiles: One profile name or an ordered sequence of names for a
                path-backed source. For a typed source, an ordered sequence of
                ``FabricProfileConfig`` values.
            base_dir: Base directory for resolving relative paths in a typed
                config. Valid only when ``agent`` is a ``FabricConfig``.
            service_id: Reserved caller-owned service identifier.
            overrides: JSON-compatible service-scoped config overrides.

        Raises:
            FabricConfigError: If inputs or overrides are invalid.
            FabricNativeUnavailableError: If the native extension is not
                installed.
            FabricCapabilityError: Always, because service creation is not yet
                implemented.
        """

        _json_mapping(overrides, "service overrides")
        plan = await _call_blocking(
            lambda: self.plan(  # type: ignore[arg-type]
                agent, profiles=profiles, base_dir=base_dir
            )
        )
        raise FabricCapabilityError(
            "service mode is not implemented by this Fabric runtime",
            stage="start",
            code="service_not_supported",
            details={"service": plan.capabilities.service, "service_id": service_id},
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
