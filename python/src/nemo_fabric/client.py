# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native Python client for NeMo Fabric."""

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


class FabricClient:
    """Entrypoint for Fabric resolution, planning, diagnostics, and execution."""

    async def __aenter__(self) -> "FabricClient":
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
        """Resolve config and ordered profiles without planning execution."""

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
        """Resolve a source into an immutable execution plan."""

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
        """Diagnose a resolved plan without starting a runtime."""

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
        """Execute one complete runtime lifecycle."""

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
        """Create a session runtime from a path-backed or typed source."""

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
        """Reject service creation until the selected runtime declares support."""

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
