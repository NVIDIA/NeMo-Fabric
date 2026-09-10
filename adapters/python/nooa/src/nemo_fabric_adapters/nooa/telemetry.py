# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NeMo Relay telemetry lifecycle for the OO Agents adapter."""

from __future__ import annotations

import asyncio
import importlib.metadata
import inspect
import logging
import os
import re
from collections.abc import Awaitable
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import TypeVar

from nemo_fabric_adapter_contract.models import AgentConfig
from nemo_fabric_adapter_contract.models import RuntimeContext
from nemo_fabric_adapters.common import relay_artifacts
import nemo_fabric_adapters.common.utils as common_utils

LOGGER = logging.getLogger(__name__)

_RELAY_MINIMUM = (0, 7, 2)
_RELAY_MAXIMUM = (0, 8, 0)
_RELAY_VERSION = re.compile(r"^(\d+)\.(\d+)\.(\d+)(?:\+[^\s]+)?$")
_QUARANTINE_NOTE = (
    "telemetry is disabled for later turns because an earlier Relay scope "
    "did not return to its original state"
)
_UNREADABLE = object()
T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class RelayReport:
    """Safe telemetry details attached to one normalized adapter result."""

    enabled: bool
    artifacts: tuple[dict[str, str], ...] = ()
    error: str | None = None
    quarantine_cause: str | None = None


@dataclass(frozen=True, slots=True)
class RelayInvocation:
    """Functional outcome and its independently managed telemetry outcome."""

    called: bool
    result: Any | None
    report: RelayReport | None


def _safe_fault(stage: str, error: BaseException) -> str:
    return f"{stage} failed ({type(error).__name__})"


def _join_faults(*faults: str | None) -> str | None:
    present = [fault for fault in faults if fault]
    return "; ".join(present) if present else None


def _relay_version() -> tuple[int, int, int]:
    try:
        value = importlib.metadata.version("nemo-relay")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(
            "a compatible nemo-relay package is not installed"
        ) from error
    match = _RELAY_VERSION.fullmatch(value)
    if match is None:
        raise RuntimeError("the installed nemo-relay version is not a stable release")
    version = tuple(int(part) for part in match.groups())
    if not (_RELAY_MINIMUM <= version < _RELAY_MAXIMUM):
        raise RuntimeError("nemo-relay must satisfy >=0.7.2,<0.8")
    return version


def _validate_config_path(runtime_context: RuntimeContext) -> None:
    telemetry = runtime_context.telemetry
    assert telemetry is not None
    if telemetry.config_path is None:
        raise RuntimeError("Relay telemetry config_path is required")
    configured = os.environ.get("FABRIC_RELAY_CONFIG_PATH")
    if configured is None:
        raise RuntimeError("FABRIC_RELAY_CONFIG_PATH is required")
    if Path(configured).resolve() != Path(telemetry.config_path).resolve():
        raise RuntimeError("Relay telemetry config_path does not match the environment")


def _scope_handle() -> Any:
    try:
        from nemo_relay import scope

        return scope.get_handle()
    except Exception:
        return None


def _scope_identity(handle: Any) -> Any:
    if handle is None:
        return None
    return getattr(handle, "uuid", _UNREADABLE)


def _scope_unchanged(baseline: Any) -> bool:
    baseline_identity = _scope_identity(baseline)
    current_identity = _scope_identity(_scope_handle())
    if baseline_identity is None:
        return current_identity is None
    if baseline_identity is _UNREADABLE:
        return False
    if current_identity is None or current_identity is _UNREADABLE:
        return False
    return bool(baseline_identity == current_identity)


def _artifact_snapshot(
    plugin_config: dict[str, Any],
) -> dict[Path, tuple[int, int, int, int]]:
    snapshot: dict[Path, tuple[int, int, int, int]] = {}
    for artifact in common_utils.collect_relay_artifacts(plugin_config):
        try:
            path = Path(artifact["path"])
            status = path.stat()
        except (KeyError, OSError, RuntimeError, TypeError, ValueError):
            continue
        snapshot[path] = (
            status.st_dev,
            status.st_ino,
            status.st_size,
            status.st_mtime_ns,
        )
    return snapshot


def _changed_artifacts(
    plugin_config: dict[str, Any],
    before: dict[Path, tuple[int, int, int, int]],
) -> tuple[dict[str, str], ...]:
    changed = []
    for artifact in common_utils.collect_relay_artifacts(plugin_config):
        try:
            path = Path(artifact["path"])
            status = path.stat()
        except (KeyError, OSError, RuntimeError, TypeError, ValueError):
            continue
        current = (
            status.st_dev,
            status.st_ino,
            status.st_size,
            status.st_mtime_ns,
        )
        if before.get(path) != current:
            changed.append(artifact)
    return tuple(changed)


class RelayTelemetry:
    """Runtime-local Relay policy and stale-scope quarantine state."""

    def __init__(
        self,
        *,
        agent_name: str,
        base_dir: Path,
        config: AgentConfig,
        scope_name: str = "nooa-interactive-agent-request",
    ) -> None:
        self._agent_name = agent_name
        self._base_dir = base_dir
        self._config = config
        self._scope_name = scope_name
        self._quarantine: str | None = None
        self._quarantine_cause: str | None = None

    def _plugin_config(self, runtime_context: RuntimeContext) -> dict[str, Any]:
        return common_utils.load_relay_plugin_config(
            {
                "agent_name": self._agent_name,
                "base_dir": str(self._base_dir),
                "config": self._config.to_mapping(),
                "runtime_context": runtime_context.to_mapping(),
            }
        )

    async def invoke(
        self,
        *,
        agent: Any,
        runtime_context: RuntimeContext,
        call: Callable[[], Awaitable[T]],
    ) -> RelayInvocation:
        telemetry = runtime_context.telemetry
        if telemetry is None or not telemetry.relay_enabled:
            return RelayInvocation(called=True, result=await call(), report=None)

        providers = telemetry.metadata.get("telemetry_providers", ["relay"])
        if not isinstance(providers, list) or any(
            provider != "relay" for provider in providers
        ):
            return RelayInvocation(
                called=False,
                result=None,
                report=RelayReport(
                    enabled=True,
                    error="OO Agents supports only Relay telemetry",
                ),
            )

        if self._quarantine is not None:
            return RelayInvocation(
                called=True,
                result=await call(),
                report=RelayReport(
                    enabled=True,
                    error=self._quarantine,
                    quarantine_cause=self._quarantine_cause,
                ),
            )

        try:
            _validate_config_path(runtime_context)
            _relay_version()
            common_utils.reject_ambient_relay_plugin_config()
            plugin_config = self._plugin_config(runtime_context)
            before = _artifact_snapshot(plugin_config)
            atif_before = relay_artifacts.snapshot_atif_files(plugin_config)
            from nemo_relay import ScopeType
            from nemo_relay import plugin
            from nemo_relay import scope
            from nooa.nemo_relay_middleware import install_nemo_relay
        except Exception as error:
            LOGGER.error(
                "OO Agents Relay setup failed (error_type=%s)",
                type(error).__name__,
            )
            return RelayInvocation(
                called=False,
                result=None,
                report=RelayReport(
                    enabled=True,
                    error=_safe_fault("Relay setup", error),
                ),
            )

        baseline = _scope_handle()
        called = False
        result: T | None = None
        scope_fault: str | None = None
        plugin_fault: str | None = None
        uninstall_fault: str | None = None
        try:
            async with plugin.plugin(plugin_config) as activation_report:
                common_utils.reject_inherited_relay_plugin_config(activation_report)
                uninstall = install_nemo_relay(agent.event_manager)
                try:
                    metadata = {
                        "nemo_fabric_request_id": runtime_context.request_id,
                        "nemo_fabric_invocation_id": runtime_context.invocation_id,
                        "nemo_fabric_runtime_id": runtime_context.runtime_id,
                    }
                    try:
                        with scope.scope(
                            self._scope_name,
                            ScopeType.Agent,
                            metadata=metadata,
                        ):
                            called = True
                            result = await call()
                    except asyncio.CancelledError:
                        raise
                    except Exception as error:
                        scope_fault = _safe_fault("Relay request scope", error)
                finally:
                    try:
                        await _await_if_needed(uninstall())
                    except Exception as error:
                        uninstall_fault = _safe_fault("Relay middleware cleanup", error)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            plugin_fault = _safe_fault("Relay plugin lifecycle", error)

        telemetry_fault = _join_faults(scope_fault, uninstall_fault, plugin_fault)
        if telemetry_fault is not None and not _scope_unchanged(baseline):
            self._quarantine = _QUARANTINE_NOTE
            self._quarantine_cause = telemetry_fault
            telemetry_fault = _join_faults(telemetry_fault, _QUARANTINE_NOTE)

        artifact_fault: str | None = None
        artifacts: tuple[dict[str, str], ...] = ()
        if called:
            try:
                if relay_artifacts.expects_local_atif(plugin_config):
                    finalized = await relay_artifacts.wait_for_finalized_atif(
                        plugin_config,
                        atif_before,
                    )
                    if finalized is None:
                        raise RuntimeError("Relay did not finalize the current ATIF")
                artifacts = _changed_artifacts(plugin_config, before)
            except Exception as error:
                artifact_fault = _safe_fault("Relay artifact finalization", error)
        telemetry_fault = _join_faults(telemetry_fault, artifact_fault)
        return RelayInvocation(
            called=called,
            result=result,
            report=RelayReport(
                enabled=True,
                artifacts=artifacts,
                error=telemetry_fault,
            ),
        )

    async def close(self) -> None:
        """Release runtime-local references; middleware is invocation-scoped."""

        self._quarantine = None
        self._quarantine_cause = None


async def _await_if_needed(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value
