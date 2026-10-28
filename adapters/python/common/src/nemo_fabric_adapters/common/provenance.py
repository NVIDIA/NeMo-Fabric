# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Selected adapter-owned package observations at runtime initialization."""

import sys
from collections.abc import Mapping
from typing import Any

from nemo_fabric_adapter_contract.evidence import (
    ExecutionEvidence,
    SoftwareEntry,
    SoftwareEnvironment,
)
from nemo_fabric_adapter_contract.metadata import (
    collect_selected,
    observation,
    observed_at,
    combine,
)


def collection_enabled(payload: Mapping[str, Any]) -> bool:
    """Treat absent or null optional controls as the default-enabled setting."""
    control = payload.get("provenance")
    return not isinstance(control, Mapping) or control.get("enabled", True) is not False


def collect_runtime(runtime, payload) -> ExecutionEvidence | None:
    """Honor disable before looking up metadata; do not scan dependencies."""
    if not collection_enabled(payload):
        return None
    try:
        module = sys.modules.get(type(runtime).__module__)
        targets = [
            (
                "contract",
                "nemo-fabric-adapter-contract",
                "nemo_fabric_adapter_contract.models",
            ),
            ("contract", "nemo-fabric-evidence", "nemo_fabric_evidence.evidence"),
            (
                "shared_library",
                "nemo-fabric-adapters-common",
                "nemo_fabric_adapters.common.lifecycle",
            ),
            *(
                (role, name, type(runtime).__module__ if role == "adapter" else owner)
                for role, name, owner in getattr(module, "PROVENANCE_PACKAGES", ())
            ),
        ]
        telemetry = payload.get("telemetry_plan") or {}
        if telemetry.get("providers") and "nemo_relay" in sys.modules:
            # This names the integration actually enabled by this runtime.
            # Executable identity is forwarded only from an existing startup observation.
            targets.append(
                (
                    "telemetry"
                    if "native" in telemetry.get("providers", [])
                    else "relay",
                    "nemo-relay",
                    "nemo_relay",
                )
            )
        report = collect_selected("adapter", targets)
        expected_harness = getattr(module, "PROVENANCE_HARNESS", None)
        if expected_harness and report.software_environments[0].status == "available":
            report.software_environments[0].entries.append(
                SoftwareEntry(
                    role="harness",
                    kind="executable",
                    name=expected_harness,
                    version=observation(
                        None,
                        "adapter:startup",
                        observed_at(),
                        reason="no_existing_version_report",
                    ),
                )
            )
        if (
            not getattr(module, "PROVENANCE_PACKAGES", ())
            and report.software_environments[0].status == "available"
        ):
            report.software_environments[0].entries.append(
                SoftwareEntry(
                    role="adapter",
                    kind="component",
                    name=type(runtime).__module__,
                    version=observation(
                        None,
                        "adapter:startup",
                        observed_at(),
                        reason="no_selected_package_report",
                    ),
                )
            )
        forwarded = (payload.get("service") or {}).get("execution")
        if forwarded is not None:
            report = combine(report, ExecutionEvidence.receive(forwarded, "service"))
        return ExecutionEvidence.receive(report.to_mapping(), "adapter")
    except Exception:
        return ExecutionEvidence.gap("adapter", "collection_failed")


# The lifecycle coroutine owns this bounded sink. Child tasks inherit its source
# context; concurrent lifecycle hosts never share observations.
from contextvars import ContextVar

_passive: ContextVar[dict | None] = ContextVar("fabric_passive_versions", default=None)


def enable_capture(enabled: bool):
    _passive.set({} if enabled else None)


def record_version(
    name: str,
    value: str,
    source: str,
    *,
    role: str = "harness",
    kind: str = "executable",
    origin: str = "reported",
    participant: str = "adapter",
):
    """Forward an existing startup observation. This never initiates a probe."""
    sink = _passive.get()
    if (
        sink is None
        or not isinstance(value, str)
        or not value.strip()
        or len(sink) >= 8
    ):
        return
    key = (participant, role, kind, name)
    if key not in sink:
        try:
            sink[key] = SoftwareEntry(
                role=role,
                kind=kind,
                name=name,
                version=observation(value, source, observed_at(), origin=origin),
            )
        except Exception:
            pass


def refresh_passive(report: dict) -> dict:
    """Attach already captured values without refreshing their capture times."""
    try:
        decoded = ExecutionEvidence.from_mapping(report)
        for (participant, role, kind, name), entry in (_passive.get() or {}).items():
            env = next(
                (
                    env
                    for env in decoded.software_environments
                    if participant in env.participants and env.status == "available"
                ),
                None,
            )
            if env is None:
                refs = {env.context_ref for env in decoded.software_environments}
                index = 0
                while f"passive-{index}" in refs:
                    index += 1
                env = SoftwareEnvironment(
                    context_ref=f"passive-{index}",
                    participants=[participant],
                    status="available",
                    complete=False,
                    reason="selected_passive_observations_only",
                    entries=[],
                )
                decoded.software_environments.append(env)
            replacement = next(
                (
                    i
                    for i, old in enumerate(env.entries)
                    if (old.role, old.kind, old.name) == (role, kind, name)
                ),
                None,
            )
            if replacement is None:
                env.entries.append(entry)
            elif env.entries[replacement].version.value != entry.version.value:
                # An existing source observed a changed version. Equal values
                # retain the shared service's original capture time.
                env.entries[replacement] = entry
        return combine(decoded).to_mapping()
    except Exception:
        return ExecutionEvidence.gap(
            "adapter", "passive_report_invalid_or_over_budget"
        ).to_mapping()
