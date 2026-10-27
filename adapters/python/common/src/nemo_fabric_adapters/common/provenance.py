# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Selected adapter-owned package observations at runtime initialization."""

import sys

from nemo_fabric_adapter_contract.evidence import ExecutionEvidence, SoftwareEntry
from nemo_fabric_adapter_contract.metadata import (
    collect_selected,
    observation,
    observed_at,
)


def collect_runtime(runtime, payload) -> ExecutionEvidence | None:
    """Honor disable before looking up metadata; do not scan dependencies."""
    if payload.get("provenance", {}).get("enabled", True) is False:
        return None
    try:
        module = sys.modules.get(type(runtime).__module__)
        targets = [
            (
                "contract",
                "nemo-fabric-adapter-contract",
                "nemo_fabric_adapter_contract.models",
            ),
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
    key = (role, kind, name)
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
        env = decoded.software_environments[0]
        if env.status != "available":
            return report
        for key, entry in (_passive.get() or {}).items():
            entries = env.entries
            replacement = next(
                (
                    i
                    for i, old in enumerate(entries)
                    if (old.role, old.kind, old.name) == key
                ),
                None,
            )
            if replacement is None:
                entries.append(entry)
            else:
                entries[replacement] = entry
        return ExecutionEvidence.receive(decoded.to_mapping(), "adapter").to_mapping()
    except Exception:
        return ExecutionEvidence.gap(
            "adapter", "passive_report_invalid_or_over_budget"
        ).to_mapping()
