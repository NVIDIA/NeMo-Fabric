# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metadata-only adapter inspection for host-side admission."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from nemo_fabric.errors import FabricConfigError
from nemo_fabric.models import FabricConfig
from nemo_fabric.types import RunPlan


@dataclass(frozen=True)
class AdapterCapabilityProfile:
    """Declared runtime support, not observed execution provenance.

    A missing descriptor produces conservative false claims and no fingerprint.
    The fingerprint covers the core-normalized descriptor, not discovery paths.
    Registered workflows require both adapter and target metadata; their
    fingerprint covers both normalized descriptors. Attached services require
    separate route-specific admission.
    """

    adapter_id: str
    descriptor_sha256: str | None = None
    skills: bool = False
    mcp: bool = False
    atif: bool = False


def _adapter_descriptor_sha256(plan: RunPlan) -> str:
    """Fingerprint the descriptor selected by a plan, independently of its origin."""
    mapping = plan.to_mapping()
    resolved = mapping.get("adapter_descriptor")
    if resolved is None:
        raise FabricConfigError("The execution plan has no adapter descriptor")
    target = mapping.get("adapter_target_descriptor")
    return _descriptor_sha256(
        resolved["descriptor"], None if target is None else target["descriptor"]
    )


def _descriptor_sha256(
    descriptor: Mapping[str, Any], target_descriptor: Mapping[str, Any] | None = None
) -> str:
    value = (
        descriptor
        if target_descriptor is None
        else {"adapter": descriptor, "target": target_descriptor}
    )
    canonical = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _verify_adapter_descriptor(plan: RunPlan, expected_sha256: str) -> None:
    """Reject host/task descriptor drift before starting the planned runtime."""
    actual = _adapter_descriptor_sha256(plan)
    if actual != expected_sha256:
        raise FabricConfigError(
            f"Host/task adapter descriptor mismatch for {plan.adapter.adapter_id}: "
            f"expected {expected_sha256}, found {actual}. Install matching host "
            "catalog and task adapter versions, or supply matching external metadata."
        )


def _relay_atif_enabled(component: Mapping[str, Any]) -> bool:
    if not component.get("enabled", True):
        return False
    atif = component.get("config", {}).get("atif")
    if atif is None:
        return False
    if not isinstance(atif, Mapping) or not isinstance(
        atif.get("enabled", False), bool
    ):
        raise FabricConfigError(
            "Relay ATIF admission requires an object with a boolean enabled field"
        )
    return atif.get("enabled", False)


def inspect_adapter(
    config: FabricConfig,
    descriptor: Mapping[str, Any] | None,
    *,
    target_descriptor: Mapping[str, Any] | None = None,
) -> AdapterCapabilityProfile:
    """Inspect supplied adapter and optional workflow metadata without loading its SDK.

    Uses the existing native planner and validation contract. A temporary metadata
    copy is used only for inspection; it is never an execution origin. Task-local
    discovery paths are not accessed on the host. Callers supply official catalog
    metadata or an external adapter's canonical descriptor.

    Missing metadata permits basic execution with false optional-feature claims;
    requested skills, MCP, or telemetry fail rather than being silently dropped.
    Configured registered workflows require their canonical target metadata as
    well as adapter metadata. Their fingerprint covers both descriptors.
    """
    from nemo_fabric import Fabric

    if not isinstance(config, FabricConfig):
        raise FabricConfigError("Adapter inspection requires a typed FabricConfig")

    if config.workflow is None and target_descriptor is not None:
        raise FabricConfigError("Target metadata requires a configured workflow")
    if config.workflow is not None and (
        not isinstance(target_descriptor, Mapping) or descriptor is None
    ):
        raise FabricConfigError(
            "Workflow inspection requires adapter and target metadata"
        )
    adapter_id = (
        config.harness.adapter_id
        if config.harness is not None
        else target_descriptor.get("adapter_id")
        if target_descriptor is not None
        else None
    )
    if not isinstance(adapter_id, str) or not adapter_id:
        raise FabricConfigError("Adapter inspection requires a selected adapter")
    if descriptor is None:
        if (
            (config.skills is not None and config.skills.paths)
            or (config.mcp is not None and config.mcp.servers)
            or (config.telemetry is not None and config.telemetry.providers)
        ):
            raise FabricConfigError(
                f"Cannot confirm requested skills, MCP, or telemetry for {adapter_id}: "
                "host descriptor metadata is unavailable"
            )
        return AdapterCapabilityProfile(adapter_id=adapter_id)
    if (
        not isinstance(descriptor, Mapping)
        or descriptor.get("adapter_id") != adapter_id
    ):
        raise FabricConfigError(f"Host descriptor does not match adapter {adapter_id}")
    with TemporaryDirectory(prefix="nemo-fabric-inspection-") as directory:
        path = Path(directory) / "host.fabric-adapter.json"
        native = Fabric()._require_native_module("inspect_adapter")
        try:
            path.write_text(
                json.dumps(dict(descriptor), allow_nan=False), encoding="utf-8"
            )
            arguments = [json.dumps(config.to_mapping()), str(path)]
            if target_descriptor is not None:
                target_path = Path(directory) / "host.fabric-target.json"
                target_path.write_text(
                    json.dumps(dict(target_descriptor), allow_nan=False),
                    encoding="utf-8",
                )
                arguments.append(str(target_path))
            inspection = json.loads(native.inspect_adapter_metadata(*arguments))
        except Exception as error:
            if config.workflow is not None:
                raise FabricConfigError(
                    "Workflow target metadata or configuration is invalid"
                ) from None
            raise FabricConfigError(str(error)) from error
    normalized = inspection["descriptor"]
    accepts = normalized["config"].get("accepts", [])
    providers = normalized["telemetry"].get("providers", {})
    selected = config.telemetry.providers if config.telemetry is not None else {}
    atif = (
        "atif" in providers.get("native", {}).get("outputs", [])
        and "native" in selected
    )
    relay_atif = False
    if "relay" in selected:
        relay = (inspection.get("telemetry_plan") or {}).get("relay_config") or {}
        observability = [
            component
            for component in relay.get("components", [])
            if component.get("kind") == "observability"
        ]
        # Do not infer ambient configuration or ambiguous duplicate-component
        # precedence. A single explicitly enabled output confirms the claim.
        enabled_outputs = [
            _relay_atif_enabled(component) for component in observability
        ]
        requested_atif = any(enabled_outputs)
        if requested_atif and (
            len(observability) != 1
            or "atif" not in providers.get("relay", {}).get("outputs", [])
        ):
            raise FabricConfigError(
                f"Cannot confirm requested Relay ATIF output for {adapter_id}: "
                "the provider must declare ATIF and exactly one observability "
                "component must enable it"
            )
        if len(observability) == 1:
            relay_atif = enabled_outputs[0]
    return AdapterCapabilityProfile(
        adapter_id=adapter_id,
        descriptor_sha256=_descriptor_sha256(
            normalized, inspection.get("target_descriptor")
        ),
        skills="skills" in accepts,
        mcp="mcp" in accepts,
        atif=bool(atif or relay_atif),
    )
