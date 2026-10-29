# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Selected software report shared by SDK and private lifecycle transports.

Reports describe observations, not an SBOM or an attestation. Transport readers
must isolate invalid optional evidence from the original invocation outcome.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from typing import Any
from typing import Literal

from nemo_fabric_evidence.codec import ContractValidationError
from nemo_fabric_evidence.models import ContractModel

MAX_REPORT_BYTES = 64 * 1024
MAX_ENVIRONMENTS = 16
MAX_ENTRIES = 128
EvidenceStatus = Literal[
    "available", "unavailable", "unsupported", "not_collected", "not_applicable"
]
EvidenceOrigin = Literal["observed", "reported"]


def _optional():
    return field(default=None, metadata={"omit_none": True})


def _nonblank(value: str) -> None:
    if not value.strip():
        raise ContractValidationError("evidence strings must be non-blank")


@dataclass(slots=True, kw_only=True)
class VersionObservation(ContractModel):
    """Full version or a reason for missing evidence, with original capture time."""

    status: EvidenceStatus
    source: str
    origin: EvidenceOrigin
    observed_at: str
    value: str | None = _optional()
    reason: str | None = _optional()

    def _validate(self) -> None:
        _nonblank(self.source)
        if not re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|\+00:00)",
            self.observed_at,
        ):
            raise ContractValidationError("observed_at must be a UTC timestamp")
        try:
            datetime.fromisoformat(self.observed_at)
        except ValueError:
            raise ContractValidationError(
                "observed_at must be a UTC timestamp"
            ) from None
        if self.status == "available":
            if self.value is None or self.reason is not None:
                raise ContractValidationError(
                    "available evidence requires only a value"
                )
            _nonblank(self.value)
        else:
            if self.value is not None or self.reason is None:
                raise ContractValidationError("missing evidence requires only a reason")
            _nonblank(self.reason)


@dataclass(slots=True, kw_only=True)
class SoftwareEntry(ContractModel):
    """One selected component; version is never truncated."""

    role: str
    kind: str
    name: str
    version: VersionObservation

    def _validate(self) -> None:
        for value in (self.role, self.kind, self.name):
            _nonblank(value)


@dataclass(slots=True, kw_only=True)
class SoftwareEnvironment(ContractModel):
    """Inventory for one observed installation/resolution context."""

    context_ref: str
    participants: list[str]
    status: Literal["available", "unavailable", "unsupported", "not_collected"]
    complete: bool
    entries: list[SoftwareEntry]
    reason: str | None = _optional()

    def _validate(self) -> None:
        _nonblank(self.context_ref)
        if not self.participants:
            raise ContractValidationError("inventory requires participants")
        for participant in self.participants:
            _nonblank(participant)
        if self.status != "available" and (self.complete or self.entries):
            raise ContractValidationError(
                "unavailable inventory cannot claim coverage or entries"
            )
        if self.status != "available" or not self.complete:
            if self.reason is None:
                raise ContractValidationError("incomplete inventory requires a reason")
            _nonblank(self.reason)


@dataclass(slots=True, kw_only=True)
class SandboxSoftwareReport(ContractModel):
    """Existing provider and gateway version report."""

    context_refs: list[str]
    provider: VersionObservation
    service_version: VersionObservation


@dataclass(slots=True, kw_only=True)
class ContainerSoftwareReport(ContractModel):
    """Existing owner report of an engine server, never its CLI version."""

    context_refs: list[str]
    engine: VersionObservation
    engine_version: VersionObservation


@dataclass(slots=True, kw_only=True)
class InfrastructureSoftware(ContractModel):
    """Conditional infrastructure reports; no discovery is performed."""

    sandbox: SandboxSoftwareReport | None = _optional()
    container: ContainerSoftwareReport | None = _optional()


@dataclass(slots=True, kw_only=True)
class ExecutionEvidence(ContractModel):
    """Bounded software inventories transported inline with each RunResult."""

    software_environments: list[SoftwareEnvironment]
    infrastructure: InfrastructureSoftware | None = _optional()

    def _validate(self) -> None:
        refs = {env.context_ref for env in self.software_environments}
        if len(refs) != len(self.software_environments):
            raise ContractValidationError("context_ref values must be unique")
        if (
            len(refs) > MAX_ENVIRONMENTS
            or sum(len(env.entries) for env in self.software_environments) > MAX_ENTRIES
        ):
            raise ContractValidationError(
                "software report exceeds entry/context budget"
            )
        if self.infrastructure is not None:
            for report in (self.infrastructure.sandbox, self.infrastructure.container):
                if report is not None and (
                    not report.context_refs or not set(report.context_refs) <= refs
                ):
                    raise ContractValidationError(
                        "infrastructure must reference reported contexts"
                    )
        if (
            len(
                json.dumps(
                    self.to_mapping(), ensure_ascii=False, separators=(",", ":")
                ).encode("utf-8")
            )
            > MAX_REPORT_BYTES
        ):
            raise ContractValidationError("software report exceeds byte budget")

    @classmethod
    def gap(cls, participant: str, reason: str) -> ExecutionEvidence:
        """Expose a missing/invalid report at an implemented receiving boundary."""
        return cls(
            software_environments=[
                SoftwareEnvironment(
                    context_ref="env-0",
                    participants=[participant],
                    status="unavailable",
                    complete=False,
                    reason=reason,
                    entries=[],
                )
            ]
        )

    @classmethod
    def receive(cls, value: Any, participant: str) -> ExecutionEvidence:
        """Invalid optional evidence cannot change an invocation's outcome."""
        if value is None:
            return cls.gap(participant, "report_missing")
        try:
            if isinstance(value, cls):
                value = value.to_mapping()
            # Check size before constructing nested typed values.
            if (
                len(
                    json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode(
                        "utf-8"
                    )
                )
                > MAX_REPORT_BYTES
            ):
                return cls.gap(participant, "report_over_budget")
            return cls.from_mapping(value)
        except Exception:
            return cls.gap(participant, "report_invalid_or_over_budget")
