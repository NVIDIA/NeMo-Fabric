# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SDK-owned initialization snapshots forwarded through normal configuration."""

from nemo_fabric_evidence.evidence import ExecutionEvidence
from nemo_fabric_evidence.metadata import collect_selected, combine
from nemo_fabric.models import FabricConfig


def sdk_snapshot(participant: str) -> ExecutionEvidence:
    """Read only this interpreter's selected SDK distribution and contract."""
    return collect_selected(
        participant,
        [
            ("fabric_sdk", "nemo-fabric-runtime", "nemo_fabric.client"),
            (
                "contract",
                "nemo-fabric-evidence",
                "nemo_fabric_evidence.evidence",
            ),
        ],
    )


def with_snapshot(config: FabricConfig, snapshot: ExecutionEvidence) -> FabricConfig:
    """Forward both caller reports and this installation; never inspect a host."""
    updated = config.model_copy(deep=True)
    report = config.provenance.reports
    updated.provenance.reports = (
        snapshot if report is None else combine(report, snapshot)
    )
    return updated


def _restore_dispatch_evidence(
    document: dict, forwarded: ExecutionEvidence | None, *, enabled: bool
) -> dict:
    """Receiving boundary for an explicit caller-to-guest configuration dispatch.

    Only execution evidence changes. The original terminal result and all other
    fields remain intact, including when an old intermediary dropped evidence.
    """
    if not enabled:
        document.pop("execution", None)
        return document
    guest = ExecutionEvidence.receive(document.get("execution"), "guest_sdk")
    if not any("guest_sdk" in env.participants for env in guest.software_environments):
        guest = combine(
            guest,
            ExecutionEvidence.gap("guest_sdk", "expected_guest_sdk_report_missing"),
        )
    if forwarded is not None:
        prefix = forwarded.to_mapping()["software_environments"]
        received = guest.to_mapping()
        retained = received["software_environments"][: len(prefix)] == prefix and (
            forwarded.infrastructure is None
            or received.get("infrastructure")
            == forwarded.to_mapping().get("infrastructure")
        )
        if not retained:
            guest = combine(forwarded, guest)
    document["execution"] = guest.to_mapping()
    return document
