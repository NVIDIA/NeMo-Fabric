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
