# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility imports for the shared NeMo Fabric evidence contract."""

from nemo_fabric_evidence.evidence import MAX_REPORT_BYTES as MAX_REPORT_BYTES
from nemo_fabric_evidence.evidence import MAX_ENVIRONMENTS as MAX_ENVIRONMENTS
from nemo_fabric_evidence.evidence import MAX_ENTRIES as MAX_ENTRIES
from nemo_fabric_evidence.evidence import EvidenceStatus as EvidenceStatus
from nemo_fabric_evidence.evidence import EvidenceOrigin as EvidenceOrigin
from nemo_fabric_evidence.evidence import VersionObservation as VersionObservation
from nemo_fabric_evidence.evidence import SoftwareEntry as SoftwareEntry
from nemo_fabric_evidence.evidence import SoftwareEnvironment as SoftwareEnvironment
from nemo_fabric_evidence.evidence import SandboxSoftwareReport as SandboxSoftwareReport
from nemo_fabric_evidence.evidence import (
    ContainerSoftwareReport as ContainerSoftwareReport,
)
from nemo_fabric_evidence.evidence import (
    InfrastructureSoftware as InfrastructureSoftware,
)
from nemo_fabric_evidence.evidence import ExecutionEvidence as ExecutionEvidence
