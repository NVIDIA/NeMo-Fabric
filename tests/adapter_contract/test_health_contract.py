# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python adapters share strict health probe data with the native core."""

import pytest

from nemo_fabric_adapter_contract.models import AdapterCheckRequest, AdapterCheckResult
from nemo_fabric_adapter_contract.codec import ContractValidationError


@pytest.mark.parametrize(
    "source",
    [
        "fabric.adapter.pi.session",
        "fabric.adapter.openclaw.gateway",
        "fabric.adapter.openclaw.interface",
        "fabric.adapter.openclaw.readiness",
    ],
)
def test_check_probe_contract_round_trips_required_nulls(source):
    result = {
        "outcome": "passed",
        "reason_code": None,
        "activity": "busy",
        "source": source,
        "observed_at": "2026-10-19T12:00:00Z",
    }
    assert AdapterCheckResult.from_mapping(result).to_mapping() == result
    request = {"runtime_id": "runtime-1", "level": "live", "timeout_seconds": 0.5}
    assert AdapterCheckRequest.from_mapping(request).to_mapping() == request


@pytest.mark.parametrize(
    "fields",
    [
        {"runtime_id": "runtime-1", "level": "operational", "timeout_seconds": 1},
        {"runtime_id": "runtime-1", "level": "ready", "timeout_seconds": False},
        {"runtime_id": " ", "level": "live", "timeout_seconds": 1},
        {"runtime_id": "runtime-1", "level": "live", "timeout_seconds": 0},
        {"runtime_id": "runtime-1", "level": "live", "timeout_seconds": float("inf")},
        {
            "runtime_id": "runtime-1",
            "level": "live",
            "timeout_seconds": 1,
            "extra": "PRIVATE",
        },
    ],
)
def test_check_probe_requests_reject_invalid_data(fields):
    with pytest.raises(ContractValidationError):
        AdapterCheckRequest.from_mapping(fields)


@pytest.mark.parametrize(
    "changes",
    [
        {"source": "PRIVATE"},
        {"raw": "PRIVATE"},
        {"reason_code": "probe_error"},
        {"observed_at": None},
        {"observed_at": "2026-10-19T12:00:00"},
    ],
)
def test_check_probe_results_reject_invalid_evidence(changes):
    with pytest.raises(ContractValidationError):
        AdapterCheckResult.from_mapping(
            {
                "outcome": "passed",
                "reason_code": None,
                "activity": "idle",
                "source": "fabric.adapter.pi.session",
                "observed_at": "2026-10-19T12:00:00Z",
                **changes,
            }
        )
