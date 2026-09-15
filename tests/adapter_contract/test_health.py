# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for optional adapter health contracts."""

import pytest

from nemo_fabric_adapter_contract.codec import ContractValidationError
from nemo_fabric_adapter_contract.models import AdapterHealthRequest
from nemo_fabric_adapter_contract.models import AdapterHealthResult
from nemo_fabric_adapter_contract.models import AdapterReadiness
from nemo_fabric_adapter_contract.models import HealthCheck
from nemo_fabric_adapter_contract.models import HealthCheckStatus
from nemo_fabric_adapter_contract.models import RuntimeReadiness


def test_adapter_health_models_round_trip():
    result = AdapterHealthResult(
        readiness=AdapterReadiness(
            state=RuntimeReadiness.READY,
            reason_code="ready",
        ),
        checks=[
            HealthCheck(
                name="dependency.cache",
                status=HealthCheckStatus.OK,
                reason_code="cache_ready",
                observed_at_millis=10,
                age_millis=0,
                metadata={"region": "local"},
            )
        ],
    )

    restored = AdapterHealthResult.from_mapping(result.to_mapping())

    assert restored == result
    assert restored.checks[0].status is HealthCheckStatus.OK


def test_adapter_health_request_requires_positive_timeout():
    with pytest.raises(ContractValidationError, match="greater than zero"):
        AdapterHealthRequest(runtime_id="runtime-1", timeout_millis=0)
