# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Normalized response-limit authoring and task transport budgets."""

import json

import pytest
from pydantic import ValidationError

from nemo_fabric import ResponseLimits, RuntimeConfig
from nemo_fabric.runtime_server.protocol import encode_response

FIELDS = (
    "max_response_bytes",
    "max_artifact_bytes",
    "max_artifact_files",
    "max_artifact_file_bytes",
)


@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("value", [0, -1, 1.5, True, "12", None, 1 << 32])
def test_response_limits_reject_invalid_values(field, value):
    with pytest.raises(ValidationError):
        ResponseLimits(**{field: value})


def test_response_limits_omitted_fields_retain_defaults():
    assert RuntimeConfig().response_limits is None
    limits = RuntimeConfig(
        response_limits={"max_response_bytes": 8_000_000}
    ).response_limits
    assert limits == ResponseLimits(max_response_bytes=8_000_000)
    assert limits.max_artifact_bytes == 2 * 1024 * 1024
    assert limits.max_artifact_files == 16
    assert limits.max_artifact_file_bytes == 1024 * 1024
    with pytest.raises(ValidationError):
        ResponseLimits(max_response_byte=100)


@pytest.mark.parametrize("delta", [-1, 0, 1])
def test_bridge_encoding_uses_the_configured_transport_budget(delta):
    response = {"operation": "invoke", "value": "é😀"}
    encoded = (
        json.dumps(response, ensure_ascii=False, separators=(",", ":")).encode() + b"\n"
    )
    wire = encode_response(response, limit=len(encoded) + delta)
    if delta >= 0:
        assert wire == encoded
    else:
        assert json.loads(wire)["error"]["code"] == "response_too_large"


def test_bridge_can_carry_response_above_old_four_mib_cap():
    response = {"operation": "invoke", "value": "x" * (4 * 1024 * 1024)}
    wire = encode_response(response, limit=8 * 1024 * 1024)
    assert json.loads(wire) == response
