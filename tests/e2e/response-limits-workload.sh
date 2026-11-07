#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

WORKLOAD_NAME=response-limits
SUPPORTED_MODES=(fabric_owned)
IMAGE_DOCKERFILE="${FABRIC_ROOT}/tests/e2e/Dockerfile.response-limits"
POLICY_FILE="${FABRIC_ROOT}/examples/openshell_echo/policy.yaml"
IMAGE_TAG=fabric-response-limits:test
stage_workload_image() { :; }
run_workload_consumer() {
  RESPONSE_LIMIT_TEST_LIVE=1 \
    RESPONSE_LIMIT_TEST_GATEWAY="http://127.0.0.1:${GATEWAY_PORT}" \
    RESPONSE_LIMIT_TEST_IMAGE="${RUNTIME_IMAGE}" \
    "${DEMO_PYTHON_ENV}/bin/python" -m pytest tests/e2e/test_response_limits.py -v -s
}
