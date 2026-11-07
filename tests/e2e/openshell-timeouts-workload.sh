#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Workload hooks for examples/openshell/run-demo.sh. The host Python environment
# must include pytest and the current SDK; the launcher builds the task image.
WORKLOAD_NAME=openshell-timeouts
SUPPORTED_MODES=(fabric_owned)
IMAGE_DOCKERFILE="${FABRIC_ROOT}/examples/openshell_echo/runtime-image.Dockerfile"
POLICY_FILE="${FABRIC_ROOT}/examples/openshell_echo/policy.yaml"
IMAGE_TAG=fabric-openshell-echo:timeout-test

stage_workload_image() { :; }

run_workload_consumer() {
  OPENSHELL_TIMEOUT_TEST_GATEWAY="http://127.0.0.1:${GATEWAY_PORT}" \
    OPENSHELL_TIMEOUT_TEST_IMAGE="${RUNTIME_IMAGE}" \
    "${DEMO_PYTHON_ENV}/bin/python" -m pytest tests/e2e/test_openshell_timeouts.py -v -s
}
