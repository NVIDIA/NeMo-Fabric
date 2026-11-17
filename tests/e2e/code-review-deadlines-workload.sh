#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

WORKLOAD_NAME=code-review-deadlines
SUPPORTED_MODES=(fabric_owned)
IMAGE_DOCKERFILE="${FABRIC_ROOT}/tests/e2e/Dockerfile.code-review-deadlines"
POLICY_FILE="${CODE_REVIEW_POLICY:?Set CODE_REVIEW_POLICY to the endpoint-specific sandbox policy}"
IMAGE_TAG=fabric-code-review:deadline-qualification

stage_workload_image() {
  docker build --build-arg HERMES_COMMIT=bf0203a794a07aac3ff764972ab9087baa4e4f4e \
    -f "${FABRIC_ROOT}/examples/code_review_agent/Dockerfile.openshell-hermes-sdk" \
    -t fabric-code-review-hermes-sdk:deadline-qualification-base "${FABRIC_ROOT}"
}

run_workload_consumer() {
  CODE_REVIEW_GATEWAY="http://127.0.0.1:${GATEWAY_PORT}" \
    CODE_REVIEW_IMAGE="${RUNTIME_IMAGE}" \
    CODE_REVIEW_DEADLINE_LIVE=1 \
    "${DEMO_PYTHON_ENV}/bin/python" -m pytest tests/e2e/test_code_review_deadlines.py -v -s
}
