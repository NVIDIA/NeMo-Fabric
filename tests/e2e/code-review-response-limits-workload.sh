#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Reuse the two-harness code-review image and controlled MCP source reader.
source "${FABRIC_ROOT}/tests/e2e/code-review-deadlines-workload.sh"
WORKLOAD_NAME=code-review-response-limits
EVIDENCE_FILE="${CODE_REVIEW_RESPONSE_EVIDENCE:?Set CODE_REVIEW_RESPONSE_EVIDENCE to a local report path}"
run_workload_consumer() {
  CODE_REVIEW_GATEWAY="http://127.0.0.1:${GATEWAY_PORT}" \
    CODE_REVIEW_IMAGE="${RUNTIME_IMAGE}" \
    CODE_REVIEW_RESPONSE_EVIDENCE="${EVIDENCE_FILE}" \
    CODE_REVIEW_RESPONSE_LIMIT_LIVE=1 \
    "${DEMO_PYTHON_ENV}/bin/python" -m pytest tests/e2e/test_code_review_response_limits.py -v -s
}
