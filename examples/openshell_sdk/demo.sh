#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Workload hooks for the reusable examples/openshell/run-demo.sh launcher.
WORKLOAD_NAME=openshell-sdk
SUPPORTED_MODES=(caller_owned fabric_owned)
IMAGE_DOCKERFILE="${WORKLOAD_ROOT}/runtime-image.Dockerfile"
POLICY_FILE="${WORKLOAD_ROOT}/policy.yaml"
IMAGE_TAG="fabric-openshell-sdk-example:local"

stage_workload_image() { :; }

run_workload_consumer() {
  local ownership="$1"
  shift
  uv run --no-sync python -m examples.openshell_sdk.consumer \
    --ownership "${ownership}" \
    --gateway "http://127.0.0.1:${GATEWAY_PORT}" \
    --image "${RUNTIME_IMAGE}" \
    --policy-file "${POLICY_FILE}" \
    --results-file "${RESULT_DIR}/${ownership}-results.json" "$@"
}
