#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Configure the OpenClaw code-review workload for the shared OpenShell launcher.
# These hooks set up model access and run the normal host SDK consumer.
WORKLOAD_NAME=code-review
SUPPORTED_MODES=(caller_owned fabric_owned)
IMAGE_DOCKERFILE="${WORKLOAD_ROOT}/Dockerfile.openshell-sdk"
POLICY_FILE="${WORKLOAD_ROOT}/openshell-sdk-policy.yaml"
IMAGE_TAG="fabric-code-review-sdk:local"
SERVER_ENV_ARGS=(--env ADAPTER_PYTHON=/opt/fabric/bin/python --env TMPDIR=/sandbox/tmp)
MODEL_ENV_ARGS=(--provider hosted)

stage_workload_image() { :; }

configure_workload_gateway() {
  if [[ -z "${NVIDIA_API_KEY:-}" ]]; then
    echo "ERROR: NVIDIA_API_KEY is required to configure the hosted model provider" >&2
    return 2
  fi
  local endpoint="http://127.0.0.1:${GATEWAY_PORT}"
  "${OPENSHELL_CLI}" --gateway-endpoint "${endpoint}" \
    provider profile import --file "${WORKLOAD_ROOT}/openshell-provider.yaml"
  OPENAI_API_KEY="${NVIDIA_API_KEY}" "${OPENSHELL_CLI}" \
    --gateway-endpoint "${endpoint}" provider create --name hosted \
    --type nvidia-code-review --credential OPENAI_API_KEY
}

run_workload_consumer() {
  local ownership="$1"
  shift
  uv run --no-sync python -m examples.code_review_agent.openshell_sdk_consumer \
    --ownership "${ownership}" \
    --gateway "http://127.0.0.1:${GATEWAY_PORT}" \
    --image "${RUNTIME_IMAGE}" \
    --policy-file "${POLICY_FILE}" \
    --results-file "${RESULT_DIR}/${ownership}-report.json" "$@"
}
