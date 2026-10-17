#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Configure the Hermes code-review workload for the shared OpenShell launcher.
# These hooks set up model access and run the normal host SDK consumer.
WORKLOAD_NAME=code-review-hermes
SUPPORTED_MODES=(caller_owned fabric_owned)
IMAGE_DOCKERFILE="${WORKLOAD_ROOT}/Dockerfile.openshell-hermes-sdk"
POLICY_FILE="${WORKLOAD_ROOT}/openshell-sdk-policy.yaml"
IMAGE_TAG="fabric-code-review-hermes-sdk:local"
SERVER_ENV_ARGS=(--env ADAPTER_PYTHON=/opt/fabric/bin/python --env TMPDIR=/sandbox/tmp)
MODEL_ENV_ARGS=(--provider hosted)

stage_workload_image() { :; }

configure_workload_gateway() {
  local provider_profile="${OPENSHELL_MODEL_PROVIDER_PROFILE:-${WORKLOAD_ROOT}/openshell-provider.yaml}"
  local model_api_key required_key provider_type
  if [[ -n "${OPENSHELL_MODEL_PROVIDER_PROFILE:-}" ]]; then
    model_api_key="${OPENAI_API_KEY:-}"
    required_key=OPENAI_API_KEY
  else
    model_api_key="${NVIDIA_API_KEY:-}"
    required_key=NVIDIA_API_KEY
  fi
  if [[ -z "${model_api_key}" ]]; then
    echo "ERROR: ${required_key} is required to configure the model provider" >&2
    return 2
  fi
  provider_type="$(sed -nE 's/^id:[[:space:]]*([[:alnum:]][[:alnum:]_.-]*)[[:space:]]*$/\1/p' "${provider_profile}")"
  if [[ -z "${provider_type}" ]]; then
    echo "ERROR: provider profile must contain a top-level unquoted id" >&2
    return 2
  fi
  local endpoint="http://127.0.0.1:${GATEWAY_PORT}"
  "${OPENSHELL_CLI}" --gateway-endpoint "${endpoint}" \
    provider profile import --file "${provider_profile}"
  OPENAI_API_KEY="${model_api_key}" "${OPENSHELL_CLI}" \
    --gateway-endpoint "${endpoint}" provider create --name hosted \
    --type "${provider_type}" --credential OPENAI_API_KEY
}

run_workload_consumer() {
  local ownership="$1"
  shift
  local relay_args=()
  if [[ "${OPENSHELL_CODE_REVIEW_RELAY:-false}" == "true" ]]; then
    relay_args=(--relay)
  fi
  local model_args=()
  if [[ -n "${OPENSHELL_CODE_REVIEW_MODEL:-}" ]]; then
    model_args+=(--model "${OPENSHELL_CODE_REVIEW_MODEL}")
  fi
  if [[ -n "${OPENSHELL_CODE_REVIEW_BASE_URL:-}" ]]; then
    model_args+=(--base-url "${OPENSHELL_CODE_REVIEW_BASE_URL}")
  fi
  uv run --no-sync python -m examples.code_review_agent.openshell_sdk_consumer \
    --ownership "${ownership}" \
    --harness hermes \
    "${relay_args[@]}" \
    --gateway "http://127.0.0.1:${GATEWAY_PORT}" \
    --image "${RUNTIME_IMAGE}" \
    --policy-file "${POLICY_FILE}" \
    --results-file "${RESULT_DIR}/${ownership}-report.json" \
    "${model_args[@]}" "$@"
}
