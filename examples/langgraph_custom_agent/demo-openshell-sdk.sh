#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Configure the custom LangGraph workload for the shared OpenShell launcher.
# These hooks set up model access and run the normal host SDK consumer.
WORKLOAD_NAME=langgraph-custom-agent
SUPPORTED_MODES=(caller_owned fabric_owned)
IMAGE_DOCKERFILE="${WORKLOAD_ROOT}/Dockerfile.openshell-sdk"
POLICY_FILE="${WORKLOAD_ROOT}/openshell-sdk-policy.yaml"
IMAGE_TAG="fabric-langgraph-custom-agent-sdk:local"
SERVER_ENV_ARGS=(--env ADAPTER_PYTHON=/opt/fabric/bin/python --env TMPDIR=/sandbox/tmp)
MODEL_ENV_ARGS=(--provider hosted)

stage_workload_image() { :; }

configure_workload_gateway() {
  local model_api_key="${OPENAI_API_KEY:-${NVIDIA_API_KEY:-}}"
  local provider_profile="${OPENSHELL_MODEL_PROVIDER_PROFILE:-${WORKLOAD_ROOT}/openshell-provider.yaml}"
  if [[ -z "${model_api_key}" ]]; then
    echo "ERROR: OPENAI_API_KEY or NVIDIA_API_KEY is required to configure the model provider" >&2
    return 2
  fi
  local endpoint="http://127.0.0.1:${GATEWAY_PORT}"
  "${OPENSHELL_CLI}" --gateway-endpoint "${endpoint}" \
    provider profile import --file "${provider_profile}"
  OPENAI_API_KEY="${model_api_key}" "${OPENSHELL_CLI}" \
    --gateway-endpoint "${endpoint}" provider create --name hosted \
    --type nvidia-langgraph --credential OPENAI_API_KEY
}

run_workload_consumer() {
  local ownership="$1"
  shift
  local relay_args=()
  local mcp_args=()
  local model_args=()
  if [[ "${OPENSHELL_LANGGRAPH_RELAY:-false}" == "true" ]]; then
    relay_args=(--relay)
  fi
  if [[ "${OPENSHELL_LANGGRAPH_MCP:-false}" == "true" ]]; then
    mcp_args=(--mcp)
  fi
  if [[ -n "${OPENSHELL_LANGGRAPH_MODEL:-}" ]]; then
    model_args+=(--model "${OPENSHELL_LANGGRAPH_MODEL}")
  fi
  if [[ -n "${OPENSHELL_LANGGRAPH_BASE_URL:-}" ]]; then
    model_args+=(--base-url "${OPENSHELL_LANGGRAPH_BASE_URL}")
  fi
  uv run --no-sync python -m examples.langgraph_custom_agent.consumer.openshell \
    --ownership "${ownership}" \
    --gateway "http://127.0.0.1:${GATEWAY_PORT}" \
    --image "${RUNTIME_IMAGE}" \
    --policy-file "${POLICY_FILE}" \
    --results-file "${RESULT_DIR}/${ownership}-report.json" \
    "${relay_args[@]}" "${mcp_args[@]}" "${model_args[@]}" "$@"
}
