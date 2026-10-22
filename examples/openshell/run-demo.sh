#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Build a Fabric task image, create a local OpenShell gateway, and run one SDK
# consumer against a caller-owned and/or Fabric-owned sandbox.
set -euo pipefail

OPEN_SHELL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FABRIC_ROOT="$(cd "${OPEN_SHELL_ROOT}/../.." && pwd)"
OPENSHELL_VERSION="${OPENSHELL_VERSION:-v0.1.2}"
OPEN_SHELL_IMAGE_VERSION="${OPENSHELL_VERSION#v}"
GATEWAY_PORT="${OPENSHELL_EXAMPLE_PORT:-18080}"
GATEWAY_START_TIMEOUT="${OPENSHELL_EXAMPLE_GATEWAY_START_TIMEOUT:-120}"
EXAMPLE_MODE="${OPENSHELL_EXAMPLE_MODE:-both}"
DEMO_PYTHON_ENV="${UV_PROJECT_ENVIRONMENT:-${FABRIC_ROOT}/.tmp/host-stack-tests}"
if [[ $# -lt 1 || ! -f "$1" ]]; then
  echo "Usage: bash examples/openshell/run-demo.sh <workload/demo.sh> [consumer-options]" >&2
  exit 2
fi
if (true >/dev/tcp/127.0.0.1/"${GATEWAY_PORT}") 2>/dev/null; then
  echo "ERROR: 127.0.0.1:${GATEWAY_PORT} is already in use; set OPENSHELL_EXAMPLE_PORT to a free port" >&2
  exit 2
fi
WORKLOAD_ROOT="$(cd "$(dirname "$1")" && pwd)"
WORKLOAD_FILE="${WORKLOAD_ROOT}/$(basename "$1")"
shift
SUPPORTED_MODES=()
MODEL_ENV_ARGS=()
SERVER_ENV_ARGS=(--env ADAPTER_PYTHON=/opt/fabric/bin/python --env TMPDIR=/sandbox/tmp)
IMAGE_DOCKERFILE="${WORKLOAD_ROOT}/runtime-image.Dockerfile"
POLICY_FILE="${WORKLOAD_ROOT}/policy.yaml"
source "${WORKLOAD_FILE}"
if [[ ! "${WORKLOAD_NAME:-}" =~ ^[a-z0-9-]+$ ]]; then
  echo "ERROR: workload must declare WORKLOAD_NAME" >&2
  exit 2
fi
for hook in stage_workload_image run_workload_consumer; do
  if ! declare -F "${hook}" >/dev/null; then
    echo "ERROR: workload must define ${hook}" >&2
    exit 2
  fi
done
IMAGE_TAG="${IMAGE_TAG:-fabric-${WORKLOAD_NAME}:example}"
RESULT_DIR="${OPENSHELL_EXAMPLE_RESULTS_DIR:-${RESULT_DIR:-${FABRIC_ROOT}/.tmp/${WORKLOAD_NAME}}}"
mkdir -p "${RESULT_DIR}" "${FABRIC_ROOT}/.tmp"
RESULT_DIR="$(cd "${RESULT_DIR}" && pwd)"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${FABRIC_ROOT}/.tmp/uv-cache}"
GATEWAY_STATE_DIR="$(mktemp -d "${FABRIC_ROOT}/.tmp/openshell-demo.XXXXXX")"
OPEN_SHELL_CACHE_DIR="${FABRIC_ROOT}/.tmp/openshell-releases/${OPENSHELL_VERSION}"
OPEN_SHELL_BIN_DIR="${OPEN_SHELL_CACHE_DIR}/bin"
HOST_ARCH="$(uname -m)"
case "${HOST_ARCH}" in
  x86_64|amd64) HOST_ARCH=x86_64 ;;
  aarch64|arm64) HOST_ARCH=aarch64 ;;
  *) echo "ERROR: unsupported host architecture ${HOST_ARCH}" >&2; exit 2 ;;
esac
if [[ "$(uname -s)" != Linux ]]; then
  echo "ERROR: this local Docker-backed example currently requires Linux" >&2
  exit 2
fi
CLI_ASSET="openshell-${HOST_ARCH}-unknown-linux-musl.tar.gz"
GATEWAY_ASSET="openshell-gateway-${HOST_ARCH}-unknown-linux-gnu.tar.gz"
SANDBOX_ASSET="openshell-sandbox-${HOST_ARCH}-unknown-linux-musl.tar.gz"
OPENSHELL_CLI="${OPEN_SHELL_BIN_DIR}/openshell"
OPENSHELL_GATEWAY="${OPEN_SHELL_BIN_DIR}/openshell-gateway"
WORKLOAD_SLUG="${WORKLOAD_NAME:0:3}"
SANDBOX_NAME="fab-${WORKLOAD_SLUG}-$(python3 -c 'import uuid; print(uuid.uuid4().hex[:8])')"
FABRIC_SANDBOX_NAME="fab-${WORKLOAD_SLUG}-o-$(python3 -c 'import uuid; print(uuid.uuid4().hex[:8])')"
FABRIC_SANDBOX_STATE_FILE="${RESULT_DIR}/fabric-owned-sandbox.json"
rm -f -- "${FABRIC_SANDBOX_STATE_FILE}"
RUNTIME_IMAGE=""
GATEWAY_PID=""
EVIDENCE_PID=""
CALLER_SANDBOX_ATTEMPTED=""

for dependency in cargo curl docker openssl python3 tar uv; do
  command -v "${dependency}" >/dev/null || {
    echo "ERROR: ${dependency} is required" >&2
    exit 2
  }
done
docker info >/dev/null
if [[ "${EXAMPLE_MODE}" != caller_owned && "${EXAMPLE_MODE}" != fabric_owned && "${EXAMPLE_MODE}" != both ]]; then
  echo "ERROR: OPENSHELL_EXAMPLE_MODE must be caller_owned, fabric_owned, or both" >&2
  exit 2
fi
for requested_mode in caller_owned fabric_owned; do
  if [[ "${EXAMPLE_MODE}" == both || "${EXAMPLE_MODE}" == "${requested_mode}" ]]; then
    if [[ " ${SUPPORTED_MODES[*]} " != *" ${requested_mode} "* ]]; then
      echo "ERROR: ${WORKLOAD_NAME} does not support ${requested_mode}" >&2
      exit 2
    fi
  fi
done

download_asset() {
  local asset="$1"
  local sums="$2"
  local release_url="https://github.com/NVIDIA/OpenShell/releases/download/${OPENSHELL_VERSION}"
  mkdir -p "${OPEN_SHELL_CACHE_DIR}" "${OPEN_SHELL_BIN_DIR}"
  if [[ ! -s "${OPEN_SHELL_CACHE_DIR}/${sums}" ]]; then
    curl --fail --location --retry 3 --show-error --silent \
      "${release_url}/${sums}" -o "${OPEN_SHELL_CACHE_DIR}/${sums}.part"
    mv "${OPEN_SHELL_CACHE_DIR}/${sums}.part" "${OPEN_SHELL_CACHE_DIR}/${sums}"
  fi
  if [[ ! -s "${OPEN_SHELL_CACHE_DIR}/${asset}" ]]; then
    curl --fail --location --retry 3 --show-error --silent \
      "${release_url}/${asset}" -o "${OPEN_SHELL_CACHE_DIR}/${asset}.part"
    mv "${OPEN_SHELL_CACHE_DIR}/${asset}.part" "${OPEN_SHELL_CACHE_DIR}/${asset}"
  fi
  python3 - "${OPEN_SHELL_CACHE_DIR}/${asset}" "${OPEN_SHELL_CACHE_DIR}/${sums}" <<'PY'
import hashlib
from pathlib import Path
import sys

asset = Path(sys.argv[1])
checksums = Path(sys.argv[2])
expected = {
    name: digest
    for digest, name in (line.split() for line in checksums.read_text().splitlines())
}.get(asset.name)
if expected is None or hashlib.sha256(asset.read_bytes()).hexdigest() != expected:
    raise SystemExit(f"ERROR: missing or invalid release checksum for {asset.name}")
PY
  tar -xzf "${OPEN_SHELL_CACHE_DIR}/${asset}" -C "${OPEN_SHELL_BIN_DIR}"
}

cleanup() {
  local status=$?
  if [[ -n "${CALLER_SANDBOX_ATTEMPTED}" && -x "${OPENSHELL_CLI}" ]]; then
    if [[ -n "${EVIDENCE_PID}" ]]; then
      kill "${EVIDENCE_PID}" 2>/dev/null || true
      wait "${EVIDENCE_PID}" 2>/dev/null || true
    fi
    if [[ "${status}" -ne 0 ]] && declare -F collect_workload_diagnostics >/dev/null; then
      collect_workload_diagnostics || true
    fi
    if "${OPENSHELL_CLI}" --gateway-endpoint "http://127.0.0.1:${GATEWAY_PORT}" \
      --workspace default sandbox get "${SANDBOX_NAME}" --output json >/dev/null 2>&1; then
      if [[ "${status}" -ne 0 ]]; then
        "${OPENSHELL_CLI}" --gateway-endpoint "http://127.0.0.1:${GATEWAY_PORT}" \
          --workspace default logs "${SANDBOX_NAME}" -n 200 \
          >"${RESULT_DIR}/sandbox-logs.txt" 2>&1 || true
      fi
      if ! "${OPENSHELL_CLI}" --gateway-endpoint "http://127.0.0.1:${GATEWAY_PORT}" \
          --workspace default sandbox delete "${SANDBOX_NAME}" >/dev/null 2>&1; then
        echo "ERROR: could not delete caller-owned sandbox ${SANDBOX_NAME}" >&2
        status=1
      fi
    fi
  fi
  if [[ "${status}" -ne 0 && -s "${FABRIC_SANDBOX_STATE_FILE}" && -x "${OPENSHELL_CLI}" && -n "${GATEWAY_PID}" ]]; then
    local recorded_name recorded_id current_id current_json
    read -r recorded_name recorded_id < <(python3 -c 'import json,sys; state=json.load(open(sys.argv[1])); print(state.get("name", ""), state.get("id", ""))' "${FABRIC_SANDBOX_STATE_FILE}")
    if [[ -n "${recorded_name}" && -n "${recorded_id}" ]] && \
      current_json="$(${OPENSHELL_CLI} --gateway-endpoint "http://127.0.0.1:${GATEWAY_PORT}" --workspace default sandbox get "${recorded_name}" --output json 2>/dev/null)"; then
      current_id="$(python3 -c 'import json,sys; print(json.load(sys.stdin).get("id", ""))' <<<"${current_json}")"
      if [[ "${current_id}" == "${recorded_id}" ]]; then
        if ! "${OPENSHELL_CLI}" --gateway-endpoint "http://127.0.0.1:${GATEWAY_PORT}" \
          --workspace default sandbox delete "${recorded_name}" >/dev/null 2>&1; then
          echo "ERROR: could not clean up failed Fabric-owned sandbox ${recorded_name} (${recorded_id})" >&2
          status=1
        fi
      else
        echo "Skipped cleanup of ${recorded_name}: current sandbox ID does not match recorded ID ${recorded_id}." >&2
      fi
    fi
  fi
  if [[ -n "${GATEWAY_PID}" ]]; then
    kill "${GATEWAY_PID}" 2>/dev/null || true
    wait "${GATEWAY_PID}" 2>/dev/null || true
  fi
  if [[ -f "${GATEWAY_STATE_DIR}/gateway.log" ]]; then
    cp "${GATEWAY_STATE_DIR}/gateway.log" "${RESULT_DIR}/openshell-gateway.log"
    if [[ "${status}" -ne 0 ]]; then
      tail -n 80 "${GATEWAY_STATE_DIR}/gateway.log" >&2 || true
    fi
  fi
  rm -rf -- "${GATEWAY_STATE_DIR}"
  trap - EXIT
  exit "${status}"
}
trap cleanup EXIT

cd "${FABRIC_ROOT}"
if [[ ! -x "${DEMO_PYTHON_ENV}/bin/python" ]]; then
  echo "ERROR: set UV_PROJECT_ENVIRONMENT to an installed NeMo Fabric Python environment" >&2
  exit 2
fi
VIRTUAL_ENV="${DEMO_PYTHON_ENV}" UV_PROJECT_ENVIRONMENT="${DEMO_PYTHON_ENV}" \
  PATH="${DEMO_PYTHON_ENV}/bin:${PATH}" cargo build -p nemo-fabric-openshell-provider --locked
PROVIDER_BINARY="${FABRIC_ROOT}/target/debug/fabric-environment-openshell"
if [[ ! -x "${PROVIDER_BINARY}" ]]; then
  echo "ERROR: provider build did not produce ${PROVIDER_BINARY}" >&2
  exit 1
fi
docker build -f "${OPEN_SHELL_ROOT}/Dockerfile.sdk" -t fabric-openshell-sdk:source .
stage_workload_image
docker build -f "${IMAGE_DOCKERFILE}" -t "${IMAGE_TAG}" .
RUNTIME_IMAGE="$(docker image inspect "${IMAGE_TAG}" --format '{{.Id}}')"
printf '%s\n' "${RUNTIME_IMAGE}" >"${RESULT_DIR}/image-id.txt"

download_asset "${CLI_ASSET}" openshell-checksums-sha256.txt
download_asset "${GATEWAY_ASSET}" openshell-gateway-checksums-sha256.txt
download_asset "${SANDBOX_ASSET}" openshell-sandbox-checksums-sha256.txt
mkdir -p "${GATEWAY_STATE_DIR}/jwt"
(umask 077 && openssl genpkey -algorithm Ed25519 -out "${GATEWAY_STATE_DIR}/jwt/signing.pem")
openssl pkey -in "${GATEWAY_STATE_DIR}/jwt/signing.pem" -pubout -out "${GATEWAY_STATE_DIR}/jwt/public.pem"
openssl rand -hex 16 >"${GATEWAY_STATE_DIR}/jwt/kid"
cat >"${GATEWAY_STATE_DIR}/gateway.toml" <<EOF
[openshell]
version = 2

[openshell.gateway]
name = "fabric-example"
bind_address = "127.0.0.1:${GATEWAY_PORT}"
compute_driver = "docker"
disable_tls = true

[openshell.gateway.auth]
allow_unauthenticated_users = true

[openshell.gateway.gateway_jwt]
signing_key_path = "${GATEWAY_STATE_DIR}/jwt/signing.pem"
public_key_path = "${GATEWAY_STATE_DIR}/jwt/public.pem"
kid_path = "${GATEWAY_STATE_DIR}/jwt/kid"
gateway_id = "fabric-example"

[openshell.drivers.docker]
default_image = "${RUNTIME_IMAGE}"
image_pull_policy = "never"
sandbox_label = "fabric-example"
grpc_endpoint = "http://127.0.0.1:${GATEWAY_PORT}"
sandbox_runtime_image = "ghcr.io/nvidia/openshell/sandbox:${OPEN_SHELL_IMAGE_VERSION}"
supervisor_image = "ghcr.io/nvidia/openshell/supervisor:${OPEN_SHELL_IMAGE_VERSION}"
app_armor_profile = "Unconfined"
EOF
OPENSHELL_DB_URL="sqlite:${GATEWAY_STATE_DIR}/gateway.db?mode=rwc" \
  "${OPENSHELL_GATEWAY}" --config "${GATEWAY_STATE_DIR}/gateway.toml" \
  --bind-address 127.0.0.1 --port "${GATEWAY_PORT}" --compute-driver docker --disable-tls \
  >"${GATEWAY_STATE_DIR}/gateway.log" 2>&1 &
GATEWAY_PID=$!
gateway_deadline=$((SECONDS + GATEWAY_START_TIMEOUT))
until { true >/dev/tcp/127.0.0.1/"${GATEWAY_PORT}"; } 2>/dev/null; do
  if ! kill -0 "${GATEWAY_PID}" 2>/dev/null; then
    echo "ERROR: OpenShell gateway exited during startup" >&2
    exit 1
  fi
  if (( SECONDS >= gateway_deadline )); then
    echo "ERROR: OpenShell gateway did not become ready in ${GATEWAY_START_TIMEOUT}s" >&2
    exit 1
  fi
  sleep 1
done
if ! kill -0 "${GATEWAY_PID}" 2>/dev/null; then
  echo "ERROR: OpenShell gateway exited during startup" >&2
  exit 1
fi

if declare -F configure_workload_gateway >/dev/null; then
  configure_workload_gateway
fi

run_consumer() {
  OPENSHELL_FABRIC_SANDBOX_STATE_FILE="${FABRIC_SANDBOX_STATE_FILE}" \
    VIRTUAL_ENV="${DEMO_PYTHON_ENV}" UV_PROJECT_ENVIRONMENT="${DEMO_PYTHON_ENV}" \
    PATH="${DEMO_PYTHON_ENV}/bin:${PATH}" PYTHONPATH="${FABRIC_ROOT}" \
    NEMO_FABRIC_OPEN_SHELL_PROVIDER="${PROVIDER_BINARY}" \
    run_workload_consumer "$@"
}

if [[ "${EXAMPLE_MODE}" == caller_owned || "${EXAMPLE_MODE}" == both ]]; then
  CALLER_SANDBOX_ATTEMPTED=1
  # Guard empty arrays for Bash 3.2's set -u behavior without adding empty args.
  "${OPENSHELL_CLI}" --gateway-endpoint "http://127.0.0.1:${GATEWAY_PORT}" \
    --workspace default sandbox create --name "${SANDBOX_NAME}" --from "${RUNTIME_IMAGE}" \
    --policy "${POLICY_FILE}" ${MODEL_ENV_ARGS[@]+"${MODEL_ENV_ARGS[@]}"} \
    ${SERVER_ENV_ARGS[@]+"${SERVER_ENV_ARGS[@]}"} \
    --detach --no-tty -- fabric-agent-ctl serve
  SANDBOX_JSON="$("${OPENSHELL_CLI}" --gateway-endpoint "http://127.0.0.1:${GATEWAY_PORT}" \
    --workspace default sandbox get "${SANDBOX_NAME}" --output json)"
  SANDBOX_ID="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])' <<<"${SANDBOX_JSON}")"
  run_consumer caller_owned --sandbox-name "${SANDBOX_NAME}" \
    --sandbox-id "${SANDBOX_ID}" "$@"
  "${OPENSHELL_CLI}" --gateway-endpoint "http://127.0.0.1:${GATEWAY_PORT}" \
    --workspace default sandbox get "${SANDBOX_NAME}" --output json >/dev/null
  echo "Verified caller-owned sandbox remained available after Fabric detached."
fi

if [[ "${EXAMPLE_MODE}" == fabric_owned || "${EXAMPLE_MODE}" == both ]]; then
  run_consumer fabric_owned --sandbox-name "${FABRIC_SANDBOX_NAME}" "$@"
  if "${OPENSHELL_CLI}" --gateway-endpoint "http://127.0.0.1:${GATEWAY_PORT}" \
    --workspace default sandbox get "${FABRIC_SANDBOX_NAME}" --output json >/dev/null 2>&1; then
    echo "ERROR: Fabric-owned sandbox remained after release" >&2
    exit 1
  fi
  echo "Verified Fabric-owned sandbox was deleted after explicit release."
fi
