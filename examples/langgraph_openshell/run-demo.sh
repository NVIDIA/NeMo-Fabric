#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

EXAMPLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FABRIC_ROOT="$(cd "${EXAMPLE_ROOT}/../.." && pwd)"
OPENSHELL_VERSION="${OPENSHELL_VERSION:-v0.0.116}"
OPENSHELL_RELEASE_URL="https://github.com/NVIDIA/OpenShell/releases/download/${OPENSHELL_VERSION}"
GATEWAY_PORT="${OPENSHELL_EXAMPLE_PORT:-18080}"
GATEWAY_START_TIMEOUT="${OPENSHELL_EXAMPLE_GATEWAY_START_TIMEOUT:-120}"
EXAMPLE_MODE="${OPENSHELL_EXAMPLE_MODE:-both}"
export UV_CACHE_DIR="${FABRIC_ROOT}/.tmp/uv-cache"
DEMO_PYTHON_ENV="${FABRIC_ROOT}/.tmp/langgraph-openshell-venv"
GATEWAY_PID=""
SANDBOX_NAME="fab-courier-$$"
FABRIC_SANDBOX_NAME="fab-dev-$$"
SANDBOX_CREATED=""

for dependency in cargo curl docker python3 tar uv; do
  if ! command -v "${dependency}" >/dev/null 2>&1; then
    echo "ERROR: ${dependency} is required to run the OpenShell example" >&2
    exit 2
  fi
done
if [[ "${EXAMPLE_MODE}" != "deployment" && "${EXAMPLE_MODE}" != "development" && "${EXAMPLE_MODE}" != "both" ]]; then
  echo "ERROR: OPENSHELL_EXAMPLE_MODE must be deployment, development, or both" >&2
  exit 2
fi

mkdir -p "${FABRIC_ROOT}/.tmp"
STAGE_DIR="$(mktemp -d "${FABRIC_ROOT}/.tmp/portable-courier-image.XXXXXX")"
OPENSHELL_CACHE_DIR="${FABRIC_ROOT}/.tmp/openshell-releases/${OPENSHELL_VERSION}"
OPENSHELL_BIN_DIR="${OPENSHELL_CACHE_DIR}/bin"
GATEWAY_STATE_DIR="$(mktemp -d "${FABRIC_ROOT}/.tmp/openshell-example.XXXXXX")"

normalize_arch() {
  case "$1" in
    x86_64 | amd64) echo "x86_64" ;;
    aarch64 | arm64) echo "aarch64" ;;
    *) echo "ERROR: unsupported architecture: $1" >&2; exit 2 ;;
  esac
}

HOST_ARCH="$(normalize_arch "$(uname -m)")"
DAEMON_ARCH="$(normalize_arch "$(docker info --format '{{.Architecture}}')")"
case "$(uname -s)" in
  Linux)
    CLI_ASSET="openshell-${HOST_ARCH}-unknown-linux-musl.tar.gz"
    GATEWAY_ASSET="openshell-gateway-${HOST_ARCH}-unknown-linux-gnu.tar.gz"
    ;;
  Darwin)
    if [[ "${HOST_ARCH}" != "aarch64" ]]; then
      echo "ERROR: published OpenShell artifacts do not support Intel macOS" >&2
      exit 2
    fi
    CLI_ASSET="openshell-aarch64-apple-darwin.tar.gz"
    GATEWAY_ASSET="openshell-gateway-aarch64-apple-darwin.tar.gz"
    ;;
  *)
    echo "ERROR: this demo supports Linux and Apple Silicon macOS" >&2
    exit 2
    ;;
esac
SUPERVISOR_ASSET="openshell-sandbox-${DAEMON_ARCH}-unknown-linux-musl.tar.gz"
OPENSHELL_CLI="${OPENSHELL_BIN_DIR}/openshell"
OPENSHELL_GATEWAY="${OPENSHELL_BIN_DIR}/openshell-gateway"
OPENSHELL_SUPERVISOR="${OPENSHELL_BIN_DIR}/openshell-sandbox"

download_release_asset() {
  local asset="$1"
  local destination="${OPENSHELL_CACHE_DIR}/${asset}"
  if [[ ! -f "${destination}" ]]; then
    echo "Downloading OpenShell ${OPENSHELL_VERSION} ${asset}..."
    curl --fail --location --retry 3 --show-error --silent \
      --output "${destination}.part" \
      "${OPENSHELL_RELEASE_URL}/${asset}"
    mv "${destination}.part" "${destination}"
  fi
}

verify_release_asset() {
  python3 - "$1" "$2" <<'PY'
import hashlib
from pathlib import Path
import sys

asset = Path(sys.argv[1])
checksums = Path(sys.argv[2])
expected = {
    name: digest
    for digest, name in (line.split() for line in checksums.read_text().splitlines())
}.get(asset.name)
if expected is None:
    raise SystemExit(f"ERROR: no checksum published for {asset.name}")
actual = hashlib.sha256(asset.read_bytes()).hexdigest()
if actual != expected:
    raise SystemExit(f"ERROR: checksum mismatch for {asset.name}")
PY
}

ensure_release_asset() {
  local asset="$1"
  local checksums="$2"
  download_release_asset "${asset}"
  if ! verify_release_asset "${OPENSHELL_CACHE_DIR}/${asset}" \
    "${OPENSHELL_CACHE_DIR}/${checksums}"; then
    echo "Cached ${asset} is invalid; downloading it again..." >&2
    rm -f -- "${OPENSHELL_CACHE_DIR}/${asset}"
    download_release_asset "${asset}"
    verify_release_asset "${OPENSHELL_CACHE_DIR}/${asset}" \
      "${OPENSHELL_CACHE_DIR}/${checksums}"
  fi
}

install_openshell_release() {
  mkdir -p "${OPENSHELL_CACHE_DIR}" "${OPENSHELL_BIN_DIR}"
  download_release_asset "openshell-checksums-sha256.txt"
  download_release_asset "openshell-gateway-checksums-sha256.txt"
  download_release_asset "openshell-sandbox-checksums-sha256.txt"
  ensure_release_asset "${CLI_ASSET}" "openshell-checksums-sha256.txt"
  ensure_release_asset "${GATEWAY_ASSET}" \
    "openshell-gateway-checksums-sha256.txt"
  ensure_release_asset "${SUPERVISOR_ASSET}" \
    "openshell-sandbox-checksums-sha256.txt"
  tar -xzf "${OPENSHELL_CACHE_DIR}/${CLI_ASSET}" -C "${OPENSHELL_BIN_DIR}"
  tar -xzf "${OPENSHELL_CACHE_DIR}/${GATEWAY_ASSET}" -C "${OPENSHELL_BIN_DIR}"
  tar -xzf "${OPENSHELL_CACHE_DIR}/${SUPERVISOR_ASSET}" -C "${OPENSHELL_BIN_DIR}"
}

cleanup() {
  if [[ -n "${SANDBOX_CREATED}" ]] && [[ -x "${OPENSHELL_CLI}" ]]; then
    "${OPENSHELL_CLI}" --gateway-endpoint "http://127.0.0.1:${GATEWAY_PORT}" \
      sandbox delete "${SANDBOX_NAME}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${GATEWAY_PID}" ]]; then
    kill "${GATEWAY_PID}" 2>/dev/null || true
    wait "${GATEWAY_PID}" 2>/dev/null || true
  fi
  rm -rf -- "${STAGE_DIR}"
  rm -rf -- "${GATEWAY_STATE_DIR}"
}
trap cleanup EXIT

gateway_ready() {
  { true >/dev/tcp/127.0.0.1/"${GATEWAY_PORT}"; } 2>/dev/null
}

gateway_failure() {
  echo "$1; gateway log follows" >&2
  tail -n 80 "${GATEWAY_STATE_DIR}/gateway.log" >&2 || true
  exit 1
}

cd "${FABRIC_ROOT}"
cargo build --release -p nemo-fabric-runtime-control --bins
cargo build -p nemo-fabric-openshell-provider
UV_PROJECT_ENVIRONMENT="${DEMO_PYTHON_ENV}" \
  uv sync --locked --no-default-groups --reinstall-package nemo-fabric-runtime
cp target/release/fabric-runtime-server target/release/fabric-runtime-ctl "${STAGE_DIR}/"
cp -R adapter-contract/python "${STAGE_DIR}/adapter-contract"
cp -R adapters/python/common "${STAGE_DIR}/adapters-common"
mkdir -p "${STAGE_DIR}/examples"
cp examples/__init__.py "${STAGE_DIR}/examples/"
cp -R "${EXAMPLE_ROOT}" "${STAGE_DIR}/examples/langgraph_openshell"
cp "${EXAMPLE_ROOT}/runtime-image.dockerignore" "${STAGE_DIR}/.dockerignore"
docker build -f "${EXAMPLE_ROOT}/runtime-image.Dockerfile" -t fabric-portable-courier:example "${STAGE_DIR}"
RUNTIME_IMAGE="$(docker image inspect fabric-portable-courier:example --format '{{.Id}}')"

install_openshell_release
"${OPENSHELL_GATEWAY}" generate-certs \
  --output-dir "${GATEWAY_STATE_DIR}/tls" \
  --server-san "127.0.0.1" \
  --server-san "host.openshell.internal"
cat >"${GATEWAY_STATE_DIR}/gateway.toml" <<EOF
[openshell]
version = 1

[openshell.gateway]
name = "fabric-example"
compute_drivers = ["docker"]
disable_tls = true

[openshell.gateway.auth]
allow_unauthenticated_users = true

[openshell.gateway.gateway_jwt]
signing_key_path = "${GATEWAY_STATE_DIR}/tls/jwt/signing.pem"
public_key_path = "${GATEWAY_STATE_DIR}/tls/jwt/public.pem"
kid_path = "${GATEWAY_STATE_DIR}/tls/jwt/kid"
gateway_id = "fabric-example"
ttl_secs = 3600

[openshell.drivers.docker]
default_image = "${RUNTIME_IMAGE}"
image_pull_policy = "Never"
sandbox_namespace = "fabric-example"
grpc_endpoint = "http://host.openshell.internal:${GATEWAY_PORT}"
supervisor_bin = "${OPENSHELL_SUPERVISOR}"
EOF
"${OPENSHELL_GATEWAY}" \
  --config "${GATEWAY_STATE_DIR}/gateway.toml" \
  --port "${GATEWAY_PORT}" \
  --drivers docker \
  --disable-tls \
  --db-url "sqlite:${GATEWAY_STATE_DIR}/gateway.db?mode=rwc" \
  >"${GATEWAY_STATE_DIR}/gateway.log" 2>&1 &
GATEWAY_PID=$!

gateway_deadline=$((SECONDS + GATEWAY_START_TIMEOUT))
until gateway_ready; do
  if ! kill -0 "${GATEWAY_PID}" 2>/dev/null; then
    gateway_failure "OpenShell gateway exited during startup"
  fi
  if (( SECONDS >= gateway_deadline )); then
    gateway_failure "OpenShell gateway did not become ready within ${GATEWAY_START_TIMEOUT}s"
  fi
  sleep 1
done

run_consumer() {
  cd "${FABRIC_ROOT}"
  NEMO_FABRIC_OPEN_SHELL_PROVIDER="${FABRIC_ROOT}/target/debug/fabric-environment-openshell" \
  PYTHONPATH="${FABRIC_ROOT}" \
  UV_PROJECT_ENVIRONMENT="${DEMO_PYTHON_ENV}" \
  uv run --no-sync \
    python -m examples.langgraph_openshell.consumer \
    --gateway "http://127.0.0.1:${GATEWAY_PORT}" \
    --image "${RUNTIME_IMAGE}" \
    --base-dir "${FABRIC_ROOT}/.tmp/portable-courier" \
    "$@"
}

if [[ "${EXAMPLE_MODE}" == "deployment" || "${EXAMPLE_MODE}" == "both" ]]; then
  "${OPENSHELL_CLI}" --gateway-endpoint "http://127.0.0.1:${GATEWAY_PORT}" \
    sandbox create \
    --name "${SANDBOX_NAME}" \
    --from "${RUNTIME_IMAGE}" \
    --policy "${EXAMPLE_ROOT}/policy.yaml" \
    --env PYTHONPATH=/opt/nemo-fabric \
    --detach \
    --no-tty \
    -- fabric-runtime-server serve
  SANDBOX_CREATED="yes"
  SANDBOX_JSON="$(
    "${OPENSHELL_CLI}" --gateway-endpoint "http://127.0.0.1:${GATEWAY_PORT}" \
      sandbox get "${SANDBOX_NAME}" --output json
  )"
  SANDBOX_ID="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])' <<<"${SANDBOX_JSON}")"

  run_consumer --sandbox-name "${SANDBOX_NAME}" --sandbox-id "${SANDBOX_ID}"
  "${OPENSHELL_CLI}" --gateway-endpoint "http://127.0.0.1:${GATEWAY_PORT}" \
    sandbox get "${SANDBOX_NAME}" --output json >/dev/null
  echo "Verified deployment mode: Fabric detached; the caller-owned sandbox still exists."
fi

if [[ "${EXAMPLE_MODE}" == "development" || "${EXAMPLE_MODE}" == "both" ]]; then
  run_consumer --fabric-sandbox-name "${FABRIC_SANDBOX_NAME}"
  if "${OPENSHELL_CLI}" --gateway-endpoint "http://127.0.0.1:${GATEWAY_PORT}" \
    sandbox get "${FABRIC_SANDBOX_NAME}" --output json >/dev/null 2>&1; then
    echo "ERROR: Fabric-owned sandbox still exists after release" >&2
    exit 1
  fi
  echo "Verified development mode: explicit Fabric release deleted the Fabric-owned sandbox."
fi
