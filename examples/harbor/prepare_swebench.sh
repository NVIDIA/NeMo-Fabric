#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
bundle_dir="$repo_root/examples/harbor/swebench"
wheelhouse="$bundle_dir/.wheelhouse"
relay_root="$bundle_dir/.relay"

if [[ "$(uname -m)" != "x86_64" ]]; then
    echo "This SWE-Bench example currently targets x86_64 task images." >&2
    exit 1
fi

mkdir -p "$wheelhouse" "$relay_root"
find "$wheelhouse" -maxdepth 1 -type f -name 'nemo_fabric*.whl' -delete

for project in adapters/common adapters/claude adapters/hermes .; do
    uv build --wheel --out-dir "$wheelhouse" "$repo_root/$project"
done

docker run --rm \
    -e CARGO_TARGET_DIR=/tmp/target \
    -e HOST_UID="$(id -u)" \
    -e HOST_GID="$(id -g)" \
    -v "$repo_root:/io:ro" \
    -v "$wheelhouse:/out" \
    -w /io/python \
    --entrypoint /bin/bash \
    ghcr.io/pyo3/maturin:v1.9.6 \
    -lc 'maturin build --release --locked --out /out --compatibility manylinux2014 && chown "$HOST_UID:$HOST_GID" /out/*.whl'

relay_version="$({ "$relay_root/bin/nemo-relay" --version 2>/dev/null || true; })"
if [[ "$relay_version" != "nemo-relay 0.5.0" ]]; then
    docker run --rm \
        --user "$(id -u):$(id -g)" \
        -e CARGO_HOME=/tmp/cargo \
        -e CARGO_TARGET_DIR=/tmp/target \
        -v "$relay_root:/out" \
        rust:1.94-bullseye \
        cargo install nemo-relay-cli --version 0.5.0 --locked --force --root /out
fi

fabric_wheel="$(find "$wheelhouse" -maxdepth 1 -type f -name 'nemo_fabric-*-py3-none-any.whl' -printf '%f\n' | sort | tail -n 1)"
if [[ -z "$fabric_wheel" ]]; then
    echo "The nemo-fabric wheel was not created." >&2
    exit 1
fi

printf '%s\n' \
    "nemo-fabric[claude,harbor,hermes,relay,runtime] @ file:///tmp/nemo-fabric-config/.wheelhouse/$fabric_wheel" \
    > "$bundle_dir/.fabric-package"

echo "Prepared $bundle_dir"
echo "Fabric requirement: $(< "$bundle_dir/.fabric-package")"
