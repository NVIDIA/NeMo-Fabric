#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Prepare the standalone nemo-relay executable used by the Claude walkthrough.

set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
bundle_dir="$repo_root/examples/harbor/swebench"
relay_root="$bundle_dir/.relay"

if [[ "$(uname -m)" != "x86_64" ]]; then
    echo "This SWE-Bench example currently targets x86_64 task images." >&2
    exit 1
fi

mkdir -p "$relay_root"

relay_version="$({ "$relay_root/bin/nemo-relay" --version 2>/dev/null || true; })"
if [[ "$relay_version" != "nemo-relay 0.6.0" ]]; then
    docker run --rm \
        --user "$(id -u):$(id -g)" \
        -e CARGO_HOME=/tmp/cargo \
        -e CARGO_TARGET_DIR=/tmp/target \
        -v "$relay_root:/out" \
        rust:1.94-bullseye \
        cargo install nemo-relay-cli --version 0.6.0 --locked --force --root /out
fi

echo "Prepared $bundle_dir"
