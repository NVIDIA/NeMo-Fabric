#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Prepare the standalone nemo-relay executable used by the Claude walkthrough.

set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
bundle_dir="$repo_root/examples/harbor/swebench"
relay_root="$bundle_dir/.relay"
relay_cli_version="0.6.0"

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
    echo "This SWE-Bench example currently requires an x86_64 Linux host." >&2
    exit 1
fi

mkdir -p "$relay_root"

relay_version="$({ "$relay_root/bin/nemo-relay" --version 2>/dev/null || true; })"
if [[ "$relay_version" != "nemo-relay $relay_cli_version" ]]; then
    docker run --rm \
        --user "$(id -u):$(id -g)" \
        -e CARGO_HOME=/tmp/cargo \
        -e CARGO_TARGET_DIR=/tmp/target \
        -v "$relay_root:/out" \
        rust:1.94-bullseye \
        cargo install nemo-relay-cli --version "$relay_cli_version" --locked --force --root /out
fi

echo "Prepared $bundle_dir"
