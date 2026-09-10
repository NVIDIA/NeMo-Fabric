#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(git -C "$script_dir" rev-parse --show-toplevel)"
example_dir="$script_dir"
stage_dir="$(mktemp -d "${TMPDIR:-/tmp}/fabric-nooa-bench.XXXXXX")"

cleanup() {
    rm -rf "$stage_dir"
}
trap cleanup EXIT

wheelhouse="$stage_dir/wheelhouse"
mkdir -p "$wheelhouse"

uv build --wheel --out-dir "$wheelhouse" "$repo_root/sdk/python/nemo-fabric"
uv build --wheel --out-dir "$wheelhouse" "$repo_root/adapter-contract/python"
uv build --wheel --out-dir "$wheelhouse" "$repo_root/adapters/python/common"
uv build --wheel --out-dir "$wheelhouse" "$repo_root/adapters/python/nooa"
(
    cd "$repo_root/sdk/python/nemo-fabric-runtime"
    uvx --from 'maturin[zig]>=1.9.3,<2.0' maturin build \
        --release \
        --locked \
        --compatibility manylinux_2_17 \
        --zig \
        --out "$wheelhouse"
)

rm -rf "$example_dir/task/environment/vendor"
mkdir -p "$example_dir/task/environment/vendor"
mv "$wheelhouse" "$example_dir/task/environment/vendor/"

echo "Built and prepared the BenchAgent Harbor context with packaged adapters."
