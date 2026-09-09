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
adapter_source="$stage_dir/nooa-adapter"
constraints_file="$stage_dir/nooa-constraints.txt"
bundle="$stage_dir/bundle"
mkdir -p "$wheelhouse" "$adapter_source" "$bundle/adapters"

uv build --wheel --out-dir "$wheelhouse" "$repo_root/sdk/python/nemo-fabric"
uv build --wheel --out-dir "$wheelhouse" "$repo_root/adapter-contract/python"
uv build --wheel --out-dir "$wheelhouse" "$repo_root/adapters/python/common"
(
    cd "$repo_root/sdk/python/nemo-fabric-runtime"
    uvx --from 'maturin[zig]>=1.9.3,<2.0' maturin build \
        --release \
        --locked \
        --compatibility manylinux_2_17 \
        --zig \
        --out "$wheelhouse"
)

git -C "$repo_root" archive HEAD:external/nooa src | tar -x -C "$adapter_source"
cp "$repo_root/external/nooa/constraints.txt" "$constraints_file"
cp "$repo_root/external/nooa/nooa-bench.fabric-adapter.json" "$bundle/adapters/"

rm -rf "$example_dir/task/environment/vendor" "$example_dir/.bundle"
mkdir -p "$example_dir/task/environment/vendor"
mv "$wheelhouse" "$example_dir/task/environment/vendor/"
mv "$adapter_source" "$example_dir/task/environment/vendor/"
mv "$constraints_file" "$example_dir/task/environment/vendor/"
mv "$bundle" "$example_dir/.bundle"

echo "Built and prepared the BenchAgent Harbor context with published NOOA packages."
