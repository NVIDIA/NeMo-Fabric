#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(git -C "$script_dir" rev-parse --show-toplevel)"
example_dir="$script_dir"
oo_agents_repo="${1:-$repo_root/../labs-OO-Agents}"
stage_dir="$(mktemp -d "${TMPDIR:-/tmp}/fabric-nooa-bench.XXXXXX")"

cleanup() {
    rm -rf "$stage_dir"
}
trap cleanup EXIT

if [[ ! -d "$oo_agents_repo/.git" ]]; then
    echo "OO Agents repository not found: $oo_agents_repo" >&2
    exit 1
fi

wheelhouse="$stage_dir/wheelhouse"
adapter_source="$stage_dir/nooa-adapter"
oo_agents_source="$stage_dir/labs-OO-Agents"
bundle="$stage_dir/bundle"
mkdir -p "$wheelhouse" "$adapter_source" "$oo_agents_source" "$bundle/adapters"

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
git -C "$oo_agents_repo" archive HEAD | tar -x -C "$oo_agents_source"
cp "$repo_root/external/nooa/nooa-bench.fabric-adapter.json" "$bundle/adapters/"

rm -rf "$example_dir/task/environment/vendor" "$example_dir/.bundle"
mkdir -p "$example_dir/task/environment/vendor"
mv "$wheelhouse" "$example_dir/task/environment/vendor/"
mv "$adapter_source" "$example_dir/task/environment/vendor/"
mv "$oo_agents_source" "$example_dir/task/environment/vendor/"
mv "$bundle" "$example_dir/.bundle"

echo "Built and prepared the BenchAgent Harbor context from committed source."
