#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(git -C "$script_dir" rev-parse --show-toplevel)"
oo_agents_repo="${1:-$repo_root/../labs-OO-Agents}"
runs_dir="$script_dir/runs"
task_dir="$runs_dir/prepared-django__django-13741"
mkdir -p "$runs_dir"
stage_dir="$(mktemp -d "$runs_dir/.prepare-swebench.XXXXXX")"

cleanup() {
    rm -rf "$stage_dir"
}
trap cleanup EXIT

"$script_dir/prepare.sh" "$oo_agents_repo"

uv run --extra harbor harbor download \
    swe-bench/django__django-13741 \
    --output-dir "$stage_dir" \
    --export \
    --overwrite

downloaded_task="$stage_dir/django__django-13741"
if [[ ! -f "$downloaded_task/task.toml" ]]; then
    echo "Downloaded SWE-bench task is incomplete: $downloaded_task" >&2
    exit 1
fi

rm -rf "$task_dir"
mv "$downloaded_task" "$task_dir"
cp "$script_dir/swebench/Dockerfile" "$task_dir/environment/Dockerfile"
cp -R "$script_dir/task/environment/vendor" "$task_dir/environment/vendor"

echo "Prepared the SWE-bench task at $task_dir"
