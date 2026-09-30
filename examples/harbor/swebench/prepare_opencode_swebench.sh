#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(git -C "$script_dir" rev-parse --show-toplevel)"
runs_dir="$repo_root/.tmp/harbor/fabric-swebench"
task_dir="$runs_dir/prepared-django__django-13741-opencode"
mkdir -p "$runs_dir"
stage_dir="$(mktemp -d "$runs_dir/.prepare-opencode.XXXXXX")"

cleanup() {
    rm -rf "$stage_dir"
}
trap cleanup EXIT

uv run --extra harbor harbor download \
    swe-bench/django__django-13741 \
    --output-dir "$stage_dir" \
    --export \
    --overwrite

downloaded_task="$stage_dir/django__django-13741"
if [[ ! -f "$downloaded_task/task.toml" ]]; then
    echo "Downloaded SWE-Bench task is incomplete: $downloaded_task" >&2
    exit 1
fi

mkdir -p "$downloaded_task/environment/vendor"
git -C "$repo_root" ls-files -z | \
    tar -C "$repo_root" --null -czf \
        "$downloaded_task/environment/vendor/nemo-fabric-source.tar.gz" -T -
cp "$script_dir/opencode/Dockerfile" "$downloaded_task/environment/Dockerfile"

rm -rf "$task_dir"
mv "$downloaded_task" "$task_dir"

echo "Prepared the OpenCode SWE-Bench task at $task_dir"
