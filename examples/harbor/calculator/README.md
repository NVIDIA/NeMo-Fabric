<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Harbor Calculator Smoke Test

This self-contained calculator task is the fastest way to check the complete
Harbor → `FabricAgent` → NeMo Fabric → verifier path. Start with the deterministic,
credential-free scripted run, then use the same task to try Hermes Agent, Relay
telemetry, or Claude. `FabricAgent` translates Harbor options into a complete
typed `FabricConfig`; Harbor owns the task, container, verifier, reward,
concurrency, and run layout.

## Before You Start

Complete the shared host setup in the
[Harbor landing page](../README.md#shared-host-setup), then continue in the
same shell. Commit the NeMo Fabric revision you want to run because the build context
is created from `HEAD`.

The credential-free smoke does not require an API key. Export `NVIDIA_API_KEY`
for Hermes Agent runs or `ANTHROPIC_API_KEY` for the Claude run before using that
harness. The first image build can take several minutes.

## Prepare the Build Context

Harbor builds `task/environment/Dockerfile` with the environment directory as
its Docker context. Export committed `HEAD` so the image installs the exact
NeMo Fabric revision from your checkout:

```bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

CALCULATOR_DIR="$PWD/examples/harbor/calculator"
TASK_DIR="$CALCULATOR_DIR/task"
RUNS_DIR="$CALCULATOR_DIR/runs"
STAGING_DIR="$(mktemp -d "$TASK_DIR/environment/.vendor.XXXXXX")"
trap 'rm -rf "$STAGING_DIR"' EXIT

mkdir -p "$STAGING_DIR/nemo-fabric"
git archive HEAD | tar -x -C "$STAGING_DIR/nemo-fabric"
rm -rf "$TASK_DIR/environment/vendor"
mv "$STAGING_DIR" "$TASK_DIR/environment/vendor"
trap - EXIT
```

The existing vendor tree is replaced only after the new archive has been
created successfully.

Keep this shell open for the commands below. Use a new `--job-name`, or remove
the matching generated directory under `$RUNS_DIR`, before repeating a run.

## 1. Credential-Free Smoke

This run checks Harbor setup, spec upload, sandbox-local SDK execution,
workspace mutation, result download, and verification:

```bash
uv run --extra runtime --extra harbor harbor run \
  --path "$TASK_DIR" \
  --agent nemo_fabric.integrations.harbor:FabricAgent \
  --ak fabric_adapter_id=demo.fabric.scripted \
  --ak fabric_config_base_dir=/opt/fabric-calculator \
  --ak fabric_workspace=/app \
  --job-name fabric-smoke \
  --jobs-dir "$RUNS_DIR" \
  --n-concurrent 1 \
  --n-attempts 1 \
  --force-build
```

Expected Harbor summary: one trial, zero exceptions, and mean reward `1.000`.

## 2. Hermes Agent

```bash
: "${NVIDIA_API_KEY:?Export NVIDIA_API_KEY before running Hermes Agent}"

uv run --extra runtime --extra harbor harbor run \
  --path "$TASK_DIR" \
  --agent nemo_fabric.integrations.harbor:FabricAgent \
  --model nvidia/nemotron-3-nano-30b-a3b \
  --ak fabric_adapter_id=nvidia.fabric.hermes \
  --ak fabric_config_base_dir=/opt/fabric-calculator \
  --ak fabric_workspace=/app \
  --ak fabric_model_base_url=https://integrate.api.nvidia.com/v1 \
  --ak fabric_max_turns=20 \
  --ae "NVIDIA_API_KEY=$NVIDIA_API_KEY" \
  --job-name fabric-hermes \
  --jobs-dir "$RUNS_DIR" \
  --n-concurrent 1 \
  --n-attempts 1 \
  --force-build
```

The Harbor model and agent arguments become the model and harness fields in the
typed config. The API key is passed separately as a task credential.

## 3. Hermes Agent with Relay Telemetry

```bash
: "${NVIDIA_API_KEY:?Export NVIDIA_API_KEY before running Hermes Agent}"

uv run --extra runtime --extra harbor harbor run \
  --path "$TASK_DIR" \
  --agent nemo_fabric.integrations.harbor:FabricAgent \
  --model nvidia/nemotron-3-nano-30b-a3b \
  --ak fabric_adapter_id=nvidia.fabric.hermes \
  --ak fabric_config_base_dir=/opt/fabric-calculator \
  --ak fabric_workspace=/app \
  --ak fabric_telemetry=relay \
  --ak fabric_model_base_url=https://integrate.api.nvidia.com/v1 \
  --ak fabric_max_turns=4 \
  --ak 'fabric_harness_settings={"terminal_timeout":120}' \
  --ae "NVIDIA_API_KEY=$NVIDIA_API_KEY" \
  --job-name fabric-hermes-relay \
  --jobs-dir "$RUNS_DIR" \
  --n-concurrent 1 \
  --n-attempts 1 \
  --force-build
```

The completed run writes direct Relay ATOF and ATIF records into the Harbor
agent logs:

```bash
find "$RUNS_DIR/fabric-hermes-relay" \
  -name 'events.atof.jsonl' \
  -print -exec sed -n '1,5p' {} \;

find "$RUNS_DIR/fabric-hermes-relay" \
  -name '*.atif.json' \
  -print -exec python -m json.tool {} \;
```

## 4. Claude

The Claude Agent SDK supplies its compatible Claude Code executable. Harbor
passes the API key into the task environment; NeMo Fabric forwards only the model's
configured credential variable to Claude.

```bash
: "${ANTHROPIC_API_KEY:?Export ANTHROPIC_API_KEY before running Claude}"

uv run --extra runtime --extra harbor harbor run \
  --path "$TASK_DIR" \
  --agent nemo_fabric.integrations.harbor:FabricAgent \
  --model anthropic/claude-sonnet-4-5 \
  --ak fabric_adapter_id=nvidia.fabric.claude \
  --ak fabric_config_base_dir=/opt/fabric-calculator \
  --ak fabric_workspace=/app \
  --ak fabric_max_turns=20 \
  --ak fabric_runtime_timeout_seconds=600 \
  --ae "ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY" \
  --job-name fabric-claude \
  --jobs-dir "$RUNS_DIR" \
  --n-concurrent 1 \
  --n-attempts 1 \
  --force-build
```

The config uses `bypassPermissions` and `IS_SANDBOX=1` because Harbor runs the
harness as root inside an ephemeral task container and the benchmark expects it
to edit `/app`. Do not reuse this combination outside a deliberately isolated
evaluation container.

## Inspect Results

NeMo Fabric result files use unique names in each trial's agent logs:

```bash
find "$RUNS_DIR/fabric-smoke" -path '*/agent/fabric-result-*.json' -print -exec cat {} \;
cat "$RUNS_DIR/fabric-smoke/result.json"
uv run --extra runtime --extra harbor harbor view "$RUNS_DIR"
```

Check NeMo Fabric status, harness and adapter identity, runtime and invocation IDs,
artifacts, telemetry, Harbor exceptions, and reward. A successful smoke run has
NeMo Fabric status `succeeded` and Harbor mean reward `1.0`.

After the runs, remove the generated build-context copy:

```bash
rm -rf "$TASK_DIR/environment/vendor"
```
