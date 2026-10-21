<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Harbor Calculator Smoke Test

This self-contained calculator task is the fastest way to check the complete
Harbor → `FabricAgent` → NeMo Fabric → verifier path. Start with the deterministic,
credential-free scripted run, then use the same task to try OpenClaw, Claude,
Pi, or Codex. `FabricAgent` translates Harbor
options into a complete typed `FabricConfig`; Harbor owns the task, container, verifier, reward,
concurrency, and run layout.

## Before You Start

Complete the shared host setup in the
[Harbor landing page](../README.md#shared-host-setup), then continue in the
same shell. Commit the NeMo Fabric revision you want to run because the build context
is created from `HEAD`.

The credential-free smoke does not require an API key. Export `NVIDIA_API_KEY`
for OpenClaw and Pi runs; `ANTHROPIC_API_KEY` for Claude; or
`OPENAI_API_KEY` for Codex. The first image build can take several minutes.

Harbor 0.23 resolves literal `${VAR}` values in `--ae` against the host
environment. The commands below keep those templates quoted so the shell does
not put API keys in the Harbor process arguments.

## Prepare the Build Context

Harbor builds `task/environment/Dockerfile` with the environment directory as
its Docker context. Export committed `HEAD` so the image installs the exact
NeMo Fabric revision from your checkout:

```bash
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
uv run --extra harbor harbor run \
  --path "$TASK_DIR" \
  --agent nemo_fabric.integrations.harbor:FabricAgent \
  --ak fabric_adapter_id=demo.fabric.scripted \
  --ak "fabric_config_bundle=$TASK_DIR/environment/fabric" \
  --ak fabric_workspace=/app \
  --job-name fabric-smoke \
  --jobs-dir "$RUNS_DIR" \
  --n-concurrent 1 \
  --n-attempts 1 \
  --force-build
```

Expected Harbor summary: one trial, zero exceptions, and mean reward `1.000`.

## 2. OpenClaw

The task image installs Node.js, OpenClaw, and the OpenClaw adapter. This run
uses the NVIDIA API through OpenClaw's OpenAI-compatible provider.

```bash
: "${NVIDIA_API_KEY:?Export NVIDIA_API_KEY before running OpenClaw}"

uv run --extra harbor harbor run \
  --path "$TASK_DIR" \
  --agent nemo_fabric.integrations.harbor:FabricAgent \
  --model nvidia/nemotron-3-nano-omni-30b-a3b-reasoning \
  --ak fabric_adapter_id=nvidia.fabric.openclaw \
  --ak fabric_config_base_dir=/opt/fabric-calculator \
  --ak fabric_workspace=/app \
  --ak fabric_model_base_url=https://integrate.api.nvidia.com/v1 \
  --ak fabric_model_api_key_env=NVIDIA_API_KEY \
  --ak fabric_runtime_timeout_seconds=600 \
  --ae 'NVIDIA_API_KEY=${NVIDIA_API_KEY}' \
  --job-name fabric-openclaw \
  --jobs-dir "$RUNS_DIR" \
  --n-concurrent 1 \
  --n-attempts 1 \
  --force-build
```

## 3. Claude

The Claude Agent SDK supplies its compatible Claude Code executable. Harbor
passes the API key into the task environment. NeMo Fabric forwards the supported
Claude authentication variables selected by the adapter; this command uses
`ANTHROPIC_API_KEY`.

```bash
: "${ANTHROPIC_API_KEY:?Export ANTHROPIC_API_KEY before running Claude}"

uv run --extra harbor harbor run \
  --path "$TASK_DIR" \
  --agent nemo_fabric.integrations.harbor:FabricAgent \
  --model anthropic/claude-sonnet-4-5 \
  --ak fabric_adapter_id=nvidia.fabric.claude \
  --ak 'fabric_harness_settings={"permission_mode":"bypassPermissions"}' \
  --ak 'fabric_environment_env={"IS_SANDBOX":"1"}' \
  --ak fabric_config_base_dir=/opt/fabric-calculator \
  --ak fabric_workspace=/app \
  --ak fabric_max_turns=20 \
  --ak fabric_runtime_timeout_seconds=600 \
  --ae 'ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}' \
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

## 4. Pi

The task image builds the Pi adapter and its pinned SDK harness from the same
vendored Fabric revision. `fabric_discovery_paths` points to its task-local
TypeScript descriptor; the NVIDIA model and API key stay the same as the other
NVIDIA-backed calculator runs.

```bash
: "${NVIDIA_API_KEY:?Export NVIDIA_API_KEY before running Pi}"

uv run --extra harbor harbor run \
  --path "$TASK_DIR" \
  --agent nemo_fabric.integrations.harbor:FabricAgent \
  --model nvidia/nemotron-3-nano-omni-30b-a3b-reasoning \
  --ak fabric_adapter_id=nvidia.fabric.pi \
  --ak fabric_config_base_dir=/opt/fabric-calculator \
  --ak 'fabric_discovery_paths=["/opt/nemo-fabric/adapters/typescript/pi/pi.fabric-adapter.json"]' \
  --ak fabric_workspace=/app \
  --ak fabric_model_base_url=https://integrate.api.nvidia.com/v1 \
  --ak fabric_model_api_key_env=NVIDIA_API_KEY \
  --ak fabric_runtime_timeout_seconds=600 \
  --ae 'NVIDIA_API_KEY=${NVIDIA_API_KEY}' \
  --job-name fabric-pi \
  --jobs-dir "$RUNS_DIR" \
  --n-concurrent 1 \
  --n-attempts 1 \
  --force-build
```

## 5. Codex

The task image installs the Codex Python adapter and its SDK-managed app-server.
This recipe explicitly gives Codex `workspace-write` sandboxing and denies
interactive approval requests so it can edit the isolated `/app` workspace.
The Harbor bridge does not inject harness-specific permission defaults.
Codex uses the OpenAI API credential named by `fabric_model_api_key_env`
for a noninteractive, task-local login. A host Codex login is not copied into
the container.

```bash
: "${OPENAI_API_KEY:?Export OPENAI_API_KEY before running Codex}"

uv run --extra harbor harbor run \
  --path "$TASK_DIR" \
  --agent nemo_fabric.integrations.harbor:FabricAgent \
  --model openai/gpt-5.4 \
  --ak fabric_adapter_id=nvidia.fabric.codex \
  --ak 'fabric_harness_settings={"sandbox":"workspace-write","approval_mode":"deny_all"}' \
  --ak fabric_config_base_dir=/opt/fabric-calculator \
  --ak fabric_workspace=/app \
  --ak fabric_model_api_key_env=OPENAI_API_KEY \
  --ak fabric_runtime_timeout_seconds=600 \
  --ae 'OPENAI_API_KEY=${OPENAI_API_KEY}' \
  --job-name fabric-codex \
  --jobs-dir "$RUNS_DIR" \
  --n-concurrent 1 \
  --n-attempts 1 \
  --force-build
```

The OpenClaw, Claude, Pi, and Codex runs use the same task and verifier as the
scripted smoke. Compare their `result.json` rewards and `agent/` metadata under
`$RUNS_DIR` to see the selected harness without changing Harbor task setup.

## Inspect Results

NeMo Fabric result files use unique names in each trial's agent logs:

```bash
find "$RUNS_DIR/fabric-smoke" -path '*/agent/fabric-result-*.json' -print -exec cat {} \;
cat "$RUNS_DIR/fabric-smoke/result.json"
uv run --extra harbor harbor view "$RUNS_DIR"
```

Check NeMo Fabric status, harness and adapter identity, runtime and invocation IDs,
artifacts, telemetry, Harbor exceptions, and reward. A successful calculator run
has one completed trial, zero errored trials, NeMo Fabric status `succeeded`, and
Harbor mean reward `1.0`.

After the runs, remove the generated build-context copy:

```bash
rm -rf "$TASK_DIR/environment/vendor"
```
