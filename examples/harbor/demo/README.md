<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Harbor Multi-Harness Demo

This demo keeps one Harbor task and one `FabricAgent` class while complete
Fabric configs select the execution harness and telemetry behavior. Harbor owns
the task, container, verifier, reward, concurrency, and run layout. Fabric runs
one independent harness runtime for each Harbor agent run.

## Requirements

- Python 3.12+
- `uv`
- Docker
- this repository checkout, with the changes under test committed
- `ANTHROPIC_API_KEY` for the Claude run

The first image build can take several minutes.

## Prepare the Build Context

Harbor builds `task/environment/Dockerfile` with the environment directory as
its Docker context. Export committed `HEAD` so the image installs the exact
Fabric revision under test:

```bash
set -euo pipefail

DEMO_DIR="$PWD/examples/harbor/demo"
TASK_DIR="$DEMO_DIR/task"
RUNS_DIR="$DEMO_DIR/runs"
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

## Harbor Arguments

| Argument | Meaning |
| --- | --- |
| `--path` | Harbor task directory containing the environment and verifier |
| `--agent` | Harbor agent class imported from Fabric |
| `--ak` | Constructor argument passed to `FabricAgent` |
| `fabric_config_path` | Complete Fabric config inside the task container |
| `--model` | Harbor model selection applied to a copy of the Fabric config |
| `--ae` | Environment variable passed to the Harbor agent |
| `--mounts` | Host-to-container mounts managed by Harbor |
| `--extra-docker-compose` | Compose overlay for the task environment |
| `--job-name` | Harbor output directory name for this run |
| `--force-build` | Rebuild the task image from the prepared context |

## 1. Credential-Free Smoke

This run checks Harbor setup, spec upload, sandbox-local SDK execution,
workspace mutation, result download, and verification:

```bash
uv run --extra runtime --extra harbor harbor run \
  --path "$TASK_DIR" \
  --agent nemo_fabric.integrations.harbor:FabricAgent \
  --ak fabric_config_path=/opt/fabric-demo/configs/smoke.yaml \
  --job-name fabric-smoke \
  --jobs-dir "$RUNS_DIR" \
  --n-concurrent 1 \
  --n-attempts 1 \
  --force-build
```

Expected Harbor summary: one trial, zero exceptions, and mean reward `1.000`.

## 2. Hermes

```bash
export NVIDIA_API_KEY=...

uv run --extra runtime --extra harbor harbor run \
  --path "$TASK_DIR" \
  --agent nemo_fabric.integrations.harbor:FabricAgent \
  --ak fabric_config_path=/opt/fabric-demo/configs/hermes.yaml \
  --model nvidia/nemotron-3-nano-30b-a3b \
  --ae "NVIDIA_API_KEY=$NVIDIA_API_KEY" \
  --job-name fabric-hermes \
  --jobs-dir "$RUNS_DIR" \
  --n-concurrent 1 \
  --n-attempts 1 \
  --force-build
```

Harbor's model value replaces `models.default` in the config copy used for this
run.

## 3. Hermes with Relay Telemetry

Start Phoenix on the host:

```bash
docker rm -f fabric-phoenix 2>/dev/null || true
docker run --rm --detach \
  --name fabric-phoenix \
  --publish 6006:6006 \
  arizephoenix/phoenix:latest

until curl --fail --silent http://localhost:6006 >/dev/null; do sleep 1; done
```

Visit `http://localhost:6006`. The Compose overlay maps
`host.docker.internal` to the host gateway on Linux.

```bash
uv run --extra runtime --extra harbor harbor run \
  --path "$TASK_DIR" \
  --agent nemo_fabric.integrations.harbor:FabricAgent \
  --ak fabric_config_path=/opt/fabric-demo/configs/hermes-relay.yaml \
  --model nvidia/nemotron-3-nano-30b-a3b \
  --ae "NVIDIA_API_KEY=$NVIDIA_API_KEY" \
  --extra-docker-compose "$DEMO_DIR/host-gateway.compose.yaml" \
  --job-name fabric-hermes-relay \
  --jobs-dir "$RUNS_DIR" \
  --n-concurrent 1 \
  --n-attempts 1 \
  --force-build
```

The completed run appears in Phoenix and writes ATOF and ATIF records into the
Harbor agent logs:

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
passes the API key into the task environment; Fabric forwards only the model's
configured credential variable to Claude.

```bash
export ANTHROPIC_API_KEY=...

uv run --extra runtime --extra harbor harbor run \
  --path "$TASK_DIR" \
  --agent nemo_fabric.integrations.harbor:FabricAgent \
  --ak fabric_config_path=/opt/fabric-demo/configs/claude.yaml \
  --model anthropic/claude-sonnet-4-5 \
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

Fabric result files use unique names in each trial's agent logs:

```bash
find "$RUNS_DIR/fabric-smoke" -path '*/agent/fabric-result-*.json' -print -exec cat {} \;
cat "$RUNS_DIR/fabric-smoke/result.json"
uv run --extra runtime --extra harbor harbor view "$RUNS_DIR"
```

Check Fabric status, harness and adapter identity, runtime and invocation IDs,
artifacts, telemetry, Harbor exceptions, and reward. A successful smoke run has
Fabric status `succeeded` and Harbor mean reward `1.0`.

After the demo, remove the generated build-context copy:

```bash
rm -rf "$TASK_DIR/environment/vendor"
```

## Recording Flow

1. Show the common `--agent` argument and the four complete config files.
2. Run the credential-free smoke and inspect its Fabric result.
3. Run Hermes and Claude, changing the config, model, and credential inputs.
4. Start Phoenix, run the Hermes Relay config, and open its trace.
5. Show the same run's ATOF and ATIF records.
6. Open all four jobs with `harbor view`.
