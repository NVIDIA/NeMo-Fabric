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
- a host `codex login` for the Codex run

The first image build can take several minutes.

## Prepare the build context

Harbor builds `task/environment/Dockerfile` with the environment directory as
its Docker context. Export committed `HEAD` so the image installs the exact
Fabric revision under test:

```bash
DEMO_DIR="$PWD/examples/harbor/demo"
TASK_DIR="$DEMO_DIR/task"
RUNS_DIR="$DEMO_DIR/runs"
VENDOR_DIR="$TASK_DIR/environment/vendor/nemo-fabric"

rm -rf "$TASK_DIR/environment/vendor"
mkdir -p "$VENDOR_DIR"
git archive HEAD | tar -x -C "$VENDOR_DIR"
```

Keep this shell open for the commands below. Use a new `--job-name`, or remove
the matching generated directory under `$RUNS_DIR`, before repeating a run.

## Harbor arguments

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

## 1. Credential-free smoke

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

## 3. Hermes with Relay telemetry

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
  -path '*/agent/fabric-artifacts/hermes-relay/relay/events.atof.jsonl' \
  -print -exec sed -n '1,5p' {} \;

find "$RUNS_DIR/fabric-hermes-relay" \
  -path '*/agent/fabric-artifacts/hermes-relay/relay/*.atif.json' \
  -print -exec python -m json.tool {} \;
```

## 4. Codex SDK

Harbor mounts the host Codex login as a read-only secret. The setup command
copies it into a writable container-local `CODEX_HOME`; the Codex SDK app-server
uses that environment without requiring a separately installed `codex` command.

```bash
CODEX_HOME_DIR="${CODEX_HOME:-$HOME/.codex}"
test -f "$CODEX_HOME_DIR/auth.json"
CODEX_AUTH_MOUNT="[{\"type\":\"bind\",\"source\":\"$CODEX_HOME_DIR/auth.json\",\"target\":\"/run/secrets/codex-auth.json\",\"read_only\":true}]"

uv run --extra runtime --extra harbor harbor run \
  --path "$TASK_DIR" \
  --agent nemo_fabric.integrations.harbor:FabricAgent \
  --ak fabric_config_path=/opt/fabric-demo/configs/codex.yaml \
  --ak 'fabric_install_command=mkdir -p "$CODEX_HOME" && cp /run/secrets/codex-auth.json "$CODEX_HOME/auth.json"' \
  --model openai/gpt-5.4 \
  --ae CODEX_HOME=/tmp/fabric-codex-home \
  --mounts "$CODEX_AUTH_MOUNT" \
  --job-name fabric-codex \
  --jobs-dir "$RUNS_DIR" \
  --n-concurrent 1 \
  --n-attempts 1 \
  --force-build
```

The Fabric Codex adapter pins the Python SDK and its bundled app-server runtime.
The config uses `danger-full-access` because Harbor's task container is the
outer sandbox and nested Linux namespace creation is unavailable there.

## Inspect results

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

## Recording flow

1. Show the common `--agent` argument and the four complete config files.
2. Run the credential-free smoke and inspect its Fabric result.
3. Run Hermes and Codex, changing the config, model, and credential inputs.
4. Start Phoenix, run the Hermes Relay config, and open its trace.
5. Show the same run's ATOF and ATIF records.
6. Open all four jobs with `harbor view`.
