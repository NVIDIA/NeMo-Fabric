<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Local Harbor smoke task

This task is the fast, deterministic first stage of the canonical
[Fabric–Harbor guide](../README.md#fast-local-smoke). Its Dockerfile copies
`task/environment/fabric/` to `/opt/fabric-demo/`, so these source configs map
to the following task-container paths:

| Source config | Container path | Purpose |
| --- | --- | --- |
| [`smoke.yaml`](task/environment/fabric/configs/smoke.yaml) | `/opt/fabric-demo/configs/smoke.yaml` | credential-free integration smoke |
| [`hermes.yaml`](task/environment/fabric/configs/hermes.yaml) | `/opt/fabric-demo/configs/hermes.yaml` | Hermes harness |
| [`hermes-relay.yaml`](task/environment/fabric/configs/hermes-relay.yaml) | `/opt/fabric-demo/configs/hermes-relay.yaml` | Hermes plus ATOF/ATIF |
| [`codex.yaml`](task/environment/fabric/configs/codex.yaml) | `/opt/fabric-demo/configs/codex.yaml` | Codex harness |

Prepare the committed source as the Docker build context:

```bash
DEMO_DIR="$PWD/examples/harbor/demo"
TASK_DIR="$DEMO_DIR/task"
RUNS_DIR="$DEMO_DIR/runs"
VENDOR_DIR="$TASK_DIR/environment/vendor/nemo-fabric"

rm -rf "$TASK_DIR/environment/vendor"
mkdir -p "$VENDOR_DIR"
git archive HEAD | tar -x -C "$VENDOR_DIR"
```

Run the credential-free smoke:

```bash
uv run --extra runtime --extra harbor harbor run \
  --path "$TASK_DIR" \
  --agent-import-path nemo_fabric.integrations.harbor:FabricAgent \
  --ak fabric_config_path=/opt/fabric-demo/configs/smoke.yaml \
  --job-name fabric-smoke \
  --jobs-dir "$RUNS_DIR" \
  --n-concurrent 1 --n-attempts 1 \
  --force-build
```

Expected result: one trial, no exception, Fabric status `succeeded`, and Harbor
reward `1.0`. The canonical guide contains the model-backed harness comparison,
telemetry inspection, SWE-Bench progression, and resume workflow.
