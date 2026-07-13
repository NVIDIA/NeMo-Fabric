<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Run Fabric agents with Harbor

This is the canonical guide for the Fabric–Harbor integration. Start with the
local [calculator smoke](demo/README.md), then use the
[SWE-Bench assets](swebench/) to compare harnesses and capabilities on one
unchanged task.

Harbor owns task materialization, containers, verification, rewards, retries,
concurrency, and job layout. Fabric owns final config validation, harness
execution, normalized results, artifacts, and telemetry. One Harbor trial uses
one complete Fabric config and therefore one harness:

```text
Harbor task -> FabricAgent -> Fabric.run -> selected adapter -> harness
            -> Harbor verifier and reward
            -> Fabric result + ATOF/ATIF evidence
```

Run the same task again with another complete config to compare harnesses. A
skill, MCP, tool, or telemetry variant is a separate run so results remain
attributable.

## Install and preflight

Harbor 0.18 and Python 3.12 or later are supported:

```bash
uv sync --extra runtime --extra harbor
uv run harbor --version
uv run python -c 'from nemo_fabric.integrations.harbor import FabricAgent; print(FabricAgent.import_path())'
```

`FabricAgent` has two environment modes:

- **preinstalled**: `fabric_config_path` is an absolute path already in the task
  image; this is fastest for repeated evaluations;
- **portable**: `fabric_config_bundle` is a host directory and
  `fabric_config_path` is a relative entrypoint within it. Harbor uploads the
  bundle with `BaseEnvironment.upload_dir()` to
  `/tmp/nemo-fabric-config` before the run.

Set `fabric_package` to a PEP 508 requirement when the task image needs Fabric
installed. For example, a released package can use
`nemo-fabric[harbor,hermes,relay,runtime]==<version>`, while branch testing can
use a pinned Git requirement. Fabric creates an isolated environment at
`/tmp/nemo-fabric-venv` and puts its console scripts on the runner `PATH`. The
deprecated `fabric_install_command` remains only for experimental images that
also need non-Python setup.

Run Harbor's installation-only gate before spending model tokens:

```bash
uv run harbor run \
  --task swe-bench/django__django-13741 \
  --agent nemo_fabric.integrations.harbor:FabricAgent \
  --ak fabric_config_bundle="$PWD/examples/harbor/swebench" \
  --ak fabric_config_path=configs/hermes.yaml \
  --ak "fabric_package=$FABRIC_PACKAGE" \
  --install-only \
  --n-concurrent 1
```

This gate verifies Fabric and the selected adapter can be installed in the real
task image. Harness binaries and credentials must also be available; use a
purpose-built evaluation image for large runs.

## Fast local smoke

The [calculator demo](demo/README.md) includes a credential-free scripted
harness plus Hermes, Hermes with Relay, and Codex configs. It verifies spec
upload, config loading, workspace mutation, result download, and Harbor reward
without first downloading SWE-Bench.

## One SWE-Bench entry

The task `django__django-13741` is available from Harbor's
`swe-bench/swe-bench-verified` dataset. These inputs stay fixed across the
comparison:

```bash
export FABRIC_BUNDLE="$PWD/examples/harbor/swebench"
export FABRIC_AGENT='nemo_fabric.integrations.harbor:FabricAgent'
export FABRIC_PACKAGE='nemo-fabric[harbor,hermes,relay,runtime]==<version>'
export RUNS_DIR="$PWD/.tmp/harbor/fabric-swebench"
export NVIDIA_API_KEY=...
```

Run Hermes:

```bash
uv run harbor run \
  --dataset swe-bench/swe-bench-verified \
  --include-task-name django__django-13741 \
  --n-tasks 1 \
  --agent "$FABRIC_AGENT" \
  --ak fabric_config_bundle="$FABRIC_BUNDLE" \
  --ak fabric_config_path=configs/hermes.yaml \
  --ak "fabric_package=$FABRIC_PACKAGE" \
  --model nvidia/nemotron-3-nano-30b-a3b \
  --ae "NVIDIA_API_KEY=$NVIDIA_API_KEY" \
  --job-name django-13741-hermes \
  --jobs-dir "$RUNS_DIR" \
  --n-concurrent 1 --n-attempts 1 --max-retries 1
```

For Codex, keep the dataset, task, attempts, and concurrency fixed, and change
only the config, model, and required credential/bootstrap inputs:

```bash
export OPENAI_API_KEY=...
export FABRIC_CODEX_PACKAGE='nemo-fabric[codex,harbor,runtime]==<version>'

uv run harbor run \
  --dataset swe-bench/swe-bench-verified \
  --include-task-name django__django-13741 \
  --n-tasks 1 \
  --agent "$FABRIC_AGENT" \
  --ak fabric_config_bundle="$FABRIC_BUNDLE" \
  --ak fabric_config_path=configs/codex.yaml \
  --ak 'fabric_install_command=python3 -m pip install "nemo-fabric[codex,harbor,runtime]==<version>" && npm install --global @openai/codex@0.142.4' \
  --model openai/gpt-5.4 \
  --ae "OPENAI_API_KEY=$OPENAI_API_KEY" \
  --job-name django-13741-codex \
  --jobs-dir "$RUNS_DIR" \
  --n-concurrent 1 --n-attempts 1 --max-retries 1
```

The arbitrary install command is shown only because the registry task also
needs the Codex npm binary. A full evaluation should bake Fabric and the CLI
into a pinned image instead.

## Hold the harness fixed and vary one capability

Use the Hermes command above and change exactly one dimension:

| Variant | Change |
| --- | --- |
| Skill | add `--skill "$FABRIC_BUNDLE/skills/swebench-debugging"` |
| MCP | add `--mcp-config "$FABRIC_BUNDLE/mcp.json"` |
| Tools | change the entrypoint to [`configs/hermes-tools.yaml`](swebench/configs/hermes-tools.yaml) |
| Relay | change the entrypoint to [`configs/hermes-relay.yaml`](swebench/configs/hermes-relay.yaml) |

Harbor-supplied skills and MCP servers replace those sections in the complete
Fabric config. Tool selection remains config-owned because Harbor does not have
a generic tool flag. A Harbor `--model` replaces provider and model identity
while preserving other model settings such as temperature.

The MCP variant uploads a dependency-free, read-only
[`repo_inspector.py`](swebench/mcp/repo_inspector.py). Its absolute command path
is the portable bundle target inside the task, not a workstation path.

## Verify reward and ATOF/ATIF

Harbor's verifier remains the correctness authority. Fabric telemetry is a
separate quality gate and never changes the SWE-Bench reward.

Relay runs validate ATOF JSONL, ATIF schema and session correlation, scan for
obvious credential leakage, and write:

- the original Fabric ATOF and ATIF files;
- `agent/trajectory.json`, Harbor's canonical ATIF path;
- `agent/telemetry-validation.json`, a concise machine-readable summary;
- `agent/fabric-result-<id>.json`, the normalized Fabric result.

Inspect a completed job:

```bash
find "$RUNS_DIR/django-13741-hermes-relay" \
  -path '*/agent/telemetry-validation.json' -exec python -m json.tool {} \;
find "$RUNS_DIR/django-13741-hermes-relay" \
  -path '*/agent/trajectory.json' -exec python -m json.tool {} \;
cat "$RUNS_DIR/django-13741-hermes-relay/result.json"
uv run harbor view "$RUNS_DIR"
```

The standalone in-environment quality gate is also available:

```bash
python -m nemo_fabric.integrations.harbor.verify_telemetry \
  --result /tmp/fabric-result.json \
  --logs-dir /logs/agent
```

## Progress from a spot check to a full run

Do not begin with all 500 tasks:

1. Run `--install-only` on the chosen task image.
2. Run the credential-free calculator smoke.
3. Run `django__django-13741` once with one harness.
4. Repeat it with the second harness.
5. Exercise the skill, MCP, tool, and Relay variants individually.
6. Run a five-task shard by keeping the same command, removing
   `--include-task-name`, and setting `--n-tasks 5`.
7. Inspect every exception and reward plus at least one Fabric result and
   telemetry summary before scaling.
8. Start the full dataset by removing `--n-tasks` and choosing concurrency that
   respects model and environment limits.

For a long run, use a stable job name and directory. Spot-check without changing
the running job:

```bash
find "$RUNS_DIR/<job-name>" -name result.json -print | head
find "$RUNS_DIR/<job-name>" -path '*/agent/telemetry-validation.json' -print | head
uv run harbor view "$RUNS_DIR/<job-name>"
```

After interruption or infrastructure failures, resume the recorded job config
instead of launching a differently configured replacement:

```bash
uv run harbor job resume --job-path "$RUNS_DIR/<job-name>"
```

## Source and task paths

| Repository/host asset | Task-environment path |
| --- | --- |
| `examples/harbor/swebench/` | `/tmp/nemo-fabric-config/` in portable mode |
| `configs/hermes.yaml` | `/tmp/nemo-fabric-config/configs/hermes.yaml` |
| `mcp/repo_inspector.py` | `/tmp/nemo-fabric-config/mcp/repo_inspector.py` |
| `examples/harbor/demo/task/environment/fabric/` | `/opt/fabric-demo/` via the demo Dockerfile `COPY` |
| Harbor agent logs | `/logs/agent/` in the task and `<trial>/agent/` on the host |

## Integration contract

`FabricAgent` writes a strict `HarborRunSpec`, uploads it, and invokes
`nemo_fabric.integrations.harbor.runner` inside the task. The runner loads one
complete YAML config, deep-copies it, applies explicit Harbor model/MCP/skill
overrides, and calls `Fabric.run()`. It never composes Fabric filesystem
profiles.

Harbor `session_id` and `context_id` are propagated through
`RunRequest.context`. Valid ATIF final metrics populate Harbor input, cached,
output-token, and cost fields during post-run processing.

Run the lightweight coverage with:

```bash
uv run --extra runtime --extra harbor pytest \
  tests/python/test_harbor_integration.py \
  tests/integrations/test_harbor_runner.py \
  tests/integrations/test_harbor_telemetry.py
```

