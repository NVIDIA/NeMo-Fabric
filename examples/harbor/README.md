<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Run Fabric Agents with Harbor

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

## Install and Preflight

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
  `/tmp/nemo-fabric-config` before installation and the run. Portable bundles
  include config-local adapter descriptors under `configs/adapters/`; the
  selected adapter package still supplies the executable Python code.

Set `fabric_package` to a PEP 508 requirement when the task image needs Fabric
installed. For example, a released package can use
`nemo-fabric[claude,harbor,hermes,relay,runtime]==<version>`, while branch testing can
use a pinned Git requirement. Fabric creates an isolated environment at
`/tmp/nemo-fabric-venv` and puts its console scripts on the runner `PATH`. The
deprecated `fabric_install_command` remains only for experimental images that
also need non-Python setup.

The Fabric runtime includes a native extension. A wheel built directly on a
new workstation may require a newer glibc than an older SWE-Bench image. Use a
published manylinux wheel, build the branch runtime for the image's compatible
manylinux baseline, or preinstall Fabric in a purpose-built evaluation image.
Do not treat a locally tagged `linux_x86_64` wheel as a portable bundle.

Run Harbor's installation-only gate before spending model tokens:

```bash
uv run harbor run \
  --task swe-bench/django__django-13741 \
  --agent-import-path nemo_fabric.integrations.harbor:FabricAgent \
  --ak fabric_config_bundle="$PWD/examples/harbor/swebench" \
  --ak fabric_config_path=configs/hermes.yaml \
  --ak "fabric_package=$FABRIC_PACKAGE" \
  --install-only \
  --n-concurrent 1
```

This gate verifies Fabric and the selected adapter can be installed in the real
task image. Harness binaries and credentials must also be available; use a
purpose-built evaluation image for large runs.

## Fast Local Smoke

The [calculator demo](demo/README.md) includes a credential-free scripted
harness plus Hermes, Hermes with Relay, and Claude configs. It verifies spec
upload, config loading, workspace mutation, result download, and Harbor reward
without first downloading SWE-Bench.

## One SWE-Bench Entry

The task `django__django-13741` is available from Harbor's
`swe-bench/swe-bench-verified` dataset. These inputs stay fixed across the
comparison:

```bash
export FABRIC_BUNDLE="$PWD/examples/harbor/swebench"
export FABRIC_AGENT='nemo_fabric.integrations.harbor:FabricAgent'
export FABRIC_PACKAGE='nemo-fabric[claude,harbor,hermes,relay,runtime]==<version>'
export RUNS_DIR="$PWD/.tmp/harbor/fabric-swebench"
export NVIDIA_API_KEY=...
```

Run Hermes:

```bash
uv run harbor run \
  --task swe-bench/django__django-13741 \
  --agent-import-path "$FABRIC_AGENT" \
  --ak fabric_config_bundle="$FABRIC_BUNDLE" \
  --ak fabric_config_path=configs/hermes.yaml \
  --ak "fabric_package=$FABRIC_PACKAGE" \
  --model nvidia/nemotron-3-nano-30b-a3b \
  --ae "NVIDIA_API_KEY=$NVIDIA_API_KEY" \
  --job-name django-13741-hermes \
  --jobs-dir "$RUNS_DIR" \
  --n-concurrent 1 --n-attempts 1 --max-retries 1
```

For Claude, keep the task, attempts, and concurrency fixed, and change only the
config, model, and credential:

```bash
export ANTHROPIC_API_KEY=...

uv run harbor run \
  --task swe-bench/django__django-13741 \
  --agent-import-path "$FABRIC_AGENT" \
  --ak fabric_config_bundle="$FABRIC_BUNDLE" \
  --ak fabric_config_path=configs/claude.yaml \
  --ak "fabric_package=$FABRIC_PACKAGE" \
  --model anthropic/claude-sonnet-4-5 \
  --ae "ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY" \
  --job-name django-13741-claude \
  --jobs-dir "$RUNS_DIR" \
  --n-concurrent 1 --n-attempts 1 --max-retries 1
```

The checked-in Harbor example treats Hermes and Claude as the two qualified
harnesses. Other adapters remain available in Fabric but are intentionally not
presented as Harbor-qualified here.

The baseline configs were live-verified on July 14, 2026 against the same
Harbor task and task checksum:

| Harness | Verification Model | Fabric Status | Harbor Reward | Review Bundle |
| --- | --- | --- | --- | --- |
| Hermes | self-hosted `nvidia/nemotron-3-nano` | `succeeded` | `1.0` | [`sample-artifacts/hermes/`](swebench/sample-artifacts/hermes/) |
| Claude | `anthropic/claude-sonnet-4-5` | `succeeded` | `1.0` | [`sample-artifacts/claude/`](swebench/sample-artifacts/claude/) |

Claude's config uses `bypassPermissions` and marks the process with
`IS_SANDBOX=1` because Harbor executes it as root inside an ephemeral task
container. Keep that marker scoped to a deliberately isolated evaluation
container; do not copy this permission mode into a normal host environment.

For a self-hosted Nemotron 3 Nano NIM, Hermes requires OpenAI-compatible
automatic tool calling. The current image accepts the vLLM options through
`NIM_PASSTHROUGH_ARGS`; publish the container's port 8000 even when the host
port is different:

```bash
export NIM_IMAGE='nvcr.io/nim/nvidia/nemotron-3-nano@sha256:<approved-digest>'

docker run -d --rm \
  --name nemotron-3-nano \
  --gpus '"device=2"' \
  --shm-size=16GB \
  -e NGC_API_KEY \
  -e 'NIM_PASSTHROUGH_ARGS=--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser nemotron_v3' \
  -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
  -p 8010:8000 \
  "$NIM_IMAGE"
```

Replace `<approved-digest>` with the immutable digest used for qualification
and retain that value with the run metadata. This prevents a mutable image tag
from silently changing the server between runs.

Confirm `/v1/models` and one request containing `tools` with
`tool_choice: "auto"` before starting Harbor. A plain chat completion can
succeed even when this required agent capability is disabled. See NVIDIA's
[tool-calling guidance](https://docs.nvidia.com/nim/large-language-models/latest/advanced-use-cases/tool-calling-and-mcp.html).

## Hold the Harness Fixed and Vary One Capability

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

## Verify Reward and ATOF/ATIF

Harbor's verifier remains the correctness authority. Fabric telemetry is a
separate quality gate and never changes the SWE-Bench reward.

Relay runs validate ATOF JSONL and the native ATIF structure, scan for obvious
credential leakage, and write:

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

The standalone quality gate works in the task environment and against a
collected Harbor trial. For a collected trial, pass its downloaded Fabric
result and `agent` directory; container paths under `/logs/agent` are resolved
to the collected directory automatically:

```bash
python -m nemo_fabric.integrations.harbor.verify_telemetry \
  --result "$TRIAL_DIR/agent/fabric-result-<id>.json" \
  --logs-dir "$TRIAL_DIR/agent"
```

Direct Relay output is the telemetry contract. Validate the emitted ATOF stream
and ATIF independently; do not derive ATIF from ATOF through a separate
converter. Fabric promotes Relay's ATIF to `agent/trajectory.json`, and Harbor
validates that canonical file with its trajectory model.

## Review Sample Artifacts

Curated, sanitized outputs from the qualified one-task runs live under
[`swebench/sample-artifacts/`](swebench/sample-artifacts/). Each harness bundle
contains a compact result summary, verifier summary, representative workspace
patch, and telemetry summary when emitted. These files are quick references,
not substitutes for the complete Harbor trial directory: raw prompts, secrets,
large logs, and full token-heavy trajectories are deliberately excluded.

## Progress from a Spot Check to a Full Run

Do not begin with all 500 tasks:

1. Run `--install-only` on the chosen task image.
2. Run the credential-free calculator smoke.
3. Run `django__django-13741` once with one harness.
4. Repeat it with the second harness.
5. Exercise the skill, MCP, tool, and Relay variants individually.
6. Run a five-task shard by replacing `--task swe-bench/django__django-13741`
   with `--dataset swe-bench/swe-bench-verified --n-tasks 5`.
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

## Source and Task Paths

| Repository/Host Asset | Task-Environment Path |
| --- | --- |
| `examples/harbor/swebench/` | `/tmp/nemo-fabric-config/` in portable mode |
| `configs/hermes.yaml` | `/tmp/nemo-fabric-config/configs/hermes.yaml` |
| `configs/claude.yaml` | `/tmp/nemo-fabric-config/configs/claude.yaml` |
| SWE-Bench checkout | `/testbed/` |
| `mcp/repo_inspector.py` | `/tmp/nemo-fabric-config/mcp/repo_inspector.py` |
| `examples/harbor/demo/task/environment/fabric/` | `/opt/fabric-demo/` via the demo Dockerfile `COPY` |
| Harbor agent logs | `/logs/agent/` in the task and `<trial>/agent/` on the host |

## Integration Contract

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
