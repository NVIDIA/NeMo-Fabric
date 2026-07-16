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
concurrency, and job layout. Fabric owns typed configuration, harness execution,
normalized results, artifacts, and telemetry. One Harbor trial calls one Python
factory that returns a complete `FabricConfig` and therefore selects one
harness:

```text
Harbor task -> FabricAgent -> Fabric.run -> selected adapter -> harness
            -> Harbor verifier and reward
            -> Fabric result + ATOF/ATIF evidence
```

Run the same task again with another factory to compare harnesses. A skill,
MCP, tool, or telemetry variant is an ordinary Python function returning a
copied `FabricConfig`; each variant runs separately so results remain
attributable.

## Install and Preflight

Harbor 0.18 and Python 3.12 or later are supported:

```bash
uv sync --extra runtime --extra harbor
uv run --extra runtime --extra harbor harbor --version
uv run --extra runtime --extra harbor python -c \
  'from nemo_fabric.integrations.harbor import FabricAgent; print(FabricAgent.import_path())'
```

The version command must report Harbor 0.18.x. The explicit extras make each
command independent of whichever optional dependencies are already installed
in the active environment.

### Docker installed with Snap

The Snap build of Docker sees a private `/tmp`, while Harbor generates temporary
Docker Compose overlays in the host temporary directory. If `command -v docker`
prints `/snap/bin/docker`, select a host directory that both processes can see
and verify it before running Harbor:

```bash
mkdir -p "$HOME/harbor-tmp"
export TMPDIR="$HOME/harbor-tmp"
uv run --extra runtime --extra harbor python -c \
  'import tempfile; print(tempfile.gettempdir())'
```

The verification command must print the directory exported above. Installing
Docker Engine and the Compose plugin outside Snap avoids this workaround.

### Make Fabric available inside the task

`FabricAgent` has two environment modes:

- **preinstalled**: `fabric_config_factory` identifies a `module:callable`
  already importable in the task image, and `fabric_config_base_dir` explicitly
  anchors relative assets;
- **portable**: `fabric_config_bundle` is a host directory containing the
  factory module and its assets. Harbor uploads it to
  `/tmp/nemo-fabric-config`, which becomes the factory import path and explicit
  base directory.

The callable is invoked once inside the task and must return a `FabricConfig`.
Fabric does not load a persisted YAML, TOML, or JSON agent configuration.
The selected adapter package supplies its executable Python code.

Set `fabric_package` to a PEP 508 requirement when the task image needs Fabric
installed. The requirement must identify the exact Fabric revision under test
and be reachable from the task container. Fabric installs it in an isolated
environment at `/tmp/nemo-fabric-venv`.

The Fabric runtime includes a native extension. A wheel built directly on a
new workstation may require a newer glibc than an older SWE-Bench image. Use a
wheel built for the image's manylinux baseline, or preinstall Fabric in a
purpose-built evaluation image. Do not assume a locally tagged `linux_x86_64`
wheel will run in SWE-Bench images.

Set `FABRIC_PACKAGE` before using the commands below:

```bash
export FABRIC_PACKAGE='<PEP-508-requirement-for-the-Fabric-revision-under-test>'
```

### Verify installation in the task image

Run Harbor's installation-only gate before spending model tokens:

```bash
: "${FABRIC_PACKAGE:?Set FABRIC_PACKAGE to the Fabric revision under test}"

uv run --extra runtime --extra harbor harbor run \
  --task swe-bench/django__django-13741 \
  --agent nemo_fabric.integrations.harbor:FabricAgent \
  --ak fabric_config_bundle="$PWD/examples/harbor/swebench" \
  --ak fabric_config_factory=harbor_swebench_config:build_hermes \
  --ak "fabric_package=$FABRIC_PACKAGE" \
  --install-only \
  --n-concurrent 1
```

This gate verifies Fabric and the selected adapter can be installed in the real
task image. Harness binaries and credentials must also be available; use a
purpose-built evaluation image for large runs.

## Run One SWE-Bench Task with FabricConfig Variants

The task `django__django-13741` is available from Harbor's
`swe-bench/swe-bench-verified` dataset. Each run uses that same Harbor task.
The selected factory controls the harness, model, skills, MCP servers, tool
policy, and telemetry. Harbor continues to own task execution, concurrency,
retries, and job output; credentials enter through the agent environment.

The factories live in
[`harbor_swebench_config.py`](swebench/harbor_swebench_config.py):

| Experiment | Factory | Configuration difference |
| --- | --- | --- |
| Hermes baseline | `build_hermes` | Baseline Hermes harness and model |
| Claude baseline | `build_claude` | Claude harness and model |
| Skill | `build_hermes_skill` | Adds `skills.paths` |
| MCP | `build_hermes_mcp` | Adds the repository-inspector MCP server |
| Tools | `build_hermes_tools` | Adds normalized `tools.blocked` policy |
| Relay | `build_hermes_relay` | Enables Relay with ATOF and ATIF output |

Set the shared execution inputs, then select one factory, credential, and job
name:

```bash
export FABRIC_BUNDLE="$PWD/examples/harbor/swebench"
export FABRIC_AGENT='nemo_fabric.integrations.harbor:FabricAgent'
export RUNS_DIR="$PWD/.tmp/harbor/fabric-swebench"
export NVIDIA_API_KEY=...
export ANTHROPIC_API_KEY=...

export FABRIC_CONFIG_FACTORY=harbor_swebench_config:build_hermes_skill
export MODEL_CREDENTIAL="NVIDIA_API_KEY=$NVIDIA_API_KEY"
export JOB_NAME=django-13741-hermes-skill
```

Run the fixed task:

```bash
uv run --extra runtime --extra harbor harbor run \
  --task swe-bench/django__django-13741 \
  --agent "$FABRIC_AGENT" \
  --ak fabric_config_bundle="$FABRIC_BUNDLE" \
  --ak fabric_config_factory="$FABRIC_CONFIG_FACTORY" \
  --ak "fabric_package=$FABRIC_PACKAGE" \
  --ae "$MODEL_CREDENTIAL" \
  --job-name "$JOB_NAME" \
  --jobs-dir "$RUNS_DIR" \
  --n-concurrent 1 \
  --n-attempts 1 \
  --max-retries 1
```

For Claude, select its factory and credential, then rerun the same command:

```bash
export FABRIC_CONFIG_FACTORY=harbor_swebench_config:build_claude
export MODEL_CREDENTIAL="ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY"
export JOB_NAME=django-13741-claude
```

Do not add Harbor `--model`, `--skill`, or `--mcp-config` overrides to these
comparison runs. The experiment intentionally selects those capabilities
through the typed `FabricConfig` factory.

Claude's config uses `bypassPermissions` and marks the process with
`IS_SANDBOX=1` because Harbor executes it as root inside an ephemeral task
container. Keep that marker scoped to a deliberately isolated evaluation
container; do not copy this permission mode into a normal host environment.

For a self-hosted model, Hermes requires an OpenAI-compatible endpoint with
automatic tool calling enabled. Record the immutable server image version or
digest with the run metadata, and verify one request containing `tools` with
`tool_choice: "auto"` before starting Harbor. A plain chat completion can
succeed even when this required agent capability is disabled. Refer to NVIDIA's
[tool-calling guidance](https://docs.nvidia.com/nim/large-language-models/latest/advanced-use-cases/tool-calling-and-mcp.html).

The MCP factory references the dependency-free, read-only
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
uv run --extra runtime --extra harbor harbor view "$RUNS_DIR"
```

Direct Relay output is the telemetry contract. Validate the emitted ATOF stream
and ATIF independently; do not derive ATIF from ATOF through a separate
converter. Fabric promotes Relay's ATIF to `agent/trajectory.json`, and Harbor
validates that canonical file with its trajectory model.

## Review Sample Artifacts

Artifacts from a successful Hermes Relay run live under
[`swebench/sample-artifacts/`](swebench/sample-artifacts/). The bundle includes
the complete Relay ATOF and ATIF outputs, Harbor's canonical ATIF copy, a
telemetry summary, a verifier summary, and the resulting workspace patch. Large
telemetry files use Git LFS.

## Progress from a Spot Check to a Full Run

Do not begin with all 500 tasks:

### Spot-Check

1. Run `--install-only` on the chosen task image.
2. Run the credential-free calculator smoke.
3. Run `django__django-13741` once with one harness.
4. Repeat it with the second harness.
5. Exercise the skill, MCP, tool, and Relay variants individually.

### Scale Up

1. Run a five-task shard by replacing `--task swe-bench/django__django-13741`
   with `--dataset swe-bench/swe-bench-verified --n-tasks 5`.
2. Inspect every exception and reward plus at least one Fabric result and
   telemetry summary before scaling.
3. Start the full dataset by removing `--n-tasks` and choosing concurrency that
   respects model and environment limits.

For a long run, use a stable job name and directory. Spot-check without changing
the running job:

```bash
find "$RUNS_DIR/<job-name>" -name result.json -print | head
find "$RUNS_DIR/<job-name>" -path '*/agent/telemetry-validation.json' -print | head
uv run --extra runtime --extra harbor harbor view "$RUNS_DIR/<job-name>"
```

After interruption or infrastructure failures, resume the recorded job config
instead of launching a differently configured replacement:

```bash
uv run --extra runtime --extra harbor harbor job resume \
  --job-path "$RUNS_DIR/<job-name>"
```
