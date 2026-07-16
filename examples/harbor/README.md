<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Run Fabric Agents with Harbor

This example runs one unchanged Harbor SWE-Bench task while Harbor inputs vary
the resulting in-memory `FabricConfig`. Harbor owns the task, container,
verifier, reward, retries, concurrency, and job layout. `FabricAgent` translates
the selected adapter, model, skills, MCP servers, tool policy, and telemetry
mode into one typed config and asks Fabric to run it.

```text
Harbor task + agent options
  -> FabricAgent
  -> one FabricConfig
  -> Fabric.run
  -> selected adapter and harness
  -> Harbor verifier and reward
  -> Fabric result + optional Relay ATOF/ATIF
```

The [calculator demo](demo/README.md) remains available as a small,
credential-free integration smoke. The SWE-Bench workflow below is the primary
example.

## Install and Preflight

Harbor 0.18 and Python 3.12 or later are supported:

```bash
uv sync --extra runtime --extra harbor
uv run --extra runtime --extra harbor harbor --version
uv run --extra runtime --extra harbor python -c \
  'from nemo_fabric.integrations.harbor import FabricAgent; print(FabricAgent.import_path())'
```

The version command must report Harbor 0.18.x.

### Docker installed with Snap

The Snap build of Docker sees a private `/tmp`, while Harbor generates temporary
Docker Compose overlays in the host temporary directory. If `command -v docker`
prints `/snap/bin/docker`, use a directory visible to both processes:

```bash
mkdir -p "$HOME/harbor-tmp"
export TMPDIR="$HOME/harbor-tmp"
uv run --extra runtime --extra harbor python -c \
  'import tempfile; print(tempfile.gettempdir())'
```

The final command must print `$HOME/harbor-tmp`. Installing Docker Engine and
the Compose plugin outside Snap avoids this workaround.

### Make Fabric available inside the task

Set `fabric_package` to a PEP 508 requirement for the exact Fabric revision
under test. The requirement must be reachable from the task container. Harbor
installs it into `/tmp/nemo-fabric-venv`:

```bash
export FABRIC_PACKAGE='<PEP-508-requirement-for-the-Fabric-revision-under-test>'
```

The Fabric runtime includes a native extension. Use a wheel compatible with the
SWE-Bench image's manylinux baseline; a wheel built directly on a newer
workstation may require a newer glibc.

Verify installation before spending model tokens:

```bash
: "${FABRIC_PACKAGE:?Set FABRIC_PACKAGE to the Fabric revision under test}"

uv run --extra runtime --extra harbor harbor run \
  --task swe-bench/django__django-13741 \
  --agent nemo_fabric.integrations.harbor:FabricAgent \
  --ak fabric_adapter_id=nvidia.fabric.hermes \
  --ak fabric_config_bundle="$PWD/examples/harbor/swebench" \
  --ak "fabric_package=$FABRIC_PACKAGE" \
  --install-only \
  --n-concurrent 1
```

## How Harbor Inputs Become FabricConfig

`FabricAgent` starts with the selected adapter and the Harbor task workspace,
then applies the run inputs through typed Fabric models:

| Harbor input | `FabricConfig` field |
| --- | --- |
| `--ak fabric_adapter_id=...` | `harness.adapter_id` |
| `--model` | `models.default` |
| `--skill` | `skills.paths` |
| `--mcp-config` | `mcp.servers` |
| `--ak fabric_blocked_tools='[...]'` | `tools.blocked` |
| `--ak fabric_telemetry=relay` | `telemetry` and Relay ATOF/ATIF configuration |
| `--ak fabric_harness_settings='{...}'` | `harness.settings` for adapter-specific runtime controls |

The task, verifier, and `FabricAgent` stay fixed. Each command changes only the
input named by the experiment, so the effective config and resulting evidence
remain attributable.

## Run One SWE-Bench Task

Set the shared inputs:

```bash
export FABRIC_AGENT='nemo_fabric.integrations.harbor:FabricAgent'
export FABRIC_BUNDLE="$PWD/examples/harbor/swebench"
export RUNS_DIR="$PWD/.tmp/harbor/fabric-swebench"
export NVIDIA_API_KEY=...
export ANTHROPIC_API_KEY=...
```

`fabric_config_bundle` uploads the example-owned adapter descriptors and MCP
implementation. It does not contain or load a persisted Fabric configuration;
`FabricAgent` constructs that config in memory from the Harbor inputs.

### Hermes

```bash
uv run --extra runtime --extra harbor harbor run \
  --task swe-bench/django__django-13741 \
  --agent "$FABRIC_AGENT" \
  --model nvidia/nemotron-3-nano-30b-a3b \
  --ak fabric_adapter_id=nvidia.fabric.hermes \
  --ak fabric_config_bundle="$FABRIC_BUNDLE" \
  --ak "fabric_package=$FABRIC_PACKAGE" \
  --ae "NVIDIA_API_KEY=$NVIDIA_API_KEY" \
  --job-name django-13741-hermes \
  --jobs-dir "$RUNS_DIR" \
  --n-concurrent 1 \
  --n-attempts 1 \
  --max-retries 1
```

### Claude

The task and verifier are unchanged. Only the adapter, model, credential, and
job name differ. Claude's Relay integration also uses the standalone
`nemo-relay` gateway CLI. Build Relay 0.5.0 in an older Linux image so the
binary is compatible with the SWE-Bench task image, then place it in the
uploaded example bundle:

```bash
mkdir -p "$FABRIC_BUNDLE/.relay"
docker run --rm \
  --user "$(id -u):$(id -g)" \
  -e CARGO_HOME=/tmp/cargo \
  -e CARGO_TARGET_DIR=/tmp/target \
  -v "$FABRIC_BUNDLE/.relay:/out" \
  rust:1.94-bullseye \
  cargo install nemo-relay-cli --version 0.5.0 --root /out
```

The `.relay` directory is ignored by Git. `fabric_config_bundle` uploads it
with the rest of the example inputs; the harness setting below selects the
task-local executable.

```bash
uv run --extra runtime --extra harbor harbor run \
  --task swe-bench/django__django-13741 \
  --agent "$FABRIC_AGENT" \
  --model anthropic/claude-sonnet-4-5 \
  --ak fabric_adapter_id=nvidia.fabric.claude \
  --ak fabric_config_bundle="$FABRIC_BUNDLE" \
  --ak fabric_telemetry=relay \
  --ak 'fabric_harness_settings={"nemo_relay_command":"/tmp/nemo-fabric-config/.relay/bin/nemo-relay"}' \
  --ak "fabric_package=$FABRIC_PACKAGE" \
  --ae "ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY" \
  --job-name django-13741-claude \
  --jobs-dir "$RUNS_DIR" \
  --n-concurrent 1 \
  --n-attempts 1 \
  --max-retries 1
```

Relay is enabled so the resulting ATIF confirms that the harness switch reached
Claude. Claude uses unattended permissions only inside Harbor's ephemeral task
container. Do not apply that permission mode to a normal host environment.

For a self-hosted Hermes model, use an OpenAI-compatible endpoint with
automatic tool calling enabled. A plain chat completion can succeed even when
tool calling is unavailable.

## Vary One Capability Through Harbor

Keep the Hermes command fixed and add one of the following variations. Relay is
enabled for the skill, MCP, and tool-policy runs so their ATIF trajectories can
confirm that the capability reached the harness.

| Experiment | Add to the Hermes command | Job name |
| --- | --- | --- |
| Skill | `--skill "$PWD/examples/harbor/swebench/skills/swebench-debugging" --ak fabric_telemetry=relay` | `django-13741-hermes-skill` |
| MCP | `--mcp-config "$FABRIC_BUNDLE/mcp/repo-inspector.mcp.json" --ak fabric_telemetry=relay` | `django-13741-hermes-mcp` |
| Blocked tool | `--ak 'fabric_blocked_tools=["browser"]' --ak fabric_telemetry=relay` | `django-13741-hermes-tools` |
| Relay telemetry | `--ak fabric_telemetry=relay` | `django-13741-hermes-relay` |

For example, the skill variation is:

```bash
uv run --extra runtime --extra harbor harbor run \
  --task swe-bench/django__django-13741 \
  --agent "$FABRIC_AGENT" \
  --model nvidia/nemotron-3-nano-30b-a3b \
  --skill "$PWD/examples/harbor/swebench/skills/swebench-debugging" \
  --ak fabric_adapter_id=nvidia.fabric.hermes \
  --ak fabric_config_bundle="$FABRIC_BUNDLE" \
  --ak fabric_telemetry=relay \
  --ak "fabric_package=$FABRIC_PACKAGE" \
  --ae "NVIDIA_API_KEY=$NVIDIA_API_KEY" \
  --job-name django-13741-hermes-skill \
  --jobs-dir "$RUNS_DIR" \
  --n-concurrent 1 \
  --n-attempts 1 \
  --max-retries 1
```

The MCP variation uploads this example directory because the MCP config starts
the local, dependency-free [`repo_inspector.py`](swebench/mcp/repo_inspector.py)
inside the task container. The MCP definition itself still enters through
Harbor's `--mcp-config` option.

For a pure telemetry comparison, run the Hermes baseline once without
`fabric_telemetry`, then repeat it with `--ak fabric_telemetry=relay`. No model,
harness, skill, MCP, tool-policy, task, or verifier input changes.

## Verify Reward and Relay Evidence

Harbor's verifier remains the correctness authority. Relay telemetry is a
separate run-evidence check and never changes the SWE-Bench reward.

Every completed variant must have a normal Harbor result with one trial, no
exception, and a verifier reward:

```bash
python -m json.tool "$RUNS_DIR/<job-name>/result.json"
uv run --extra runtime --extra harbor harbor view "$RUNS_DIR"
```

Relay-enabled runs preserve the direct Relay ATOF and ATIF files and publish:

- `agent/trajectory.json`, Harbor's canonical ATIF path;
- `agent/telemetry-validation.json`, the telemetry validation summary;
- `agent/fabric-result-<id>.json`, the normalized Fabric result.

Inspect them with:

```bash
find "$RUNS_DIR/<job-name>" \
  -path '*/agent/telemetry-validation.json' -exec python -m json.tool {} \;
find "$RUNS_DIR/<job-name>" \
  -path '*/agent/trajectory.json' -exec python -m json.tool {} \;
find "$RUNS_DIR/<job-name>" \
  -name '*.atof.jsonl' -o -name '*.atif.json'
```

Validate the direct ATOF stream and direct ATIF document independently. Fabric
promotes Relay's ATIF to `agent/trajectory.json`; it does not derive that file
from ATOF through a separate converter.

Sample output from successful Relay-enabled Hermes and Claude runs is checked
in under [`swebench/sample-artifacts/`](swebench/sample-artifacts/). Large
telemetry files use Git LFS.

## Progress from One Task to a Full Run

Before scaling, run the install-only gate and complete each single-task
variation above. Then replace the one `--task` argument with a five-task shard:

```bash
--dataset swe-bench/swe-bench-verified --n-tasks 5
```

Inspect every exception and reward plus at least one Fabric result and telemetry
summary. Remove `--n-tasks` only after the shard is healthy, and choose
concurrency that respects model and environment limits.

Spot-check a running job without changing it:

```bash
find "$RUNS_DIR/<job-name>" -name result.json -print | head
find "$RUNS_DIR/<job-name>" \
  -path '*/agent/telemetry-validation.json' -print | head
uv run --extra runtime --extra harbor harbor view "$RUNS_DIR/<job-name>"
```

Resume an interrupted job from its recorded configuration:

```bash
uv run --extra runtime --extra harbor harbor job resume \
  --job-path "$RUNS_DIR/<job-name>"
```
