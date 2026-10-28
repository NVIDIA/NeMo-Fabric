<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Harbor SWE-Bench Walkthrough

This walkthrough runs the unchanged Harbor task
`swe-bench/django__django-13741` while Harbor options vary the harness, skills,
MCP servers, tool policy, and Relay telemetry in the resulting
`FabricConfig`. It starts with one task and progresses to a resumable
multi-task run.

## Before You Start

Complete the shared host setup in the [Harbor landing page](../README.md#shared-host-setup),
then continue in the same shell. Export `NVIDIA_API_KEY` for Hermes Agent,
OpenCode, or Cline runs, or `ANTHROPIC_API_KEY` for Claude runs before using
that harness.

## Prepare the Task Bundle

Build the standalone Relay executable that will be uploaded into the isolated
task container for the Claude walkthrough:

```bash
cd "$(git rev-parse --show-toplevel)"

export FABRIC_AGENT='nemo_fabric.integrations.harbor:FabricAgent'
export FABRIC_BUNDLE="$PWD/examples/harbor/swebench"
export FABRIC_PACKAGE='nemo-fabric[claude,hermes-agent,relay]==0.6.0'
export RUNS_DIR="$PWD/.tmp/harbor/fabric-swebench"

curl -fsSL https://raw.githubusercontent.com/NVIDIA/NeMo-Relay/main/install.sh |
  NEMO_RELAY_VERSION=0.11.0 sh -s -- \
    --install-dir "$FABRIC_BUNDLE/.relay/bin"
```

The `hermes-agent` extra in `FABRIC_PACKAGE` installs only the NeMo Fabric
Hermes adapter; it does not install Hermes Agent. Hermes Agent 0.20 and later is
no longer installable from PyPI. Before running the Hermes examples, prepare the
task image by following the
[Hermes Agent installation guide](https://hermes-agent.nousresearch.com/docs/installation)
and ensure that the Fabric runner uses the Python environment containing Hermes
Agent and `nemo-fabric-adapters-hermes`. Relay-enabled Hermes runs require a
Hermes build compatible with Relay 0.10, such as the upstream source used by
`just install-hermes-agent`; set `ADAPTER_PYTHON` to that task environment's
interpreter. The dynamic `fabric_package` install is not a substitute for that
image preparation.

The curl command downloads and installs the standalone NeMo Relay 0.11.0 CLI tool
and verifies its checksum. For other installation methods, refer to the
[NeMo Relay installation instructions](../../../docs/getting-started/install.mdx#install-nemo-relay).

The generated files are ignored by Git. The uploaded bundle contains the example
MCP server and Relay; it does not contain a persisted `FabricConfig`.

Keep this shell open for the commands below.

## Verify Installation Inside SWE-Bench

Run the credential-free installation gate before spending model tokens:

```bash
uv run --extra harbor harbor run \
  --task swe-bench/django__django-13741 \
  --agent "$FABRIC_AGENT" \
  --ak fabric_adapter_id=nvidia.fabric.hermes \
  --ak fabric_config_bundle="$FABRIC_BUNDLE" \
  --ak "fabric_package=$FABRIC_PACKAGE" \
  --install-only \
  --job-name django-13741-install \
  --jobs-dir "$RUNS_DIR" \
  --n-concurrent 1
```

The job must complete with one trial and no exception.

## Run Deep Agents with Resource Stats

This recipe runs the Django task with Deep Agents, a self-hosted Nemotron model, and native resource sampling. Use a checkout containing the `fabric_system_stats` bridge option. Run from the NeMo Fabric repository root with Docker available:

```bash
uv sync --frozen --group adapters --extra harbor --extra deepagents
mkdir -p "$PWD/outputs"
export RUNS_DIR="$(mktemp -d "$PWD/outputs/stats-swebench.XXXXXX")"
mkdir -p "$RUNS_DIR/tmp"
export TMPDIR="$RUNS_DIR/tmp"
export JOB_NAME="stats-django-$(date +%Y%m%d-%H%M%S)"
.venv/bin/harbor download swe-bench/django__django-13741 \
  --output-dir "$RUNS_DIR/task" --export
export TASK_DIR="$RUNS_DIR/task/django__django-13741"

mkdir -p "$TASK_DIR/environment/vendor"
git ls-files -z | tar --null -czf \
  "$TASK_DIR/environment/vendor/nemo-fabric-source.tar.gz" -T -
uv export --frozen --no-default-groups --extra deepagents \
  --no-emit-workspace --no-emit-local --no-hashes \
  --output-file "$TASK_DIR/environment/vendor/harness-constraints.txt"
cp examples/harbor/swebench/deepagents/Dockerfile "$TASK_DIR/environment/Dockerfile"
docker build -t fabric-stats-django:dev "$TASK_DIR/environment"
sed -i '/^\[environment\]$/a docker_image = "fabric-stats-django:dev"' "$TASK_DIR/task.toml"

git rev-parse HEAD > "$RUNS_DIR/source-commit.txt"
docker image inspect fabric-stats-django:dev --format '{{.Id}}' > "$RUNS_DIR/image-id.txt"
docker run --rm fabric-stats-django:dev \
  uv pip freeze --python /opt/nemo-fabric-venv/bin/python > "$RUNS_DIR/task-packages.txt"
```

Each preparation creates a fresh task and results directory. To run the prepared task again, keep `TASK_DIR` and `RUNS_DIR` and choose a new `JOB_NAME`; skip the download and image preparation.

The image builds the native runtime and Python adapter from this checkout and constrains harness dependencies to `uv.lock`. The source archive includes tracked files with their current working-tree contents; include any new runtime source files in the archive before testing them. Task instructions and verification remain unchanged. Record the task revision, image ID, and resolved package versions with the experiment; the task and base image tags can change over time.

Check installation before invoking the model:

```bash
.venv/bin/harbor run --path "$TASK_DIR" \
  --agent nemo_fabric.integrations.harbor:FabricAgent \
  --ak fabric_adapter_id=nvidia.fabric.langchain.deepagents \
  --ak fabric_python=/opt/nemo-fabric-venv/bin/python \
  --install-only --job-name "${JOB_NAME}-install" \
  --jobs-dir "$RUNS_DIR/jobs" --n-concurrent 1
```

Set the endpoint to a server reachable from the task container, and provide its credential or a nonempty placeholder for an unauthenticated server:

```bash
export MODEL_BASE_URL="http://your-model-host:8000/v1"
export MODEL_API_KEY="unused"

.venv/bin/harbor run --path "$TASK_DIR" \
  --agent nemo_fabric.integrations.harbor:FabricAgent \
  --model nvidia/nemotron-3-ultra-550b-a55b \
  --ak fabric_adapter_id=nvidia.fabric.langchain.deepagents \
  --ak fabric_python=/opt/nemo-fabric-venv/bin/python \
  --ak "fabric_model_base_url=$MODEL_BASE_URL" \
  --ak fabric_model_api_key_env=MODEL_API_KEY \
  --ae 'MODEL_API_KEY=${MODEL_API_KEY}' \
  --ak 'fabric_harness_settings={"deepagents":{"backend":{"type":"local_shell"}}}' \
  --ak 'fabric_enabled_tools=["execute","ls","read_file","write_file","edit_file","glob","grep","write_todos"]' \
  --ak 'fabric_environment_env={"PATH":"/opt/miniconda3/envs/testbed/bin:/opt/miniconda3/bin:/usr/local/bin:/usr/bin:/bin"}' \
  --ak fabric_max_turns=200 \
  --ak fabric_timeout_sec=3300 \
  --ak 'fabric_system_stats={"sampling":{"interval_ms":1000,"max_samples":10000}}' \
  --job-name "$JOB_NAME" --jobs-dir "$RUNS_DIR/jobs" \
  --n-concurrent 1 --n-attempts 1 --max-retries 0
```

The local-shell backend lets Deep Agents run commands in the task container. It requires the explicit tool allowlist shown above, which includes `execute` and the file tools. Its command environment is explicit; the testbed Python comes first on `PATH`. For this harness, `fabric_max_turns` maps to the graph recursion limit. Harbor's verifier runs after the measured invocation. `TMPDIR` keeps temporary Compose files in the workspace so Docker launchers with a private `/tmp` can access the same files as Harbor.

`fabric_system_stats` accepts the SDK configuration object. Omit it for all groups with before/after collection, use `--ak fabric_system_stats=null` to disable invocation stats, or add `"collect":["process","cgroup.cpu"]` to select groups. Invalid settings fail host preflight. Host and task runtime versions must both support the selected configuration.

Inspect the downloaded result and locate its raw samples:

```bash
RESULT_PATH="$(find "$RUNS_DIR/jobs/$JOB_NAME" -path '*/agent/fabric-result-*.json' -print -quit)"
jq '{status, runtime_id, invocation_id, request_id, system_stats}' "$RESULT_PATH"
AGENT_DIR="$(dirname "$RESULT_PATH")"
REMOTE_SAMPLES="$(jq -r '.system_stats.sampled.samples.path' "$RESULT_PATH")"
cat "$AGENT_DIR/${REMOTE_SAMPLES#/logs/agent/}"
```

Paths inside `RunResult` refer to the task filesystem. Harbor downloads `/logs/agent` into the trial's `agent` directory, retaining the `fabric-artifacts` subtree. Keep the runner payload (`fabric-run-*.json`), result, samples, summary, and Harbor verifier result together. Agent completion and verifier reward answer different questions.

### Captured Container Run

A Linux run with the locked Deep Agents 0.7.9 environment and the model above produced the following observations. The agent reached `deepagents_recursion_limit_reached`, so Harbor recorded an execution error and did not run the verifier. The finalized stats and raw samples remain available despite that failure; there is no verifier reward to interpret as zero.

| Measurement | Observed value |
| --- | --- |
| Invocation duration | 239.198 seconds |
| Process-tree CPU estimate | 71.95 CPU-seconds |
| Sampled process RSS peak / mean / final | 985.99 / 355.20 / 201.79 MiB |
| Sampled process count peak / final | 12 / 1 |
| Visible CPU / memory limits | 1 core / 4 GiB |
| Cgroup CPU usage delta | 73.185 CPU-seconds |
| Cgroup throttled-time delta | 360.323 seconds |
| Samples / dropped samples / RSS coverage | 236 / 0 / 100% |

In this run, peak RSS is nearly five times the final reading. Actual measurements and agent outcomes vary between runs. For metric definitions, collection scope, and coverage, see [Resource Usage and Stats](../../../docs/sdk/system-stats.mdx).

## Run One Task with Hermes Agent

The default Hermes Agent command uses NVIDIA's hosted API:

```bash
: "${NVIDIA_API_KEY:?Export NVIDIA_API_KEY before running Hermes Agent}"

uv run --extra harbor harbor run \
  --task swe-bench/django__django-13741 \
  --agent "$FABRIC_AGENT" \
  --model nvidia/nemotron-3-nano-omni-30b-a3b-reasoning \
  --ak fabric_adapter_id=nvidia.fabric.hermes \
  --ak fabric_config_bundle="$FABRIC_BUNDLE" \
  --ak "fabric_package=$FABRIC_PACKAGE" \
  --ae 'NVIDIA_API_KEY=${NVIDIA_API_KEY}' \
  --job-name django-13741-hermes \
  --jobs-dir "$RUNS_DIR" \
  --n-concurrent 1 \
  --n-attempts 1 \
  --max-retries 1
```

For a self-hosted OpenAI-compatible model, change `--model` and add
`--ak fabric_model_base_url=<url>`. The server must support automatic tool
calling; a successful plain chat completion is not sufficient for SWE-Bench.

## Run One Task with OpenCode

Prepare the task with the OpenCode adapter from the current checkout. The
SWE-Bench task image requires `linux/amd64`:

```bash
cd "$(git rev-parse --show-toplevel)"

export FABRIC_AGENT='nemo_fabric.integrations.harbor:FabricAgent'
export FABRIC_BUNDLE="$PWD/examples/harbor/swebench"
export RUNS_DIR="$PWD/.tmp/harbor/fabric-swebench"
export DOCKER_DEFAULT_PLATFORM=linux/amd64

./examples/harbor/swebench/prepare_opencode_swebench.sh

export OPENCODE_SWEBENCH_TASK="$RUNS_DIR/prepared-django__django-13741-opencode"
export OPENCODE_JOB_NAME="django-13741-opencode-$(date +%Y%m%d-%H%M%S)"
```

Verify the prepared image without credentials or a model call:

```bash
uv run --extra harbor harbor run \
  --path "$OPENCODE_SWEBENCH_TASK" \
  --agent "$FABRIC_AGENT" \
  --ak fabric_adapter_id=nvidia.fabric.opencode \
  --ak fabric_config_bundle="$FABRIC_BUNDLE" \
  --ak fabric_config_target=/opt/nemo-fabric-config \
  --ak fabric_python=/opt/nemo-fabric-venv/bin/python \
  --install-only \
  --job-name "${OPENCODE_JOB_NAME}-install" \
  --jobs-dir "$RUNS_DIR" \
  --n-concurrent 1 \
  --force-build
```

Run the prepared task with NVIDIA's OpenAI-compatible endpoint:

```bash
: "${NVIDIA_API_KEY:?Export NVIDIA_API_KEY before running OpenCode}"

uv run --extra harbor harbor run \
  --path "$OPENCODE_SWEBENCH_TASK" \
  --agent "$FABRIC_AGENT" \
  --model nvidia/nemotron-3-nano-omni-30b-a3b-reasoning \
  --ak fabric_adapter_id=nvidia.fabric.opencode \
  --ak fabric_config_bundle="$FABRIC_BUNDLE" \
  --ak fabric_config_target=/opt/nemo-fabric-config \
  --ak fabric_python=/opt/nemo-fabric-venv/bin/python \
  --ak fabric_model_base_url=https://integrate.api.nvidia.com/v1 \
  --ak fabric_model_api_key_env=NVIDIA_API_KEY \
  --ak fabric_runtime_timeout_seconds=1800 \
  --ak fabric_timeout_sec=1900 \
  --ae 'NVIDIA_API_KEY=${NVIDIA_API_KEY}' \
  --job-name "$OPENCODE_JOB_NAME" \
  --jobs-dir "$RUNS_DIR" \
  --n-concurrent 1 \
  --n-attempts 1 \
  --max-retries 1 \
  --force-build
```

Inspect the OpenCode result with its generated job name:

```bash
uv run --extra harbor python -m json.tool \
  "$RUNS_DIR/$OPENCODE_JOB_NAME/result.json"
```

## Run One Task with Cline

Prepare the task with the Cline adapter and supported Cline SDK from the current
checkout. The SWE-Bench task image requires `linux/amd64`:

```bash
cd "$(git rev-parse --show-toplevel)"

export FABRIC_AGENT='nemo_fabric.integrations.harbor:FabricAgent'
export FABRIC_BUNDLE="$PWD/examples/harbor/swebench"
export RUNS_DIR="$PWD/.tmp/harbor/fabric-swebench"
export DOCKER_DEFAULT_PLATFORM=linux/amd64

./examples/harbor/swebench/prepare_cline_swebench.sh

export CLINE_SWEBENCH_TASK="$RUNS_DIR/prepared-django__django-13741-cline"
export CLINE_JOB_NAME="django-13741-cline-$(date +%Y%m%d-%H%M%S)"
```

Verify the prepared image without credentials or a model call:

```bash
uv run --extra harbor harbor run \
  --path "$CLINE_SWEBENCH_TASK" \
  --agent "$FABRIC_AGENT" \
  --ak fabric_adapter_id=nvidia.fabric.cline \
  --ak fabric_config_bundle="$FABRIC_BUNDLE" \
  --ak fabric_config_target=/opt/nemo-fabric-config \
  --ak fabric_python=/opt/nemo-fabric-venv/bin/python \
  --install-only \
  --job-name "${CLINE_JOB_NAME}-install" \
  --jobs-dir "$RUNS_DIR" \
  --n-concurrent 1 \
  --force-build
```

Run the prepared task with NVIDIA's OpenAI-compatible endpoint:

```bash
: "${NVIDIA_API_KEY:?Export NVIDIA_API_KEY before running Cline}"

uv run --extra harbor harbor run \
  --path "$CLINE_SWEBENCH_TASK" \
  --agent "$FABRIC_AGENT" \
  --model nvidia/nemotron-3-super-120b-a12b \
  --ak fabric_adapter_id=nvidia.fabric.cline \
  --ak fabric_config_bundle="$FABRIC_BUNDLE" \
  --ak fabric_config_target=/opt/nemo-fabric-config \
  --ak fabric_python=/opt/nemo-fabric-venv/bin/python \
  --ak fabric_model_base_url=https://integrate.api.nvidia.com/v1 \
  --ak fabric_model_api_key_env=NVIDIA_API_KEY \
  --ak fabric_runtime_timeout_seconds=1800 \
  --ak fabric_timeout_sec=1900 \
  --ae 'NVIDIA_API_KEY=${NVIDIA_API_KEY}' \
  --job-name "$CLINE_JOB_NAME" \
  --jobs-dir "$RUNS_DIR" \
  --n-concurrent 1 \
  --n-attempts 1 \
  --max-retries 1 \
  --force-build
```

Inspect the Cline result with its generated job name:

```bash
uv run --extra harbor python -m json.tool \
  "$RUNS_DIR/$CLINE_JOB_NAME/result.json"
```

## Run the Same Task with Claude

The task and verifier are unchanged. Only the adapter, model, credential, and
job name differ. Relay is enabled so the resulting ATIF confirms that the
harness switch reached Claude.

```bash
: "${ANTHROPIC_API_KEY:?Export ANTHROPIC_API_KEY before running Claude}"

uv run --extra harbor harbor run \
  --task swe-bench/django__django-13741 \
  --agent "$FABRIC_AGENT" \
  --model anthropic/claude-sonnet-4-5 \
  --ak fabric_adapter_id=nvidia.fabric.claude \
  --ak 'fabric_harness_settings={"permission_mode":"bypassPermissions"}' \
  --ak 'fabric_environment_env={"IS_SANDBOX":"1"}' \
  --ak fabric_max_turns=75 \
  --ak fabric_runtime_timeout_seconds=1800 \
  --ak fabric_config_bundle="$FABRIC_BUNDLE" \
  --ak fabric_telemetry=relay \
  --ak "fabric_package=$FABRIC_PACKAGE" \
  --ae "PATH=/tmp/nemo-fabric-config/.relay/bin:$PATH" \
  --ae 'ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}' \
  --job-name django-13741-claude \
  --jobs-dir "$RUNS_DIR" \
  --n-concurrent 1 \
  --n-attempts 1 \
  --max-retries 1
```

Claude uses unattended permissions only inside Harbor's ephemeral task
container. Do not apply that permission mode to a normal host environment.

## Vary One NeMo Fabric Capability

Start from the Hermes Agent command, replace its `--job-name`, and add the option in
the middle column. Relay is enabled for every capability variation so its ATIF
can confirm that the input reached the harness.

| Experiment | Add to the Hermes Agent command | Replacement job name |
| --- | --- | --- |
| Skill | `--skill "$PWD/examples/harbor/swebench/skills/swebench-debugging" --ak fabric_telemetry=relay` | `django-13741-hermes-skill` |
| MCP | `--mcp-config "$FABRIC_BUNDLE/mcp/repo-inspector.mcp.json" --ak fabric_telemetry=relay` | `django-13741-hermes-mcp` |
| Blocked tool | `--ak 'fabric_blocked_tools=["browser"]' --ak fabric_telemetry=relay` | `django-13741-hermes-tools` |
| Telemetry only | `--ak fabric_telemetry=relay` | `django-13741-hermes-relay` |

For example, the complete skill variation is:

```bash
: "${NVIDIA_API_KEY:?Export NVIDIA_API_KEY before running Hermes Agent}"

uv run --extra harbor harbor run \
  --task swe-bench/django__django-13741 \
  --agent "$FABRIC_AGENT" \
  --model nvidia/nemotron-3-nano-omni-30b-a3b-reasoning \
  --skill "$PWD/examples/harbor/swebench/skills/swebench-debugging" \
  --ak fabric_adapter_id=nvidia.fabric.hermes \
  --ak fabric_config_bundle="$FABRIC_BUNDLE" \
  --ak fabric_telemetry=relay \
  --ak "fabric_package=$FABRIC_PACKAGE" \
  --ae 'NVIDIA_API_KEY=${NVIDIA_API_KEY}' \
  --job-name django-13741-hermes-skill \
  --jobs-dir "$RUNS_DIR" \
  --n-concurrent 1 \
  --n-attempts 1 \
  --max-retries 1
```

The MCP definition starts the dependency-free
[`repo_inspector.py`](mcp/repo_inspector.py) inside the task container. The MCP
definition itself enters through Harbor's `--mcp-config` option.

For a pure telemetry comparison, run the Hermes Agent baseline once without
`fabric_telemetry`, then repeat it with `--ak fabric_telemetry=relay`. No other
model, harness, capability, task, or verifier input changes.

## Verify Reward and Relay Evidence

Harbor's verifier is the correctness authority. Relay telemetry is separate run
evidence and never replaces or changes the SWE-Bench reward.

Select a completed job and verify that it has one completed trial, no errors,
and a verifier reward. A reward of `0.0` is still a valid verifier result; it
means the proposed patch did not pass the task tests.

```bash
export JOB_NAME=django-13741-hermes
export RESULT_PATH="$RUNS_DIR/$JOB_NAME/result.json"

uv run --extra harbor python - "$RESULT_PATH" <<'PY'
import json
import sys

result = json.load(open(sys.argv[1], encoding="utf-8"))
stats = result["stats"]
assert stats["n_completed_trials"] == 1, stats
assert stats["n_errored_trials"] == 0, stats
rewards = {
    reward
    for evaluation in stats["evals"].values()
    for reward, trials in evaluation["reward_stats"]["reward"].items()
    if trials
}
assert rewards, "Harbor did not record a verifier reward"
print("verifier reward:", ", ".join(sorted(rewards)))
PY

uv run --extra harbor harbor view "$RUNS_DIR/$JOB_NAME"
```

For a Relay-enabled job, select its job name and inspect the published evidence:

```bash
export JOB_NAME=django-13741-hermes-relay

find "$RUNS_DIR/$JOB_NAME" \
  -path '*/agent/telemetry-validation.json' \
  -exec uv run --extra harbor python -m json.tool {} \;
find "$RUNS_DIR/$JOB_NAME" \
  -path '*/agent/trajectory.json' \
  -exec uv run --extra harbor python -m json.tool {} \;
find "$RUNS_DIR/$JOB_NAME" \
  \( -name '*.atof.jsonl' -o -name '*.atif.json' \) -print
```

Relay-enabled runs preserve the direct Relay ATOF and ATIF files. NeMo Fabric
promotes Relay's ATIF to `agent/trajectory.json`, Harbor's canonical ATIF path,
and also publishes `agent/telemetry-validation.json` plus the normalized
`agent/fabric-result-<id>.json`. Validate ATOF and ATIF independently.

Sample output from successful Relay-enabled Hermes Agent and Claude runs is checked
in under [`sample-artifacts/`](sample-artifacts/).

## Progress to a Full Run

Complete the install gate and the single-task experiments first. Then run a
five-task Hermes Agent shard with Relay enabled:

```bash
: "${NVIDIA_API_KEY:?Export NVIDIA_API_KEY before running Hermes Agent}"

uv run --extra harbor harbor run \
  --dataset swe-bench/swe-bench-verified \
  --n-tasks 5 \
  --agent "$FABRIC_AGENT" \
  --model nvidia/nemotron-3-nano-omni-30b-a3b-reasoning \
  --ak fabric_adapter_id=nvidia.fabric.hermes \
  --ak fabric_config_bundle="$FABRIC_BUNDLE" \
  --ak fabric_telemetry=relay \
  --ak "fabric_package=$FABRIC_PACKAGE" \
  --ae 'NVIDIA_API_KEY=${NVIDIA_API_KEY}' \
  --job-name swebench-verified-hermes-5 \
  --jobs-dir "$RUNS_DIR" \
  --n-concurrent 1 \
  --n-attempts 1 \
  --max-retries 1
```

Spot-check progress and evidence without changing the running job:

```bash
export JOB_NAME=swebench-verified-hermes-5
uv run --extra harbor harbor view "$RUNS_DIR/$JOB_NAME"
find "$RUNS_DIR/$JOB_NAME" -name result.json -print | head
find "$RUNS_DIR/$JOB_NAME" \
  -path '*/agent/telemetry-validation.json' -print | head
```

Inspect every exception and reward plus at least one NeMo Fabric result and telemetry
summary. Resume an interrupted job from its recorded configuration:

```bash
export JOB_NAME=swebench-verified-hermes-5
uv run --extra harbor harbor job resume \
  --job-path "$RUNS_DIR/$JOB_NAME"
```

Remove `--n-tasks` only after the shard is healthy, and choose concurrency that
respects model and Docker resource limits.
