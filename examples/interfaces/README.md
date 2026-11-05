<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Fabric Native Interface Examples

Use native harness interfaces alongside a Fabric-managed runtime. Each harness example uses the public Python SDK for configuration and lifecycle, then sends input through the harness's native interface.

## Hermes API and Relay

[`hermes.py`](hermes.py) starts one managed Hermes API server, submits a real file task, verifies the generated file, and checks Relay ATOF events and ATIF tool steps. Its API response remains native; the SDK manages the runtime but does not convert this response into a Fabric `RunResult`.

### Prepare the Environment

Follow the [Hermes adapter installation guide](../../docs/integrations/harness/hermes.mdx#install-the-adapter). For source development, run from the repository root:

```bash
just install-hermes-agent
export ADAPTER_PYTHON="$PWD/.venv-hermes/bin/python"
```

This prepares pinned Hermes v0.22.5, the adapter, and Relay 0.10. The application interpreter also needs the NeMo Fabric SDK. The command below uses the prepared Hermes environment for both application and adapter.

Set your OpenAI-compatible model endpoint, served model name, and credential:

```bash
export MODEL_BASE_URL="http://MODEL_HOST:8000/v1"
export MODEL_NAME="YOUR_SERVED_MODEL_NAME"
export MODEL_API_KEY="YOUR_MODEL_API_KEY"
```

Use the value required by your endpoint. An endpoint without authentication can use a nonempty placeholder credential. To reference an existing credential variable instead, pass its name with `--api-key-env`.

### Run a Real Task

Run the example from the repository root:

```bash
.venv-hermes/bin/python examples/interfaces/hermes.py \
  --base-url "$MODEL_BASE_URL" \
  --model "$MODEL_NAME" \
  --port 8642
```

The example copies sample release notes into a new workspace, asks the native API to read them and write a rollout checklist, and stops the runtime. To use your own notes, add `--input-file /path/to/release-notes.md`. `--output-dir` selects a new evidence directory; the example refuses to overwrite an existing directory. `--keep-running` leaves the API available for additional requests until you press Enter.

The model and port must be reachable from the adapter environment. The native API binds to loopback and uses a generated bearer token; the example reads the token file and sends the header without printing the token.

The example prints its evidence directory and checks that:

- The output file exists and is nonempty.
- ATOF records show `read_file` and `write_file` execution.
- A populated ATIF trajectory contains those tool steps.
- Stop removes the API token.

It exits unsuccessfully if any check fails. An ATOF file alone does not establish ATIF export.

### Inspect the Result and Trajectory

Set `RUN_DIR` to the evidence directory printed by the example:

```bash
RUN_DIR="outputs/interfaces/<printed-run-directory>"
cat "$RUN_DIR/workspace/rollout-checklist.md"
jq . "$RUN_DIR/summary.json"

ATIF=$(jq -r '.atif_files[0]' "$RUN_DIR/summary.json")
jq -r '.steps[] |
  [.step_id, .source, ((.tool_calls // []) | map(.function_name) | join(","))] |
  @tsv' "$ATIF"
```

For this task, expect user input, an agent `read_file` step with its observation, an agent `write_file` step with its observation, and a final response. Inspect complete tool arguments and observations with:

```bash
jq '.steps[] | select(.tool_calls != null) |
  {tool_calls, observation}' "$ATIF"
```

To inspect the raw event sequence:

```bash
ATOF=$(jq -r '.atof_files[0]' "$RUN_DIR/summary.json")
jq -r 'select(.kind == "scope" and
  (.category == "llm" or .category == "tool")) |
  [.timestamp, .scope_category, .category, .name] | @tsv' "$ATOF"
```

### ATIF Finalization and Limits

The pinned Hermes API path leaves its Relay conversation scope open after a turn. ATOF events can therefore exist without ATIF export. The managed native launcher finalizes and flushes an API turn's Relay segment when execution returns, including failed turns. It does not fire native conversation-end plugin hooks or delete conversation state. The conversation can continue; subsequent turns create new Relay segments and ATIF files. Finalization errors are logged without replacing the execution outcome, so consumers should check their artifacts explicitly.

This example exercises the managed API path. It does not qualify browser chat, TUI, independently restarted gateways, approvals, or complete native-result and artifact normalization. Refer to [Hermes Native Interfaces](../../docs/integrations/harness/hermes-interfaces.mdx) for the configuration and ownership boundaries.
