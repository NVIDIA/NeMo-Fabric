<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Fabric Hermes Testing Examples

These examples are a work in progress. Their structure and user experience will be reworked as part of OpenClaw TUI support.

Start with the [Hermes architecture guide](../../../docs/integrations/harness/hermes/hermes.mdx) for the invocation diagram, shared configuration, backend ownership, and result/artifact boundaries. Use this README to test each of the four client paths:

| Test Path | Entry Point | Architecture and Setup |
| --- | --- | --- |
| [1. SDK](#1-sdk) | [Code Review Example](../../code_review_agent/README.md) | [SDK guide](../../../docs/integrations/harness/hermes/hermes-sdk.mdx) |
| [2. HTTP API](#2-http-api) | `hermes.py` | [API guide](../../../docs/integrations/harness/hermes/hermes-api.mdx) |
| [3. Browser dashboard](#3-browser-dashboard) | `hermes_tui.py --dashboard`, then browser chat | [Dashboard guide](../../../docs/integrations/harness/hermes/hermes-dashboard.mdx) |
| [4. Terminal TUI](#4-terminal-tui) | `hermes_tui.py`, then terminal attachment | [Terminal guide](../../../docs/integrations/harness/hermes/hermes-tui.mdx) |

## Shared Setup for Native Interfaces

Paths 2–4 require a POSIX host and use the following setup. For SDK testing, follow the code-review example linked in [Path 1](#1-sdk).

Run commands from the NeMo Fabric repository root, including commands in additional terminals. Follow the [shared installation guide](../../../docs/integrations/harness/hermes/hermes.mdx#install-the-adapter). For repository development:

```bash
just install-hermes-agent
export ADAPTER_PYTHON="${ADAPTER_PYTHON:-$PWD/.venv-hermes/bin/python}"
```

The export preserves a separately selected adapter interpreter; otherwise it uses `.venv-hermes`. The example scripts below run in `.venv-hermes`, which provides their NeMo Fabric SDK host.

This prepares pinned Hermes v0.22.5, the NeMo Fabric adapter/SDK, and NeMo Relay 0.10 in `.venv-hermes`. The API dependency `aiohttp` and the Python web-serving dependencies needed by the TUI backend are included. Use `jq` for the JSON inspection commands below.

Set the model endpoint, served model name, and credential used by all three native tests:

```bash
export MODEL_BASE_URL="https://MODEL_HOST/v1"
export MODEL_NAME="YOUR_SERVED_MODEL_NAME"
export MODEL_API_KEY="YOUR_MODEL_API_KEY"
```

Use the credential required by your endpoint; an endpoint without authentication can use a nonempty placeholder. To use another credential environment variable, pass its name with `--api-key-env`. The endpoint must be reachable from the adapter environment. Use HTTPS for credential-bearing remote requests, or an authenticated encrypted tunnel; a local model can use a loopback URL such as `http://127.0.0.1:8000/v1`.

Each test creates a fresh evidence directory. The commands below include a timestamp; use a new directory when repeating a test because the scripts refuse to overwrite an existing one. Choose available ports. The native services bind to loopback in the adapter environment; accessing a remote adapter requires an appropriate port forward.

### Build the TUI Frontends

Complete this step for Paths 3 and 4. SDK and API testing do not require frontend builds.

Use the same pinned Hermes checkout selected during installation. If `HERMES_AGENT_CHECKOUT` was set for the installer, keep that value. Otherwise the default is `external/hermes-agent`:

```bash
export HERMES_AGENT_CHECKOUT="${HERMES_AGENT_CHECKOUT:-$PWD/external/hermes-agent}"
"$ADAPTER_PYTHON" -c "import aiohttp, fastapi, uvicorn"
(
  cd "$HERMES_AGENT_CHECKOUT"
  npm ci --ignore-scripts --workspace ui-tui --workspace web --include-workspace-root
  npm run build --workspace ui-tui
  npm run build --workspace web
)
export HERMES_TUI_DIR="$HERMES_AGENT_CHECKOUT/ui-tui"
export HERMES_WEB_DIST="$HERMES_AGENT_CHECKOUT/hermes_cli/web_dist"
```

For terminal-only testing, omit `npm run build --workspace web` and the `HERMES_WEB_DIST` export. Use Node supported by the pinned release (`^22.22.0`, `^24.11.0`, or `>=26.0.0`); the live client checks used Node 22.22.2. For a separately prepared Python environment, follow the additional dependency steps in the [dashboard](../../../docs/integrations/harness/hermes/hermes-dashboard.mdx#installation) or [terminal](../../../docs/integrations/harness/hermes/hermes-tui.mdx#installation) guide.

## 1. SDK

Use the [Code Review Example](../../code_review_agent/README.md#run-the-default-demo), which selects Hermes by default. That example covers normalized configuration, SDK invocation, output, and artifacts. Its [NeMo Relay instructions](../../code_review_agent/README.md#enable-relay-telemetry-and-streaming) describe telemetry and streaming options.

The SDK path runs the embedded Hermes `AIAgent` and returns a normalized NeMo Fabric `RunResult`. Refer to the [SDK architecture guide](../../../docs/integrations/harness/hermes/hermes-sdk.mdx) for its lifecycle and conversation behavior.

## 2. HTTP API

Architecture: [Hermes API](../../../docs/integrations/harness/hermes/hermes-api.mdx). [`hermes.py`](hermes.py) starts the native API, submits a real file task, and verifies its output and NeMo Relay evidence. The script manages the runtime through the SDK but sends the task through HTTP; the response remains native. This example retains the default embedded SDK agent, but does not invoke it.

### Start and Run the Task

After shared setup, run:

```bash
export RUN_DIR="$PWD/outputs/interfaces/api-$(date +%Y%m%d-%H%M%S)"
.venv-hermes/bin/python examples/interfaces/hermes/hermes.py \
  --base-url "$MODEL_BASE_URL" \
  --model "$MODEL_NAME" \
  --port 18642 \
  --output-dir "$RUN_DIR"
```

The script copies sample release notes into the workspace, asks Hermes to read them and write `rollout-checklist.md`, then stops the runtime. Use `--input-file /path/to/release-notes.md` for your own notes.

### Inspect the Result

The command exits successfully when the generated file, tool events, ATIF tool steps, and API-token cleanup checks pass:

```bash
cat "$RUN_DIR/workspace/rollout-checklist.md"
jq . "$RUN_DIR/summary.json"
jq -e '.checks | all' "$RUN_DIR/summary.json"
```

The last command should print `true`. The summary lists the ATOF and ATIF files; see [Inspect NeMo Relay Trajectories](#inspect-nemo-relay-trajectories) for their contents.

### Send Your Own API Input

To leave the API open after its initial task, add `--keep-running` to the start command, using a fresh `RUN_DIR`. The owner terminal waits for Enter after the task completes.

In a second terminal at the repository root, set `RUN_DIR` to the evidence directory printed by the owner, then submit another request:

```bash
export RUN_DIR="/absolute/path/to/the/printed/evidence-directory"
export API_PROMPT="Read $RUN_DIR/workspace/release-notes.md using file tools and summarize the release changes."
.venv-hermes/bin/python - <<'PY'
import os
from pathlib import Path
from examples.interfaces.hermes import call_api

root = Path(os.environ["RUN_DIR"])
[token_file] = (root / "artifacts/.fabric/hermes/runtimes").glob("*/interfaces/api.token")
response = call_api("http://127.0.0.1:18642", token_file.read_text(), os.environ["API_PROMPT"])
print(response["choices"][0]["message"]["content"])
PY
```

Edit `API_PROMPT` to test another input. The helper reads the token file and sends `Authorization: Bearer <token>` without printing the token. If you changed the API port, use the same port in this request.

Press Enter in the owner terminal to stop the runtime. The script then writes `summary.json` and checks that the token was removed. Without `--keep-running`, stop and verification happen automatically.

## 3. Browser Dashboard

Architecture: [Hermes Dashboard and Browser TUI](../../../docs/integrations/harness/hermes/hermes-dashboard.mdx). This path uses the dashboard's TUI backend rather than the API gateway or embedded SDK agent.

### Start the Dashboard

Complete shared setup and both frontend builds. In the owner terminal, run:

```bash
export RUN_DIR="$PWD/outputs/interfaces/browser-$(date +%Y%m%d-%H%M%S)"
.venv-hermes/bin/python examples/interfaces/hermes/hermes_tui.py \
  --base-url "$MODEL_BASE_URL" \
  --model "$MODEL_NAME" \
  --dashboard \
  --port 19119 \
  --output-dir "$RUN_DIR"
```

The script prints a browser URL, an absolute-path file task after `Try:`, and a terminal attachment command. It stays running until you press Enter in the owner terminal; this is expected. For this test, use the browser URL and leave the terminal attachment command unused.

The host example enables terminal attachment alongside the dashboard, sharing one backend. It disables the embedded SDK agent by default and does not start an API gateway.

### Interact in the Browser

1. Open the printed URL, normally `http://127.0.0.1:19119`, and open Chat.
2. Copy the file task printed after `Try:` into chat. It reads `workspace/notes.txt` and writes a three-item `workspace/checklist.md` using file tools.
3. Wait for the final response. The chat should show the file-tool activity and the created checklist.
4. Send a follow-up in the same chat, such as “Summarize the three checks we just wrote.”
5. In another terminal, set `RUN_DIR` to the printed evidence directory. Run `cat "$RUN_DIR/workspace/checklist.md"` and follow the [trajectory inspection steps](#inspect-nemo-relay-trajectories).

### Stop the Dashboard

Press Enter in the owner terminal. The runtime stops its TUI service and removes the attachment files. Browser and terminal output remains native; it does not become a NeMo Fabric `RunResult`.

## 4. Terminal TUI

**Experimental:** The terminal configuration, attachment files, and helper may change across releases. Read the [terminal interface notice](../../../docs/integrations/harness/hermes/hermes-tui.mdx) before relying on their compatibility.

Architecture: [Hermes Terminal TUI](../../../docs/integrations/harness/hermes/hermes-tui.mdx). Terminal-only mode uses the managed TUI backend without serving the browser dashboard.

### Start the Backend

Complete shared setup and the terminal frontend build. In the owner terminal, run:

```bash
export RUN_DIR="$PWD/outputs/interfaces/terminal-$(date +%Y%m%d-%H%M%S)"
.venv-hermes/bin/python examples/interfaces/hermes/hermes_tui.py \
  --base-url "$MODEL_BASE_URL" \
  --model "$MODEL_NAME" \
  --port 19119 \
  --output-dir "$RUN_DIR"
```

The script starts a headless TUI service, with the embedded SDK agent and API gateway disabled. It prints an attachment command and file task, then waits for Enter in the owner terminal. Stop the browser test first if you want to reuse port 19119.

### Attach and Interact

1. In a second interactive terminal at the repository root, run the exact attachment command printed by the owner. It uses the Hermes interpreter and that runtime's private `interfaces/tui.json` file.
2. Paste the file task printed after `Try:` into the Hermes terminal UI.
3. Wait for the final response and inspect the displayed file-tool activity.
4. Send a follow-up in the same conversation, such as “Summarize the three checks we just wrote.”
5. In another terminal, set `RUN_DIR` to the printed evidence directory. Run `cat "$RUN_DIR/workspace/checklist.md"` and follow the [trajectory inspection steps](#inspect-nemo-relay-trajectories).

The attachment helper reads a TUI credential separate from the API bearer token. It attaches to the existing service and does not start another backend.

### Detach and Stop

Enter `/quit` in the Hermes terminal UI to close the client. The owner process and backend remain running. Press Enter in the owner terminal to stop the runtime and remove its attachment files. For this test, let the turn complete before detaching.

## Inspect NeMo Relay Trajectories

NeMo Relay is enabled by default in both native example scripts. Inspect evidence while a TUI client stays connected, or after the API script finishes. In a separate terminal, set `RUN_DIR` to the evidence directory printed by the owner:

```bash
export RUN_DIR="/absolute/path/to/the/printed/evidence-directory"
find "$RUN_DIR/relay/atif" -type f -name '*.json' -print
```

Choose a trajectory from the list and inspect its tool calls and observations:

```bash
ATIF="/absolute/path/to/a/listed/trajectory.json"
jq '.steps[] | {step_id, source, tool_calls, observation}' "$ATIF"
```

For the API example, the generated file is `workspace/rollout-checklist.md`. Its summary also gives the trajectory path directly:

```bash
ATIF=$(jq -r '.atif_files[0]' "$RUN_DIR/summary.json")
jq '.steps[] | {step_id, source, tool_calls, observation}' "$ATIF"
```

Expect the file task's `read_file` and `write_file` calls with observations, followed by a final response. A follow-up turn produces another segment while preserving the native conversation history. To inspect the ATOF event sequence:

```bash
find "$RUN_DIR/relay/atof" -type f -name '*.jsonl' -exec \
  jq -r 'select(.kind == "scope" and (.category == "llm" or .category == "tool")) |
    [.timestamp, .scope_category, .category, .name] | @tsv' {} \;
```

The [architecture guide's normalization table](../../../docs/integrations/harness/hermes/hermes.mdx#what-nemo-fabric-normalizes) explains the boundary: NeMo Relay provides common evidence formats, while native API and TUI responses retain native session identities and response types.

## Test Browser and Terminal Together

Start the browser example in Path 3 and run its printed terminal attachment command in a second terminal. Submit a task in each client, choosing different output filenames. Both clients use one backend and separate conversations. Follow the same inspection and owner-stop steps.

The host script also accepts `--sdk` to enable the embedded SDK agent and `--api-port 18642` to enable the separate API gateway. Enabling SDK execution does not submit an SDK task automatically; use the code-review example for the standalone SDK test. API and TUI ports must differ.

## Limitations

Interactive approvals, detach during active work, remote terminal attachment, named-profile switching, same-conversation multi-client control, and independently replaced gateways are outside these manual flows. The adapter defaults `HERMES_YOLO_MODE` to enabled. Complete native-result, usage, and session-artifact normalization remain planned work; see [Limitations and Planned Work](../../../docs/integrations/harness/hermes/hermes.mdx#limitations-and-planned-work).
