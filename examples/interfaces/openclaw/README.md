<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Fabric OpenClaw Interface Examples

Start with the [OpenClaw architecture guide](../../../docs/integrations/harness/openclaw/openclaw.mdx). These four paths share one managed Gateway, its selected agent, and translated Fabric configuration. SDK responses are normalized; other responses remain native. NeMo Relay integration and trajectory export are deferred.

| Path | Example | Architecture |
| --- | --- | --- |
| SDK | [Code Review Example](../../code_review_agent/README.md) with `--variant openclaw` | [SDK](../../../docs/integrations/harness/openclaw/openclaw-sdk.mdx) |
| HTTP | `openclaw.py --http-input ... --once` | [HTTP API](../../../docs/integrations/harness/openclaw/openclaw-api.mdx) |
| Browser | `openclaw.py --interface dashboard`, then native chat | [Dashboard](../../../docs/integrations/harness/openclaw/openclaw-dashboard.mdx) |
| Terminal | `openclaw.py --interface terminal`, then generated launcher | [Terminal TUI](../../../docs/integrations/harness/openclaw/openclaw-tui.mdx) |

## Shared Setup

Run all commands from the repository root. Install pinned OpenClaw `2026.9.34` with Node `>=24.22.1 <25` or `>=26.12.1` and npm `11.16+`:

```bash
npm install --global openclaw@2026.9.34 --allow-scripts=openclaw
openclaw --version
uv sync --frozen --no-default-groups --group adapters --group adapter-tests --group test
export ADAPTER_PYTHON="${ADAPTER_PYTHON:-$PWD/.venv/bin/python}"
export MODEL_BASE_URL="https://MODEL_HOST/v1"
export MODEL_NAME="YOUR_SERVED_MODEL_NAME"
export MODEL_API_KEY="YOUR_MODEL_API_KEY"
```

The npm package provides browser and terminal assets; these tests do not build frontends. The model endpoint must be reachable from the adapter and support OpenAI-compatible Chat Completions with file-tool calls. Use a nonempty placeholder only if the endpoint does not require authentication. A separately selected adapter interpreter must contain this checkout's adapter and dependencies.

Use a new output directory for each run; the example refuses to overwrite one. `--openclaw-command /path/to/openclaw` selects a separately prepared pinned executable. Use `--api-key-env NAME` when your credential uses another environment variable. No command prints the Gateway token.

## 1. SDK

Follow the [Code Review Example](../../code_review_agent/README.md) with `--variant openclaw`. To compare an SDK turn with the native clients in this example, add `--sdk-input 'Read notes.txt and write sdk-result.md with a short checklist.'` to an owner command below. The result is saved as `sdk-result.json`.

## 2. HTTP API

Run a real file task, then stop the owner automatically:

```bash
export RUN_DIR="$PWD/outputs/interfaces/openclaw-http-$(date +%Y%m%d-%H%M%S)"
.venv/bin/python examples/interfaces/openclaw/openclaw.py \
  --base-url "$MODEL_BASE_URL" --model "$MODEL_NAME" \
  --interface terminal --output-dir "$RUN_DIR" \
  --http-input 'Read notes.txt and write api-result.md with a short checklist.' --once
cat "$RUN_DIR/workspace/api-result.md"
cat "$RUN_DIR/http-result.json"
cat "$RUN_DIR/cleanup.json"
```

Terminal selection publishes private Gateway connection material without enabling the browser. This command does not launch the terminal client. The HTTP request uses that Gateway, an independent stable `user`, and the configured agent. Inspect a native HTTP response and confirm all cleanup fields are true. There is no additional API server or API port setting.

## 3. Browser Dashboard

In the owner terminal, choose an available Gateway base port and keep the owner running:

```bash
export RUN_DIR="$PWD/outputs/interfaces/openclaw-browser-$(date +%Y%m%d-%H%M%S)"
.venv/bin/python examples/interfaces/openclaw/openclaw.py \
  --base-url "$MODEL_BASE_URL" --model "$MODEL_NAME" \
  --interface dashboard --port 18789 --output-dir "$RUN_DIR"
```

Open the printed `http_url` in a browser. Read the printed token-file path privately and enter its value in the native Control UI sign-in field, or use OpenClaw's native token-fragment sign-in flow. The token authenticates to this Gateway; it is not the model API key. Select agent `reviewer` and create a distinct chat.

Send `Read notes.txt and write browser-result.md with a short checklist.` After completion, send `Read browser-result.md and add one verification step.`

In another terminal, set `RUN_DIR` to the same output directory used by the owner. Use the parent directory of the `workspace` path printed by the owner, replacing the placeholder below with that exact path:

```bash
export RUN_DIR="/absolute/path/to/openclaw-browser-YYYYMMDD-HHMMSS"
cat "$RUN_DIR/workspace/browser-result.md"
```

Press Enter in the owner terminal to stop. Check `cleanup.json` and confirm the browser can no longer submit work. Browser tabs share a native session when they select the same chat; another tab does not guarantee isolation.

## 4. Terminal TUI

In the owner terminal:

```bash
export RUN_DIR="$PWD/outputs/interfaces/openclaw-terminal-$(date +%Y%m%d-%H%M%S)"
.venv/bin/python examples/interfaces/openclaw/openclaw.py \
  --base-url "$MODEL_BASE_URL" --model "$MODEL_NAME" \
  --interface terminal --output-dir "$RUN_DIR"
```

In another terminal, run the exact `terminal_launcher` path printed by the owner. Terminal attachment requires POSIX and a real TTY. The launcher reads private credentials and attaches to the owned Gateway; it does not start another backend. It has no port argument. Add `--session manual-review` to the launcher to intentionally reuse that conversation on agent `reviewer`; omit it for a new session.

Send `Read notes.txt and write terminal-result.md with a short checklist.` Follow with `Read terminal-result.md and add one verification step.` Inspect the output, then enter `/quit`. The owner should remain usable; press Enter there when finished. The pinned native client can exit with code 130 through its own disconnect fallback. Verify Gateway survival separately from that native exit status.

Use `--interface both --port 18789` on the owner to attach browser and terminal concurrently to one Gateway. Select distinct chats unless you intend to share a live conversation.

## Automated Native Qualification

The ordinary OpenClaw tests use a deterministic mock executable. The opt-in native tests use the pinned Gateway, a recording model endpoint, actual file tools and MCP, a real browser, and a terminal PTY:

```bash
uv pip install --python .venv/bin/python playwright pexpect
.venv/bin/python -m playwright install chromium
export FABRIC_OPENCLAW_TEST_COMMAND="$(command -v openclaw)"
mkdir -p outputs/interfaces
.venv/bin/python -m pytest tests/adapters/test_openclaw_native.py -q --basetemp outputs/interfaces/openclaw-native-tests
```

Use a fresh `--basetemp` directory: pytest removes an existing one. `FABRIC_TEST_CHROMIUM` can select an already installed browser executable. Retained evidence includes captured model requests, a browser screenshot, a terminal transcript, and native skill inventory. Credentials used by the recorder are test placeholders. These tests assert actual endpoint/model/parameter/prompt parity, workspace/tool policy, MCP execution, skills, conversation behavior, health, and cleanup; they do not assert Relay trajectories.
