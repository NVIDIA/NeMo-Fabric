<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Codex Streaming POC — Findings

**Harness:** `nvidia.fabric.codex` · **Relay mode:** gateway (Relay CLI, `nemo-relay`
0.6.0) · **Model:** `gpt-5.6-sol` (reasoning effort medium) · **Auth:** Codex /
ChatGPT **subscription / SSO** forwarded through the gateway — **no
`OPENAI_API_KEY`** (the gateway relayed the OAuth session from
`CODEX_HOME=~/.codex/auth.json`). Requires **Codex CLI ≥0.145.0** (older CLIs reject
the ChatGPT account's `gpt-5.6-sol`).

## Scenario
Prompt: *"Reply with a one-sentence greeting."* A single Codex CLI turn against the
OpenAI Responses API through the Relay gateway, captured via `invoke_stream` with
both recorders active. Final answer: *"Hello! It’s great to meet you."*

## Fixtures & How They Were Captured
- [`native-events.jsonl`](native-events.jsonl) — the native stream, teed *before*
  Relay at the Codex SDK `AsyncTurnHandle.stream()` notification loop (the same
  stream `handle.run()` consumes), via `common/native_recorder.py`. 26 Codex
  app-server notifications; delta bodies verbatim, no PII.
- [`events.atof.jsonl`](events.atof.jsonl) — Relay's ATOF from the *same* run
  (22 records; oversized request/terminal snapshots elided for size — model, shape,
  counts, usage, IDs, and the terminal answer text preserved).

> **Layer note.** The two fixtures are captured at **different layers**, so this is
> not a byte-identical source→projection pair like Claude. The Codex adapter sees
> **Codex app-server notifications** (`item/agentMessage/delta`, …); Relay taps the
> **OpenAI Responses SSE** at the gateway (`response.output_text.delta`, …) — two
> representations of the same turn. Either way, the token text is present in the
> native stream and absent from Relay's ATOF.

## Native Event Units (Real Codex App-Server Notifications)
The 26 notifications, by `method`: `turn/started` ×1, `hook/started`/`hook/completed`
×3 each, `item/started`/`item/completed` ×2 each, **`item/agentMessage/delta` ×9**
(the token stream, each with a `payload.delta` text fragment),
`thread/tokenUsage/updated` ×1, `turn/completed` ×1, and `error` ×4 (benign — the CLI
probes a `ws://…/responses` transport that returns 405 and falls back to HTTP; the
turn still succeeds). **Unit = one app-server notification.**

## Native → ATOF Diff (Same Run)
Both streams carry exactly **9 text deltas**, one per token fragment, in order — but
only the native stream carries the **text**:

| native notification (`native-events.jsonl`) | Relay ATOF (`events.atof.jsonl`) | delta text? |
|---|---|---|
| `item/agentMessage/delta` `payload.delta` = `Hello`, `!`, ` It`, `’s`, ` great`, ` to`, ` meet`, ` you`, `.` (×9) | `llm.chunk` `event_type=response.output_text.delta`, `indices` only (×9) | **native YES → ATOF NO** |
| `thread/tokenUsage/updated` / `turn/completed` (usage) | `llm.chunk` `event_type=response.completed` (+ `usage`) | n/a — usage preserved in both |
| `item/completed` (assembled message text) | `scope openai.responses end` `output[].content[].text` | terminal text in both |

Assembling the 9 native `payload.delta` fragments yields exactly the terminal
answer `"Hello! It’s great to meet you."`; the 9 ATOF `response.output_text.delta`
`llm.chunk` records carry `event_type` + `indices` + `provider` and **no `delta`/
`text` field**.

## What Is Preserved vs. Lost
- **Preserved in ATOF:** the full Responses event **sequence**
  (`response.created` … `response.completed`), per-event **timing**,
  **output/content indices**, and terminal **usage** + assembled text.
- **Lost in ATOF:** the **per-delta token text**, which the native adapter stream
  carries (`item/agentMessage/delta.payload.delta`) and every ATOF
  `response.output_text.delta` omits.
- So the gateway stream gives, live, **delta-event structure + timing + usage** —
  **not renderable incremental text**; authoritative text arrives at the end. Same
  result as Claude — a Relay ATOF-projection choice, not a per-harness limitation.

## Streamed Events vs. Terminal Response · Duplicate-Rendering Risk
As with Claude, delta text is **absent** from the live ATOF today, so only the
terminal `openai.responses end` `output` carries it — duplication is *latent*. If
Relay later projects delta text, render live for progress/cadence and treat
`await stream.result()` (→ terminal `output` text) as authoritative — **replace,
don't append**.

## Recommendation
**Raw ATOF pass-through (v0.1)** — see [the cross-harness recommendation](../synthesis/README.md).
ATOF preserves enough for a **structural** progress UI (event sequence, timing,
usage, indices, terminal text), **but not incremental text rendering**; the
rendering-relevant loss is the per-delta text. Document the delta-vs-terminal
contract above.

## Reproduce This Experiment
Prereqs: a native extension matching the Python SDK (`just build-python`, or
`PYO3_PYTHON=$PWD/.venv/bin/python cargo build -p fabric-python --release`, then copy
`target/release/lib_native.dylib` → `python/src/nemo_fabric/_native.abi3.so`), the
`nemo-relay` gateway CLI (≥0.6.0), **Codex CLI ≥0.145.0**, and **either** a signed-in
Codex/ChatGPT subscription (`~/.codex/auth.json`) **or** a funded `OPENAI_API_KEY`.
The seam is a checked-in, reversible patch
([`../patches/codex-native-tee.patch`](../patches/codex-native-tee.patch)); output
goes to a scratch directory so the committed fixtures are never overwritten.

Subscription/SSO path (what this fixture used — no API key):
```bash
out=$(mktemp -d)
git apply streaming-poc/patches/codex-native-tee.patch
unset OPENAI_API_KEY
export CODEX_HOME="$HOME/.codex"                     # signed-in Codex session (auth.json)
export FABRIC_RELAY_CLI="$(command -v nemo-relay)"   # gateway forwards the OAuth session
export FABRIC_MODEL="gpt-5.6-sol"
POC_RECORDER_DIR="$PWD/streaming-poc/common" \
POC_NATIVE_RECORD="$out/native-events.jsonl" \
python streaming-poc/common/run_harness.py nvidia.fabric.codex \
  "$out/events.atof.jsonl" "Reply with a one-sentence greeting."
git apply -R streaming-poc/patches/codex-native-tee.patch
```
`run_harness` streams the Relay ATOF via `invoke_stream` → `$out/events.atof.jsonl`;
the seam writes the notifications → `$out/native-events.jsonl`. For the API-key path,
`export OPENAI_API_KEY=…` (funded) instead. The committed fixtures are these outputs
with oversized request/terminal snapshots truncated before check-in.

Cross-harness recommendation: [the synthesis](../synthesis/README.md).
