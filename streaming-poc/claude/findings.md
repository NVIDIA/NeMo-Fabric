<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Claude streaming POC — findings

**Harness:** `nvidia.fabric.claude` · **Relay mode:** gateway (Relay CLI, `nemo-relay`) ·
**Model:** `claude-sonnet-4-5` (extended thinking on) · **Auth:** Claude Code
**subscription / SSO** forwarded through the gateway — **no `ANTHROPIC_API_KEY`**
(the gateway relayed the OAuth session from `ANTHROPIC_CONFIG_DIR=~/.claude`).

## Scenario (real run, token streaming through the gateway)
Prompt: *"Reply with a one-sentence greeting."* Captured live via `invoke_stream`
while the real Claude Agent SDK ran a single turn against the Anthropic Messages
API through the Relay gateway. The turn produced a thinking block and a text block;
final answer: *"Hello! I'm Claude, ready to help you with coding, research, and any
other tasks you have in mind."*

## Fixture & how it was captured
- [`events.atof.jsonl`](events.atof.jsonl) — Relay's ATOF from the real run, 18
  records, streamed one-per-line to the SDK loopback listener as they occurred.
- **No separate `native-events.jsonl` is needed here.** In gateway mode Relay taps
  the Anthropic SSE wire and **embeds each native event verbatim** into an
  `llm.chunk` mark: `data.event_type` is the raw Anthropic stream event
  (`message_start`, `content_block_delta`, …) and `data.indices.index` is the
  content-block index. So the native event stream *is* captured, inline, inside the
  ATOF — unlike the in-process harnesses (Hermes / Deep Agents) where the native
  callbacks had to be teed before Relay.

(The oversized request snapshot on record #2 — full system prompt + 27 tool schemas
— has its `data.content` and `category_profile` elided for size, matching the other
fixtures; model, shape, counts, IDs, and all streaming deltas are preserved.)

## Native event units (real Anthropic SSE, embedded in ATOF)
The 13 `llm.chunk` marks carry the canonical Anthropic Messages streaming sequence,
in order (`chunk_index` · `event_type`):
```
0 message_start        4 content_block_delta   8 content_block_start   11 message_delta
1 content_block_start  5 content_block_delta    9 content_block_delta   12 message_stop
2 ping                 6 content_block_delta   10 content_block_stop
3 content_block_delta  7 content_block_stop
```
Two content blocks: block 0 = **thinking** deltas (3–6), block 1 = **text** delta
(9). **Unit = one SSE event** (message / content-block / delta) — true token-level
granularity, which the in-process harnesses do not have.

## Prototype crossing the Fabric boundary
Identical to the other harnesses: `common/run_harness.py` →
`start_streaming_runtime` injects a loopback ndjson ATOF endpoint into
`config.relay.observability.atof.endpoints`; the **gateway** (`nemo-relay`) pushes
ATOF live to the SDK listener; `invoke_stream` yields each raw record. The gateway
also forwarded the subscription OAuth, so the whole path ran **without an API key**.

## Native (SSE) → ATOF mapping (same run)
| native SSE event | ATOF record | preserved | dropped / changed |
|---|---|---|---|
| request (model, tools, system, thinking, `stream:true`) | `scope anthropic.messages start` | model, `max_tokens`, thinking config, message/tool **counts**, header key names | full system prompt + tool schemas elided in the committed fixture (present live) |
| `message_start` … `message_stop` (13 SSE events) | 13 `llm.chunk` marks | **event type**, **content-block index**, **timing** (per-event `timestamp`), **usage** (on `message_start`/`message_delta`) | **the per-delta token TEXT** (`text_delta` / `thinking_delta` payloads) — see below |
| `message_delta`(stop_reason,usage) + `message_stop` | `scope anthropic.messages end` (+ `claude-code-turn end`) carrying the assembled `content`, `stop_reason`, `usage` | the final text, thinking block, stop reason, token usage | the streaming boundary between the two terminal SSE events collapses into scope-end |

Pairing key: `uuid`/`parent_uuid` (every `llm.chunk` has
`parent_uuid = anthropic.messages` scope uuid) + `chunk_index` + event order.

## What is preserved vs. lost (comparison-based, from the fixture alone)
- **Faithfully preserved live:** the full native event **sequence**, per-event
  **timing** (13 events over ~0.88 s — real cadence), **usage/token accounting**,
  **content-block indices**, and the request shape.
- **Dropped from the live stream:** the **actual streamed text**. Each
  `content_block_delta` `llm.chunk` carries `event_type` + `indices` + `provider`
  but **no `text`/`thinking` field** — the assembled text appears only in the
  terminal `anthropic.messages end` scope `content`. So a live consumer sees
  *when* and *how many* tokens arrive and the block structure, but the **text
  itself is terminal-only**.
- This is exactly the **"degraded granularity" (option a)** contract accepted for
  v0.1: token-level *structure and timing* live, authoritative *text* at the end.
  It is a property of Relay's current ATOF projection (event structure, not delta
  bodies), not of the Anthropic wire, which does carry `text_delta`.

## Streamed events vs. terminal response · duplicate-rendering risk
**HIGH** if Relay later carries delta text: the same tokens would appear both in
the `content_block_delta` stream and in the terminal `content`. Contract: render
the live stream for progress/cadence; treat `await stream.result()` (→ terminal
`content`) as authoritative — **replace, don't append**. Today, because delta text
is absent live, only the terminal scope has the text, so duplication is latent
rather than active — but the consumer contract must assume it.

## Recommendation
**Raw ATOF pass-through (v0.1).** The gateway already embeds the native Anthropic
event stream inside ATOF `llm.chunk` marks with faithful type/index/timing/usage,
and wraps the turn in the same `scope`/`mark` envelope as every other harness —
so the cross-harness uniformity goal is met at the ATOF layer with no bespoke
schema. Ship raw ATOF; document the delta-vs-terminal contract above.

## Reproduce this experiment
Prereqs: a native extension matching the Python SDK (`just build-python`, or
`PYO3_PYTHON=$PWD/.venv/bin/python cargo build -p fabric-python --release` then copy
`target/release/lib_native.dylib` → `python/src/nemo_fabric/_native.abi3.so`), the
`nemo-relay` gateway CLI (≥0.6.0), and **either** a Claude Code subscription signed
in (`claude` CLI, `~/.claude`) **or** an `ANTHROPIC_API_KEY`.

Subscription / SSO path (what this fixture used — no API key):
```bash
unset ANTHROPIC_API_KEY
export ANTHROPIC_CONFIG_DIR="$HOME/.claude"        # signed-in Claude Code session
export FABRIC_RELAY_CLI="$(command -v nemo-relay)" # gateway forwards the OAuth session
export FABRIC_MODEL="claude-sonnet-4-5"
python streaming-poc/common/run_harness.py nvidia.fabric.claude \
  streaming-poc/claude/events.atof.jsonl "Reply with a one-sentence greeting."
```
`run_harness` sets `permission_mode=bypassPermissions` for Claude so the turn runs
non-interactively, streams the Relay ATOF via `invoke_stream` → `events.atof.jsonl`.
For the API-key path instead, `export ANTHROPIC_API_KEY=…` and skip the SSO vars.

Cross-harness recommendation: [../synthesis/README.md](../synthesis/README.md).
