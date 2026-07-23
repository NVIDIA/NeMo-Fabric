<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Claude streaming POC — findings

**Harness:** `nvidia.fabric.claude` · **Relay mode:** gateway (Relay CLI, `nemo-relay`) ·
**Model:** `claude-sonnet-4-5` (extended thinking on) · **Auth:** Claude Code
**subscription / SSO** forwarded through the gateway — **no `ANTHROPIC_API_KEY`**
(the gateway relayed the OAuth session from `ANTHROPIC_CONFIG_DIR=~/.claude`).

## Scenario (real run, live delta-event stream through the gateway)
Prompt: *"Reply with a one-sentence greeting."* Captured live via `invoke_stream`
while the real Claude Agent SDK ran a single turn against the Anthropic Messages
API through the Relay gateway, with **both** recorders active. The turn produced a
thinking block and a text block; final answer: *"Hello! I'm Claude, ready to help
you with your NeMo-Fabric project or any other tasks you have in mind."*

## Fixtures & how they were captured
- [`native-events.jsonl`](native-events.jsonl) — **genuine native evidence**, the
  raw Anthropic SSE teed *before* Relay by enabling `include_partial_messages=True`
  and recording every `StreamEvent.event` in `ClaudeSDKClient.receive_response()`
  (POC-only seam, via `common/native_recorder.py`). 15 raw stream events. Secrets
  are redacted by key; **one email address in the model's thinking was redacted**;
  the delta bodies (token text) are preserved verbatim.
- [`events.atof.jsonl`](events.atof.jsonl) — Relay's ATOF from the *same* run
  (21 records; the oversized request snapshot #2 has its `data.content` +
  `category_profile` elided for size — model/shape/counts/IDs preserved).

## Native event units (real Anthropic SSE)
`StreamEvent.event` is the raw Anthropic Messages stream event. The 15 captured:
`message_start` → `content_block_start` → `content_block_delta` ×7 (thinking block)
→ `content_block_stop` → `content_block_start` → `content_block_delta` ×1 (text
block) → `content_block_stop` → `message_delta` → `message_stop`. **Unit = one SSE
event.**

## Native (SSE) → ATOF diff (same run — this is the point)
Pairing the two fixtures by event order shows ATOF is a **projection** that keeps
the event *structure* but **drops the delta bodies**:

| native SSE (`native-events.jsonl`) | Relay ATOF (`events.atof.jsonl`) | delta text? |
|---|---|---|
| `content_block_delta` `delta.thinking="The"`, `" user is asking…"`, … (7 thinking deltas) | `llm.chunk` `event_type=content_block_delta`, `indices.index=0` | **native YES → ATOF NO** |
| `content_block_delta` `delta.text="Hello! I'm Claude, ready to help you with your NeMo-Fabric project…"` | `llm.chunk` `event_type=content_block_delta`, `indices.index=1` | **native YES → ATOF NO** |
| `message_start` / `message_delta` (usage) | `llm.chunk` `event_type=message_start`/`message_delta` (+ `usage`) | n/a — usage preserved in both |
| *(not surfaced by the SDK stream)* | `llm.chunk` `event_type=ping` (1 extra) | ATOF carries a `ping` the SDK's `StreamEvent` stream does not |

The ATOF `llm.chunk` records carry `event_type` + `indices` + `provider` + (on
start/delta of the message) `usage`, but **no `text`/`thinking` field** — verified
by grepping both files: every token string lives in `native-events.jsonl` and in
the terminal `anthropic.messages end` scope, and in **none** of the ATOF deltas.

## What is preserved vs. lost (measured, not inferred)
- **Preserved in ATOF:** the full event **sequence**, per-event **timing**,
  **usage/token accounting**, **content-block indices**, and the terminal assembled
  message. (ATOF even adds the `ping` the SDK hides.)
- **Lost in ATOF:** the **per-delta token text** — present in the native SSE
  (`delta.text` / `delta.thinking`), absent from every ATOF `llm.chunk`. The text
  reappears only in the terminal `anthropic.messages end` scope `content`.
- So the gateway stream gives, live, **delta-event structure + timing + usage** —
  **not renderable incremental text**. That is the "degraded granularity" (option a)
  contract accepted for v0.1: per-delta *cadence and shape* live, authoritative
  *text* at the end. The loss is a property of Relay's current ATOF projection, not
  of the Anthropic wire (which carries `text_delta`, as `native-events.jsonl`
  proves).

## Streamed events vs. terminal response · duplicate-rendering risk
Today the delta text is **absent** from the live ATOF, so only the terminal scope
carries it — duplication is *latent*, not active. If Relay later projects delta
text into `llm.chunk`, it would appear both live and in the terminal `content`
(HIGH risk); the contract is fixed regardless: render live for progress/cadence,
treat `await stream.result()` (→ terminal `content`) as authoritative — **replace,
don't append**.

## Recommendation
**Raw ATOF pass-through (v0.1).** ATOF wraps the turn in the same `scope`/`mark`
envelope as every harness and preserves the event structure, timing, and usage a
progress UI needs; the one measured loss (delta text) is uniform with Codex and is
a Relay-projection choice, not a per-harness schema problem. Ship raw ATOF; document
the delta-vs-terminal contract above.

## Reproduce this experiment
Prereqs: a native extension matching the Python SDK (`just build-python`, or
`PYO3_PYTHON=$PWD/.venv/bin/python cargo build -p fabric-python --release` then copy
`target/release/lib_native.dylib` → `python/src/nemo_fabric/_native.abi3.so`), the
`nemo-relay` gateway CLI (≥0.6.0), and **either** a Claude Code subscription signed
in (`~/.claude`) **or** an `ANTHROPIC_API_KEY`.

1. **Apply the POC native tee** (temporary — revert after capture) in
   `adapters/claude/.../adapter.py`: add
   `include_partial_messages=bool(os.environ.get("POC_NATIVE_RECORD"))` to the
   `ClaudeAgentOptions(...)` in `build_options`, and in the
   `client.receive_response()` loop record each `StreamEvent` before it is dropped:
   `record("claude-sdk", ev.event.get("type"), ev.event)` (see
   [`../common/native_recorder.py`](../common/native_recorder.py)).
2. **Run with both recorders active** (subscription / SSO path — no API key):
   ```bash
   unset ANTHROPIC_API_KEY
   export ANTHROPIC_CONFIG_DIR="$HOME/.claude"
   export FABRIC_RELAY_CLI="$(command -v nemo-relay)"
   export FABRIC_MODEL="claude-sonnet-4-5"
   export POC_RECORDER_DIR="$PWD/streaming-poc/common"
   export POC_NATIVE_RECORD="$PWD/streaming-poc/claude/native-events.jsonl"; rm -f "$POC_NATIVE_RECORD"
   python streaming-poc/common/run_harness.py nvidia.fabric.claude \
     streaming-poc/claude/events.atof.jsonl "Reply with a one-sentence greeting."
   ```
   `run_harness` sets `permission_mode=bypassPermissions` and streams the Relay ATOF
   via `invoke_stream` → `events.atof.jsonl`; the tee writes the raw SSE →
   `native-events.jsonl`. For the API-key path, `export ANTHROPIC_API_KEY=…` instead.
3. **Revert** the seam:
   `git checkout -- adapters/claude/src/nemo_fabric_adapters/claude/adapter.py`.
   (Redact any PII, e.g. an email in the model's thinking, before committing.)

Cross-harness recommendation: [../synthesis/README.md](../synthesis/README.md).
