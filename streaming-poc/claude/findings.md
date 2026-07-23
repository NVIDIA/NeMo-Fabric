<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Claude Streaming POC — Findings

**Harness:** `nvidia.fabric.claude` · **Relay mode:** gateway (Relay CLI, `nemo-relay`) ·
**Model:** `claude-sonnet-4-5` (extended thinking on) · **Auth:** Claude Code
**subscription / SSO** forwarded through the gateway — **no `ANTHROPIC_API_KEY`**
(the gateway relayed the OAuth session from `ANTHROPIC_CONFIG_DIR=~/.claude`).

## Scenario
Prompt: *"Reply with a one-sentence greeting."* A single Claude Agent SDK turn
against the Anthropic Messages API through the Relay gateway, captured via
`invoke_stream` with both recorders active. The turn produced a thinking block and a
text block; final answer: *"Hello! I'm Claude, ready to help you with your
NeMo-Fabric project or any other tasks you have in mind."*

## Fixtures & How They Were Captured
- [`native-events.jsonl`](native-events.jsonl) — the raw Anthropic SSE, teed
  *before* Relay by enabling `include_partial_messages=True` and recording every
  `StreamEvent.event` in `ClaudeSDKClient.receive_response()` (POC-only seam, via
  `common/native_recorder.py`). 15 stream events; delta bodies (token text)
  verbatim. Secrets redacted by key; one email in the model's thinking redacted.
- [`events.atof.jsonl`](events.atof.jsonl) — Relay's ATOF from the *same* run
  (21 records; the oversized request snapshot #2 has its `data.content` +
  `category_profile` elided for size — model/shape/counts/IDs preserved).

## Native Event Units (Real Anthropic SSE)
`StreamEvent.event` is the raw Anthropic Messages stream event. The 15 captured:
`message_start` → `content_block_start` → `content_block_delta` ×7 (thinking block)
→ `content_block_stop` → `content_block_start` → `content_block_delta` ×1 (text
block) → `content_block_stop` → `message_delta` → `message_stop`. **Unit = one SSE
event.**

## Native (SSE) → ATOF Diff (Same Run)
Pairing the two fixtures by event order shows ATOF is a **projection** that keeps
the event *structure* but **drops the delta bodies**:

| native SSE (`native-events.jsonl`) | Relay ATOF (`events.atof.jsonl`) | delta text? |
|---|---|---|
| `content_block_delta` `delta.thinking="The"`, `" user is asking…"`, … (7 thinking deltas) | `llm.chunk` `event_type=content_block_delta`, `indices.index=0` | **native YES → ATOF NO** |
| `content_block_delta` `delta.text="Hello! I'm Claude, ready to help you with your NeMo-Fabric project…"` | `llm.chunk` `event_type=content_block_delta`, `indices.index=1` | **native YES → ATOF NO** |
| `message_start` / `message_delta` (usage) | `llm.chunk` `event_type=message_start`/`message_delta` (+ `usage`) | n/a — usage preserved in both |
| *(not surfaced by the SDK stream)* | `llm.chunk` `event_type=ping` (1 extra) | ATOF carries a `ping` the SDK's `StreamEvent` stream does not |

The ATOF `llm.chunk` records carry `event_type` + `indices` + `provider` + (on the
message's start/delta) `usage`, but **no `text`/`thinking` field**: every token
string lives in `native-events.jsonl` and the terminal `anthropic.messages end`
scope, and in **none** of the ATOF deltas.

## What Is Preserved vs. Lost
- **Preserved in ATOF:** the full event **sequence**, per-event **timing**,
  **usage/token accounting**, **content-block indices**, and the terminal assembled
  message. (ATOF even adds the `ping` the SDK hides.)
- **Lost in ATOF:** the **per-delta token text** — present in the native SSE
  (`delta.text` / `delta.thinking`), absent from every ATOF `llm.chunk`. The text
  reappears only in the terminal `anthropic.messages end` scope `content`. (ATOF also
  keeps only the content-block *index*, not the `content_block` start body/type —
  but that isn't needed to render progress.)
- So the gateway stream gives, live, **delta-event structure + timing + usage** —
  **not renderable incremental text**; authoritative text arrives at the end. This
  is a property of Relay's current ATOF projection, not of the Anthropic wire (which
  carries `text_delta`, as `native-events.jsonl` shows).

## Streamed Events vs. Terminal Response · Duplicate-Rendering Risk
Today the delta text is **absent** from the live ATOF, so only the terminal scope
carries it — duplication is *latent*, not active. If Relay later projects delta
text into `llm.chunk`, it would appear both live and in the terminal `content`
(HIGH risk); the contract is fixed regardless: render live for progress/cadence,
treat `await stream.result()` (→ terminal `content`) as authoritative — **replace,
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
`nemo-relay` gateway CLI (≥0.6.0), and **either** a signed-in Claude Code
subscription (`~/.claude`) **or** an `ANTHROPIC_API_KEY`. The seam is a checked-in,
reversible patch ([`../patches/claude-native-tee.patch`](../patches/claude-native-tee.patch));
output goes to a scratch directory so the committed fixtures are never overwritten.

Subscription/SSO path (what this fixture used — no API key):
```bash
out=$(mktemp -d)
git apply streaming-poc/patches/claude-native-tee.patch
unset ANTHROPIC_API_KEY
export ANTHROPIC_CONFIG_DIR="$HOME/.claude"          # signed-in Claude Code session
export FABRIC_RELAY_CLI="$(command -v nemo-relay)"   # gateway forwards the OAuth session
export FABRIC_MODEL="claude-sonnet-4-5"
POC_RECORDER_DIR="$PWD/streaming-poc/common" \
POC_NATIVE_RECORD="$out/native-events.jsonl" \
python streaming-poc/common/run_harness.py nvidia.fabric.claude \
  "$out/events.atof.jsonl" "Reply with a one-sentence greeting."
git apply -R streaming-poc/patches/claude-native-tee.patch
```
`run_harness` sets `permission_mode=bypassPermissions` and streams the Relay ATOF via
`invoke_stream` → `$out/events.atof.jsonl`; the seam writes the raw SSE →
`$out/native-events.jsonl`. For the API-key path, `export ANTHROPIC_API_KEY=…`
instead. The committed fixtures are these outputs with the oversized request snapshot
truncated and one email in the model's thinking redacted before check-in.

Cross-harness recommendation: [the synthesis](../synthesis/README.md).
