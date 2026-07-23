<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Codex streaming POC — findings

**Harness:** `nvidia.fabric.codex` · **Relay mode:** gateway (Relay CLI, `nemo-relay`
0.6.0) · **Model:** `gpt-5.6-sol` (reasoning effort medium) · **Auth:** Codex /
ChatGPT **subscription / SSO** forwarded through the gateway — **no
`OPENAI_API_KEY`** (the gateway relayed the OAuth session from
`CODEX_HOME=~/.codex/auth.json`).

## Scenario (real run, token streaming through the gateway)
Prompt: *"Reply with a one-sentence greeting."* Captured live via `invoke_stream`
while the real Codex CLI ran a single turn against the OpenAI Responses API through
the Relay gateway. Final answer: *"Hello! It's great to meet you."*

Requires Codex CLI **≥0.145.0** — the ChatGPT account's default model `gpt-5.6-sol`
is rejected by older CLIs ("requires a newer version of Codex"). See the sibling
[Claude findings](../claude/findings.md); the two gateway harnesses are twins.

## Fixture & how it was captured
- [`events.atof.jsonl`](events.atof.jsonl) — Relay's ATOF from the real run, 22
  records, streamed one-per-line to the SDK loopback listener as they occurred.
- **No separate `native-events.jsonl` is needed here.** In gateway mode Relay taps
  the OpenAI Responses SSE wire and **embeds each native event** into an `llm.chunk`
  mark: `data.event_type` is the raw Responses stream event (`response.created`,
  `response.output_text.delta`, …) and `data.indices` carries `output_index` /
  `content_index`. So the native event stream *is* captured inline in the ATOF —
  unlike the in-process harnesses (Hermes / Deep Agents) where the native callbacks
  had to be teed before Relay.

(The oversized request snapshot on record #2 — 7-item input, ~67 KB of system
prompt + context — and the two terminal response objects have their `data` and
`category_profile` elided for size, matching the other fixtures; model, shape,
counts, IDs, usage, and the terminal answer text are preserved.)

## Native event units (real OpenAI Responses SSE, embedded in ATOF)
The 17 `llm.chunk` marks carry the canonical OpenAI Responses streaming sequence,
in order (`chunk_index` · `event_type`):
```
0 response.created          4–12 response.output_text.delta (×9)   14 response.content_part.done
1 response.in_progress        13 response.output_text.done          15 response.output_item.done
2 response.output_item.added                                         16 response.completed (usage)
3 response.content_part.added
```
One output item, one content part; the 9 `response.output_text.delta` events are the
token stream. **Unit = one SSE event** (response / item / content-part / delta) —
true token-level granularity, which the in-process harnesses do not have.

## Prototype crossing the Fabric boundary
Identical to the other harnesses: `common/run_harness.py` →
`start_streaming_runtime` injects a loopback ndjson ATOF endpoint into
`config.relay.observability.atof.endpoints`; the **gateway** (`nemo-relay`) pushes
ATOF live to the SDK listener; `invoke_stream` yields each raw record. The gateway
also forwarded the ChatGPT OAuth, so the whole path ran **without an API key**.

## Native (SSE) → ATOF mapping (same run)
| native SSE event | ATOF record | preserved | dropped / changed |
|---|---|---|---|
| request (model, input, reasoning, text, `stream:true`) | `scope openai.responses start` | model, reasoning/text config, tool_choice, input **item count + bytes**, header key names | full input (system prompt + context) elided in the committed fixture (present live) |
| `response.created` … `response.completed` (17 SSE events) | 17 `llm.chunk` marks | **event type**, **output/content indices**, **timing** (per-event `timestamp`), **usage** (on `response.completed`) | **the per-delta token TEXT** (`response.output_text.delta` `delta` string) — see below |
| `response.completed` (final `output`, usage, status) | `scope openai.responses end` (+ `codex-turn end`) carrying `output[].content[].text`, `usage`, `status` | the final answer text, token usage, stop status, response id | verbose request params + instructions/tools elided in the fixture for size |

Pairing key: `uuid`/`parent_uuid` (every `llm.chunk` has
`parent_uuid = openai.responses` scope uuid) + `chunk_index` + event order;
`turn_id` is echoed in the terminal `output` metadata.

## What is preserved vs. lost (comparison-based, from the fixture alone)
- **Faithfully preserved live:** the full native event **sequence**, per-event
  **timing**, **output/content indices**, and the request shape; **usage** on the
  terminal `response.completed`.
- **Dropped from the live stream:** the **actual streamed text**. Each
  `response.output_text.delta` `llm.chunk` carries `event_type` + `indices` +
  `provider` but **no `delta`/`text` field** — the assembled text appears only in
  the terminal `openai.responses end` scope `output[].content[].text`. So a live
  consumer sees *when* and *how many* tokens arrive and the item/part structure, but
  the **text itself is terminal-only**.
- This is exactly the **"degraded granularity" (option a)** contract accepted for
  v0.1, and it matches Claude verbatim — a property of Relay's current ATOF
  projection (event structure, not delta bodies), not of the OpenAI wire, which does
  carry the `delta` text.

## Streamed events vs. terminal response · duplicate-rendering risk
**HIGH** if Relay later carries delta text: the same tokens would appear both in the
`response.output_text.delta` stream and in the terminal `output`. Contract: render
the live stream for progress/cadence; treat `await stream.result()` (→ terminal
`output` text) as authoritative — **replace, don't append**. Today, because delta
text is absent live, only the terminal scope has the text, so duplication is latent
rather than active — but the consumer contract must assume it.

## Recommendation
**Raw ATOF pass-through (v0.1).** The gateway already embeds the native OpenAI
Responses event stream inside ATOF `llm.chunk` marks with faithful
type/index/timing/usage, and wraps the turn in the same `scope`/`mark` envelope as
every other harness — Codex and Claude produce structurally identical ATOF from
different SDK event models, confirming the cross-harness uniformity holds at the
ATOF layer with no bespoke schema. Ship raw ATOF; document the delta-vs-terminal
contract above.

## Reproduce this experiment
Prereqs: a native extension matching the Python SDK (`just build-python`, or
`PYO3_PYTHON=$PWD/.venv/bin/python cargo build -p fabric-python --release` then copy
`target/release/lib_native.dylib` → `python/src/nemo_fabric/_native.abi3.so`), the
`nemo-relay` gateway CLI (≥0.6.0), **Codex CLI ≥0.145.0**, and **either** a signed-in
Codex/ChatGPT subscription (`~/.codex/auth.json`) **or** a funded `OPENAI_API_KEY`.

Subscription / SSO path (what this fixture used — no API key):
```bash
unset OPENAI_API_KEY
export CODEX_HOME="$HOME/.codex"                   # signed-in Codex session (auth.json)
export FABRIC_RELAY_CLI="$(command -v nemo-relay)" # gateway forwards the OAuth session
export FABRIC_MODEL="gpt-5.6-sol"
python streaming-poc/common/run_harness.py nvidia.fabric.codex \
  streaming-poc/codex/events.atof.jsonl "Reply with a one-sentence greeting."
```
`run_harness` streams the Relay ATOF via `invoke_stream` → `events.atof.jsonl`.
For the API-key path instead, `export OPENAI_API_KEY=…` (funded) and skip the SSO
vars.

Cross-harness recommendation: [../synthesis/README.md](../synthesis/README.md).
