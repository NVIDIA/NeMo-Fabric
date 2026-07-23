<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Codex streaming POC — findings

**Harness:** `nvidia.fabric.codex` · **Relay mode:** gateway (Relay CLI, `nemo-relay`
0.6.0) · **Model:** `gpt-5.6-sol` (reasoning effort medium) · **Auth:** Codex /
ChatGPT **subscription / SSO** forwarded through the gateway — **no
`OPENAI_API_KEY`** (the gateway relayed the OAuth session from
`CODEX_HOME=~/.codex/auth.json`). Requires **Codex CLI ≥0.145.0** (older CLIs reject
the ChatGPT account's `gpt-5.6-sol`).

## Scenario (real run, live delta-event stream through the gateway)
Prompt: *"Reply with a one-sentence greeting."* Captured live via `invoke_stream`
while the real Codex CLI ran a single turn against the OpenAI Responses API through
the Relay gateway, with **both** recorders active. Final answer: *"Hello! It's great
to meet you."*

## Fixtures & how they were captured
- [`native-events.jsonl`](native-events.jsonl) — **genuine native evidence**, teed
  *before* Relay at the Codex SDK `AsyncTurnHandle.stream()` notification loop (the
  same stream `handle.run()` consumes), via `common/native_recorder.py`. 26 raw
  Codex app-server notifications; delta bodies preserved verbatim, no PII present.
- [`events.atof.jsonl`](events.atof.jsonl) — Relay's ATOF from the *same* run
  (22 records; oversized request/terminal snapshots elided for size — model, shape,
  counts, usage, IDs, and the terminal answer text preserved).

> **Layer note (important).** The two fixtures are captured at **different layers**,
> so this is not a byte-identical source→projection pair like Claude. The Codex
> adapter sees **Codex app-server notifications** (`item/agentMessage/delta`, …);
> Relay taps the **OpenAI Responses SSE** at the gateway (`response.output_text.delta`,
> …). They are two representations of the same turn. The comparison below still holds
> the reviewer's line: the **token text is present in the native adapter stream and
> absent from Relay's ATOF**.

## Native event units (real Codex app-server notifications)
The 26 notifications, by `method`: `turn/started` ×1, `hook/started`/`hook/completed`
×3 each, `item/started`/`item/completed` ×2 each, **`item/agentMessage/delta` ×9**
(the token stream, each with a `payload.delta` text fragment),
`thread/tokenUsage/updated` ×1, `turn/completed` ×1, and `error` ×4 (benign — the CLI
probes a `ws://…/responses` transport that returns 405 and falls back to HTTP; the
turn still succeeds). **Unit = one app-server notification.**

## Native → ATOF diff (same run — this is the point)
Both streams carry exactly **9 text deltas**, one per token fragment, in order — but
only the native stream carries the **text**:

| native notification (`native-events.jsonl`) | Relay ATOF (`events.atof.jsonl`) | delta text? |
|---|---|---|
| `item/agentMessage/delta` `payload.delta` = `"Hello"`,`"!"`,`" It"`,`"’s"`,`" great"`,`" to"`,`" meet"`,`" you"`,`"."` (×9) | `llm.chunk` `event_type=response.output_text.delta`, `indices` only (×9) | **native YES → ATOF NO** |
| `thread/tokenUsage/updated` / `turn/completed` (usage) | `llm.chunk` `event_type=response.completed` (+ `usage`) | n/a — usage preserved in both |
| `item/completed` (assembled message text) | `scope openai.responses end` `output[].content[].text` | terminal text in both |

Assembling the 9 native `payload.delta` fragments yields exactly the terminal
answer `"Hello! It's great to meet you."`; the 9 ATOF `response.output_text.delta`
`llm.chunk` records carry `event_type` + `indices` + `provider` and **no `delta`/
`text` field** — verified by grepping both files.

## What is preserved vs. lost (measured, not inferred)
- **Preserved in ATOF:** the full Responses event **sequence**
  (`response.created` … `response.completed`), per-event **timing**,
  **output/content indices**, and terminal **usage** + assembled text.
- **Lost in ATOF:** the **per-delta token text**, which the native adapter stream
  carries (`item/agentMessage/delta.payload.delta`) and every ATOF
  `response.output_text.delta` omits.
- So the gateway stream gives, live, **delta-event structure + timing + usage** —
  **not renderable incremental text**; the text is terminal-only. This is the
  "degraded granularity" (option a) contract, and it matches Claude: a Relay
  ATOF-projection choice, not a per-harness limitation.

## Streamed events vs. terminal response · duplicate-rendering risk
As with Claude, delta text is **absent** from the live ATOF today, so only the
terminal `openai.responses end` `output` carries it — duplication is *latent*. If
Relay later projects delta text, render live for progress/cadence and treat
`await stream.result()` (→ terminal `output` text) as authoritative — **replace,
don't append**.

## Recommendation
**Raw ATOF pass-through (v0.1).** Codex and Claude share the same ATOF **envelope**
(`scope`/`mark`, `uuid`/`parent_uuid`, one `llm.chunk` per SSE event) though their
event vocabularies and payloads differ (`response.*` vs `message`/`content_block`),
and both exhibit the same single measured loss (delta text). The uniform *shape*
(not identical content) plus the uniform loss is strong evidence that normalizing
per-harness is unnecessary for v0.1. Ship raw ATOF; document the delta-vs-terminal
contract.

## Reproduce this experiment
Prereqs: a native extension matching the Python SDK (`just build-python`, or
`PYO3_PYTHON=$PWD/.venv/bin/python cargo build -p fabric-python --release` then copy
`target/release/lib_native.dylib` → `python/src/nemo_fabric/_native.abi3.so`), the
`nemo-relay` gateway CLI (≥0.6.0), **Codex CLI ≥0.145.0**, and **either** a signed-in
Codex/ChatGPT subscription (`~/.codex/auth.json`) **or** a funded `OPENAI_API_KEY`.

1. **Apply the POC native tee** (temporary — revert after capture) in
   `adapters/codex/.../adapter.py`, in `_invoke_thread`: when `POC_NATIVE_RECORD` is
   set, consume `handle.stream()` through `openai_codex._run._collect_async_turn_result`
   and `record("codex-sdk", event.method, {"method": …, "payload": …})` for each
   notification before collection (see
   [`../common/native_recorder.py`](../common/native_recorder.py)).
2. **Run with both recorders active** (subscription / SSO path — no API key):
   ```bash
   unset OPENAI_API_KEY
   export CODEX_HOME="$HOME/.codex"
   export FABRIC_RELAY_CLI="$(command -v nemo-relay)"
   export FABRIC_MODEL="gpt-5.6-sol"
   export POC_RECORDER_DIR="$PWD/streaming-poc/common"
   export POC_NATIVE_RECORD="$PWD/streaming-poc/codex/native-events.jsonl"; rm -f "$POC_NATIVE_RECORD"
   python streaming-poc/common/run_harness.py nvidia.fabric.codex \
     streaming-poc/codex/events.atof.jsonl "Reply with a one-sentence greeting."
   ```
   `run_harness` streams the Relay ATOF via `invoke_stream` → `events.atof.jsonl`;
   the tee writes the notifications → `native-events.jsonl`. For the API-key path,
   `export OPENAI_API_KEY=…` (funded) instead.
3. **Revert** the seam:
   `git checkout -- adapters/codex/src/nemo_fabric_adapters/codex/adapter.py`.

Cross-harness recommendation: [../synthesis/README.md](../synthesis/README.md).
