<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Streaming POC — cross-harness synthesis & recommendation

Synthesis of the child POCs. **All four were run for real**, and for each the raw
native SDK stream was **teed before Relay** and diffed against Relay's ATOF from the
same run: **Hermes** ([findings](../hermes/findings.md)) and **Deep Agents**
([findings](../deepagents/findings.md)) in-process, **Claude**
([findings](../claude/findings.md)) and **Codex** ([findings](../codex/findings.md))
through the gateway on a **subscription / SSO** session — no API key. The gateway
diff is measured, not assumed: the native stream carries the **per-delta token text**
that Relay's ATOF projection **drops** (text is terminal-only). Codex and Claude
share the **same ATOF envelope** (`scope`/`mark`, `uuid`/`parent_uuid`, one
`llm.chunk` per SSE event) even though their **event vocabularies and payloads
differ** (`response.*` items vs `message`/`content_block` events) — the *shape* is
uniform, the *content* is provider-specific. The `invoke_stream` prototype is
harness-agnostic and validated on both Relay modes.

## Cross-harness evidence
| | Hermes | Deep Agents | Claude | Codex |
|---|---|---|---|---|
| Relay mode | in-process | in-process | gateway CLI | gateway CLI |
| native API (captured layer) | AIAgent callbacks | LangGraph + middleware | Anthropic Messages SSE (`StreamEvent`) | **Codex app-server notifications** (`item/agentMessage/delta`) — Relay taps Responses SSE at the gateway |
| **stream unit** | callback scope/mark | scope tree (nested) | message → content_block → delta | response → item → delta (ATOF) / notification (native) |
| **token deltas** | ❌ (scope-level) | ❌ (scope-level) | ✅ SSE events live (`content_block_delta`); **text terminal-only** in current ATOF | ✅ SSE events live (`response.output_text.delta`); **text terminal-only** in current ATOF |
| **nesting** | session>turn>tool/llm | deep (delegated sub-agents) | message>block | response>item |
| **ordering** | temporal | temporal; **parallel subagents observed** — interleaved, keyed by `parent_uuid`/`namespace` | temporal | temporal |
| **terminal** | `session end` scope + metadata | `request end` scope (`status OK`) | `message_delta`(stop,usage)+`message_stop` | `response.completed`/`failed`(+usage) |
| duplicate risk | delta↔terminal | subagent-echo + delta↔terminal + tree | latent (text terminal-only today) | latent (text terminal-only today) |

### Differences that matter
- **Stream units differ per harness** — callback-scope vs. content-block vs.
  response/item. Codex ≠ Claude (different SDK event models); neither is inferable
  from the other.
- **Granularity splits by Relay mode:** gateway (Claude/Codex) = per-delta events
  (one ATOF `llm.chunk` per SSE event — structure/timing/usage live, but the token
  **text is terminal-only** in the current ATOF projection, measured against the
  native fixtures); in-process (Hermes/Deep Agents) = scope-level (no token deltas).
- **Nesting/ordering:** Deep Agents needs `parent_uuid` tree reconstruction —
  parallel sub-agents **do** interleave, keyed by `parent_uuid`/`namespace`. Now
  **observed** (`deepagents/parallel-*.jsonl`, `llama-3.1-70b`): two `task` calls in
  one message, 9.57s overlapping sibling scopes, interleaved namespaces
  (`43d0a4d3`↔`5c7a0931`). Stream order alone mis-nests. The others are largely
  linear.
- **Terminal semantics differ** — scope-end + metadata vs. `message_stop` vs.
  `response.completed/failed`.

### The unifying fact
Relay wraps **all four** into one ATOF envelope: `scope`/`mark` records with
`uuid`/`parent_uuid`/`timestamp`, and native events either *are* the scope
(in-process) or are **projected** into `llm.chunk` `data` (gateway — event type,
block/item index, timing, and usage inline, but **not** the delta text, which lands
only in the terminal scope, as **both** the real Claude and Codex native fixtures
confirmed). The
uniform **envelope** the effort wanted already exists at the ATOF layer; the
per-event *content* stays provider-specific, and the per-delta token text is the one
native detail the projection drops.

## Final recommendation — v0.1
**Ship raw, Relay-generated ATOF pass-through**, surfaced as sugar over
`Runtime.invoke()`:

```python
runtime = await fabric.start_runtime(config)   # relay on → loopback ATOF endpoint injected
stream  = runtime.invoke_stream(input="...")
async for atof_record in stream:   # raw canonical ATOF (dict)
    ...
result = await stream.result()     # RunResult, out of band
```
- Available **only when Relay is enabled** (`relay_enabled`); else
  `FabricCapabilityError`.
- **No Fabric-specific normalization** in v0.1; **`RunResult` out of band**.

## Why normalization is deferred (not needed for v0.1)
1. **ATOF is already the common envelope** across all four harnesses (uniform
   `scope`/`mark` + `uuid`/`parent_uuid` + `timestamp`) — the uniformity goal is
   met without a bespoke schema.
2. **A normalized model would be lossy and premature:** it must reconcile
   fundamentally different units (callback-scope vs content_block vs
   response/item) and would erase per-harness structure — Hermes scope semantics,
   the Deep Agents delegation tree, provider-specific deltas.
3. **No token-delta unification payoff** for the in-process harnesses (they emit
   none).
4. **The hard consumer concerns are contract, not schema:** delta-vs-terminal
   duplication (render deltas live, treat terminal as authoritative — *replace,
   don't append*) and tree reconstruction via `parent_uuid` are documentable on
   raw ATOF today.

A typed/normalized event layer can later sit **on top of** the raw stream, opt-in,
without changing this contract.

## Production work breakdown
See [work-breakdown.md](work-breakdown.md).
