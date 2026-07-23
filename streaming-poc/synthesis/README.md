<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Streaming POC — cross-harness synthesis & recommendation

Synthesis of the child POCs. **Hermes** ([findings](../hermes/findings.md)),
**Deep Agents** ([findings](../deepagents/findings.md)), and **Claude**
([findings](../claude/findings.md)) were run for real with ATOF capture (Claude
through the gateway on a **subscription / SSO** session — no API key; native
Anthropic events embedded in the ATOF). **Codex** ([findings](../codex/findings.md))
is still a stub — its row below is **preliminary**, drawn from a partial run and the
OpenAI Responses event model; it will be confirmed once the CLI/quota blocker
clears. The `invoke_stream` prototype is harness-agnostic and validated on both
Relay modes.

## Cross-harness evidence
| | Hermes | Deep Agents | Claude | Codex |
|---|---|---|---|---|
| Relay mode | in-process | in-process | gateway CLI | gateway CLI |
| native API | AIAgent callbacks | LangGraph + middleware | Anthropic Messages SSE | OpenAI Responses SSE |
| **stream unit** | callback scope/mark | scope tree (nested) | message → content_block → delta | response → item → delta |
| **token deltas** | ❌ (scope-level) | ❌ (scope-level) | ✅ SSE events live (`content_block_delta`); **text terminal-only** in current ATOF | ✅ `response.output_text.delta` (expected) |
| **nesting** | session>turn>tool/llm | deep (delegated sub-agents) | message>block | response>item |
| **ordering** | temporal | temporal (sequential in POC; `parent_uuid` would key any parallel) | temporal | temporal |
| **terminal** | `session end` scope + metadata | `request end` scope (`status OK`) | `message_delta`(stop,usage)+`message_stop` | `response.completed`/`failed`(+usage) |
| duplicate risk | delta↔terminal | subagent-echo + delta↔terminal + tree | latent (text terminal-only today) | HIGH (delta↔terminal, expected) |

### Differences that matter
- **Stream units differ per harness** — callback-scope vs. content-block vs.
  response/item. Codex ≠ Claude (different SDK event models); neither is inferable
  from the other.
- **Granularity splits by Relay mode:** gateway (Claude/Codex) = token-level;
  in-process (Hermes/Deep Agents) = scope-level (no token deltas).
- **Nesting/ordering:** Deep Agents needs `parent_uuid` tree reconstruction
  (parallel sub-agents *would* interleave, keyed by `parent_uuid` — but parallel
  execution was **not observed** in the POC; both subagents ran sequentially). The
  others are largely linear.
- **Terminal semantics differ** — scope-end + metadata vs. `message_stop` vs.
  `response.completed/failed`.

### The unifying fact
Relay wraps **all four** into one ATOF envelope: `scope`/`mark` records with
`uuid`/`parent_uuid`/`timestamp`, and native events either *are* the scope
(in-process) or are embedded in `llm.chunk` `data` (gateway — event type, block
index, timing, and usage inline; the per-delta text currently lands only in the
terminal scope, as the real Claude run confirmed). The cross-harness uniformity the
effort wanted **already exists at the ATOF layer**, with native detail preserved.

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
