<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Streaming POC — Cross-Harness Synthesis & Recommendation

Synthesis of the three child POCs — covering four harnesses. Each folder holds the
harness's native SDK stream (teed before Relay) and Relay's ATOF from the same run —
**Hermes**
([findings](../hermes/findings.md)) and **Deep Agents**
([findings](../deepagents/findings.md)) in-process, **Claude**
([findings](../claude/findings.md)) and **Codex** ([findings](../codex/findings.md))
through the gateway on a subscription/SSO session (no API key). The `invoke_stream`
prototype is harness-agnostic across both Relay modes.

**Conclusion.** Relay wraps all four harnesses in one ATOF envelope, so the
cross-harness uniformity is **structural — no bespoke schema needed.** The envelope
is uniform (`scope`/`mark`, `uuid`/`parent_uuid`, one `llm.chunk` per SSE event); the
per-event *content* stays provider-specific (`response.*` items vs
`message`/`content_block` events). ATOF is a lossy projection of the native stream —
it also drops, e.g., content-block start bodies/types and Codex app-server lifecycle
detail — but the **only rendering-relevant loss shared by both gateway harnesses is
the per-delta token text**, which lands only in the terminal scope.

## Cross-Harness Evidence
The captured native layer, stream unit, token-delta behavior, nesting, ordering,
terminal semantics, and duplicate-rendering risk for each harness:

| | Hermes | Deep Agents | Claude | Codex |
|---|---|---|---|---|
| Relay mode | in-process | in-process | gateway CLI | gateway CLI |
| native API (captured layer) | Hermes lifecycle/plugin hooks | LangGraph + middleware | Anthropic Messages SSE (`StreamEvent`) | **Codex app-server notifications** (`item/agentMessage/delta`) — Relay taps Responses SSE at the gateway |
| **stream unit** | callback scope/mark | scope tree (nested) | message → content_block → delta | response → item → delta (ATOF) / notification (native) |
| **token deltas** | ❌ (scope-level) | ❌ (scope-level) | ✅ SSE events live (`content_block_delta`); **text terminal-only** in current ATOF | ✅ SSE events live (`response.output_text.delta`); **text terminal-only** in current ATOF |
| **nesting** | session>turn>tool/llm | deep (delegated sub-agents) | message>block | response>item |
| **ordering** | temporal | temporal; **parallel subagents observed** — interleaved, keyed by `parent_uuid`/`namespace` | temporal | temporal |
| **terminal** | `session end` scope + metadata | `request end` scope — sequential run `OK`; parallel run `failed` at combine (after both `task` scopes close) | `message_delta`(stop,usage)+`message_stop` | `response.completed`/`failed`(+usage) |
| duplicate risk | delta↔terminal | subagent-echo + delta↔terminal + tree | latent (text terminal-only today) | latent (text terminal-only today) |

### Differences That Matter
- **Stream units differ per harness** — callback-scope vs. content-block vs.
  response/item. Codex ≠ Claude (different SDK event models); neither is inferable
  from the other.
- **Granularity splits by Relay mode:** gateway (Claude/Codex) = per-delta events
  (one ATOF `llm.chunk` per SSE event — structure/timing/usage live, token **text
  terminal-only**); in-process (Hermes/Deep Agents) = scope-level (no token deltas).
- **Nesting/ordering:** Deep Agents needs `parent_uuid` tree reconstruction; parallel
  sub-agents interleave, keyed by `parent_uuid`/`namespace` (observed in
  `deepagents/parallel-*.jsonl`: two `task` calls in one message, 9.57s overlapping
  sibling scopes, interleaved namespaces). Stream order alone mis-nests. The others
  are largely linear.
- **Terminal semantics differ** — scope-end + metadata vs. `message_stop` vs.
  `response.completed/failed`.

## Final Recommendation — v0.1
**Ship raw, Relay-generated ATOF pass-through**, surfaced as sugar over
`Runtime.invoke()`. This is the **proposed** surface — not implemented in the SDK;
the POC models it in [`../common/fabric_stream.py`](../common/README.md)
(`start_streaming_runtime()` / `StreamingRuntime`):

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

## Why Normalization Is Deferred (Not Needed for v0.1)
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

## Production Work Breakdown
See [work-breakdown.md](work-breakdown.md).
