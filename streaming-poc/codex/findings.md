# Codex streaming POC — findings (+ combined Codex+Claude recommendation)

**Harness:** `nvidia.fabric.codex` · **Relay mode:** gateway CLI
(`nemo-relay` ≥0.6.0) · **API:** OpenAI Responses (streaming SSE)

## Scenario (real run, structured streaming with Relay)
Real Codex run via `invoke_stream` through the gateway (built `nemo-relay-cli`
0.6.0). It streamed live ATOF; the *turn* then errored `Quota exceeded` (the
`OPENAI_API_KEY` account is out of quota) — the **streaming path is fully proven**;
only the completion failed on billing. Fixtures:
[`native-events.jsonl`](native-events.jsonl), [`events.atof.jsonl`](events.atof.jsonl)
(2 oversized full-request records elided; the final `response.failed` shows the
quota error propagating through the stream).

## Native event units (OpenAI Responses SSE)
From [`native-events.jsonl`](native-events.jsonl):
```
mark  session.start
scope codex-turn start
llm.chunk  response.created        (provider: openai_responses)
llm.chunk  response.in_progress
llm.chunk  response.failed         ← terminal (quota); normally response.completed
```
On a funded key the full sequence is `response.created → response.in_progress →
response.output_item.added → response.output_text.delta* → response.output_item.done
→ response.completed`. **Unit = a `response` with nested output *items*** (text,
`function_call`); text deltas arrive as `response.output_text.delta`. Wrapped by
Relay in a `codex-turn` scope and `openai.responses` LLM scope.

## Prototype crossing the Fabric boundary
`common/run_harness.py` (`provider="openai"` — **Relay requires the built-in openai
provider** for Codex) → gateway CLI pushes each Responses SSE event as an ATOF
`llm.chunk` to the SDK listener → `invoke_stream` yields raw records. Real data,
real OpenAI call through the gateway.

## Native → ATOF / candidate Fabric envelope mapping
| OpenAI Responses event | ATOF record | candidate Fabric event (v0.1 = raw ATOF) |
|---|---|---|
| `response.created`/`in_progress` | `mark llm.chunk` (data.event_type) under `scope openai.responses` | raw ATOF record |
| `response.output_text.delta` | `mark llm.chunk` (data.delta) | raw ATOF record |
| `response.output_item.*` | `mark llm.chunk` | raw ATOF record |
| `response.completed`/`failed` | `mark llm.chunk` (status, usage) | raw ATOF record |

## Streamed deltas vs. terminal & duplicate-rendering risk (HIGH)
`response.output_text.delta`s accumulate into the final response;
`response.completed` (or `failed`) carries the terminal status/usage. Same risk as
Claude: rendering deltas **and** the terminal response/`RunResult.output` double
-renders. Mitigation: render deltas live; treat terminal as replace/authoritative.

---

## Combined comparison: Codex vs. Claude (the FABRIC-103 ask)
Codex behavior is **not** inferable from Claude — different SDK event models:

| aspect | Claude — Anthropic Messages | Codex — OpenAI Responses |
|---|---|---|
| streaming API | Messages SSE | Responses SSE |
| stream unit | message → **content blocks** → deltas | response → **output items** → deltas |
| text delta event | `content_block_delta` (`delta.text`) | `response.output_text.delta` |
| tool/function | `content_block(tool_use)` + `input_json` deltas | `function_call` output items |
| keepalive | `ping` | (none observed) |
| terminal | `message_delta`(stop_reason,usage) + `message_stop` | `response.completed` / `response.failed` (+usage) |
| ordering nesting | message > block > delta | response > item > delta |

**What is the same:** Relay wraps *both* into the identical ATOF envelope — each
native SSE event becomes an `llm.chunk` mark whose `data` is the verbatim provider
event, under a provider scope (`anthropic.messages` / `openai.responses`). So at
the **ATOF layer the two are uniform**; the native differences survive in `data`.

<a name="combined-recommendation-cod--claude"></a>
## Combined recommendation (Codex + Claude)
**Raw ATOF pass-through for v0.1.** A *normalized* common mapping would have to
reconcile Claude's `content_block` model with Codex's `response/item` model
(different unit boundaries, different tool-argument encodings, different terminal
events). That reconciliation is possible but **lossy and premature** — and the
ATOF envelope already delivers a uniform transport across both. Ship raw ATOF;
consumers needing provider detail read `data.event_type`. A normalized text/tool
event layer is deferred (see synthesis). Both harnesses carry a **high
duplicate-rendering risk** that the consumer contract must call out (deltas are
incremental; terminal is authoritative — replace, don't append).
