# Deep Agents streaming POC — findings

**Harness:** `nvidia.fabric.langchain.deepagents` · **Relay mode:** in-process SDK
(`relay_api_plugin_config` + callback handler) · **Model:**
`nvidia/nemotron-3-nano-30b-a3b`

## Scenario (real run, nested + delegated subagents)
Prompt: *"Delegate two independent subtasks to two separate subagents … subagent A
computes 15*23; subagent B writes a one-line haiku … Launch both, then combine."*
Captured live via `invoke_stream` with **both** recorders active. Two delegated
subagents resulted (this model ran them sequentially, not concurrently).

## Fixtures & how they were captured
- [`native-events.jsonl`](native-events.jsonl) — **genuine native evidence**, teed
  *before* Relay at the `agent.astream(..., stream_mode=["updates","values"],
  subgraphs=True)` loop (via POC-only `common/native_recorder.py`). 26 raw
  `(namespace, mode, chunk)` tuples.
- [`events.atof.jsonl`](events.atof.jsonl) — Relay's ATOF from the *same* run
  (51 records).

## Native event units (real LangGraph astream tuples)
`(namespace, mode, chunk)` where `mode ∈ {values, updates}` and `namespace` is the
subgraph path. Two delegated subagents have distinct namespaces:
`tools:e55b5b5f-…` and `tools:e898cde5-…` — each with its own nested
`before_agent → model → after_model` sub-stream. An `updates/model` chunk carries
full LangChain messages: `{content, tool_calls, usage_metadata, response_metadata,
id, additional_kwargs, invalid_tool_calls, type, name}`.

## Nesting / parallelism / delegation
- **Nesting/delegation:** the empty-namespace `tools` node spawns a subagent whose
  entire sub-stream is tagged `namespace=tools:<tool_call_id>`.
- **Parallelism:** concurrent subagents would **interleave** in stream order,
  distinguishable *only* by `namespace` — so the native namespace (not position) is
  the parallel-safe key.

## Native → ATOF paired examples (same run)
| native event | ATOF record | preserved | dropped / changed |
|---|---|---|---|
| #7–#11 `ns=tools:e55b5b5f…` (subagent A sub-stream) | `scope general-purpose [agent]` (1st) under `scope task` | subagent ran, nested via `parent_uuid`; the **`tools:<tool_call_id>` namespace appears in ATOF too** | — (structure + namespace both preserved) |
| `mode=values` snapshots (#1,#5,#10,…) | — (**no ATOF record**) | — | LangGraph **`values` full-state snapshots** and the `values`/`updates` **mode** have no ATOF equivalent |
| #4 `updates/model` (messages w/ `tool_calls`, `usage_metadata`) | `scope model → nvidia/nemotron [llm]` | model call + **`usage_metadata`/`tool_calls` preserved in ATOF** | the raw LangChain message *envelope* (`additional_kwargs`, `invalid_tool_calls`, `response_metadata`) is reshaped |

Pairing key: native `sequence` + `namespace`; two subagents ↔ the two ATOF
`general-purpose` scopes in order; `parent_uuid` reconstructs the ATOF tree.

## What cannot be normalized without loss (comparison-based)
Verified by diffing the two fixtures from the same run:
- **Absent from ATOF:** the `values`/`updates` **mode** distinction and the
  `values` full-state snapshots — you cannot recreate the native stream_mode view
  from ATOF.
- **Faithfully preserved:** the delegation **tree** (`uuid`/`parent_uuid`), the
  subagent **`tools:<id>` namespaces**, `usage_metadata`, and `tool_calls` all
  survive into ATOF — a more faithful projection than assumed.
- ATOF's loss here is thin (mode/snapshots), which *supports* raw pass-through.

## Streamed events vs. terminal · duplicate-rendering risks
1. **Sub-agent → parent echo:** a subagent result re-appears in the parent's next
   `model` input and again in the terminal — dedup by scope `uuid`.
2. **Tree vs. stream order:** under parallel subagents, naive concatenation
   mis-nests — group by `parent_uuid` (ATOF) / `namespace` (native).
3. **Delta vs terminal:** final `model` scope == `RunResult.output` — render once.

## Recommendation
**Raw ATOF pass-through (v0.1).** ATOF's `uuid`/`parent_uuid` tree already carries
the nesting/delegation a UI needs, and the native recorder confirms it also keeps
the subagent namespaces, `usage_metadata`, and `tool_calls` — omitting only the
LangGraph `values`/`updates` mode and full-state snapshots. That thin, comparison
-verified loss justifies shipping the ATOF projection for v0.1 rather than a lossy
normalized schema.
