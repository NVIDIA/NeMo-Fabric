<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Deep Agents streaming POC — findings

**Harness:** `nvidia.fabric.langchain.deepagents` · **Relay mode:** in-process SDK
(`relay_api_plugin_config` + callback handler) · **Model:**
`nvidia/nemotron-3-nano-30b-a3b`

## Scenario (real run, nested + delegated subagents)
Prompt: *"Delegate two independent subtasks to two separate subagents … subagent A
computes 15*23; subagent B writes a one-line haiku … Launch both, then combine."*
Captured live via `invoke_stream` with **both** recorders active. Two delegated
subagents resulted; this run's model ran them sequentially (a **separate** run with
`llama-3.1-70b` captures them running **concurrently** — see Parallelism and the
`parallel-*.jsonl` fixtures).

## Fixtures & how they were captured
Two runs, for two purposes:
- **Sequential (loss analysis):** [`native-events.jsonl`](native-events.jsonl) —
  **genuine native evidence**, teed *before* Relay at the
  `agent.astream(..., stream_mode=["updates","values"], subgraphs=True)` loop (via
  POC-only `common/native_recorder.py`), 26 raw `(namespace, mode, chunk)` tuples —
  and [`events.atof.jsonl`](events.atof.jsonl), Relay's ATOF from the *same* run
  (51 records). Model `nvidia/nemotron-3-nano-30b-a3b`; the two subagents ran
  sequentially here (fine for the native→ATOF diff below).
- **Parallel (FABRIC-104 concurrency evidence):**
  [`parallel-native-events.jsonl`](parallel-native-events.jsonl) +
  [`parallel-events.atof.jsonl`](parallel-events.atof.jsonl), model
  `meta/llama-3.1-70b-instruct` — see the Parallelism section for the overlap proof.

## Native event units (real LangGraph astream tuples)
`(namespace, mode, chunk)` where `mode ∈ {values, updates}` and `namespace` is the
subgraph path. Two delegated subagents have distinct namespaces:
`tools:e55b5b5f-…` and `tools:e898cde5-…` — each with its own nested
`before_agent → model → after_model` sub-stream. An `updates/model` chunk carries
full LangChain messages: `{content, tool_calls, usage_metadata, response_metadata,
id, additional_kwargs, invalid_tool_calls, type, name}`.

## Nesting / parallelism / delegation
- **Nesting + delegation (observed):** the empty-namespace `tools` node spawns a
  subagent whose entire sub-stream is tagged `namespace=tools:<tool_call_id>`; two
  distinct subagents (`tools:e55b5b5f-…`, `tools:e898cde5-…`), each with its own
  nested `before_agent → model → after_model` sub-stream.
- **Parallelism (OBSERVED)** — captured in the parallel fixtures
  (`parallel-*.jsonl`, `meta/llama-3.1-70b-instruct`, which emits two `task` calls
  in one assistant message). Three independent signals confirm concurrent sibling
  subagents:
  1. **Two `task` calls in one message:** two `task` scopes share the same parent
     and start **5 µs apart** (`…9cc07e48fd48` and `…9cd662a71b03`, both parent
     `…49440f42ee0b`).
  2. **Overlapping scopes:** the two `task` scopes are active simultaneously for
     **9.57 s** (start 16.339 → ends 25.909 / 18.783).
  3. **Interleaved namespaces (native tee):** the two subagent namespaces alternate
     in stream order — `tools:43d0a4d3 → tools:5c7a0931 → tools:43d0a4d3 →
     tools:5c7a0931 → …` (7 transitions across 2 subagents) — the definitive
     signature of concurrent execution, not sequential blocks.
  So concurrent subagents **do** interleave, distinguished by `namespace`
  (native) / `parent_uuid` (ATOF) — now evidence, not inference. Reconstruct the
  tree by `parent_uuid`; stream order alone mis-nests parallel work.
  - *Caveat (honest):* this run's **terminal status is `failed`** — the two
    subagents ran to completion in parallel, but the final *combine* model call
    returned `400: "This model only supports single tool-calls at once"` (a
    provider limit for this model, hit after and independent of the parallel
    delegation). The parallelism evidence above is unaffected; the sibling `task`
    scopes both close normally. The prior sequential run (`nemotron-nano`) had no
    such issue, which is why it remains the loss-analysis fixture.

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
2. **Tree vs. stream order:** with parallel subagents (**observed** — see
   Parallelism; the parallel fixtures show interleaved `tools:43d0a4d3` /
   `tools:5c7a0931` events), naive concatenation mis-nests — group by `parent_uuid`
   (ATOF) / `namespace` (native). The keys and the interleaving are both in the
   `parallel-*` fixtures.
3. **Delta vs terminal:** final `model` scope == `RunResult.output` — render once.

## Recommendation
**Raw ATOF pass-through (v0.1).** ATOF's `uuid`/`parent_uuid` tree already carries
the nesting/delegation a UI needs, and the native recorder confirms it also keeps
the subagent namespaces, `usage_metadata`, and `tool_calls` — omitting only the
LangGraph `values`/`updates` mode and full-state snapshots. That thin, comparison
-verified loss justifies shipping the ATOF projection for v0.1 rather than a lossy
normalized schema.

## Reproduce this experiment
Prereqs: a native extension matching the Python SDK (`just build-python`, or
`PYO3_PYTHON=$PWD/.venv/bin/python cargo build -p fabric-python --release` then copy
`target/release/lib_native.dylib` → `python/src/nemo_fabric/_native.abi3.so`), and
`NVIDIA_API_KEY` in the environment.

1. **Apply the POC native tee** (temporary — revert after capture) in
   `adapters/deepagents/src/nemo_fabric_adapters/deepagents/adapter.py`, inside the
   `async for namespace, mode, chunk in stream:` loop (before the projection):
   `record("langgraph", str(mode), {"namespace": list(namespace), "chunk": chunk})`.
   Exact snippet: [`../common/native_recorder.py`](../common/native_recorder.py).
2. **Run with both recorders active** — this writes both fixtures in this folder:
   ```bash
   export POC_RECORDER_DIR="$PWD/streaming-poc/common"
   export POC_NATIVE_RECORD="$PWD/streaming-poc/deepagents/native-events.jsonl"; rm -f "$POC_NATIVE_RECORD"
   python streaming-poc/common/run_harness.py nvidia.fabric.langchain.deepagents \
     streaming-poc/deepagents/events.atof.jsonl \
     "Delegate two independent subtasks to two separate subagents in parallel: subagent A computes 15*23; subagent B writes a one-line haiku about mountains. Launch both, then combine their results."
   ```
   `run_harness` streams the Relay ATOF via `invoke_stream` → `events.atof.jsonl`;
   the tee writes the native `(namespace, mode, chunk)` tuples → `native-events.jsonl`.
3. **Revert** the adapter patch:
   `git checkout -- adapters/deepagents/src/nemo_fabric_adapters/deepagents/adapter.py`.

### Reproduce the parallel run (`parallel-*.jsonl`)
Parallelism is model-dependent: the model must emit **two `task` calls in one
message**. `meta/llama-3.1-70b-instruct` does (a probe of the NVIDIA endpoint found
it emits 2 parallel tool calls where `llama-3.3-70b` serializes and `gpt-oss-120b`
emits 1). With the native tee applied (step 1 above), run:
```bash
FABRIC_MODEL="meta/llama-3.1-70b-instruct" \
POC_RECORDER_DIR="$PWD/streaming-poc/common" \
POC_NATIVE_RECORD="$PWD/streaming-poc/deepagents/parallel-native-events.jsonl" \
python streaming-poc/common/run_harness.py nvidia.fabric.langchain.deepagents \
  streaming-poc/deepagents/parallel-events.atof.jsonl \
  "Make TWO 'task' tool calls in a SINGLE response, at once (subagent_type 'general-purpose'): Task 1 'Write a 4-line poem about the ocean'; Task 2 'Write a 4-line poem about mountains'. Emit both task calls together now, then combine."
```
The two subagents launch together and interleave (see Parallelism). Note: the turn
terminates `failed` because the model's *combine* step then trips the endpoint's
`"model only supports single tool-calls at once"` limit — after, and independent of,
the parallel delegation. Then revert the adapter patch (step 3).
