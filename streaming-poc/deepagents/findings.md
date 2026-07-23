<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Deep Agents Streaming POC — Findings

**Harness:** `nvidia.fabric.langchain.deepagents` · **Relay mode:** in-process SDK
(`relay_api_plugin_config` + callback handler) · **Model:**
`nvidia/nemotron-3-nano-30b-a3b`

## Scenario (Nested + Delegated Subagents)
Prompt: *"Delegate two independent subtasks to two separate subagents … subagent A
computes 15*23; subagent B writes a one-line haiku … Launch both, then combine."*
Captured live via `invoke_stream` with **both** recorders active. Two delegated
subagents resulted; this run's model ran them sequentially (a **separate** run with
`llama-3.1-70b` captures them running **concurrently** — see Parallelism and the
`parallel-*.jsonl` fixtures).

## Fixtures & How They Were Captured
Two runs, for two purposes:
- **Sequential (loss analysis):** [`native-events.jsonl`](native-events.jsonl) —
  the native stream teed *before* Relay at the
  `agent.astream(..., stream_mode=["updates","values"], subgraphs=True)` loop (via
  POC-only `common/native_recorder.py`), 26 `(namespace, mode, chunk)` tuples — and
  [`events.atof.jsonl`](events.atof.jsonl), Relay's ATOF from the *same* run (51
  records). Model `nvidia/nemotron-3-nano-30b-a3b`; the two subagents ran
  sequentially here (fine for the native→ATOF diff below).
- **Parallel (FABRIC-104 concurrency evidence):**
  [`parallel-native-events.jsonl`](parallel-native-events.jsonl) +
  [`parallel-events.atof.jsonl`](parallel-events.atof.jsonl), model
  `meta/llama-3.1-70b-instruct` — see the Parallelism section for the overlap proof.

## Native Event Units (Real LangGraph Astream Tuples)
`(namespace, mode, chunk)` where `mode ∈ {values, updates}` and `namespace` is the
subgraph path. Two delegated subagents have distinct namespaces:
`tools:e55b5b5f-…` and `tools:e898cde5-…` — each with its own nested
`before_agent → model → after_model` sub-stream. An `updates/model` chunk carries
full LangChain messages: `{content, tool_calls, usage_metadata, response_metadata,
id, additional_kwargs, invalid_tool_calls, type, name}`.

## Nesting / Parallelism / Delegation
- **Nesting + delegation (observed):** the empty-namespace `tools` node spawns a
  subagent whose entire sub-stream is tagged `namespace=tools:<tool_call_id>`; two
  distinct subagents (`tools:e55b5b5f-…`, `tools:e898cde5-…`), each with its own
  nested `before_agent → model → after_model` sub-stream.
- **Parallelism (observed)** — the parallel fixtures (`parallel-*.jsonl`,
  `meta/llama-3.1-70b-instruct`) capture two subagents running concurrently, shown
  three independent ways:
  1. **Two `task` calls in one message:** two `task` scopes share one parent and
     start **5 µs apart** (`…9cc07e48fd48`, `…9cd662a71b03`, parent `…49440f42ee0b`).
  2. **Overlapping scopes:** the two `task` scopes are active together for **9.57 s**
     (start 16.339 → ends 25.909 / 18.783).
  3. **Interleaved namespaces (native tee):** the subagent namespaces alternate —
     `tools:43d0a4d3 → tools:5c7a0931 → tools:43d0a4d3 → tools:5c7a0931 → …`
     (7 transitions), the signature of concurrent, not sequential, execution.

  Concurrent subagents interleave, distinguished by `namespace` (native) /
  `parent_uuid` (ATOF); reconstruct the tree by `parent_uuid` — stream order alone
  mis-nests. The run's **terminal status is `failed`**: the subagents completed in
  parallel (both `task` scopes close normally), but the model's later *combine* call
  returned `400: "This model only supports single tool-calls at once"` — a provider
  limit hit independently of the delegation. The sequential `nemotron-nano` run has
  no such issue, so it stays the loss-analysis fixture.

## Native → ATOF Paired Examples (Same Run)
Each native `astream` event (or group) paired with the ATOF record it produced:

| native event | ATOF record | preserved | dropped / changed |
|---|---|---|---|
| #7–#11 `ns=tools:e55b5b5f…` (subagent A sub-stream) | `scope general-purpose [agent]` (1st) under `scope task` | subagent ran, nested via `parent_uuid`; the **`tools:<tool_call_id>` namespace appears in ATOF too** | — (structure + namespace both preserved) |
| `mode=values` snapshots (#1,#5,#10,…) | — (**no ATOF record**) | — | LangGraph **`values` full-state snapshots** and the `values`/`updates` **mode** have no ATOF equivalent |
| #4 `updates/model` (messages w/ `tool_calls`, `usage_metadata`) | `scope model → nvidia/nemotron [llm]` | model call + **`usage_metadata`/`tool_calls` preserved in ATOF** | the raw LangChain message *envelope* (`additional_kwargs`, `invalid_tool_calls`, `response_metadata`) is reshaped |

Pairing key: native `sequence` + `namespace`; two subagents ↔ the two ATOF
`general-purpose` scopes in order; `parent_uuid` reconstructs the ATOF tree.

## What Cannot Be Normalized Without Loss
- **Absent from ATOF:** the `values`/`updates` **mode** distinction and the
  `values` full-state snapshots — you cannot recreate the native stream_mode view
  from ATOF.
- **Faithfully preserved:** the delegation **tree** (`uuid`/`parent_uuid`), the
  subagent **`tools:<id>` namespaces**, `usage_metadata`, and `tool_calls` all
  survive into ATOF — a more faithful projection than assumed.
- ATOF's loss here is thin (mode/snapshots), which *supports* raw pass-through.

## Streamed Events vs. Terminal · Duplicate-Rendering Risks
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

## Reproduce This Experiment
Prereqs: a native extension matching the Python SDK (`just build-python`, or
`PYO3_PYTHON=$PWD/.venv/bin/python cargo build -p fabric-python --release`, then copy
`target/release/lib_native.dylib` → `python/src/nemo_fabric/_native.abi3.so`) and
`NVIDIA_API_KEY`. The seam is a checked-in, reversible patch
([`../patches/deepagents-native-tee.patch`](../patches/deepagents-native-tee.patch));
output goes to a scratch directory so the committed fixtures are never overwritten.

**Sequential run** (`nemotron-nano`, the loss-analysis fixture):
```bash
out=$(mktemp -d)
git apply streaming-poc/patches/deepagents-native-tee.patch
POC_RECORDER_DIR="$PWD/streaming-poc/common" \
POC_NATIVE_RECORD="$out/native-events.jsonl" \
python streaming-poc/common/run_harness.py nvidia.fabric.langchain.deepagents \
  "$out/events.atof.jsonl" \
  "Delegate two independent subtasks to two separate subagents in parallel: subagent A computes 15*23; subagent B writes a one-line haiku about mountains. Launch both, then combine their results."
git apply -R streaming-poc/patches/deepagents-native-tee.patch
```

**Parallel run** (`parallel-*.jsonl`): parallelism needs a model that emits **two
`task` calls in one message**; `meta/llama-3.1-70b-instruct` does (not all do). Same
patch, a different model and prompt:
```bash
out=$(mktemp -d)
git apply streaming-poc/patches/deepagents-native-tee.patch
FABRIC_MODEL="meta/llama-3.1-70b-instruct" \
POC_RECORDER_DIR="$PWD/streaming-poc/common" \
POC_NATIVE_RECORD="$out/parallel-native-events.jsonl" \
python streaming-poc/common/run_harness.py nvidia.fabric.langchain.deepagents \
  "$out/parallel-events.atof.jsonl" \
  "Make TWO 'task' tool calls in a SINGLE response, at once (subagent_type 'general-purpose'): Task 1 'Write a 4-line poem about the ocean'; Task 2 'Write a 4-line poem about mountains'. Emit both task calls together now, then combine."
git apply -R streaming-poc/patches/deepagents-native-tee.patch
```
The two subagents launch together and interleave (see Parallelism, including the
`failed`-status note). Committed fixtures are these outputs, scrubbed and truncated
before check-in.
