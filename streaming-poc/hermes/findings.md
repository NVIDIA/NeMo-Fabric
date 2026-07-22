# Hermes streaming POC — findings

**Harness:** `nvidia.fabric.hermes` · **Relay mode:** in-process SDK plugin
(`observability/nemo_relay`) · **Model:** `nvidia/nemotron-3-nano-30b-a3b`

## Scenario (real run, callback streaming with Relay)
Prompt: *"Write and run a short Python snippet that prints the sum of 1 to 10,
then tell me the result."* — an LLM call plus tool execution. Captured live via
`invoke_stream` while the real Hermes agent ran, with **both** recorders active.

## Fixtures & how they were captured
- [`native-events.jsonl`](native-events.jsonl) — **genuine native evidence**, teed
  *before* Relay by patching `hermes_cli.plugins.PluginManager.invoke_hook` (the
  single hook-dispatch chokepoint that Relay's plugin also subscribes to) via the
  POC-only `common/native_recorder.py`. 12 native lifecycle hooks.
- [`events.atof.jsonl`](events.atof.jsonl) — Relay's ATOF from the *same* run
  (11 records; 2 oversized request snapshots have their `data` truncated).

## Native event units (real, pre-Relay)
```
1 on_session_start   5 pre_tool_call      9 transform_llm_output
2 pre_llm_call        6 post_tool_call    10 post_llm_call
3 pre_api_request     7 pre_api_request   11 on_session_end
4 post_api_request    8 post_api_request  12 on_session_finalize
```
Each hook carries structured kwargs, e.g. `pre_tool_call`:
`{tool_name:"execute_code", args:{code:"print(sum(range(1,11)))"}, task_id, session_id,
tool_call_id, turn_id, api_request_id, middleware_trace}`. **Unit = a lifecycle
hook, not a token** (Hermes has no token-level deltas).

## Prototype crossing the Fabric boundary
`common/run_harness.py` → `start_streaming_runtime` injects a loopback ndjson ATOF
endpoint; the in-process Relay plugin pushes ATOF live to the SDK listener;
`invoke_stream` yields each raw record. (Unchanged; the native recorder is
orthogonal POC instrumentation.)

## Native → ATOF paired examples (same run)
| native event | ATOF record | preserved | dropped / changed |
|---|---|---|---|
| #5 `pre_tool_call` (execute_code) | `scope execute_code start` uuid `019f8bba…`, `data:{code:"print(sum(range(1,11)))"}` | tool name, **code/args**, and **all IDs** (`tool_call_id`/`task_id`/`turn_id`/`api_request_id`/`session_id` all appear in ATOF) | **`middleware_trace`** field only |
| #2 `pre_llm_call` + #3/#4 `pre/post_api_request` + #10 `post_llm_call` | one `scope nvidia [llm] start/end` | that an LLM step happened, + its IDs | the **4 hooks collapse to 1 scope** — pre/post boundaries and the LLM-step-vs-wire-API-request distinction are lost |
| #9 `transform_llm_output` | — (**no ATOF record**) | — | the entire output-transform hook is **absent from ATOF** |
| #12 `on_session_finalize` | — (ATOF has `session end`, not finalize) | session close | the finalize step as a distinct event |

Pairing key: native `sequence` + shared `session_id` (native
`session_id="runtime-1784755641256-…"` appears verbatim in the ATOF scope name
`hermes-session-runtime-1784755641256-…`); event order matches.

## What cannot be normalized without loss (comparison-based)
Verified by diffing the two fixtures from the same run:
- **Absent from ATOF entirely:** the `transform_llm_output` hook and the
  `middleware_trace` field — you cannot recreate them from ATOF.
- **Collapsed:** ATOF folds the four native hooks around one LLM step
  (`pre_llm_call`/`pre_api_request`/`post_api_request`/`post_llm_call`) into a
  single `nvidia` scope, losing the pre/post boundaries.
- **Faithfully preserved:** every ID (`tool_call_id`/`task_id`/`turn_id`/
  `api_request_id`/`session_id`) and the tool args survive — ATOF is a faithful
  projection that drops mostly redundant bookkeeping.
- **No token-level deltas** either way — Hermes streams at hook/scope granularity.

## Streamed events vs. terminal response · duplicate-rendering risk
Streamed scopes/marks are structural; the terminal `RunResult.output` is the final
text. The final `nvidia` LLM scope's data and the terminal output **both** contain
the answer → render streamed events for progress, terminal via
`await stream.result()` as authoritative (don't re-render).

## Recommendation
**Raw ATOF pass-through (v0.1).** The native recorder shows ATOF drops only a
Hermes-specific hook (`transform_llm_output`), the `middleware_trace` field, and
the pre/post hook granularity — while faithfully keeping the IDs, tool args, and
session/turn/tool/llm structure a streaming UI needs. The loss is small and
Hermes-specific with no cross-harness peer, so normalizing isn't worth it for
v0.1; ship raw ATOF.
