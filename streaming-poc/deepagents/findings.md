# Deep Agents streaming POC — findings

**Harness:** `nvidia.fabric.langchain.deepagents` · **Relay mode:** in-process SDK
(`relay_api_plugin_config` + callback handler) · **Model:**
`nvidia/nemotron-3-nano-30b-a3b`

## Scenario (real run with nesting + delegation)
Prompt: *"Complete two independent subtasks: (1) compute 12*8; (2) write a one-line
haiku about the sea. Use subagents if helpful, then combine."* — chosen to induce
delegation. Captured live via `invoke_stream`; 33 ATOF records. Fixtures:
[`native-events.jsonl`](native-events.jsonl), [`events.atof.jsonl`](events.atof.jsonl).

## Nesting / parallelism / delegation (real, via `parent_uuid`)
The scope tree reconstructed from `parent_uuid`:
```
deepagents-request [agent]
  LangGraph [agent]
    PatchToolCallsMiddleware.before_agent [agent]
    NemoRelayDeepAgentsMiddleware.before_agent [agent]
    model [agent] → nvidia/nemotron-3-nano-30b-a3b [llm]
    TodoListMiddleware.after_model [agent]
    tools [agent]
      task [tool]                     ← DELEGATION tool call
      general-purpose [agent]         ← SUB-AGENT (delegated), own subtree:
        PatchToolCallsMiddleware.before_agent [agent]
        model [agent]
        TodoListMiddleware.after_model [agent]
    model [agent] → nvidia/nemotron-3-nano-30b-a3b [llm]
    TodoListMiddleware.after_model [agent]
```
- **Nesting:** request → graph → middleware/model/tools spans.
- **Delegation:** the `task` tool spawns a `general-purpose` sub-agent with its
  **own** nested middleware/model scopes.
- **Parallelism:** here sub-agents ran sequentially, but Deep Agents can fan out;
  concurrent sub-agents appear as **sibling subtrees interleaved in stream order**,
  distinguished only by `parent_uuid` (not by position).

## Prototype crossing the Fabric boundary
Same in-process path as Hermes (`relay_api_plugin_config`), proven live:
`invoke_stream` yields the 33 raw ATOF records as they were emitted.

## Native → ATOF / Fabric mapping (structure is the point)
| Deep Agents concept | ATOF representation | candidate Fabric event (v0.1 = raw ATOF) |
|---|---|---|
| nesting | `scope` with `parent_uuid` chain | `{kind:scope, uuid, parent_uuid, category}` |
| delegation | `scope task [tool]` + child `scope <subagent> [agent]` | raw ATOF records (linked by `parent_uuid`) |
| parallelism | interleaved sibling subtrees, same `parent_uuid` | raw ATOF; **reconstruct tree via `parent_uuid`, not stream order** |
| model call | `scope model [agent]` → `scope <model> [llm]` | raw ATOF record (no token deltas) |
Like Hermes (in-process): **no `llm.chunk`** — model calls are whole-call `llm`
scopes, not token deltas.

## Streamed events vs. terminal response
Streamed scopes give the live execution tree; the terminal response
(`RunResult.output`) is the combined final answer, emitted as the last `model`
scope closes and `deepagents-request end` (`otel.status_code: OK`) fires.

## Duplicate-rendering risks
1. **Sub-agent → parent echo:** a sub-agent's result is returned to the parent and
   re-appears in the parent's next `model` input, and again in the terminal —
   render each logical unit once, keyed by scope `uuid`.
2. **Delta-vs-terminal:** the final `model` scope's output equals the terminal
   `RunResult.output` — don't render both.
3. **Tree vs. stream order:** naively concatenating stream events double-nests or
   mis-orders parallel subtrees — a correct renderer groups by `parent_uuid`.

## Recommendation
**Raw ATOF pass-through (v0.1).** Deep Agents' nesting/parallelism/delegation is
already fully represented by ATOF's `uuid`/`parent_uuid` scope tree — a flat
"normalized delta" model would **lose the tree**. Ship raw ATOF and document that
consumers must reconstruct hierarchy from `parent_uuid` (stream order alone is
insufficient under parallelism).
