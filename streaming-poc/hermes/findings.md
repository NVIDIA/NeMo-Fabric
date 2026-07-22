# Hermes streaming POC — findings

**Harness:** `nvidia.fabric.hermes` · **Relay mode:** in-process SDK plugin
(`observability/nemo_relay`) · **Model:** `nvidia/nemotron-3-nano-30b-a3b`

## Scenario (real run, callback streaming with Relay)
Prompt: *"Write and run a short Python snippet that prints the sum of 1 to 10, then
tell me the result."* Captured live via `invoke_stream` while the real Hermes agent
ran. Fixtures: [`native-events.jsonl`](native-events.jsonl),
[`events.atof.jsonl`](events.atof.jsonl) (2 oversized full-request records elided).

## Native event units
Hermes' AIAgent callbacks are surfaced by the in-process Relay plugin as ATOF
**scopes** (start/end spans, `category` ∈ agent/llm/tool) and **marks** (point
events). Observed sequence:

```
scope hermes-session-runtime-… start   [agent]   ← session lifecycle
mark  hermes.turn.start                          ← turn boundary
scope nvidia                          end [llm]   ← model call (whole call, no token deltas)
scope execute_code                  start [tool]  ← tool execution
scope execute_code                    end [tool]
scope nvidia                          end [llm]
mark  hermes.turn.end
mark  hermes.session.end
scope hermes-session-runtime-…        end [agent]
```
A richer prior run also emits `clarify`, `delegate_task`, `cronjob`. **Unit = a
callback (turn / tool / llm scope), NOT a token.**

## Prototype crossing the Fabric runtime boundary
`common/run_harness.py` → `start_streaming_runtime` injects a loopback ndjson ATOF
endpoint into the config; `Fabric.start_runtime` spawns the Hermes adapter
subprocess (in-process Relay), which pushes ATOF **live over the socket** to the
SDK-process listener; `invoke_stream` async-yields each raw record. This is real
data crossing the adapter-subprocess → SDK boundary out-of-band, not config
inference.

## Native → ATOF / candidate Fabric envelope mapping
| Hermes callback | ATOF record | candidate Fabric event (v0.1 = raw ATOF) |
|---|---|---|
| session start/end | `scope hermes-session-… start/end` (agent) | `{kind:scope, name, scope_category, category:agent, uuid, metadata}` |
| turn start/end | `mark hermes.turn.start/end` | `{kind:mark, name}` |
| model call | `scope nvidia start/end` (llm) | `{kind:scope, category:llm, data: request/response}` |
| tool run | `scope execute_code start/end` (tool) | `{kind:scope, category:tool, name}` |
The **raw ATOF record is the Fabric stream event** for v0.1 — no reshaping.

## What cannot be normalized without loss
- **No token-level deltas.** Hermes streams at *scope* granularity — the whole
  model call is one `llm` scope, not `content_block_delta`s. A normalized
  "text-delta" event type would be **empty for Hermes**; forcing one hides that
  Hermes text arrives whole-message.
- **Hermes-native names/metadata:** `hermes.turn.start`, `execute_code`,
  `delegate_task`, and session metadata (`trajectory_id`,
  `telemetry_schema_version: hermes.observer.v1`, `session_id`) have no
  cross-harness equivalent; normalizing to a common vocabulary drops them.

## Streamed events vs. terminal response
The streamed scopes/marks are **structural** (session/turn/tool/llm boundaries).
The terminal Hermes response (the `RunResult.output` message) is the final text.
They accumulate to the final `hermes.session-… end` scope, whose metadata carries
`session_id`/`trajectory_id` tying the stream to the run.

## Duplicate-rendering risk
The final `llm` scope's `data` and the terminal `RunResult.output` **both contain
the answer text**. A consumer that renders streamed LLM-scope content *and* the
terminal output would render the answer twice. Mitigation: treat streamed events
as progress, `RunResult` (via `await stream.result()`) as authoritative; don't
re-render terminal text already shown from the stream.

## Recommendation for the synthesis
**Raw ATOF pass-through (v0.1).** Hermes ATOF is already a thin, well-structured
envelope over its callbacks; normalizing gains little (no token deltas to unify)
and loses Hermes-native structure. Surface the raw scope/mark stream and document
that Hermes granularity is **scope-level, not token-level**.
