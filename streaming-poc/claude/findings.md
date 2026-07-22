# Claude streaming POC — findings (part of the Codex+Claude child POC)

**Harness:** `nvidia.fabric.claude` · **Relay mode:** gateway CLI
(`nemo-relay` ≥0.6.0) · **API:** Anthropic Messages (streaming SSE)

> **Provenance / honesty:** a live Claude run in this session is **blocked** — the
> adapter requires `ANTHROPIC_API_KEY` and none is available here. The fixtures are
> a **genuine real Relay capture** of a Claude Code run (from the repo's
> `examples/harbor/swebench/.../claude-relay` artifacts), representative subset
> (140 records, oversized full-request snapshots elided). The `invoke_stream`
> prototype itself is proven on the **same gateway mode** via Codex (see
> [`../codex/findings.md`](../codex/findings.md)); only Claude-specific execution
> is credential-blocked. See [`../synthesis/`](../synthesis/README.md).

## Native event units (Anthropic Messages SSE)
From [`native-events.jsonl`](native-events.jsonl), the raw provider event
histogram:

```
99  content_block_delta     ← token/text deltas (delta.text | delta.partial_json)
 8  content_block_start
 7  content_block_stop
 4  message_start
 4  ping                    ← keepalive
 3  message_delta           ← stop_reason + usage
 3  message_stop
 6  Bash / 2 Read           ← Claude Code tool executions (ATOF scopes)
```
Ordering per message: `message_start → (content_block_start → content_block_delta*
→ content_block_stop)* → message_delta → message_stop`. **Unit = content block +
per-token `content_block_delta`.** A message can hold multiple blocks (text +
`tool_use`); tool arguments stream as `partial_json` deltas. Real token-level
streaming.

## Prototype crossing the Fabric boundary
Same as Codex (gateway mode): Relay renders the injected loopback endpoint as a
`{type:stream, transport:ndjson}` sink; the `nemo-relay` gateway process pushes
each SSE event as an ATOF `llm.chunk` mark to the SDK listener; `invoke_stream`
yields them raw. Mechanism proven live on Codex; Claude execution is
credential-blocked.

## Native → ATOF / candidate Fabric envelope mapping
| Anthropic SSE event | ATOF record | candidate Fabric event (v0.1 = raw ATOF) |
|---|---|---|
| `message_start` | `mark llm.chunk` (data.event_type=message_start) under `scope anthropic.messages` | raw ATOF record |
| `content_block_delta` | `mark llm.chunk` (data.delta.text) | raw ATOF record |
| `content_block_start/stop` | `mark llm.chunk` | raw ATOF record |
| `message_delta`/`message_stop` | `mark llm.chunk` (stop_reason, usage) | raw ATOF record |
| tool run (Bash/Read) | `scope`/`mark` (tool) | raw ATOF record |
The native SSE event is embedded verbatim in the ATOF `data`; nothing is lost at
the transport layer.

## Streamed deltas vs. terminal response
`content_block_delta`s accumulate into the final assistant message; `message_delta`
carries the terminal `stop_reason` + `usage`; `message_stop` closes it. The
terminal Claude response (`RunResult.output`) equals the assembled blocks.

## Duplicate-rendering risk (HIGH)
`content_block_delta` already yields the full incremental text. If a consumer also
renders the terminal message (`message_stop` payload or `RunResult.output`), the
answer renders **twice**. Mitigation: render deltas live; on `message_stop`/
terminal, **replace** (don't append) — or treat `RunResult` purely as the
authoritative record and never re-render text already streamed.

## Recommendation
**Raw ATOF pass-through (v0.1).** The Anthropic event model is rich and stable; the
ATOF envelope carries every SSE event verbatim. See the combined Codex+Claude
recommendation in [`../codex/findings.md`](../codex/findings.md#combined-recommendation-cod--claude).
