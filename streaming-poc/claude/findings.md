# Claude streaming POC — stub (pending a usable API key)

**Harness:** `nvidia.fabric.claude` · gateway (Relay CLI) mode · part of the
Codex + Claude child POC.

**Status: pending a usable `ANTHROPIC_API_KEY`.** None is available in this
environment, so a real Claude run cannot be captured. Claude runs against the
Anthropic Messages API through the Relay gateway; the relay-only `invoke_stream`
prototype ([../common/](../common/README.md)) is harness-agnostic and validated on
the same gateway path, so only a real run is pending.

**To complete when a key is available:** run the harness through
`common/run_harness.py`, and capture native evidence by teeing
`ClaudeSDKClient.receive_response()` with `include_partial_messages=True` — record
every `Message` / `StreamEvent.event` before normalization — at
[adapters/claude/.../adapter.py:889](../../adapters/claude/src/nemo_fabric_adapters/claude/adapter.py#L889),
alongside the Relay listener. Then add `native-events.jsonl`, `events.atof.jsonl`,
and the native→ATOF mapping here.

Cross-harness recommendation: [../synthesis/README.md](../synthesis/README.md).
