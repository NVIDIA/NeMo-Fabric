# Codex streaming POC — stub (pending a usable API key)

**Harness:** `nvidia.fabric.codex` · gateway (Relay CLI) mode · part of the
Codex + Claude child POC.

**Status: pending a usable OpenAI API key.** Codex runs against the OpenAI
Responses API through the Relay gateway. A clean end-to-end capture is blocked —
the available key is out of quota, and NVIDIA's inference endpoint rejects Codex's
OpenAI-proprietary tool types. The relay-only `invoke_stream` prototype
([../common/](../common/README.md)) is harness-agnostic and already validated on
the other gateway path, so no Codex-specific prototype work is pending — only a
real run.

**To complete when a key is available:** run the harness through
`common/run_harness.py`, and capture native evidence by teeing
`AsyncTurnHandle.stream()` — record every notification's `method` + `payload`
before the terminal-result collector — at
[adapters/codex/.../adapter.py:927](../../adapters/codex/src/nemo_fabric_adapters/codex/adapter.py#L927),
alongside the Relay listener. Then add `native-events.jsonl`, `events.atof.jsonl`,
and the native→ATOF mapping here.

Cross-harness recommendation: [../synthesis/README.md](../synthesis/README.md).
