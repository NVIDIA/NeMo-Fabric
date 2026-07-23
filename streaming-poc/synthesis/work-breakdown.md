<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Production implementation — work breakdown

Turns the validated POC into a shippable `Runtime.invoke_stream()`. Each item is
scoped from real POC findings; effort is relative (S/M/L).

## 1. SDK surface (M)
- Add `Runtime.invoke_stream(*, input, request) -> InvokeStream` returning an async
  iterator of **raw ATOF records**, plus `await stream.result() -> RunResult` and
  `await stream.aclose()`. (POC: `common/fabric_stream.py`.)
- Gate on Relay: raise `FabricCapabilityError` when Relay is not enabled.
- Keep `RunResult` strictly out of band (no in-band terminal event).

## 2. Endpoint injection at `start_runtime` (M)
- In `Fabric.start_runtime`, when `relay_enabled`, auto-inject a loopback ndjson
  ATOF endpoint into `relay.observability.atof.endpoints` before planning. Verified
  to survive planning into the adapter's `relay-config.json` for all four adapters
  — **no Rust/core change required**.
- One listener per runtime (fixed at start); `invoke_stream` delimits turns by
  invoke completion.

## 3. Loopback listener (M)
- **Must handle large records:** gateway ATOF embeds full request/response;
  records exceed aiohttp's 512 KB readline limit — read raw chunks, split on
  newlines. (POC bug found + fixed.)
- **Bounded queue + backpressure** (default maxsize ~1024); document that a
  consumer stalling beyond Relay's ~3 s flush/close timeout causes **Relay-side
  drops** (best-effort under sustained stall).
- **Packaging:** prefer a **stdlib `asyncio` listener** (no new dep) over aiohttp,
  or ship aiohttp behind `nemo-fabric[streaming]`. (`aiohttp` is transitive today,
  not a first-party dep.)

## 4. Capability discovery (S)
- Surface `runtime.supports_streaming` (== `relay_enabled`) for discovery.
- Do **not** overload `RuntimeCapabilities.streaming` (native progressive output;
  stays `False`).

## 5. Gateway provisioning (S/M)
- Claude/Codex require the `nemo-relay` gateway CLI with the ndjson stream sink →
  **≥0.6.0** (the adapter already version-checks). Document provisioning
  (`cargo install nemo-relay-cli`), and that streaming's floor is the
  stream-sink-capable CLI. In-process harnesses need no gateway.

## 6. Multi-turn semantics (S)
- Per-runtime listener; `invoke_stream` per turn. Two-turn isolation proven
  (temporal separation + drain; no cross-turn leakage).

## 7. Early-exit / cancellation contract (S)
- Document + implement: `aclose()`/break **detaches** the consumer but does **not**
  interrupt the turn (blocking native call on a worker thread runs to completion);
  `runtime.stop()` aborts. Unread buffered events discarded.

## 8. Consumer contract documentation (M)
- Raw ATOF record shape; granularity by mode (gateway=token-level,
  in-process=scope-level).
- **Delta-vs-terminal:** render deltas live, treat terminal/`RunResult` as
  authoritative — *replace, don't append* (HIGH duplicate risk on Claude/Codex).
- **Tree reconstruction:** group by `uuid`/`parent_uuid` (required for Deep Agents
  parallel/nested work; stream order alone is insufficient).
- Sub-agent echo dedup keyed by scope `uuid`.

## 9. Tests & fixtures (M)
- Reuse POC prototypes: two-turn isolation, early-exit, buffering, long-line.
- Per-harness capture smoke tests behind provider-cred markers.
- Commit the `streaming-poc/*/` ATOF fixtures as regression fixtures.

## 10. Docs & skills parity (S)
- Update `docs/sdk/python.mdx` and the consumer skills (`skills/`) with the
  streaming contract (repo requires SDK/docs/skills parity).

## Deferred (explicitly out of scope for v0.1)
- Normalized/typed event layer over raw ATOF (a `FabricStreamEvent` mapping) —
  optional, opt-in, on top of the raw stream; only if a consumer needs
  provider-agnostic text/tool events.
- Non-Relay (native-SDK) streaming path.

## Known follow-ups from the POC environment
- `nemo-fabric` native must be built to match the Python SDK (stale `.so` breaks
  `plan()/run()`); ensure CI builds it. Build with the venv interpreter
  (`PYO3_PYTHON`) or `just build-python`.
- **Both** gateway harnesses were captured live on a **subscription / SSO** session
  (no API key) — the gateway forwarded the OAuth session for Claude
  ([claude/findings.md](../claude/findings.md)) and Codex
  ([codex/findings.md](../codex/findings.md)). Codex additionally requires **Codex
  CLI ≥0.145.0** for the ChatGPT account's `gpt-5.6-sol` model (older CLIs reject
  it); a funded `OPENAI_API_KEY` is an alternative to SSO for both.
