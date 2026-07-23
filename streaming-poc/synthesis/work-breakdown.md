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
  newlines.
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
- Per-runtime listener; `invoke_stream` per turn. Two-turn isolation proven by a
  checked-in artifact (`../two-turn-isolation.jsonl`; runner
  `common/two_turn_isolation.py`): one persistent runtime, two turns, disjoint
  record `uuid`s (overlap 0), no sentinel leakage.

## 7. Early-exit / cancellation contract (S)
- Document + implement: to stop early, break the `async for` **then**
  `await stream.aclose()`. Breaking alone stops iteration but does **not** finalize;
  `aclose()` **waits for the turn to complete** (the blocking native call runs to
  completion — not interrupted), then drains/discards the unread records. The stream
  must be finalized (fully consumed or `aclose()`d) before the next `invoke_stream`,
  which otherwise raises. `aclose()` must be cancellation-safe: shield the invoke
  task and propagate `CancelledError` so `result()` stays valid.
- **Gap to close: there is no in-flight cancellation today.** `runtime.stop()` raises
  `FabricStateError` while a turn is active (idle-only teardown), so a running turn
  cannot be aborted from the SDK. A production cancellation path (cooperative
  interrupt through the native boundary, or an adapter-level turn interrupt) is
  required before advertising cancellation.

## 8. Consumer contract documentation (M)
- Raw ATOF record shape; granularity by mode (gateway = per-delta events, token
  **text terminal-only** in current ATOF; in-process = scope-level).
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
- Gateway harnesses take either a subscription/SSO session or an API key; Codex also
  needs Codex CLI ≥0.145.0 for the `gpt-5.6-sol` account model (see the gateway
  [findings](../codex/findings.md)).
