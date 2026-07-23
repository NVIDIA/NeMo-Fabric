<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NeMo Fabric streaming POC

Proof-of-concept for a Fabric streaming API built on **NeMo Relay-generated ATOF**.
**All four harnesses were run for real** — for each, both the raw native SDK stream
(teed *before* Relay) and Relay's ATOF were captured from the same run, and the same
`invoke_stream` prototype exercised. Both gateway harnesses (Claude, Codex) were
captured on a **subscription / SSO** session forwarded by the gateway — **no API
key**. Conclusion + production plan: [`synthesis/`](synthesis/README.md).

## The v0.1 contract (recommended)
```python
runtime = await fabric.start_runtime(config)   # Relay enabled → loopback ATOF endpoint injected
stream  = runtime.invoke_stream(input="...")
async for atof_record in stream:   # RAW canonical ATOF record (dict), one per Relay event
    ...
result = await stream.result()     # RunResult, out of band
```
Relay-only; available only when Relay is enabled; raw ATOF pass-through (no
normalization in v0.1); `RunResult` out of band. Why normalization is deferred:
[`synthesis/README.md`](synthesis/README.md).

## Layout
```
streaming-poc/
├── implementation-spec.md   architecture + end-to-end flow (mermaid diagram)
├── two-turn-isolation.jsonl checked-in artifact: two turns, one runtime, no leakage
├── common/          the experimental prototype: loopback listener, invoke_stream,
│                    run_harness, native_recorder, two_turn_isolation
├── hermes/          in-process · native-events.jsonl · events.atof.jsonl · findings.md
├── deepagents/      in-process · native-events.jsonl · events.atof.jsonl · findings.md
│                    · parallel-{native-events,events.atof}.jsonl (concurrent subagents)
├── claude/          gateway · native-events.jsonl · events.atof.jsonl · findings.md
├── codex/           gateway · native-events.jsonl · events.atof.jsonl · findings.md
└── synthesis/       cross-harness conclusion + production work breakdown
```
**Every** harness folder carries `native-events.jsonl` (the raw SDK stream teed
*before* Relay via `common/native_recorder.py`), `events.atof.jsonl` (the Relay ATOF
from the same run), and `findings.md` (native→ATOF diff, measured loss analysis,
deltas-vs-terminal, duplicate-rendering risks, recommendation). For the **gateway**
harnesses the native stream carries the per-delta token text that Relay's ATOF
projection drops — the diff is measured, not assumed.

| Harness | mode | status |
|---|---|---|
| Hermes | in-process | ✅ real run — native + ATOF captured |
| Deep Agents | in-process | ✅ real run — delegated subagents; **parallel subagents observed** (two `task` calls one message, 9.57s overlapping sibling scopes, interleaved namespaces) in a second capture ([findings](deepagents/findings.md#nesting--parallelism--delegation)) |
| Claude | gateway | ✅ real run — live delta-event structure/timing + terminal text (subscription/SSO, no API key) |
| Codex | gateway | ✅ real run — live delta-event structure/timing + terminal text (subscription/SSO, no API key; needs Codex CLI ≥0.145.0) |

## Reproduce (Hermes / Deep Agents)
Prereqs: a native extension matching the Python SDK (`just build-python`, or
`PYO3_PYTHON=$PWD/.venv/bin/python cargo build -p fabric-python --release` then copy
`target/release/lib_native.dylib` → `python/src/nemo_fabric/_native.abi3.so`), and
`NVIDIA_API_KEY`.

```bash
python streaming-poc/common/run_harness.py nvidia.fabric.hermes out.atof.jsonl "your prompt"
```
To also tee native events, set `POC_NATIVE_RECORD=<path>` and
`POC_RECORDER_DIR=streaming-poc/common` and apply the seam patch documented in
`common/native_recorder.py` (POC-only; revert after capture). The gateway harnesses
additionally need `nemo-relay` ≥0.6.0 and either a subscription (SSO) or an API key
— exact per-harness commands in
[`claude/findings.md`](claude/findings.md#reproduce-this-experiment) and
[`codex/findings.md`](codex/findings.md#reproduce-this-experiment) (Codex also needs
CLI ≥0.145.0 for the `gpt-5.6-sol` account model).

## Fixture note
Oversized full-request/response snapshot records have their `data` truncated in the
committed ATOF fixtures; the per-delta records, IDs, usage, and terminal text are
preserved. The raw per-delta token **text** lives in each folder's
`native-events.jsonl` (the ATOF projection omits it — that is the measured finding).
Any PII in a native capture (e.g. an email in model thinking) is redacted.
