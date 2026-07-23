<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Fabric Streaming POC

Proof-of-concept for a NeMo Fabric streaming API built on **NeMo Relay-generated ATOF**.
Each harness folder holds its native SDK stream and Relay's ATOF from the same
`invoke_stream` run, plus a `findings.md`; the gateway harnesses (Claude, Codex) ran
on a subscription/SSO session — **no API key**. Conclusion + production plan:
[`synthesis/`](synthesis/README.md).

## The v0.1 Contract (Proposed API)
> **Proposed v0.1 API — not implemented in the SDK.** The POC models this surface
> with [`common/fabric_stream.py`](common/README.md) (runnable equivalent:
> `start_streaming_runtime()` / `StreamingRuntime`).

```python
runtime = await fabric.start_runtime(config)   # Relay enabled → loopback ATOF endpoint injected
stream  = runtime.invoke_stream(input="...")
async for atof_record in stream:   # RAW canonical ATOF record (dict), one per Relay event
    ...
result = await stream.result()     # RunResult, out of band
```
Relay-only; available only when Relay is enabled; raw ATOF pass-through (no
normalization in v0.1); `RunResult` out of band. Why normalization is deferred:
[the synthesis](synthesis/README.md).

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
**Every** harness folder carries `native-events.jsonl` (the SDK stream teed *before*
Relay via `common/native_recorder.py`), `events.atof.jsonl` (the Relay ATOF from the
same run), and `findings.md` (native→ATOF diff, loss analysis, deltas-vs-terminal,
duplicate-rendering risks, recommendation). For the **gateway** harnesses the native
stream carries the per-delta token text that Relay's ATOF projection drops.

Status by harness:

| Harness | Mode | Status |
|---|---|---|
| Hermes | in-process | ✅ real run — native + ATOF captured |
| Deep Agents | in-process | ✅ real run — delegated subagents; **parallel subagents observed** (two `task` calls one message, 9.57s overlapping sibling scopes, interleaved namespaces) in a second capture ([findings](deepagents/findings.md#nesting--parallelism--delegation)) |
| Claude | gateway | ✅ real run — live delta-event structure/timing + terminal text (subscription/SSO, no API key) |
| Codex | gateway | ✅ real run — live delta-event structure/timing + terminal text (subscription/SSO, no API key; needs Codex CLI ≥0.145.0) |

## Reproduce (Hermes / Deep Agents)
The gateway harnesses (Claude, Codex) have their own recipes in
[claude/findings.md](claude/findings.md#reproduce-this-experiment) and
[codex/findings.md](codex/findings.md#reproduce-this-experiment). For the in-process
harnesses, with a native extension matching the Python SDK (`just build-python`, or
`PYO3_PYTHON=$PWD/.venv/bin/python cargo build -p fabric-python --release`, then copy
`target/release/lib_native.dylib` → `python/src/nemo_fabric/_native.abi3.so`) and
`NVIDIA_API_KEY`, stream a run to a scratch file:

```bash
python streaming-poc/common/run_harness.py nvidia.fabric.hermes /tmp/out.atof.jsonl "your prompt"
```

To also capture the native stream, apply the reversible seam patch and set the
recorder environment (per-harness steps are in each `findings.md`):

```bash
git apply streaming-poc/patches/hermes-native-tee.patch
POC_RECORDER_DIR=streaming-poc/common POC_NATIVE_RECORD=/tmp/native.jsonl \
  python streaming-poc/common/run_harness.py nvidia.fabric.hermes /tmp/out.atof.jsonl "your prompt"
git apply -R streaming-poc/patches/hermes-native-tee.patch
```

## Fixture Note
Oversized request/response snapshot records have their `data` truncated in the
committed ATOF fixtures; per-delta records, IDs, usage, and terminal text are
preserved. The per-delta token **text** lives in each folder's `native-events.jsonl`
(the ATOF projection omits it). PII in a native capture (e.g. an email in model
thinking) is redacted.
