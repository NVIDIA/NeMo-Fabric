<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NeMo Fabric streaming POC

Proof-of-concept for a Fabric streaming API built on **NeMo Relay-generated ATOF**.
**All four harnesses were run for real** — their ATOF (and, for the in-process
harnesses, native events teed before Relay) captured, and the same `invoke_stream`
prototype exercised. Both gateway harnesses (Claude, Codex) were captured on a
**subscription / SSO** session forwarded by the gateway — **no API key**. Conclusion
+ production plan: [`synthesis/`](synthesis/README.md).

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
├── common/          the experimental prototype: loopback listener, invoke_stream,
│                    run_harness, native_recorder
├── hermes/          in-process · native-events.jsonl · events.atof.jsonl · findings.md
├── deepagents/      in-process · native-events.jsonl · events.atof.jsonl · findings.md
├── claude/          gateway · events.atof.jsonl · findings.md (real run, SSO)
├── codex/           gateway · events.atof.jsonl · findings.md (real run, SSO)
└── synthesis/       cross-harness conclusion + production work breakdown
```
Each completed harness folder carries `events.atof.jsonl` (the Relay ATOF that
crossed the Fabric boundary) and `findings.md` (native→ATOF mapping, loss analysis,
deltas-vs-terminal, duplicate-rendering risks, recommendation). The **in-process**
harnesses also carry `native-events.jsonl` (the SDK stream teed *before* Relay via
`common/native_recorder.py`); in **gateway** mode Relay embeds the native events
inside the ATOF `llm.chunk` marks, so no separate native file is needed.

| Harness | mode | status |
|---|---|---|
| Hermes | in-process | ✅ complete — real run, native + ATOF captured |
| Deep Agents | in-process | ✅ complete — real run w/ delegated subagents |
| Claude | gateway | ✅ complete — real run, token-level ATOF (subscription/SSO, no API key) |
| Codex | gateway | ✅ complete — real run, token-level ATOF (subscription/SSO, no API key; needs Codex CLI ≥0.145.0) |

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
Oversized full-request snapshot records (>20 KB/line) have their `data` truncated
in the committed fixtures; the streaming deltas and IDs are preserved.
