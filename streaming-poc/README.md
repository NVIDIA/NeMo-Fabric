# NeMo Fabric streaming POC

Proof-of-concept for a Fabric streaming API built on **NeMo Relay-generated ATOF**.
Hermes and Deep Agents were run for real — their raw native events (teed before
Relay) and ATOF captured, and the same `invoke_stream` prototype exercised.
Codex and Claude are stubs pending a usable API key. Conclusion + production plan:
[`synthesis/`](synthesis/README.md).

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
├── common/          the experimental prototype: loopback listener, invoke_stream,
│                    run_harness, native_recorder
├── hermes/          FABRIC-102 — native-events.jsonl · events.atof.jsonl · findings.md
├── deepagents/      FABRIC-104 — native-events.jsonl · events.atof.jsonl · findings.md
├── claude/          FABRIC-103 — findings.md (stub: pending ANTHROPIC_API_KEY)
├── codex/           FABRIC-103 — findings.md (stub: pending a funded OpenAI key)
└── synthesis/       FABRIC-101 cross-harness conclusion + FABRIC-122 work breakdown
```
Each completed harness folder carries `native-events.jsonl` (the SDK stream teed
*before* Relay via `common/native_recorder.py`), `events.atof.jsonl` (the Relay
ATOF that crossed the Fabric boundary), and `findings.md` (native→ATOF mapping,
loss analysis, deltas-vs-terminal, duplicate-rendering risks, recommendation).

| Harness | ticket | status |
|---|---|---|
| Hermes | FABRIC-102 | ✅ complete — real run, native + ATOF captured |
| Deep Agents | FABRIC-104 | ✅ complete — real run w/ delegated subagents |
| Codex / Claude | FABRIC-103 | ⏸ stub — pending a usable API key (native seam identified) |

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
`common/native_recorder.py` (POC-only; revert after capture). Gateway harnesses
(Claude/Codex) additionally need `nemo-relay` ≥0.6.0 and the respective key.

## Fixture note
Oversized full-request snapshot records (>20 KB/line) have their `data` truncated
in the committed fixtures; the streaming deltas and IDs are preserved.
