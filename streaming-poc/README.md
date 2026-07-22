# NeMo Fabric streaming POC

Proof-of-concept for a Fabric streaming API built on **NeMo Relay-generated ATOF**.
Each harness was run for real, its raw native events + ATOF captured, and the same
`invoke_stream` prototype exercised across all four. The conclusion and the
production plan are in [`synthesis/`](synthesis/README.md).

## The v0.1 contract (recommended)
```python
runtime = await fabric.start_runtime(config)   # Relay enabled → loopback ATOF endpoint injected
stream  = runtime.invoke_stream(input="...")
async for atof_record in stream:   # RAW canonical ATOF record (dict), one per Relay event
    ...
result = await stream.result()     # RunResult, out of band
```
Relay-only; available only when Relay is enabled; raw ATOF pass-through (no
normalization in v0.1); `RunResult` out of band. Rationale + why normalization is
deferred: [`synthesis/README.md`](synthesis/README.md).

## Layout
```
streaming-poc/
├── common/          prototype: listener, invoke_stream, run_harness, fixture deriver
├── hermes/          in-process · callback scopes · scope-level (no token deltas)
├── claude/          gateway · Anthropic Messages SSE · token-level deltas
├── codex/           gateway · OpenAI Responses SSE · token-level deltas
├── deepagents/      in-process · nested/delegated sub-agents · scope tree
└── synthesis/       cross-harness conclusion + production work breakdown
```
Each harness folder has `events.atof.jsonl` (Relay ATOF) and `findings.md`. For
**native evidence** — the SDK stream teed *before* Relay (via
`common/native_recorder.py`) — **Hermes** and **Deep Agents** carry a real
`native-events.jsonl`. **Claude** and **Codex** are credential/billing-blocked
this session, so they carry `relay-payloads.jsonl` (payloads extracted from ATOF,
honestly *not* native evidence) with the native seam identified for later.

## Evidence status
| Harness | live run here | fixture | token deltas |
|---|---|---|---|
| Hermes | ✅ real (`execute_code`) | real | scope-level |
| Deep Agents | ✅ real (delegation/nesting) | real | scope-level |
| Codex | ✅ real (turn errored on OpenAI quota; stream captured) | real | ✅ token-level |
| Claude | ⛔ live-blocked (no `ANTHROPIC_API_KEY`) | genuine prior Relay capture | ✅ token-level |

## Reproduce
Prereqs:
- **Built native extension matching the Python SDK.** A stale `nemo_fabric/_native.abi3.so`
  breaks `plan()/run()`; build with the venv interpreter:
  `PYO3_PYTHON=$PWD/.venv/bin/python cargo build -p fabric-python --release && cp
  target/release/lib_native.dylib python/src/nemo_fabric/_native.abi3.so` (or
  `just build-python`).
- Provider creds: `NVIDIA_API_KEY` (in-process), `OPENAI_API_KEY` (Codex),
  `ANTHROPIC_API_KEY` (Claude).
- Gateway CLI for Claude/Codex: `nemo-relay` **≥0.6.0**
  (`cargo install nemo-relay-cli --version 0.6.0`).

Run one harness:
```bash
# in-process
python streaming-poc/common/run_harness.py nvidia.fabric.hermes out.atof.jsonl "your prompt"
# gateway (needs the >=0.6.0 CLI)
FABRIC_RELAY_CLI="$(command -v nemo-relay)" \
  python streaming-poc/common/run_harness.py nvidia.fabric.codex out.atof.jsonl "your prompt"
```

## Fixture notes
- Oversized full-request snapshot records (>20 KB/line) are elided from the
  committed fixtures for size; the streaming deltas are preserved.
- The Claude fixture is a representative subset of a genuine prior Relay capture
  (`examples/harbor/swebench/.../claude-relay`).
