# common/ — shared prototype

Harness-agnostic prototype used by all four child POCs (one mechanism, no
per-harness streaming code).

| file | purpose |
|---|---|
| `atof_stream.py` | `AtofStreamListener` — loopback ndjson sink; async queue of raw ATOF; bounded-queue backpressure; handles >512 KB gateway records |
| `fabric_stream.py` | `Runtime.invoke_stream()` prototype — `StreamingRuntime` / `InvokeStream`; endpoint injection at `start_runtime`; raw ATOF + out-of-band `result()`; honest early-exit |
| `run_harness.py` | run one real harness through `invoke_stream` and save its raw ATOF stream |
| `native_recorder.py` | **POC-only** recorder that tees a harness's native SDK stream *before* Relay (produces `native-events.jsonl`); documents the per-harness capture seams. Not a Fabric API. |

Requires a built native extension and provider creds — see [`../README.md`](../README.md).
