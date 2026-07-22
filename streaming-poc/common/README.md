# common/ — shared prototype

Harness-agnostic prototype used by all four child POCs (one mechanism, no
per-harness streaming code).

| file | purpose |
|---|---|
| `atof_stream.py` | `AtofStreamListener` — loopback ndjson sink; async queue of raw ATOF; bounded-queue backpressure; handles >512 KB gateway records |
| `fabric_stream.py` | `Runtime.invoke_stream()` prototype — `StreamingRuntime` / `InvokeStream`; endpoint injection at `start_runtime`; raw ATOF + out-of-band `result()`; honest early-exit |
| `run_harness.py` | run one real harness through `invoke_stream` and save its raw ATOF stream |
| `derive_native_events.py` | derive a `native-events.jsonl` fixture from a captured `events.atof.jsonl` |

Requires a built native extension and provider creds — see [`../README.md`](../README.md).
