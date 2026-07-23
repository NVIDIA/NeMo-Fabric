<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# common/ — Shared Prototype

Harness-agnostic prototype used by all four harnesses — three child POCs — with one
mechanism and no per-harness streaming code.

| File | Purpose |
|---|---|
| `atof_stream.py` | `AtofStreamListener` — loopback NDJSON sink; async queue of raw ATOF; bounded-queue backpressure; handles >512 KB gateway records |
| `fabric_stream.py` | `Runtime.invoke_stream()` prototype — `StreamingRuntime` / `InvokeStream`; endpoint injection at `start_runtime`; raw ATOF + out-of-band `result()`; one active turn per runtime; early-exit via `aclose()` |
| `run_harness.py` | run one real harness through `invoke_stream` and save its raw ATOF stream |
| `native_recorder.py` | **POC-only** recorder that tees a harness's native SDK stream *before* Relay (produces `native-events.jsonl`); documents the per-harness capture seams. Not a Fabric API. |

Requires a built native extension and provider credentials — see
[the POC README](../README.md).
