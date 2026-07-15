<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# SDK API Inventory

`Fabric()` is the primary entrypoint. It is a plain, reusable object — not a
lifecycle context manager — and can plan, diagnose, or start multiple
independent runtimes. Exact signatures live in the generated references — the
[client reference](https://github.com/NVIDIA/NeMo-Fabric/blob/main/docs/reference/api/python-library-reference/nemo_fabric.client.md)
and [runtime reference](https://github.com/NVIDIA/NeMo-Fabric/blob/main/docs/reference/api/python-library-reference/nemo_fabric.runtime.md).

## Fabric Methods

| Method | Async | Use When | Returns |
| --- | --- | --- | --- |
| `resolve(config, *, base_dir=...)` | No | You need the normalized effective config, without resolving an adapter. | `EffectiveConfig` |
| `plan(config, *, base_dir=...)` | No | You need the selected adapter, capability routing, and runtime capabilities before running. | `RunPlan` |
| `doctor(config, *, base_dir=...)` | Yes | You need preflight diagnostics for adapter availability, config support, and environment assumptions. | `DoctorReport` |
| `run(config, *, base_dir=..., input=... \| request=...)` | Yes | You need one complete start, invoke, result, and stop cycle. | `RunResult` |
| `start_runtime(config, *, base_dir=..., overrides=...)` | Yes | You need state across multiple ordered invocations. | `Runtime` |

`input` and `request` on `run(...)` are mutually exclusive. Use `input=...` for
the common case; use `request=RunRequest(...)` when the invocation needs a
caller-owned `request_id`, `context`, or overrides.

## Runtime Methods

| Member | Async | Notes |
| --- | --- | --- |
| `invoke(*, input=... \| request=...)` | Yes | One turn on an active runtime. One active invocation at a time; overlap raises `FabricStateError`. |
| `stop()` | Yes | Stop the runtime. Called automatically by `async with`. |
| `status` | No | `RuntimeStatus`: `ACTIVE`, `STOPPED`, or `FAILED`. |
| `runtime_id` | No | Opaque identifier for this runtime lifecycle. |
| `messages` / `invocations` | No | Copied harness history and per-turn IDs. |

Always use a runtime as an async context manager so cleanup runs on exit.
Shutdown is attempted, not guaranteed — `stop()`, including the automatic call at
`async with` exit, can raise `FabricRuntimeError`:

```python
async with await fabric.start_runtime(config, base_dir=base) as runtime:
    result = await runtime.invoke(input="…")
```

## Execution Model

Fabric separates configuration, planning, runtime lifecycle, and individual
invocations:

```
FabricConfig -> plan() -> RunPlan -> start_runtime() -> Runtime -> invoke() -> RunResult
```

- `Fabric` is a lightweight facade; it holds no started state and needs no
  cleanup.
- A `Runtime` owns stateful execution and shutdown, so it is the object used
  with `async with`.
- A runtime is a logical execution boundary, not necessarily an operating-system
  process. Harness-native threads, sessions, and conversations remain
  adapter-owned state associated with the runtime.
- The application owns scheduling, queues, retries, and how many runtimes to
  run. Fabric provides only the runtime contract.
