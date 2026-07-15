<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Results, Evidence, And Errors

## RunResult Fields

Every invocation that reaches the adapter boundary returns a normalized
`RunResult`. Inspect `status` before reading output. Generated reference: the
[types reference](https://github.com/NVIDIA/NeMo-Fabric/blob/main/docs/reference/api/python-library-reference/nemo_fabric.types.md).

| Field | Meaning |
| --- | --- |
| `status` | Terminal invocation status: `succeeded`, `failed`, or `cancelled`. Branch on this. |
| `error` | Structured `ErrorInfo`, or `None` — may be `None` even when `status` is not `succeeded`, so do not use it as the success signal. |
| `output` | Harness output normalized to the configured output schema. |
| `artifacts` | Output files, logs, patches, and other materialized references. |
| `telemetry` | References to Relay or other telemetry streams from the run. |
| `events` | Ordered normalized lifecycle and invocation events. |
| `metadata` | Result-specific structured metadata. |
| `runtime_id`, `invocation_id`, `request_id` | Correlation IDs across runtimes, logs, telemetry, and artifacts. |

```python
if result.status == "succeeded":
    use_output(result.output, result.artifacts, result.telemetry)
else:
    handle_failure(result.status, result.error, result.events)  # failed, cancelled, ...
```

## Correlation IDs

`runtime_id` identifies the runtime lifecycle, `invocation_id` identifies one
invocation within it, and `request_id` correlates the caller's request.
Fabric-generated values use type-specific prefixes such as `runtime-`,
`invocation-`, and `request-`; a caller may supply its own `request_id`. Store
and log each field separately and treat every value as opaque — do not parse or
reuse the encoding.

## Error Hierarchy

All public SDK errors inherit from `FabricError`. Fabric raises these when it
cannot return a normalized result; it does not return a partial `RunResult`. See
the [errors reference](https://github.com/NVIDIA/NeMo-Fabric/blob/main/docs/reference/api/python-library-reference/nemo_fabric.errors.md).

| Error | Meaning |
| --- | --- |
| `FabricConfigError` | Invalid config, request, or override. |
| `FabricCapabilityError` | Selected adapter does not support the requested operation. |
| `FabricRuntimeError` | Startup, invocation, or shutdown failed before a normalized result. |
| `FabricStateError` | Invalid runtime state transition (invoking after stop, overlapping invocations). |
| `FabricNativeUnavailableError` | Native extension is not installed or importable. |

## Cleanup And Resilience

- Prefer `run(...)` and `async with` runtimes: both attempt cleanup
  automatically. Shutdown is attempted, not guaranteed — `stop()`, including the
  automatic call at `async with` exit, can raise `FabricRuntimeError`. On a
  normal block exit that error propagates; when an invocation already failed, the
  cleanup failure is attached to the original exception rather than replacing it.
- The consumer owns job-level retries and rollout failure policy. Fabric marks a
  runtime or invocation failed and returns structured error metadata when
  possible, but does not retry by default.
- Transient failures may carry retryable error metadata. Capacity pressure
  surfaces as a structured error or event (busy, rate limited, backpressure).
  The consumer decides whether to wait, retry, start a replacement runtime, or
  escalate.
