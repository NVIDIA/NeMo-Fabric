<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Results, Evidence, And Errors

## RunResult Fields

Every invocation that reaches the adapter boundary returns a normalized
`RunResult`. Inspect the failure fields before reading output. Generated
reference: `docs/reference/api/python-library-reference/types.md`.

| Field | Meaning |
| --- | --- |
| `status` | Terminal invocation status (for example success, failure, cancellation). |
| `error` | Structured `ErrorInfo`, or `None` on success. Check this first. |
| `output` | Harness output normalized to the configured output schema. |
| `artifacts` | Output files, logs, patches, and other materialized references. |
| `telemetry` | References to Relay or other telemetry streams from the run. |
| `events` | Ordered normalized lifecycle and invocation events. |
| `metadata` | Result-specific structured metadata. |
| `runtime_id`, `invocation_id`, `request_id` | Correlation IDs across runtimes, logs, telemetry, and artifacts. |

```python
if result.error is not None:
    handle_failure(result.status, result.error, result.events)
else:
    use_output(result.output, result.artifacts, result.telemetry)
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
cannot return a normalized result; it does not return a partial `RunResult`.

| Error | Meaning |
| --- | --- |
| `FabricConfigError` | Invalid config, request, or override. |
| `FabricCapabilityError` | Selected adapter does not support the requested operation. |
| `FabricRuntimeError` | Startup, invocation, or shutdown failed before a normalized result. |
| `FabricStateError` | Invalid runtime state transition (invoking after stop, overlapping invocations). |
| `FabricNativeUnavailableError` | Native extension is not installed or importable. |

## Cleanup And Resilience

- Prefer `run(...)` and `async with` runtimes: both attempt cleanup
  automatically. A runtime used with `async with` also attempts shutdown after
  an invocation error; if cleanup then fails, that failure is attached to the
  original exception rather than replacing it.
- The consumer owns job-level retries and rollout failure policy. Fabric marks a
  runtime or invocation failed and returns structured error metadata when
  possible, but does not retry by default.
- Transient failures may carry retryable error metadata. Capacity pressure
  surfaces as a structured error or event (busy, rate limited, backpressure).
  The consumer decides whether to wait, retry, start a replacement runtime, or
  escalate.
