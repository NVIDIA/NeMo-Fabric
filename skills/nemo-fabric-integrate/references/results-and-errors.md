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
| `error` | Structured failure metadata when available. Branch on `status`, not on this field. |
| `output` | Harness output normalized to the configured output schema. |
| `usage` | Optional `RunUsage` with invocation-local input, cached-input, output, and total token counts, reported cost in USD, and provider metadata. |
| `resource_snapshot` | Optional `RuntimeResourceSnapshot` of the runtime, read once after the adapter returned the result. `None` when the read failed. |
| `artifacts` | Output files, logs, patches, and other materialized references. |
| `telemetry` | References to NVIDIA NeMo Relay or other telemetry streams from the run. |
| `events` | Ordered normalized lifecycle and invocation events. |
| `metadata` | Result-specific structured metadata. |
| `runtime_id`, `invocation_id`, `request_id` | Correlation IDs across runtimes, logs, telemetry, and artifacts. |

Handle a result with this status-first pattern:

```python
if result.status == "succeeded":
    use_output(result.output, result.artifacts, result.telemetry)
else:
    handle_failure(result.status, result.error, result.events)  # failed, cancelled, ...
```

Read usage only when the target reports it:

```python
if result.usage is not None:
    record_usage(
        input_tokens=result.usage.get("input_tokens"),
        output_tokens=result.usage.get("output_tokens"),
        total_tokens=result.usage.get("total_tokens"),
        cost_usd=result.usage.get("cost_usd"),
    )
```

Treat missing counters and cost as unknown, not zero. `cached_input_tokens` reports cached prompt tokens; `input_tokens_include_cache` declares whether `input_tokens` includes them (`True`), excludes them (`False`), or has unknown semantics (absent). Do not add cache to an inclusive input count. Adapters can put cache-write, reasoning, peak request, and per-model detail in the descriptor-declared `usage.extensions["nemo_fabric.detailed_token_usage.v1"]` object. NeMo Fabric preserves that object at `result.usage["metadata"]["nemo_fabric.detailed_token_usage.v1"]`. Treat detailed input and output independently: any detailed accounting field for a dimension makes that dimension authoritative, with no fallback to its top-level counters. The views can diverge. Calculate an inclusive total only when the selected detailed counters provide enough information; otherwise leave the total unknown. Keep estimated costs separate in provider metadata; `cost_usd` is reported cost, not an estimate.

## Run Records

Set `RuntimeConfig(artifacts=..., record=RecordConfig(enabled=True))` to have
NeMo Fabric write `run-record.jsonl` for each invocation under
`<artifact root>/.fabric/<runtime_id>/<invocation_id>/`. Recording is off by
default and needs an artifact root, the `local` environment provider, and a
`process` or `python` adapter. The result lists the file as the artifact with
kind `run_record`. The file has a `start` line, written before the request, a
`sample` line about every 10 seconds until the adapter host answers or the
timeout fires, when the start line has `sample_interval_ms`, and an `end` line with status, elapsed time, error
code, numeric usage, artifact names, and `stop`, which
gives NeMo Fabric's failure cause (`fabric_failed`, `sandbox_failed`,
`agent_exhausted_budget`, `unclassified`, or `agent_failed`), the time limit
for a timeout, and the adapter host's exit status. Only a `failed` result
reads `agent_failed`; an error NeMo Fabric raises never does. Each line
stores the runtime's resource snapshot as `resources`. When both the start and
end lines hold one, end minus start is a partial observed CPU delta that
leaves out descendants that left the process tree; read a decrease as
unknown, never as zero or negative. It holds no prompts, output, or error
messages. A recent sample shows that NeMo Fabric observed the record
recently, not that the agent is making progress. After three sample intervals
without an update, the record is stale and the invocation's outcome is
unknown until an end line exists; never count a stale record as failed. One
slow snapshot read delays samples for every record in the process, and a
suspended process can resume. A record also stops changing when a write
fails; the result's `record_write_failed` event says so. Without
`sample_interval_ms`, a record without an end line has an unknown outcome.
Recording is synchronous and best-effort: a failed record
write adds a `record_write_failed` event and does not replace the result, and
slow storage delays the invocation. The format is the `v1alpha1`
[run record line schema](https://github.com/NVIDIA/NeMo-Fabric/blob/main/schemas/sdk/run-record-line.schema.json)
and may change until a reader ships.

## Runtime Resource Snapshots

`await runtime.resource_snapshot()` returns a `RuntimeResourceSnapshot` with
`runtime_id`, `started_at` and `observed_at` (RFC 3339 UTC),
`resident_memory_bytes`, `process_count`, and `cpu_time_seconds`. It covers the
local adapter host process and its descendants (not a shared service process),
and it does not wait for an active invocation. CPU time is cumulative since the
host started, so subtract an earlier snapshot to estimate one invocation; it
can drop when a descendant leaves the tree, so treat a negative difference as
unknown. Values that cannot be read raise an error, never read as zero.
Snapshots observe only; enforce limits in the operating system or execution
environment.

## Correlation IDs

`runtime_id` identifies the runtime lifecycle, `invocation_id` identifies one
invocation within it, and `request_id` correlates the caller's request.
Values generated by NVIDIA NeMo Fabric use type-specific prefixes such as
`runtime-`, `invocation-`, and `request-`; a caller can supply its own
`request_id`. Store
and log each field separately and treat every value as opaque — do not parse or
reuse the encoding.

## Error Hierarchy

All public SDK errors inherit from `FabricError`. NeMo Fabric raises these when it
cannot return a normalized result; it does not return a partial `RunResult`.
Refer to the [errors reference](https://github.com/NVIDIA/NeMo-Fabric/blob/main/docs/reference/api/python-library-reference/nemo_fabric.errors.md).

| Error | Meaning |
| --- | --- |
| `FabricConfigError` | Invalid config, request, or override. |
| `FabricCapabilityError` | Selected adapter does not support the requested operation. |
| `FabricRuntimeError` | Startup, invocation, or shutdown failed before a normalized result. |
| `FabricStateError` | Invalid runtime state transition (invoking after stop, overlapping invocations). |
| `FabricNativeUnavailableError` | Native extension is not installed or importable. |
| `ResourceObservationUnavailableError` | A resource snapshot cannot be read because the runtime stopped, is stopping, its adapter host exited, a live process in its tree or the process table cannot be read, or the process tree kept changing during the read. Subclass of `FabricRuntimeError`. |
| `ResourceObservationUnsupportedError` | The runtime's adapter kind, environment provider, or operating system has no resource reader. Subclass of `FabricRuntimeError`. |

## Cleanup And Resilience

Core host-operation deadlines (`host_timeout`) become Python `TimeoutError`, then
`FabricRuntimeError` with `code="timeout"` and the failed lifecycle stage.
If invocation succeeds in `Fabric.run()` but stop reaches this deadline,
it returns a failed `RunResult` with `error.stage == "stop"` and
`error.code == "timeout"`. Check the returned result as well as raised errors.
Adapters can report execution deadlines in normalized results with
`result.error.code == "timeout"`. Other adapter-specific timeout codes are not
automatically reclassified. Do not classify deadlines from error text.

- Prefer `run(...)` and `async with` runtimes: both attempt cleanup
  automatically. Shutdown is attempted, not guaranteed — `stop()`, including the
  automatic call at `async with` exit, can raise `FabricRuntimeError`. On a
  normal block exit that error propagates; when an invocation already failed, the
  cleanup failure is attached to the original exception rather than replacing it.
- The consumer owns job-level retries and rollout failure policy. NeMo Fabric marks a
  runtime or invocation failed and returns structured error metadata when
  possible, but does not retry by default.
- Transient failures can carry retryable error metadata. Capacity pressure
  surfaces as a structured error or event (busy, rate limited, backpressure).
  The consumer decides whether to wait, retry, start a replacement runtime, or
  escalate.
