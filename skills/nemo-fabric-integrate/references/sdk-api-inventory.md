<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# SDK API Inventory

`Fabric()` is the primary entrypoint. It is a plain, reusable object — not a
lifecycle context manager — and can plan, diagnose, or start multiple
independent runtimes. The generated
[client reference](https://github.com/NVIDIA/NeMo-Fabric/blob/main/docs/reference/api/python-library-reference/nemo_fabric.client.md),
[runtime reference](https://github.com/NVIDIA/NeMo-Fabric/blob/main/docs/reference/api/python-library-reference/nemo_fabric.runtime.md),
[native OpenAI streaming reference](https://github.com/NVIDIA/NeMo-Fabric/blob/main/docs/reference/api/python-library-reference/nemo_fabric.openai_streaming.md),
and [Relay streaming reference](https://github.com/NVIDIA/NeMo-Fabric/blob/main/docs/reference/api/python-library-reference/nemo_fabric.streaming.md)
document the public methods with their `async` and keyword-only markers. The
installed `nemo_fabric` package also ships type information (`py.typed`) for
static analysis.

## `Fabric` Methods

The following table lists the `Fabric` methods and when to use each:

| Method | Async | Use When | Returns |
| --- | --- | --- | --- |
| `plan(config, *, base_dir=...)` | No | You need the selected adapter, capability routing, and runtime capabilities before running. | `RunPlan` |
| `doctor(config, *, base_dir=...)` | Yes | You need preflight diagnostics for adapter resolution, capability routing, declared requirements, and environment assumptions. | `DoctorReport` |
| `run(config, *, base_dir=..., input=... \| request=..., session_ref=None)` | Yes | You need one complete start, invoke, result, and stop cycle. | `RunResult` |
| `start_runtime(config, *, base_dir=..., overrides=..., session_ref=None, streaming=False, launch_collector=None, completion_wait_timeout=1.0, service=None)` | Yes | You need state across multiple ordered invocations. Pass `streaming=True` with NVIDIA NeMo Relay enabled only to provision `invoke_stream(...)`. | `Runtime` |
| `prepare_environment(config, *, base_dir=...)` | Yes | Prepare an environment independently of a runtime; preparation depends on provider support. | `EnvironmentHandle` |
| `attach_environment(config, reference, *, base_dir=...)` | Yes | Verify a caller-owned resource with a typed `EnvironmentReference` when the provider supports attachment. | `EnvironmentHandle` |
| `start_runtime_in(config, environment, *, base_dir=..., overrides=..., session_ref=None, streaming=False)` | Yes | Start a runtime in a typed environment when the selected runtime provider supports it. | `Runtime` |
| `check(*, request=..., environment=None, service=None)` | Yes | Obtain fresh server, runtime, or service observations with a typed `CheckRequest`. Unsupported or timed-out probes remain unknown or unsupported. | `CheckReport` |
| `prepare_service(config, *, base_dir=...)` | Yes | Start a Fabric-owned shared adapter service when supported. Stop connected runtimes before releasing the service. | `Service` |
| `attach_service(config, reference, *, base_dir=...)` | Yes | Validate a typed `ServiceReference` for a caller-owned service. Release detaches without stopping the external service. | `Service` |
| `runtime_status(*, runtime_id=..., agent_name=..., environment=None)` | Yes | Read recorded runtime state and a safe discovery handle. This does not probe harness health; use `check()` for fresh observations. | `RuntimeStatusReport` |
| `invocation_result(*, agent_name=..., runtime_id=..., request_id=..., artifacts=None, environment=None)` | Yes | Retrieve a saved invocation result after an uncertain response. Use the original artifacts directory for a new local client, or an environment for OpenShell. A missing result never authorizes replay. | `RunResult` or `None` |
| `release_environment(environment)` | Yes | Release the environment separately after runtime stop; local release preserves files. | `None` |

`input` and `request` on `run(...)` are mutually exclusive. Use `input=...` for
the common case; use `request=RunRequest(...)` when the invocation needs a
caller-owned `request_id`, `context`, or overrides.

The public `inspect_adapter(config, descriptor, target_descriptor=...)` helper returns `AdapterInspection`,
with an empty `issues` list and an immutable `capability_profile` on success.
Compatibility failures return structured `AdapterCompatibilityIssue` values with
`adapter_id`, `field`, and `reason`; incompatible reports have no profile.
Check issues before deployment. `diagnostics=True` is the default; pass `diagnostics=False` for exception-based rejection. Malformed inputs raise in both modes. Inspection and planning both require declared Relay ATIF output support and exactly one observability component when ATIF is requested.
The nested `AdapterCapabilityProfile` supports host admission without harness SDKs.
Both `run()` and `start_runtime()` accept optional
`expected_descriptor_sha256` to verify the task plan's descriptor before startup.
The profile is configuration-dependent for telemetry. Registered workflows
require target metadata and bind both descriptors in the fingerprint; attached
services require separate admission. The profile is not observed execution provenance.

OpenShell environments use `start_runtime_in` with a prepared or attached
handle. They support buffered invocation and multiple independent bridge
runtime IDs per sandbox. Invocations for different runtime IDs can proceed
independently. The task SDK owns each runtime ID and normalized result. Declared
files become host paths under the configured artifact destination; a failed SDK
result still carries its output, error, and artifacts. Lost responses are not
replayed. See the [OpenShell provider guide](https://github.com/NVIDIA/NeMo-Fabric/blob/main/environment-providers/openshell/README.md).

## Runtime Methods

The following table lists the `Runtime` members for driving a stateful runtime.

| Member | Async | Notes |
| --- | --- | --- |
| `invoke(*, input=... \| request=...)` | Yes | One turn on an active runtime. One active invocation at a time; overlap raises `FabricStateError`. |
| `invoke_openai_stream(*, input=... \| request=...)` | No | Start exactly one descriptor-gated native invocation and return an async `OpenAIInvokeStream` of `chat.completion.chunk` mappings. Await `stream.result()` for the separate terminal `RunResult`. |
| `invoke_stream(*, input=... \| request=...)` | No | Start one NeMo Relay turn and return an async `InvokeStream` of raw ATOF records. Await `stream.result()` for the terminal `RunResult`. |
| `resource_snapshot()` | Yes | Timestamped memory, process count, and CPU time of the local adapter host and its descendants. Does not wait for an active invocation. |
| `stop()` | Yes | Stop the runtime. Called automatically by `async with`. |
| `status` | No | `RuntimeStatus`: `ACTIVE`, `STOPPED`, or `FAILED`. |
| `supports_openai_streaming` | No | `True` when the selected descriptor declares `capabilities.streaming` for native OpenAI Chat Completions chunks. |
| `supports_streaming` | No | `True` when NeMo Relay ATOF streaming was enabled at runtime startup. |
| `runtime_id` | No | Opaque identifier for this runtime lifecycle. |
| `handle` | No | Copied `RuntimeHandle` for this runtime; discovery handles are not invocation or ownership-transfer APIs. |
| `session_ref` | No | Confirmed durable native `SessionRef`, or `None`. Persist `to_mapping()` with the retained state tree and pass `SessionRef.from_mapping(...)` explicitly at startup to resume. |
| `recovery_warnings` | No | Structured warnings for an uncertain resumed session. Inspect native and external state before sending a new `RunRequest(acknowledge_recovery=True)`; do not replay interrupted work automatically. |
| `messages` / `invocations` | No | Copied harness history and per-turn IDs. |

Always use a runtime as an async context manager so cleanup runs on exit.
Shutdown is attempted, not guaranteed — `stop()`, including the automatic call at
`async with` exit, can raise `FabricRuntimeError`:

```python
async with await fabric.start_runtime(config, base_dir=base) as runtime:
    result = await runtime.invoke(input="…")
```

## Stream Handles

The two stream handles represent independent contracts:

| Class | Async Iteration | Terminal Result | Early Close |
| --- | --- | --- | --- |
| `OpenAIInvokeStream` | Adapter-native `chat.completion.chunk` mappings; an empty stream is valid. | `await result()` drains and discards unread chunks, then returns the separate authoritative `RunResult`. | `await aclose()` drains and discards unread chunks without cancelling the invocation. |
| `InvokeStream` | Raw invocation-correlated ATOF records generated by NeMo Relay. | `await result()` returns the separate authoritative `RunResult`. | `await aclose()` drains and discards unread records without cancelling the invocation. |

Fully consume either stream or call `await stream.aclose()` before starting
another turn. For native OpenAI streaming, `result()` also finalizes and
discards unread chunks. Relay `InvokeStream.result()` does not consume unread
ATOF records.

## Execution Model

NVIDIA NeMo Fabric separates configuration, planning, runtime lifecycle, and
individual invocations:

```text
FabricConfig -> plan() -> RunPlan -> start_runtime() -> Runtime -> invoke() -> RunResult
                                                            \-> invoke_openai_stream() -> OpenAIInvokeStream
                                                            \-> invoke_stream() -> InvokeStream
```

- `Fabric` is a reusable facade that records runtimes started through that instance, retaining live objects until successful stop. It is not a lifecycle context manager; stop each runtime explicitly or through its async context manager.
- A `Runtime` owns stateful execution and shutdown, so it is the object used
  with `async with`.
- A runtime is a logical execution boundary, not necessarily an operating-system
  process. Harness-native threads, sessions, and conversations remain
  adapter-owned state associated with the runtime.
- Runtime hosting is adapter-declared, not consumer-configured. The bundled
  Claude, Codex, Deep Agents, and Hermes Agent adapters retain their native runtime
  resources in one local host. Local `process` and `python` adapters use the
  same host lifecycle. A host crash or protocol timeout terminates the runtime;
  the application can explicitly start a new runtime according to its policy.
- The application owns scheduling, queues, retries, and how many runtimes to
  run. NeMo Fabric provides only the runtime contract.
