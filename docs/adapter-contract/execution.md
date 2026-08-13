{/*
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
*/}

# Execution

NVIDIA NeMo Fabric exposes one consumer lifecycle and maps it onto an ordered
adapter-target lifecycle. A NeMo Fabric runtime is the isolation and correlation
boundary; it does not require a particular process, service, thread, or native
harness-session topology.

## Lifecycle

The abstract lifecycle contract contains these operations:

| Operation | Requirement | Contract |
| --- | --- | --- |
| `start(AgentConfig, RuntimeContext)` | Required | Initialize one isolated adapter-target runtime. |
| `invoke(AgentRunRequest, RuntimeContext)` | Preview, not negotiated | Future typed invocation boundary. The current binding uses its legacy request envelope and JSON-compatible output. |
| `stop(runtime_id)` | Required | Attempt to release all runtime resources, including after partial or failed execution. |
| `invoke_stream(...)` | NeMo Fabric-provided | Run ordinary `invoke` while NeMo Relay supplies correlated ATOF to the consumer. |
| `invoke_openai_stream(...)` | Optional | Execute exactly one adapter invocation while emitting OpenAI Chat Completions chunks. The selected descriptor must declare `capabilities.streaming`. |
| `cancel(...)` | Reserved optional surface | Request cancellation of an active invocation when a runtime binding implements it. |
| `update(...)` | Reserved optional surface | Atomically apply declared updateable fields when a runtime binding implements it. |

The required ordering is one `start`, zero or more invocation operations, then
one `stop`, regardless of whether an invocation uses `invoke`,
`invoke_openai_stream`, or the future typed boundary. The minimum profile
permits only one active invocation in a runtime. Adapters need not implement a
queue or internal concurrency; consumers start independent runtimes for
parallel work.

Each operation produces one terminal response. An invocation-level failure
does not necessarily invalidate the runtime. A lifecycle or transport failure
can make it unusable, after which NeMo Fabric proceeds to cleanup rather than
replaying the request.

## Runtime Context

NeMo Fabric creates `RuntimeContext`; consumers and adapters must treat its IDs
as opaque correlation values.

| Field | Purpose |
| --- | --- |
| `runtime_id` | Correlates all operations in one Fabric runtime. |
| `invocation_id` | Identifies one invocation attempt. |
| `request_id` | Correlates the caller's request through Fabric and the adapter. |
| `environment` | Resolved workspace, artifact root, environment values, ownership, and provider context. |
| `artifacts` | Artifacts visible when the operation begins. |
| `telemetry` | Invocation telemetry context, including generated Relay config and environment when enabled. |

Use the generated
[`runtime-context.schema.json`](https://github.com/NVIDIA/NeMo-Fabric/blob/main/schemas/adapter-contract/runtime-context.schema.json)
for the exact shape. Runtime/session identity belongs here, not in
`AgentConfig.workflow`; caller task context belongs in the invocation request.

## Relay Streaming

`Runtime.invoke_stream()` is the primary Fabric streaming API. It exposes raw,
invocation-correlated Agent Trajectory Observability Format (ATOF) records
generated through NeMo Relay. NeMo Fabric owns stream ingestion, correlation,
buffering, backpressure, and consumer stream lifecycle. The adapter continues
to execute its ordinary `invoke` operation.

The ATOF stream and terminal normalized result describe the same invocation,
but the result is obtained separately. An empty stream can still have a valid
terminal result. Stopping iteration or closing the consumer stream does not
cancel the target invocation.

The adapter reads Relay configuration and environment from
`RuntimeContext.telemetry` or uses the optional common adapter helpers. It does
not invent a second stream protocol.

## Native OpenAI Streaming

`Runtime.invoke_openai_stream()` exposes adapter-native progressive output for
an adapter whose descriptor declares `capabilities.streaming`. An adapter that
declares the capability must implement the optional operation. This capability
is narrower than a generic native stream: the adapter emits only the declared
`openai.chat_completions.chunk/v1` profile. Each mapping includes non-empty
`id` and `model` strings, a nonnegative integer `created`, the exact
`chat.completion.chunk` object discriminator, and structurally valid `choices`.
OpenAI Responses API events, target-native event objects, Server-Sent Events
framing, and terminal results are outside the progressive stream.

One call executes exactly one adapter invocation. The stream can be empty, and
its terminal normalized `RunResult` remains separate and authoritative. Ending
iteration early does not cancel the invocation. The SDK drains the invocation
when the consumer closes the stream so that the runtime can safely accept its
next turn.

`runtime.timeout_seconds` limits the total duration of the streamed invocation,
from invocation start through terminal completion. It is not an idle or
progress timeout, and receiving chunks does not reset it. If the deadline is
exceeded, NeMo Fabric invalidates and terminates the local adapter host as
required by the existing lifecycle timeout semantics; consumers must start a
new runtime instead of reusing that host.

NeMo Fabric owns the authenticated loopback HTTP transport, chunked NDJSON
framing, correlation, validation, buffering, and consumer lifecycle. Adapters
must not persist or log the bearer token. NeMo Fabric persists only redacted
transport metadata for invocation auditing. The adapter host continues to
reserve stdout for the single terminal lifecycle response.

Bindings that implement the transport without the common Python host must
read the sink from the generated
[`openai-stream-invocation.schema.json`](https://github.com/NVIDIA/NeMo-Fabric/blob/main/schemas/adapter-contract/legacy/openai-stream-invocation.schema.json)
payload and follow the generated
[`openai-stream-record.schema.json`](https://github.com/NVIDIA/NeMo-Fabric/blob/main/schemas/adapter-contract/legacy/openai-stream-record.schema.json)
envelope for monotonic chunk records and the explicit end record.

The `fabric.openai_stream/v1alpha1` wire sequence is fixed:

1. Open one connection to the sink's loopback host and port.
2. Send `POST /openai-stream HTTP/1.1` with `Authorization: Bearer <token>`,
   `Content-Type: application/x-ndjson`, `Transfer-Encoding: chunked`, and
   `Expect: 100-continue`.
3. Wait for HTTP `100 Continue` before executing the target invocation.
4. Send zero or more `chunk` records, starting at sequence zero, followed by
   exactly one `end` record at the next sequence. Encode each record as one
   newline-terminated JSON value in the chunked request body.
5. Send the zero-length HTTP chunk, wait for HTTP `200 OK`, then write the one
   terminal lifecycle response to stdout. A rejection or incomplete end
   sequence is a lifecycle transport failure, not a successful empty stream.

The bearer token is single-use for that invocation. Do not retry or replay the
target after a transport failure.

Native OpenAI streaming and Relay streaming are independent. An adapter can
support either, both, or neither. `Runtime.invoke_stream()` continues to execute
ordinary `invoke` while exposing raw ATOF from NeMo Relay; it does not call
`invoke_openai_stream`.

## Current Python Host Binding

`nemo-fabric-adapters-common` is optional. Python adapters can use its
persistent line-oriented host instead of implementing the binding themselves:

```python
from nemo_fabric_adapter_contract.models import AgentConfig
from nemo_fabric_adapters.common import lifecycle


class ExampleRuntime:
    async def start(self, payload):
        config: AgentConfig = payload["config"]

    async def invoke(self, payload):
        request = payload["request"]
        return {"answer": "..."}

    async def invoke_openai_stream(self, payload, emit):
        request = payload["request"]
        await emit(
            {
                "id": "chunk-1",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "example-model",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "..."},
                        "finish_reason": None,
                    }
                ],
            }
        )
        return {"answer": "..."}

    async def stop(self):
        pass


def main() -> None:
    lifecycle.serve(ExampleRuntime, config_loader=AgentConfig.from_mapping)
```

The host validates the start `config` as `AgentConfig`, serializes operations,
normalizes lifecycle failures, reserves stdout for its protocol, and attempts
cleanup on EOF. For native OpenAI streaming, it validates and sends each chunk
through the Fabric-owned transport before writing one terminal lifecycle
response. The adapter remains responsible for target-specific validation,
translation, state, and shutdown.

The lifecycle table describes the typed adapter contract, not the Python method
signatures. The common Python host passes one protocol payload to `start` and
`invoke`: `payload["config"]` contains `AgentConfig` during `start`, while the
protocol envelope carries `RuntimeContext` and runtime identity. It calls
`stop()` after resolving the runtime identity from that envelope.

The current invoke payload contains `RuntimeContext` plus northbound
`RunRequest`, and accepts JSON-compatible output. The common host calls the
optional native stream method as `async invoke_openai_stream(payload, emit)`;
`payload` has the same adapter-visible invocation fields, without transport
credentials, and `emit` accepts only OpenAI Chat Completions chunk mappings.
`AgentRunRequest` and `AgentRunResult` are preview-only and are not part of the
negotiated contract. Keep conversion at the edge of the adapter so adopting a
future typed invoke boundary does not affect target lifecycle code.
