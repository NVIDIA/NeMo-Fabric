<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Execution

NeMo Fabric exposes one consumer lifecycle and maps it onto an ordered
adapter-target lifecycle. A Fabric runtime is the isolation and correlation
boundary; it does not require a particular process, service, thread, or native
harness-session topology.

## Lifecycle

| Operation | Requirement | Contract |
| --- | --- | --- |
| `start(AgentConfig, RuntimeContext)` | Required | Initialize one isolated adapter-target runtime. |
| `invoke(AgentRunRequest, RuntimeContext)` | Required | Execute one invocation and produce one terminal outcome. |
| `stop(runtime_id)` | Required | Attempt to release all runtime resources, including after partial or failed execution. |
| `invoke_stream(...)` | Fabric-provided | Run ordinary `invoke` while NeMo Relay supplies correlated ATOF to the consumer. |
| `invoke_openai_stream(...)` | Reserved optional surface | A future native pass-through may expose only a declared OpenAI-compatible event profile. Other native stream formats are outside the contract. |
| `cancel(...)` | Reserved optional surface | Request cancellation of an active invocation when a runtime binding implements it. |
| `update(...)` | Reserved optional surface | Atomically apply declared updateable fields when a runtime binding implements it. |

The required ordering is one `start`, zero or more `invoke` operations, then
one `stop`. The minimum profile permits only one active invocation in a
runtime. Adapters need not implement a queue or internal concurrency; consumers
start independent runtimes for parallel work.

Each operation produces one terminal response. An invocation-level failure
does not necessarily invalidate the runtime. A lifecycle or transport failure
may make it unusable, after which NeMo Fabric proceeds to cleanup rather than
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

## Current Python Host Binding

`nemo-fabric-adapters-common` is optional. Python adapters may use its
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

    async def stop(self):
        pass


def main() -> None:
    lifecycle.serve(ExampleRuntime, config_model=AgentConfig)
```

The host validates the start `config` as `AgentConfig`, serializes operations,
normalizes lifecycle failures, reserves stdout for its protocol, and attempts
cleanup on EOF. The adapter remains responsible for target-specific validation,
translation, state, and shutdown.

The current invoke payload contains `RuntimeContext` plus northbound
`RunRequest`, and accepts JSON-compatible output. `AgentRunRequest` and
`AgentRunResult` are published southbound types but are not yet enforced by
this transport. Keep conversion at the edge of the adapter so adopting the
typed invocation boundary does not affect target lifecycle code.
