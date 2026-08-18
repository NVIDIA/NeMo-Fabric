{/*
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
*/}

# Stage 3: Implement Execution

One NVIDIA NeMo Fabric runtime is a lifecycle, state-isolation, and correlation
boundary. It does not require a particular process, service, thread, or native
target session topology.

## Implement the Required Lifecycle

The minimum adapter implements these operations:

| Operation | Adapter Responsibility |
| --- | --- |
| `start` | Validate startup-only requirements, translate `AgentConfig`, and retain one isolated target runtime. |
| `invoke` | Translate one `AgentRunRequest`, execute the retained target, and return one `AgentRunResult`. |
| `stop` | Attempt to release every adapter-owned resource, including after partial startup or failed invocation. |

The required order is one successful `start`, zero or more ordered `invoke`
operations, and one `stop` attempt. The minimum profile permits one active
invocation in a runtime. The adapter does not need a queue or internal
concurrency; consumers start independent runtimes for parallel work.

Keep mutable target state inside the runtime instance. Do not share it between
independent Fabric runtimes.

## Start From the Minimum Python Host

Python adapters can opt into `nemo-fabric-adapters-common` instead of
implementing the persistent local-host binding. The following implementation
shows the complete method surface:

```python
from nemo_fabric_adapter_contract.models import AgentConfig
from nemo_fabric_adapter_contract.models import AgentRunError
from nemo_fabric_adapter_contract.models import AgentRunRequest
from nemo_fabric_adapter_contract.models import AgentRunResult
from nemo_fabric_adapter_contract.models import AgentRunStatus
from nemo_fabric_adapter_contract.models import RuntimeContext
from nemo_fabric_adapters.common import lifecycle


class TargetRuntime:
    def __init__(self):
        self.target = None

    async def start(self, payload):
        config: AgentConfig = payload["config"]
        target = await create_target(config)
        self.target = target

    async def invoke(
        self,
        request: AgentRunRequest,
        context: RuntimeContext,
    ) -> AgentRunResult:
        try:
            native = await self.target.run(request.input)
        except TargetInvocationError:  # Use the target SDK's documented failure type.
            return AgentRunResult(
                status=AgentRunStatus.FAILED,
                error=AgentRunError(
                    code="target_failed",
                    message="The target could not complete the invocation",
                ),
            )
        return AgentRunResult(
            status=AgentRunStatus.SUCCEEDED,
            output={"response": native.text},
        )

    async def stop(self):
        target, self.target = self.target, None
        if target is not None:
            await target.close()


def main() -> None:
    lifecycle.serve(TargetRuntime, config_loader=AgentConfig.from_mapping)


if __name__ == "__main__":
    main()
```

The host creates one `TargetRuntime` per local host, validates the start config
as `AgentConfig`, passes typed request and context objects to `invoke`, requires
a typed terminal result, serializes lifecycle operations, reserves stdout for
its wire protocol, and attempts cleanup on end of file. The common host is
optional; an adapter can implement another supported binding directly.

The target factory must release resources it creates if it fails before
returning. Once the factory returns, retain the target immediately. Make
`stop` idempotent, clear retained state even if cleanup fails, and keep it safe
after a failed invocation. A no-op `stop` is valid for a thin remote-service
adapter that owns no remote lifecycle, but it must still complete successfully.

## Use RuntimeContext for Operation Context

NeMo Fabric creates `RuntimeContext`. Treat every ID as an opaque correlation
value:

| Field | Purpose |
| --- | --- |
| `runtime_id` | Correlates all operations in one Fabric runtime. |
| `invocation_id` | Identifies one invocation attempt. |
| `request_id` | Correlates the caller's request through NeMo Fabric and the adapter. |
| `environment` | Supplies the resolved workspace, artifact root, environment values, ownership, and provider context. |
| `artifacts` | Lists artifacts visible when the operation begins. |
| `telemetry` | Supplies invocation telemetry context, including generated Relay configuration when enabled. |

Use the canonical
[`runtime-context.schema.json`](https://github.com/NVIDIA/NeMo-Fabric/blob/main/schemas/adapter-contract/runtime-context.schema.json)
for the exact shape. Runtime identity belongs in `RuntimeContext`, not in
`AgentConfig.workflow`. Per-invocation task input belongs in the request, not in
workflow settings.

## Propagate Failures Safely

Use a lifecycle failure when the adapter cannot satisfy `start`, `invoke`, or
`stop`, including protocol and transport failures. A lifecycle failure can
invalidate the runtime.

A target-level failure is an `AgentRunResult` with `status=FAILED` and an
`AgentRunError`. Do not expose stack traces, credentials, complete environment
values, HTTP authorization headers, or arbitrary user input in errors or logs.
NeMo Fabric does not automatically replay an invocation after a transport
failure.

## Add Streaming Only When Needed

`Runtime.invoke_stream()` is the primary normalized streaming API. It runs the
ordinary adapter `invoke` operation while NeMo Relay exposes correlated ATOF to
the consumer. NeMo Fabric owns ingestion, correlation, buffering,
backpressure, and the consumer stream lifecycle.

The event stream and terminal result describe the same invocation but are
delivered separately. An empty stream can have a valid result, stream
exhaustion does not imply success, and closing the consumer stream does not
cancel the target invocation.

An adapter that needs native progressive output can implement the narrower
[`invoke_openai_stream`](openai-streaming.md) capability. No other
target-native event formats are part of v1alpha2.

The descriptor also contains reserved `cancellation`, `updates`, and `service`
capability flags. Do not claim them until the selected NeMo Fabric runtime
binding exposes and tests the corresponding adapter operation.

After the lifecycle works, [normalize its terminal outcomes](results.md).
