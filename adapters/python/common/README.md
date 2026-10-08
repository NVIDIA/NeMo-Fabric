<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Fabric Adapter Utilities

`nemo-fabric-adapters-common` provides shared Python helpers for NeMo Fabric
adapter implementations. Adapter packages normally install
this package as a dependency.

Install the package directly when developing an adapter:

```bash
pip install nemo-fabric-adapters-common
```

Use the maintained
[adapter contract documentation](https://github.com/NVIDIA/NeMo-Fabric/tree/main/docs/adapter-contract)
for adapter and configuration guidance. Source code is available in the
[NVIDIA NeMo Fabric repository](https://github.com/NVIDIA/NeMo-Fabric/).

## Persistent Local Hosts

Every local Process or Python adapter implements the ordered persistent-host
wire protocol. Python adapters can use
`nemo_fabric_adapters.common.lifecycle`. Supply a factory that creates one
adapter-owned runtime with asynchronous `start`, `invoke`, and `stop` methods:

```python
from nemo_fabric_adapter_contract.models import AgentConfig
from nemo_fabric_adapter_contract.models import AgentRunRequest
from nemo_fabric_adapter_contract.models import AgentRunResult
from nemo_fabric_adapter_contract.models import AgentRunStatus
from nemo_fabric_adapter_contract.models import RuntimeContext
from nemo_fabric_adapters.common import lifecycle


class AdapterRuntime:
    async def start(self, payload):
        self.client = await connect_client(payload)

    async def invoke(
        self,
        request: AgentRunRequest,
        context: RuntimeContext,
    ) -> AgentRunResult:
        native = await self.client.run(request.input)
        return AgentRunResult(
            status=AgentRunStatus.SUCCEEDED,
            output={"response": native.text},
        )

    async def stop(self):
        await self.client.close()


lifecycle.serve(AdapterRuntime, config_loader=AgentConfig.from_mapping)
```

If the adapter descriptor declares `capabilities.streaming`, the runtime must
also implement native OpenAI Chat Completions streaming:

```python
class AdapterRuntime:
    async def invoke_openai_stream(self, request, context, emit):
        async for chunk in self.client.stream(request.input):
            await emit(chunk)
        return AgentRunResult(
            status=AgentRunStatus.SUCCEEDED,
            output={"response": self.client.final_text},
        )
```

The optional method must have the signature
`async invoke_openai_stream(request, context, emit)`. Execute the target exactly once,
await `emit(chunk)` only for mappings in the
`openai.chat_completions.chunk/v1` profile, and return one `AgentRunResult`.
Each chunk requires non-empty `id` and `model` strings, a
nonnegative integer `created`, the exact `chat.completion.chunk` object
discriminator, and structurally valid `choices`. An empty chunk stream is
valid.

The SDK owns the authenticated loopback HTTP transport with chunked NDJSON
framing. The common host validates its credentials and framing, removes the
transport from the adapter payload, and supplies `emit`. Do not write chunks to
stdout or log stream credentials. This method is not used for Relay-backed
`Runtime.invoke_stream()`, which continues to execute ordinary `invoke`.
Process bindings that implement the wire protocol directly must follow the
generated
[`openai-stream-record.schema.json`](https://github.com/NVIDIA/NeMo-Fabric/blob/main/schemas/adapter-contract/openai-stream-record.schema.json)
chunk and explicit-end envelopes.

Adapters ask the host to validate the southbound config before `start`:

```python
from nemo_fabric_adapter_contract.models import AgentConfig

lifecycle.serve(AdapterRuntime, config_loader=AgentConfig.from_mapping)
```

The runtime then receives an `AgentConfig` instance in `payload["config"]`.
New adapters provide this loader; `FabricConfig` never crosses the supported
southbound boundary.

Adapters that call an OpenAI-compatible Chat Completions endpoint can use
`nemo_fabric_adapters.common.openai_chat`. The helper requests an SSE stream,
waits for its terminal event, and normalizes response text and token usage while
the adapter continues to own its HTTP client, authentication, lifecycle, and
error mapping.

NeMo Fabric calls the factory once per local host to create one runtime instance and
serializes invocations through that instance. The host keeps one event loop
alive for the complete lifecycle so SDK clients, compiled graphs,
checkpointers, and harness databases can remain live safely. NeMo Fabric sends the
resolved configuration and capability plan during `start`. Each subsequent
`invoke` wire payload contains only `runtime_context` and `request`, and the
helper validates those values and calls `AdapterRuntime.invoke(request,
context)` with typed models. It also requires a typed `AgentRunResult`. An
adapter that needs configuration during invocation retains it as runtime-owned
state during `start`. Adapter stdout is reserved for the protocol; diagnostics
are redirected to stderr. A host crash or protocol timeout terminates that
runtime.

## Relay Request Correlation

In-process Relay SDK adapters that own their Agent scope can use
`nemo_fabric_adapters.common.utils.relay_request_context()` around each Agent
scope. A UUID `RuntimeContext.request_id` becomes the propagated Relay root;
other request IDs remain available as `nemo_fabric_request_id` metadata. The
helper always preserves that metadata so Relay-backed streaming can identify
the active turn.

Pass `request.relay_session_root` as the helper's `session_root` to let a
caller group invocations into one Relay session. SDK callers set
`RunRequest.relay_session_root`; core forwards it as the typed
`AgentRunRequest.relay_session_root` field. Context keys do not control Relay
propagation.
When that value is a UUID string Relay accepts, it becomes the propagated root
and is recorded as `nemo_fabric_session_root` metadata. A UUID request ID stays
the parent; otherwise the session root is also the parent. Relay derives ATIF
`session_id` from the root, so invocations that share a session root export
separate trajectories under one session. A missing value, a non-string, a
string that is not a UUID, the nil UUID, or a UUID whose final eight bytes are
zero falls back to the request root without an error.

This helper does not apply to adapters that send telemetry through an external
Relay gateway or whose upstream integration creates an isolated scope context.
Those adapters retain their native session correlation until their Relay
boundary accepts a per-turn propagation context.
