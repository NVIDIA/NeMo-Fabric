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

Refer to the [NeMo Fabric documentation](https://docs.nvidia.com/nemo/fabric)
for adapter and configuration guidance. Source code is available in the
[NVIDIA NeMo Fabric repository](https://github.com/NVIDIA/nemo-fabric/).

## Persistent Local Hosts

Every local Process or Python adapter implements the ordered persistent-host
wire protocol. Python adapters can use
`nemo_fabric_adapters.common.lifecycle`. Supply a factory that creates one
adapter-owned runtime with asynchronous `start`, `invoke`, and `stop` methods:

```python
from nemo_fabric_adapters.common import lifecycle


class AdapterRuntime:
    async def start(self, payload):
        self.client = await connect_client(payload)

    async def invoke(self, payload):
        return await self.client.run(payload["request"]["input"])

    async def stop(self):
        await self.client.close()


lifecycle.serve(AdapterRuntime)
```

If the adapter descriptor declares `capabilities.streaming`, the runtime must
also implement native OpenAI Chat Completions streaming:

```python
class AdapterRuntime:
    async def invoke_openai_stream(self, payload, emit):
        async for chunk in self.client.stream(payload["request"]["input"]):
            await emit(chunk)
        return {"answer": "..."}
```

The optional method must have the signature
`async invoke_openai_stream(payload, emit)`. Execute the target exactly once,
await `emit(chunk)` only for mappings in the
`openai.chat_completions.chunk/v1` profile, and return one JSON-compatible
terminal outcome. Each chunk requires non-empty `id` and `model` strings, a
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
[`openai-stream-record.schema.json`](https://github.com/NVIDIA/NeMo-Fabric/blob/main/schemas/adapter-contract/legacy/openai-stream-record.schema.json)
chunk and explicit-end envelopes.

Adapters whose descriptor sets `config.input` to `agent_config` can ask the
host to validate the southbound contract before `start`:

```python
from nemo_fabric_adapter_contract.models import AgentConfig

lifecycle.serve(AdapterRuntime, config_loader=AgentConfig.from_mapping)
```

The runtime then receives an `AgentConfig` instance in `payload["config"]`.
Omitting `config_loader` preserves the legacy `FabricConfig` mapping.

NeMo Fabric calls the factory once per local host to create one runtime instance and
serializes invocations through that instance. The host keeps one event loop
alive for the complete lifecycle so SDK clients, compiled graphs,
checkpointers, and harness databases can remain live safely. NeMo Fabric sends the
resolved configuration and capability plan during `start`. Each subsequent
`invoke` wire payload contains only `runtime_context` and `request`, and the
helper passes that payload to `AdapterRuntime.invoke` unchanged. An adapter that
needs configuration during invocation retains it as runtime-owned state during
`start`. Adapter stdout is reserved for the protocol; diagnostics are redirected
to stderr. A host crash or protocol timeout terminates that runtime.
