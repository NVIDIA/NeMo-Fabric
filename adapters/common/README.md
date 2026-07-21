<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Fabric Adapter Utilities

`nemo-fabric-adapters-common` provides shared Python helpers for NVIDIA NeMo
Fabric adapter implementations. NeMo Fabric adapter packages normally install
this package as a dependency.

Install the package directly when developing an adapter:

```bash
pip install nemo-fabric-adapters-common
```

Alternately through the NeMo Fabric metapackage:

```bash
pip install "nemo-fabric[adapters-common]"
```

## Persistent Local Hosts

Every local Process or Python adapter declares `runtime.local_host` in
`fabric-adapter.json` and implements the ordered local-host wire protocol.
Python adapters can use `nemo_fabric_adapters.common.lifecycle`. Supply a
factory that creates one adapter-owned runtime with asynchronous `start`,
`invoke`, and `stop` methods:

`runtime.local_host` is an unversioned capability marker, not a separate
adapter contract. The nesting is intentionally provisional while local hosting
is the only supported mode; a shared runtime protocol can replace it when
remote adapter hosting is introduced.

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

Fabric calls the factory once per local host to create one runtime instance and
serializes invocations through that instance. The host keeps one event loop
alive for the complete lifecycle so SDK clients, compiled graphs,
checkpointers, and harness databases can remain live safely. Fabric sends the
resolved configuration and capability plan during `start`. Each subsequent
`invoke` wire payload contains only `runtime_context` and `request`; the helper
retains the start payload and supplies a merged view to `AdapterRuntime.invoke`.
Adapter stdout is reserved for the protocol; diagnostics are redirected to
stderr. A host crash or protocol timeout is terminal for that runtime. Fabric
does not fall back to a new process.

Refer to the [NeMo Fabric documentation](https://nvidia-nemo-fabric.docs.buildwithfern.com/nemo/fabric)
for adapter and configuration guidance. Source code is available in the
[NVIDIA NeMo Fabric repository](https://github.com/NVIDIA/nemo-fabric/).
