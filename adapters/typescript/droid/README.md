<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Fabric Factory Droid Adapter

This package provides the bundled `nvidia.fabric.droid` process adapter. It
uses Factory's supported TypeScript SDK and retains one Droid session for the
lifetime of each NeMo Fabric runtime.

Install the supported SDK and caller-owned Droid CLI separately, then install
the adapter in the same Node.js project:

```bash
npm install --global --save-exact droid@0.233.0
npm install --save-exact @factory/droid-sdk@0.9.1
npm install nemo-fabric-adapters-droid
```

The adapter never installs the Droid CLI at runtime. A missing SDK returns
`droid_harness_unavailable`; a missing CLI returns `droid_cli_unavailable`.
Set the Factory API key referenced by the Fabric model configuration before
starting a runtime.

For source development, install and build the focused workspace from the
repository root:

```bash
just install-typescript-droid
npm run build --prefix adapter-contract/typescript
npm run build --prefix adapters/typescript --workspace nemo-fabric-adapters-common
npm run build --prefix adapters/typescript --workspace nemo-fabric-adapters-droid
```

Point `DiscoveryConfig.local_paths` at the installed descriptor:

```python
from nemo_fabric import DiscoveryConfig, FabricConfig, HarnessConfig
from nemo_fabric import InstructionConfig, InstructionsConfig
from nemo_fabric import MetadataConfig, ModelConfig, ToolsConfig

config = FabricConfig(
    metadata=MetadataConfig(name="droid-agent"),
    discovery=DiscoveryConfig(
        local_paths=[
            "./node_modules/nemo-fabric-adapters-droid/droid.fabric-adapter.json"
        ]
    ),
    harness=HarnessConfig(adapter_id="nvidia.fabric.droid"),
    models={
        "default": ModelConfig(
            provider="factory",
            model="auto",
            api_key_env="FACTORY_API_KEY",
        )
    },
    instructions=InstructionsConfig(
        system=InstructionConfig(
            content="Review code for correctness risks.",
            mode="append",
        )
    ),
    tools=ToolsConfig(enabled=["Read", "Grep", "Glob"]),
)
config.add_skill_path("./skills/code-review")
```

For a source build, use
`adapters/typescript/droid/droid.fabric-adapter.json`.

The initial adapter supports Factory-managed model IDs, a Factory API-key
environment reference, replacement or appended system instructions, exact
Droid built-in tool enable/block policy, normalized skill paths, and
unauthenticated stdio, SSE, and streamable HTTP MCP servers. Tool values are
Droid tool IDs; inspect the IDs
available to a model with `droid exec --model MODEL_ID --list-tools
--output-format json`.

Each NeMo Fabric runtime owns one SDK session. Ordered invocations call
`session.stream()` on that retained session, so the caller does not replay
history. `stop()` closes the session and its CLI subprocess. Independent
runtimes create separate sessions.

Configured skill directories are copied into a runtime-scoped temporary home
using Droid's compatible personal `.agents/skills` layout. The adapter verifies
them with `listSkills()` and removes the temporary home during shutdown. This
keeps Fabric-provided skills isolated without modifying the workspace.
The isolated profile preserves Factory `settings.json` for custom model
definitions but does not copy ambient MCP configuration or authentication
state.

The adapter intentionally does not accept normalized custom model endpoints.
Droid custom endpoints are configured outside a session in Droid settings. It
also does not advertise streaming, telemetry, or Relay integration. Unsupported
normalized fields, MCP authentication and per-server tool filters, unknown
settings, and unknown built-in tool IDs fail instead of being ignored.
