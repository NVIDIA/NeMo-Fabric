<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Fabric Cline Adapter

This package provides the bundled `nvidia.fabric.cline` process adapter. It
uses Cline's supported TypeScript SDK and keeps one Cline session alive for the
lifetime of each NeMo Fabric runtime.

Install Cline's supported SDK harness separately, then install the NeMo Fabric
adapter in the same Node.js project:

```bash
npm install --save-exact @cline/sdk@0.0.83
npm install nemo-fabric-adapters-cline
```

The SDK is not an adapter package dependency, so installing the NeMo Fabric
adapter does not pull Cline or its dependency graph into the project. Starting
the adapter without the supported SDK returns `cline_harness_unavailable` with
the Cline installation command.

For source development, install and build the focused workspace from the
repository root:

```bash
just install-typescript-cline
npm install --prefix adapters/typescript \
  --workspace nemo-fabric-adapters-cline \
  --include-workspace-root \
  --no-save \
  --package-lock=false \
  --ignore-scripts \
  --no-audit \
  --no-fund \
  @cline/sdk@0.0.83
npm run build --prefix adapter-contract/typescript
npm run build --prefix adapters/typescript --workspace nemo-fabric-adapters-common
npm run build --prefix adapters/typescript --workspace nemo-fabric-adapters-cline
```

Point `DiscoveryConfig.local_paths` at the installed descriptor:

```python
from nemo_fabric import DiscoveryConfig, FabricConfig, HarnessConfig
from nemo_fabric import MetadataConfig, ModelConfig, ToolsConfig

config = FabricConfig(
    metadata=MetadataConfig(name="cline-agent"),
    discovery=DiscoveryConfig(
        local_paths=[
            "./node_modules/nemo-fabric-adapters-cline/cline.fabric-adapter.json"
        ]
    ),
    harness=HarnessConfig(adapter_id="nvidia.fabric.cline"),
    models={
        "default": ModelConfig(
            provider="nvidia",
            model="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
            api_key_env="NVIDIA_API_KEY",
            base_url="https://integrate.api.nvidia.com/v1",
        )
    },
    tools=ToolsConfig(enabled=["read_files", "skills"]),
)
config.add_skill_path("./skills/code-review")
```

For a source build, use
`adapters/typescript/cline/cline.fabric-adapter.json`. Relative skill paths
resolve from the `base_dir` passed to `plan()`, `doctor()`, or `run()`.

The initial adapter supports a configured model, an API-key environment
reference, an optional model base URL, replacement system instructions, Cline
built-in tool enable/block policy, runtime-scoped native skills, and
unauthenticated stdio, SSE, and streamable HTTP MCP servers. It does not
advertise streaming, telemetry, or Relay integration.

Each NeMo Fabric runtime owns one local Cline core and one interactive Cline
session. The first invocation calls `ClineCore.start()`. Later invocations call
`ClineCore.send()` with the retained session ID, and `stop()` closes the
session, disposes the core, and removes runtime-scoped state. Independent
NeMo Fabric runtimes do not share Cline state.

The adapter accepts no Cline-specific `harness.settings`. Unsupported
normalized fields, unknown settings, unknown built-in tool names, authenticated
MCP configuration, and MCP tool allowlists or blocklists fail instead of being
ignored.
