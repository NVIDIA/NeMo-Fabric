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
session. Runtime startup calls `ClineCore.start()` without a prompt. Every invocation calls
`ClineCore.send()` with the retained session ID, and `stop()` closes the
session, disposes the core, and removes runtime-scoped state. Independent
NeMo Fabric runtimes do not share Cline state.

## Usage evidence

Results from `ClineCore.send()` retain positive input, output, cache-read, and
cache-write counts in `RunResult.usage.metadata["nvidia.fabric.cline.usage_evidence.v1"]`.
The `observed` object holds these Cline counts. The `gaps` list identifies
missing or unverified usage. Cline can fill unavailable counts with zero, so a
zero in the Cline SDK result does not prove that the provider reported zero.

When the selected provider is `nvidia` and `base_url` is set, the adapter captures
model requests through Cline's configured HTTP hook during each send. It sets
input, output, and total token counts only when every captured request returns
valid terminal usage that does not conflict with nonzero SDK counts. A zero
reported by the provider stays zero. The OpenAI-compatible input count includes
cache reads, and its output count includes reasoning tokens. The adapter reports
cache reads only when the provider supplies that count. If a request fails,
omits usage, or follows an unexpected path, complete counts remain
unknown and the observations stay available. Other Cline providers return
observations with unverified coverage. Provider-reported cost remains unknown.
Harbor uses complete counts when present and retains observations in NeMo Fabric
result metadata.

The adapter accepts no Cline-specific `harness.settings`. Unsupported
normalized fields, unknown settings, unknown built-in tool names, authenticated
MCP configuration, and MCP tool allowlists or blocklists fail instead of being
ignored.

## Runtime Health

`Fabric.check()` supports `live`, `active`, and `ready` for the selected Cline runtime. Runtime startup initializes an interactive native session without a prompt or model request. Checks use a separate health connection and can finish while the first or any later invocation waits for a model response.

- `live` verifies the owned local host still contains the same resident session and native agent identity, and the native agent has not shut down.
- `active` also observes `SessionRuntime.canStartRun()` and the native running flag. An active run counts as busy and remains healthy.
- `ready` also obtains a fresh `ClineCore.get()` response for that resident session and validates the native session identity and live status.

These observations are qualified against `@cline/sdk@0.0.83` and `@cline/core@0.0.83`. The adapter guards the local host's resident-session map and native agent closure fields because `ClineCore.get()` can return historical rows after shutdown. An incompatible native layout returns `unsupported/capabilities_unverified`. Session replacement or closure during a probe invalidates the observation. Checks return fixed safe evidence fields and exclude native snapshots, configuration, prompts, and credentials.

One caller deadline covers the health request. A timeout leaves the invocation and session running. At most one native snapshot read remains pending after a timeout; later checks never reuse an earlier response. Passing health establishes local session and request-interface availability. Model access, credential validity, and tool-server availability require separate validation.

Run the native qualification test after installing the supported Cline SDK:

```bash
node --test adapters/typescript/cline/test/native-health.mjs
```

Refer to the [SDK health-check overview](../../../docs/sdk/health-checks.mdx) for shared behavior and a comparison of native evidence across adapters.
