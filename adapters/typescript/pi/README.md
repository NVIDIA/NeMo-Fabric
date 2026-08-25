<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Fabric Pi Adapter

This package provides a Pi harness adapter for NVIDIA NeMo Fabric. It embeds the
Pi SDK in the Node process of the adapter and maps one NeMo Fabric runtime to one
in-memory Pi session.

The adapter supports:

- One explicit Pi-known model selected from the `default` role or the sole
  configured role
- Runtime API-key credentials named by `models.<role>.api_key_env`
- Optional `models.<role>.base_url`
- Optional replacement system instructions
- Tool allow and block policy
- NeMo Fabric custom tools loaded through normalized `tools.definitions`
- Explicit normalized `skills.paths`
- Explicit local `.ts` or `.js` extension files contained by the NeMo Fabric
  workspace
- Slash commands registered by those explicit extensions
- Ordered plain-text invocations with a `{ "response": "..." }` terminal
  output

Ambient Pi settings, context files, packages, extensions, skills, prompts,
themes, model files, credentials, and session files are disabled. Explicitly
configured extensions are trusted code.

## Install the Adapter

Pi 0.84.x requires Node.js 22.19.0 or newer.

### Install for Consumers

Install the adapter in the project that owns the NeMo Fabric configuration,
then install the compatible Pi SDK harness version selected by that project:

```bash
npm install nemo-fabric-adapters-pi
npm install @earendil-works/pi-ai@^0.84.2 @earendil-works/pi-coding-agent@^0.84.2
```

The adapter declares the Pi packages as optional peers. Installing the adapter
alone does not install a harness, and starting it without compatible Pi packages
returns `pi_harness_unavailable`.

### Install for Source Development

For focused Pi development in the NeMo Fabric source tree, install the Pi
adapter workspace and its pinned harness from the repository root:

```bash
just install-typescript-pi
```

To install and build all maintained TypeScript workspaces instead, run:

```bash
just build-typescript
```

The full build installs its own dependencies, so you do not need to run
`just install-typescript-pi` first.

## Configure the Adapter

The npm package includes its adapter descriptor as `pi.fabric-adapter.json`.
NeMo Fabric descriptor discovery is path-based. Point discovery at the installed
descriptor and select the Pi adapter with the Python SDK:

```python
from nemo_fabric import DiscoveryConfig, HarnessConfig

discovery = DiscoveryConfig(
    local_paths=[
        "./node_modules/nemo-fabric-adapters-pi/pi.fabric-adapter.json"
    ]
)
harness = HarnessConfig(adapter_id="nvidia.fabric.pi")
```

For a source build, set `discovery.local_paths` to
`adapters/typescript/pi/pi.fabric-adapter.json` instead.

## Custom Tool Modules

The adapter accepts custom tools with `kind: "module"`. The `ref` is a
workspace-relative JavaScript or TypeScript file with an optional named export,
for example `tools/review.ts#createTool`. Without a fragment, the adapter uses
the default export.

The export is called with `{ name, settings, workspace }` and must return a Pi
`ToolDefinition` whose name matches the normalized `tools.definitions` key.
Tool modules are trusted executable code. Their real paths must remain inside
the NeMo Fabric workspace, and their names may not replace Pi built-in or
extension tools. The following configuration registers a custom tool module:

```json
{
  "tools": {
    "definitions": {
      "review_context": {
        "kind": "module",
        "ref": "tools/review-context.ts#createTool",
        "settings": {"format": "brief"}
      }
    },
    "enabled": ["read", "review_context"]
  }
}
```

## Run the Code-Review Example

The maintained code-review example exercises the Pi adapter with an explicit
NeMo Fabric skill and an example-specific tool policy. After building the
TypeScript packages, inspect the plan from the repository root:

```bash
.venv/bin/python -m examples.code_review_agent --variant pi --plan
```

Refer to the
[code-review example](../../../examples/code_review_agent/README.md) for the
live NVIDIA-backed run command. Relay and MCP are not currently supported.

## Dependency Rationale

`@earendil-works/pi-coding-agent` provides the native Pi session, resources,
skills, extensions, and tools. Using the SDK keeps these integration points in
process; maintaining a second JSON-RPC translation was rejected for the bundled
adapter. `@earendil-works/pi-ai` supplies Pi's model catalog and credential
store, which the coding-agent SDK expects. Both packages are optional peer
dependencies so deployments control the compatible harness version. Exact
0.84.2 development dependencies keep repository builds and tests reproducible.

`jiti` loads explicitly configured, trusted JavaScript and TypeScript tool
modules. Native Node.js loading cannot execute TypeScript modules, while a
custom transpiler would duplicate this focused loader. The adapter contract
package supplies normalized types, and `nemo-fabric-adapters-common` supplies
the shared lifecycle host; copying either surface into the Pi package would
create divergent implementations.

`typescript` and `@types/node` are exact-pinned build inputs and are absent from
the published production dependency graph.
