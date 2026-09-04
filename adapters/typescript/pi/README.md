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
- NeMo Relay 0.9 telemetry through a runtime-owned gateway and an explicitly
  configured Relay Pi extension
- Ordered plain-text invocations with a `{ "response": "..." }` terminal
  output, Relay runtime details, and collected ATOF and ATIF artifacts

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

### Install NeMo Relay

Relay-enabled Pi runs require `nemo-relay>=0.9.0,<0.10.0` on `PATH`. The Relay
0.9 CLI is not yet published to PyPI. Clone a matching NeMo Relay 0.9 source
checkout and install the CLI separately from the npm adapter:

```bash
git clone https://github.com/NVIDIA/NeMo-Relay.git
cd NeMo-Relay
git checkout 30b684dbb09231ee956d40abad9af253596a81ad
cargo install --path crates/cli --locked
```

The adapter does not bundle the Relay Pi extension. Obtain the extension from
that same source revision, such as
[`crates/cli/assets/pi-extension`](https://github.com/NVIDIA/NeMo-Relay/tree/30b684dbb09231ee956d40abad9af253596a81ad/crates/cli/assets/pi-extension),
and configure its path as described in the next section.

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

## Configure NeMo Relay

Enable Relay with the standard NeMo Fabric configuration and provide the Relay
Pi extension as an adapter setting:

```python
config.runtime.artifacts = "./artifacts/pi"
config.enable_relay(output_dir="./artifacts/relay")
config.harness.settings["relay_extension_path"] = (
    "/path/to/NeMo-Relay/crates/cli/assets/pi-extension"
)
```

Relay requires `runtime.artifacts` so NeMo Fabric can create the runtime-owned
configuration passed to the adapter. The extension path can be absolute or
relative to `environment.workspace`. It can identify a JavaScript or TypeScript
file or a Pi extension package directory. Unlike user-configured Pi extensions,
the Relay extension does not need to remain inside `environment.workspace` when
an absolute path is used.

When the runtime starts, the adapter validates the Relay 0.9 CLI, writes an
explicit `plugins.toml`, starts a loopback gateway, and loads the extension into
the isolated Pi session. The result includes `relay_runtime` and
`relay_artifacts` in `output`. The gateway can produce ATOF, ATIF,
OpenTelemetry, and OpenInference output from the Relay observability
configuration.

The adapter waits up to five seconds for a local ATIF trajectory to finalize.
If it does not finalize, the invocation result remains usable, a warning is
written to the adapter log, and `relay_artifacts` omits ATIF entries while
retaining any ATOF files that were written. The runtime remains available for
subsequent turns.

Session, turn, and tool telemetry does not depend on model redirection. Model
telemetry is available only when Relay supports the selected model API and the
gateway upstream matches the model endpoint. A skipped redirect is recorded as
a `model_redirect` mark with the reason. Relay-backed
`Runtime.invoke_stream()` correlation is not yet supported for Pi.

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
live NVIDIA-backed run command. For a Relay-enabled Pi run, pass the extension
path explicitly:

```bash
.venv/bin/python -m examples.code_review_agent \
  --variant pi \
  --relay \
  --pi-relay-extension-path /path/to/NeMo-Relay/crates/cli/assets/pi-extension \
  --input "Review calculator.py"
```

MCP is not currently supported. Do not combine the Pi variant with `--stream`.

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
