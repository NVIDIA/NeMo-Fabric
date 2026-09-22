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
- Live model-turn ATOF records for successful Relay redirects through the
  default embedded NeMo Fabric collector. Startup `model_redirect` marks remain
  in the configured Relay ATOF artifacts and are not included in per-invocation
  `invoke_stream()` records
- Ordered plain-text invocations with a `{ "response": "..." }` terminal
  output, Relay runtime details, and collected ATOF artifacts

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

Relay-enabled Pi runs require `nemo-relay>=0.9.0,<0.10.0` on `PATH`. Install it
separately from the npm adapter:

```bash
pip install "nemo-relay-cli-bin>=0.9.0,<0.10.0"
```

To select a specific Relay executable instead of relying on `PATH`, set
`FABRIC_NEMO_RELAY_COMMAND` to its absolute path:

```bash
FABRIC_NEMO_RELAY_COMMAND="/absolute/path/to/nemo-relay" uv run python your_app.py
```

The adapter does not bundle the Relay Pi extension. Obtain the
[`crates/cli/assets/pi-extension`](https://github.com/NVIDIA/NeMo-Relay/tree/0.9.0/crates/cli/assets/pi-extension)
directory from the Relay 0.9 release and configure its path as described in the
next section.

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

The adapter supports one Relay-enabled Pi runtime per adapter process because
the Relay 0.9 extension receives its gateway and upstream configuration through
process environment variables. NeMo Fabric starts each runtime in a separate
adapter process, so concurrent NeMo Fabric runtimes remain isolated. Direct
embedders must likewise place concurrent Relay-enabled Pi runtimes in separate
processes.

Local ATIF trajectories are finalized only after the Pi session closes and are
therefore not included in `relay_artifacts`. The adapter does not wait for local
ATIF during invocation; it collects per-invocation artifacts such as ATOF and
returns the result. After runtime shutdown, retrieve the finalized file directly
from the ATIF output directory. For the configuration above and the default
filename template, it is written to
`./artifacts/relay/<runtime_id>/trajectory-<session_id>.atif.json`. Invocation
results do not prevent subsequent turns.

Session, turn, and tool telemetry does not depend on model redirection. Model
telemetry is available only when Relay supports the selected model API and the
gateway upstream matches the model endpoint. Relay records a skipped redirect
as a `model_redirect` mark with the reason in configured ATOF artifacts. Pi
emits its startup redirect marks before NeMo Fabric registers an invocation, so
they are not included in the per-invocation records from `invoke_stream()`.

Install the matching collector for the embedded streaming path:

```bash
pip install "nemo-fabric[streaming]"
```

Start the runtime with streaming enabled to consume live model-turn ATOF
records for successful Relay redirects in one Pi invocation:

```python
from nemo_fabric import Fabric

async with await Fabric().start_runtime(config, streaming=True) as runtime:
    stream = runtime.invoke_stream(input="Review the latest patch")
    async for record in stream:
        print(record)
    result = await stream.result()
```

The terminal `RunResult` remains separate from the ATOF records. Fully consume
each stream, or call `await stream.aclose()` if iteration stops early, before
starting another invocation; the same runtime can then alternate
`invoke_stream()` and `invoke()` calls. The embedded collector serializes both
methods behind one Pi invocation lease. Streaming capture begins at the first
Pi `turn_start` and closes at `agent_settled`. If Relay output is interrupted or
late, the collector waits for a bounded interval, then admits the next native
invocation and uses Pi's cumulative turn count to discard ambiguous delayed
records until a higher `turn_start` arrives. This can thin the ATOF stream but
does not block later invocations. Increase `completion_wait_timeout` from its
one-second default when Relay delivery can take longer. Use the default embedded
collector for Pi streaming. The Pi extension does not attach NeMo Fabric
request IDs, so
`start_runtime(..., streaming=True, launch_collector=False)` cannot correlate
its records through an externally managed collector.

This Relay-backed path runs the adapter's ordinary `invoke` operation. It is
independent of native OpenAI streaming, so the adapter descriptor's
`capabilities.streaming` value remains `false`.

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
  --stream \
  --pi-relay-extension-path /path/to/NeMo-Relay/crates/cli/assets/pi-extension \
  --input "Review calculator.py"
```

The command collects per-invocation model-turn ATOF records for successful Relay
redirects, then prints one JSON document containing `atof_records` and the
separate terminal `result`. Redirect-decision marks remain in configured Relay
ATOF artifacts; Pi's startup marks are not included in `atof_records`. MCP is
not currently supported.

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
