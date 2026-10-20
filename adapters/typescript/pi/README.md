<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Fabric Pi Adapter

This package provides a Pi harness adapter for NVIDIA NeMo Fabric. It embeds the
Pi SDK in the Node process of the adapter and maps one NeMo Fabric runtime to one
in-memory Pi session.

The adapter supports:

- One explicit selected model from the `default` role or the sole configured
  role: either a model Pi already ships, or a model served by an
  OpenAI-compatible gateway Pi does not ship (set its wire protocol with the
  model's `api` field)
- Runtime API-key credentials named by `models.<role>.api_key_env`
- Optional `models.<role>.base_url` (the gateway endpoint, or an endpoint
  override for a Pi-known model)
- Optional Pi-specific model metadata set directly on the model
  (`api`, `context_window`, `max_tokens`, `cost`, `reasoning`, `input`, `name`);
  a field you set overrides, a field you omit keeps a Pi-known model's existing
  value. The adapter adds no defaults: a gateway model Pi does not ship must
  supply `api`, `base_url`, `context_window`, and `max_tokens`.
- Optional replacement system instructions
- Tool allow and block policy
- NeMo Fabric custom tools loaded through normalized `tools.definitions`
- Explicit normalized `skills.paths`
- Native stdio and streamable HTTP MCP servers with per-server tool filters
- Explicit local `.ts` or `.js` extension files contained by the NeMo Fabric
  workspace
- Slash commands registered by those explicit extensions
- NeMo Relay 0.9 or 0.10 telemetry through a runtime-owned gateway and an explicitly
  configured Relay Pi extension
- Live model-turn ATOF records for successful Relay redirects through the
  default embedded NeMo Fabric collector. Startup `model_redirect` marks remain
  in the configured Relay ATOF artifacts and are not included in per-invocation
  `invoke_stream()` records
- Ordered plain-text invocations with a `{ "response": "..." }` terminal
  output, Relay runtime details, and collected ATOF and OTLP artifacts

Ambient Pi settings, context files, packages, extensions, skills, prompts,
themes, model files, credentials, and session files are disabled. Explicitly
configured extensions are trusted code. The adapter raises the compaction
reserve toward the selected model's maximum output while retaining at least
half of the context window for input.

## Install the Adapter

Pi 1.0 requires Node.js 22.19.0 or newer.

### Install for Consumers

Install the adapter in the project that owns the NeMo Fabric configuration,
then install the compatible Pi SDK harness version selected by that project:

```bash
npm install nemo-fabric-adapters-pi
npm install @earendil-works/pi-ai@^1.0.0 @earendil-works/pi-coding-agent@^1.0.0 @earendil-works/pi-mcp@^1.0.0
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

Relay-enabled Pi runs require `nemo-relay>=0.9.0,<0.11.0` on `PATH`. Install it
separately from the npm adapter:

```bash
pip install "nemo-relay-cli-bin>=0.9.0,<0.11.0"
```

To select a specific Relay executable instead of relying on `PATH`, set
`FABRIC_NEMO_RELAY_COMMAND` to its absolute path:

```bash
FABRIC_NEMO_RELAY_COMMAND="/absolute/path/to/nemo-relay" uv run python your_app.py
```

The adapter does not bundle the Relay Pi extension. Obtain the
[`crates/cli/assets/pi-extension`](https://github.com/NVIDIA/NeMo-Relay/tree/0.10.0/crates/cli/assets/pi-extension)
directory from the Relay release that matches the CLI and configure its path as described in the
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

When the runtime starts, the adapter validates the Relay CLI version, writes an
explicit `plugins.toml`, starts a loopback gateway, and loads the extension into
the isolated Pi session. The result includes `relay_runtime` and
`relay_artifacts` in `output`. The gateway can produce ATOF, ATIF,
OpenTelemetry, and OpenInference output from the Relay observability
configuration.

The adapter supports one Relay-enabled Pi runtime per adapter process because
the Relay Pi extension receives its gateway and upstream configuration through
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

## Token Usage

Each result's `usage` sums the tokens Pi records during the invocation, counted
the way Pi's own usage totals count them: assistant messages, tool results that
report usage, finished compaction summaries, and usage entries other than cache
warm-ups, which leave `usage` unknown (see below). It holds `input_tokens` with
cache reads and cache writes included, `cached_input_tokens`, `output_tokens`,
and `total_tokens`. Pi computes its cost figures from the model catalog's
rates, so the adapter does not report them as `cost_usd`.

Pi fills a count that the provider did not report with 0, so the adapter reads
the provider's raw stream events for each assistant message and compaction
summary call and keeps only the counts the provider reported. A reported zero
counts as 0, including a report of all zeros. When a call left out its input
count, the result leaves out `input_tokens`, `cached_input_tokens`, and
`total_tokens`. Anthropic Messages usage reports input without the cache, so
its input count also needs both cache fields. Bedrock Converse also reports
input without the cache, but its API documents the cache fields as optional, so
its input count needs only `inputTokens` rather than going unknown on every
call that omits them. When a call left out its output count, the result leaves
out `output_tokens` and `total_tokens`; for Anthropic Messages, only a
`message_delta` reports the final output count. When a call sent no cache
details, the result leaves out `cached_input_tokens`. A wire protocol outside
Pi's built-in ones reports no counts the adapter can check, so its calls leave
the result without `usage`.

A failed call counts as 0 tokens only when the provider answered each of its
attempts with a 4xx or 5xx HTTP status, such as a rate limit (429), an overload
or a server error, before sending a successful response or any stream event.
A call that Pi retries after such an error keeps the result's `usage` known.
The adapter reads that status through the request's `fetch`, which the
`anthropic-messages`, `azure-openai-responses`, `mistral-conversations`,
`openai-codex-responses`, `openai-completions`, `openai-responses` and
`pi-messages` protocols use. A failed call of any other protocol, including the
Google protocols and Bedrock, or a Codex call sent over a WebSocket, leaves its
counts unknown. A call with an attempt that got no HTTP response, such as a
connection failure or a timeout before headers, leaves its counts unknown even
when a later attempt succeeds, because the provider may have billed the prompt.
A call that fails after its stream started can have billed tokens that were
never reported, so its counts stay unknown too.

When a compaction cuts inside a turn, Pi summarizes the earlier history and the
start of that turn in separate calls and reports only their combined usage. The
adapter counts each of those calls from its own report, after checking that the
calls it saw are the ones the compaction needed and that they add up to the
combined usage; otherwise the result has no `usage`.

A tool result's `usage` counts each count it holds: its input needs `input`,
`cacheRead`, and `cacheWrite`, its cache reads need `cacheRead`, and its output
needs `output`. A count the tool left out, or that is not a non-negative
safe integer, is left out of the result. A tool result without a `usage` field adds
nothing and leaves the result's `usage` unchanged. An extension's `message_end`
handler that replaces an assistant message's usage does not change which
counts the provider reported.

The result has no `usage` when Pi produced no assistant message, when a model
call reported none of the counts, when an assistant message was aborted, when a
compaction was aborted or reported no usage, or when Pi retried a compaction
summary call, because Pi keeps no usage for the failed attempt. It also has no
`usage` when Pi recorded a cache warm-up during the invocation: Pi replays a
warm-up outside the agent's stream function, so the adapter cannot tie its raw
report to the warm-up's usage entry.

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

## Configure MCP Servers

The adapter maps normalized stdio and streamable HTTP servers to Pi's native MCP extension. Stdio servers can define `args` and `env`, but not HTTP headers. Streamable HTTP servers can define `custom_headers`, but not process arguments or environment variables. Remote endpoints require HTTPS, except for loopback development endpoints.

Header values can reference `${NAME}`. Resolution checks the NeMo Fabric runtime environment first and then the adapter process environment. The adapter maps `allowed_tools` and `blocked_tools` to Pi's per-server tool exposure. Normalized MCP authentication and MCP extensions are not supported.

Only servers supplied by NeMo Fabric are loaded. Ambient Pi MCP configuration remains disabled.

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
ATOF artifacts; Pi's startup marks are not included in `atof_records`.

## Dependency Rationale

`@earendil-works/pi-coding-agent` provides the native Pi session, resources,
skills, extensions, and tools. Using the SDK keeps these integration points in
process; maintaining a second JSON-RPC translation was rejected for the bundled
adapter. `@earendil-works/pi-ai` supplies Pi's model catalog and credential
store, which the coding-agent SDK expects. Both packages are optional peer
dependencies so deployments control the compatible harness version. Exact
1.0.3 development dependencies keep repository builds and tests reproducible.

`jiti` loads explicitly configured, trusted JavaScript and TypeScript tool
modules. Native Node.js loading cannot execute TypeScript modules, while a
custom transpiler would duplicate this focused loader. The adapter contract
package supplies normalized types, and `nemo-fabric-adapters-common` supplies
the shared lifecycle host; copying either surface into the Pi package would
create divergent implementations.

`typescript` and `@types/node` are exact-pinned build inputs and are absent from
the published production dependency graph.
