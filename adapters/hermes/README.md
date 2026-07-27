<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Fabric Hermes Agent Adapter

This adapter runs Hermes Agent through its Python SDK when telemetry is disabled.
When NeMo Relay telemetry is enabled, the same adapter uses the public
`nemo-relay run` CLI so Relay owns the loopback gateway and Hermes child process.
There is no user-selectable launch mode.

## Install

To install just the Hermes Agent adapter by itself:

```bash
pip install "nemo-fabric[hermes]"
```

To install the Hermes Agent adapter along with the NeMo Fabric Runtime:

```bash
pip install "nemo-fabric[hermes, runtime]"
```

To install the Hermes Agent adapter along with a compatible version of Hermes Agent:

```bash
pip install "nemo-fabric[hermes, hermes-agent]"
```

## What It Maps

The adapter receives a normalized payload from NeMo Fabric and materializes a native Hermes Agent configuration for:

- model provider, model name, base URL, temperature, and token settings;
- workspace and terminal settings;
- NeMo Fabric skills as external skill directories for Hermes Agent;
- NeMo Fabric MCP servers as Hermes Agent MCP server config;
- `tools.blocked` as disabled toolsets for Hermes Agent, unioned with
  `harness.settings.disabled_toolsets`;
- optional NeMo Relay 0.6 telemetry configuration.

`hermes_home` configures a base directory. The adapter creates a child under
`runtimes/<runtime_id>` so invocations in one NeMo Fabric runtime share Hermes Agent state
without sharing config or the session database with another runtime.

## Execution Model

For a runtime without Relay, the adapter constructs one Hermes Agent `AIAgent`
and opens one `SessionDB`. Ordered `Runtime.invoke(...)` calls reuse those
native objects and pass the prior transcript back to `run_conversation(...)`.

For a Relay-enabled runtime, Fabric writes a Hermes config that excludes the
native `observability/nemo_relay` plugin. Each invocation gets a distinct
directory containing colocated `config.toml` and `plugins.toml`, then executes:

```text
nemo-relay run --config <config.toml> --agent hermes -- <hermes chat args>
```

The Relay CLI owns gateway startup, the Hermes process, telemetry flush, and
configuration-overlay restoration. Fabric sends cancellation to the whole
process group and bounds captured stdout/stderr. The persistent Fabric runtime
is mapped to a stable Hermes session, so separate invocation processes retain
conversation history. Relay mode requires NeMo Relay `>=0.6,<0.7`, Hermes Agent
`>=0.18.2,<0.19`, and an OpenAI-compatible upstream endpoint.

The Relay gateway's `openai.chat_completions` scopes are the canonical source
for model requests, cost, and cache accounting. Hermes hook scopes provide
lifecycle and tool-call context only and must not be counted as another model
request. Adapter output records this as `relay_runtime.model_event_source` and
`relay_runtime.hook_event_policy`.

The Relay/Hermes process receives an allowlisted child environment: portable
process settings, `harness.settings.env`, the selected model credential, and
environment variables explicitly referenced by Relay sinks. Unrelated host
credentials are not inherited.

Hermes 0.18.x accepts the non-interactive query only as an argument. Fabric
redacts that argument from results and logs, but it can remain visible to
same-host process inspection while the command is running. Do not place
credentials in prompts on a shared host until Hermes exposes a versioned
stdin/file-descriptor query surface.

## Maintaining The Adapter

Keep `fabric-adapter.json` aligned with the Python implementation:

- `contract_version` must match the adapter contract supported by NeMo Fabric core.
- `adapter_id` is the stable id selected by `harness.adapter_id`.
- `adapter_kind` is `python` because NeMo Fabric can invoke it through Python.
- `runner.module` names the persistent host module that NeMo Fabric invokes with
  `python -m`.
- `requirements` powers `fabric doctor`; keep required env vars, binaries, or
  packages current.
- `config.accepts` must match the NeMo Fabric sections this adapter maps into Hermes Agent.
- `telemetry.providers` declares provider-specific outputs and integration modes
  the adapter can produce or forward.

Do not put end-user agent settings in this directory. Users vary harness,
model, skills, MCP, tools, telemetry, and runtime behavior through complete
typed `FabricConfig` values and ordinary Python composition. The adapter
descriptor describes adapter capabilities; it is not an agent configuration.
Add descriptor fields only when NeMo Fabric core or the SDK actually uses them.
