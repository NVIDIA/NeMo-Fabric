<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Fabric Kilo Code Adapter

This package provides the Kilo Code harness adapter for NVIDIA NeMo Fabric.
One persistent Node.js adapter process starts and manages a separate, isolated
`kilo serve` process for each NeMo Fabric runtime. The adapter communicates with
that server over loopback HTTP through `@kilocode/sdk`.

## Install the Adapter

Install Node.js 22.19 or later. Then install the adapter and its exact-pinned,
consumer-managed Kilo Code packages in the project that owns the NeMo Fabric
configuration:

```bash
npm install --save-exact nemo-fabric-adapters-kilo @kilocode/cli@7.7.12 @kilocode/sdk@7.7.12
```

The Kilo CLI and SDK are optional peers. Installing the adapter alone does not
install them. Starting the adapter without the compatible packages reports
`kilo_harness_unavailable`.

## Supported Configuration

The adapter supports the following normalized configuration:

- **Models:** Selects the `default` role, or the only configured role. It maps
  `provider`, `model`, `api_key_env`, `base_url`, `temperature`, and `top_p`.
- **Instructions:** Supports `instructions.system` with `mode: replace`.
- **Runtime:** Maps a positive `runtime.max_turns` to Kilo Code's build-agent
  step limit.
- **Tools:** Maps native Kilo Code tool names in `tools.enabled` and
  `tools.blocked`. Tool definitions are unsupported.
- **Skills:** Each `skills.paths` entry must be a directory containing
  `SKILL.md`. Startup verifies that Kilo Code loaded every configured skill.
- **MCP:** Supports stdio, streamable HTTP, and SSE server configurations.
  Startup requires every configured server to report `connected`.

### Model Endpoints

When `base_url` is set, it must identify an OpenAI-compatible model-provider
endpoint. It is not the address of the Kilo Code server. Without `base_url`,
the configured provider and model must be supported by Kilo Code.

### MCP Configuration

Remote MCP endpoints require HTTPS except for loopback development endpoints.
They may define headers but not process arguments or environment variables.
Stdio servers may define arguments and environment variables but not headers.

Header values may reference `${NAME}`. Resolution checks NeMo Fabric
`environment.env` first and then the parent process environment. Normalized MCP
authentication and per-server tool filters are unsupported.

### Runtime Behavior

The adapter accepts plain-text input and returns the terminal assistant text in
`output.response`. Ordered invocations reuse one warm Kilo Code session and its
conversation history. A prompt has a 30-minute deadline; on timeout the adapter
asks Kilo Code to abort the prompt and reports `kilo_prompt_timeout`.

The adapter disables project Kilo Code configuration and uses temporary Kilo
and XDG directories. The child process receives required operating-system
variables, explicit NeMo Fabric `environment.env` values, and the selected model
credential. Other parent variables are not forwarded.

The adapter is non-interactive. It denies follow-up questions, access outside
the workspace, doom-loop continuation, and direct reads of `.env` files. These
permission rules are not a filesystem sandbox and cannot prevent an enabled
shell tool from reading workspace files.

`models.max_tokens`, provider-specific model settings, append-mode system
instructions, streaming, cancellation, service mode, and Relay telemetry are
unsupported.

## Understand the Runtime Lifecycle

The Kilo Code harness is not embedded in the adapter process. `start` launches
`kilo serve` as a separate child process on an available loopback port and
creates one session. `stop` deletes the session, terminates that server, and
removes its temporary profile.

```mermaid
flowchart TB
  Fabric["NeMo Fabric runtime"]

  subgraph AdapterProcess["Node.js adapter process"]
    Adapter["Kilo Code adapter"]
    SDK["@kilocode/sdk client"]
    Adapter --> SDK
  end

  subgraph KiloProcess["Separate Kilo Code server process"]
    Server["kilo serve<br/>loopback only"]
    Session["One warm Kilo Code session"]
    Server --> Session
  end

  Fabric -->|"Adapter contract<br/>NDJSON over stdio"| Adapter
  Adapter -.->|"Starts, configures, and stops"| Server
  SDK -->|"HTTP over loopback"| Server
  Session -->|"Model requests"| Provider["Model-provider endpoint"]
  Server -->|"stdio, HTTP, or SSE"| MCP["Configured MCP servers"]
```

For installation, configuration, and operational guidance, see the
[Kilo Code integration guide](../../../docs/integrations/harness/kilo.mdx).
