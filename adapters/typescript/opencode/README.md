<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Fabric OpenCode Adapter

This package provides the OpenCode v2 harness adapter for NVIDIA NeMo Fabric.
It embeds `@opencode/sdk` in the adapter process and maps one NeMo Fabric
runtime to one isolated OpenCode host and session.

The adapter process runs with Bun 1.4.2 or newer. Install Bun before running
NeMo Fabric with this adapter.

To use a published adapter release, install the adapter and its compatible
OpenCode SDK in the project that owns the NeMo Fabric configuration:

```bash
npm install nemo-fabric-adapters-opencode @opencode/core@2.0.3 @opencode/sdk@2.0.3
```

OpenCode Core and the OpenCode SDK are optional peers, exact-pinned to the
supported OpenCode release. Starting the adapter without the compatible
packages reports a stable harness-unavailable error.

OpenCode 2.0.3 does not support npm's `install-strategy=nested`. The documented
command requires npm's default hoisted layout.

## Supported Configuration

The adapter supports the following configuration:

- **Models:** Selects the `default` role, or the only configured role. `api_key_env`
  may use any portable environment-variable name.
- **Instructions:** Supports `instructions.system` with `mode: replace`.
- **Skills:** Each `skills.paths` entry must be a directory containing `SKILL.md`.
  Startup verifies that every skill loads and that names are unique.
- **MCP:** Supports `stdio` and `streamable-http`. Startup fails if a configured
  server cannot connect.

### Model Endpoints

`models.<role>.base_url` must point to an OpenAI-compatible Chat Completions
endpoint, not an OpenCode server. Remote endpoints require HTTPS. HTTP is
allowed only for loopback development endpoints.

`temperature` and `top_p` are supported only with `base_url`. The adapter removes
OpenCode's `prompt_cache_key` extension for compatibility with providers that
implement only the core protocol.

### MCP Configuration

Streamable-HTTP servers may define headers but not command arguments. Stdio
servers may define command arguments and environment variables but not HTTP
headers.

Header values may reference `${NAME}`. Resolution checks `environment.env`
first, then the parent process environment. Unresolved references fail startup.
For MCP credentials, prefer parent-only variables so they are used to construct
the header without being added to OpenCode's tool environment.

SSE, MCP authentication objects, and per-server tool filters are unsupported.

### Runtime Behavior

The adapter accepts plain-text input and returns the final response in
`output.response`. It uses the Fabric workspace while disabling ambient OpenCode
project configuration, user configuration, and instruction discovery.

Streaming, Relay, tool policy, subagents, provider-specific model settings, and
`max_tokens` are not supported.

One Bun adapter process owns one embedded OpenCode host and session for each
NeMo Fabric runtime. Ordered invocations reuse that session. Stopping the
runtime removes the session and closes the host.

When OpenCode reports a diff for an invocation and the runtime has an artifact
root, the adapter writes it to a runtime- and invocation-scoped path beneath
`opencode/` and returns it as a `patch` artifact. Without an artifact root, it
returns no patch artifact.
