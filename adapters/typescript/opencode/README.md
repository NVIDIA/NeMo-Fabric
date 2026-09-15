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

Install the adapter and its compatible OpenCode SDK in the project that owns
the NeMo Fabric configuration:

```bash
npm install nemo-fabric-adapters-opencode @opencode/sdk@^2.0.3
```

The OpenCode SDK is an optional peer dependency. Starting the adapter without a
compatible SDK reports a stable harness-unavailable error.

The adapter supports `models` and an optional `models.<role>.base_url`. The
adapter selects the `default` model role or the only configured role, accepts
plain-text input, and returns a terminal result with `output.response`. A
configured `api_key_env` name is passed explicitly to OpenCode, including for
native OpenCode providers, so it does not need to be that provider's usual
environment-variable name.
A configured endpoint must implement the OpenAI-compatible Chat Completions
protocol; it is a model-provider endpoint, not an OpenCode server endpoint.
For configured endpoints, the adapter omits OpenCode's `prompt_cache_key`
extension so providers that implement the core protocol but reject that
OpenAI-specific field remain compatible.
The adapter does not expose streaming, Relay, MCP, skills, tool policy,
subagents, system instructions, or model settings.
It keeps the Fabric workspace as OpenCode's working location while disabling
ambient OpenCode project and user configuration and instruction discovery.

One Bun adapter process owns one embedded OpenCode host and session for each
NeMo Fabric runtime. Ordered invocations reuse that session. Stopping the
runtime removes the session and closes the host.

When OpenCode reports a diff for an invocation, the adapter writes it under the
Fabric artifact root as `opencode/turn-<n>.patch` and returns it as a `patch`
artifact.
