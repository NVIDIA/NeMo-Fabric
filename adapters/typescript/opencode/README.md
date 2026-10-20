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
npm install nemo-fabric-adapters-opencode @opencode/core@2.0.23 @opencode/sdk@2.0.23
```

OpenCode Core and the OpenCode SDK are optional peers, exact-pinned to the
supported OpenCode release. Starting the adapter without the compatible
packages reports a stable harness-unavailable error.

OpenCode 2.0.23 does not support npm's `install-strategy=nested`. The documented
command requires npm's default hoisted layout.

## Supported Configuration

The adapter supports the following configuration:

- **Models:** Selects the `default` role, or the only configured role. `api_key_env`
  may use any portable environment-variable name.
- **Instructions:** Supports `instructions.system` with `mode: replace`.
- **Skills:** Each `skills.paths` entry must be a directory containing `SKILL.md`.
  Startup verifies that only those `SKILL.md` files load and that names are unique.
- **MCP:** Supports `stdio` and `streamable-http`. Startup fails if a configured
  server cannot connect.

### Model Endpoints

`models.<role>.base_url` must point to an OpenAI-compatible Chat Completions
endpoint, not an OpenCode server.

HTTPS is required by default, except for loopback HTTP. For a trusted Docker or Kubernetes gateway, set `harness.settings.insecure_http_model_hosts` to the exact hostnames that can receive HTTP model requests. Entries contain only hostnames or IPv4 addresses, without schemes, ports, paths, or wildcards. Matching ignores case and a trailing DNS dot, and permits any port or path on the listed host. The adapter does not resolve DNS to infer trust. Allowlisted HTTP endpoints receive model credentials and prompt content over an unencrypted connection. This setting applies only to model endpoints; MCP endpoint restrictions are unchanged.

For example, allow the gateway hosts used by your deployment:

```yaml
harness:
  settings:
    insecure_http_model_hosts:
      - host.docker.internal
      - release-nhx-api
      - gateway.namespace.svc.cluster.local
```

`temperature` and `top_p` are supported only with `base_url`. The adapter removes
OpenCode's `prompt_cache_key` extension for compatibility with providers that
implement only the core protocol.

### MCP Configuration

Streamable-HTTP servers may define headers but not command arguments. Stdio
servers may define command arguments and environment variables but not HTTP
headers.

Streamable-HTTP server URLs require HTTPS except for loopback development
endpoints.

Header values may reference `${NAME}`. Resolution checks `environment.env`
first, then the parent process environment. Unresolved references fail startup.
For MCP credentials, prefer parent-only variables so they are used to construct
the header without being added to OpenCode's tool environment.

SSE, MCP authentication objects, and per-server tool filters are unsupported.
MCP elicitation is also unsupported. Configure only non-interactive MCP servers as
NeMo Fabric cannot reply to an elicitation request, so it can block an
invocation until its configured timeout.

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

### Token Usage

Each result's `usage` is the growth of OpenCode's session token counters
during the invocation. A subagent runs in a child session with its own
counters, so the adapter adds the growth of every subagent session below the
Fabric session. The counters include every model call these sessions record,
including context compaction. The adapter gives the session a fixed title, so
OpenCode makes no title-generation call.

`input_tokens` includes cache reads and cache writes, and
`input_tokens_include_cache` is `true`. `cached_input_tokens` holds the cache
reads. `output_tokens` includes reasoning tokens, and `total_tokens` is their
sum. OpenCode computes its cost figures from the model catalog's rates, so the
adapter does not report them as `cost_usd`.

The adapter checks new assistant and compaction records that the invocation
added to each session's full message history, including records that a
compaction removed from the session context. OpenCode stores a count the provider left
out as 0. A real model call always has input, so a call with no input tokens
leaves `input_tokens` unknown, and a call with no output or reasoning tokens
leaves `output_tokens` unknown. Either case also leaves out `total_tokens`, and
so does a session counter that is missing or goes down. A `total_tokens` past
JavaScript's safe integer range is left out.

Counter growth from earlier background work is excluded. Usage remains unknown
when a delayed completion or resumed child prevents reliable attribution. The
result has no `usage` in these cases:

- A model call or compaction failed and OpenCode recorded no usage for it.
- A subagent session that the invocation started was still running when it
  ended, or the Fabric session was already running when it started.
- The invocation continued a subagent that was still running from earlier
  work, or made a model call after the completion notice of a subagent that
  existed before the invocation.
- The session counters or message history cannot be read.

OpenCode replaces individual missing counters with zero before the adapter
sees them. Those omissions and internal retries without recorded usage can
still undercount totals.
