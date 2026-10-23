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

## Model usage

NeMo Fabric observes physical HTTP requests through OpenCode 2.0.23's public session
hooks, before OpenCode replaces omitted provider counts with zero. Supported
OpenAI Chat Completions, Responses, and Anthropic Messages reports retain field
presence: a reported zero stays zero and an omitted count stays unknown. Main,
compaction, and child-session requests share the initiating invocation's owner.

An unfinished request, malformed report, failed HTTP attempt without usage, or
WebSocket request leaves affected complete totals unknown. Positive session
counter growth remains an observation when raw HTTP evidence is unavailable;
the projected counters cannot prove completeness or a reported zero. Accounting
observations and gaps travel in `nemo_fabric.usage_accounting.v1`. Core normalizes
provider conventions. Catalog estimation is a separate shared feature. OpenCode's local
cost estimate is not provider-reported `cost_usd`. Custom endpoints retain their
actual model identifier without inheriting native-provider pricing identity.

Results freeze at return. A background child's late usage stays with its
initiating invocation and never transfers to a later one. NeMo Fabric retains the
existing fixed session title; it does not change OpenCode's retry, compaction,
or delegation behavior. A transport failure after submission invalidates the
session and returns its frozen usage evidence with a failed result.

A new completion notice from an earlier background child can wake the parent
inside a later invocation. Requests proven to precede that notice remain
observations. Ambiguous parent continuations and subsequent descendant requests
are excluded, with an unverified-scope gap. Notices already present when the
invocation starts do not disable otherwise attributable work. The adapter checks
this boundary only when earlier child sessions exist; an unreadable or stalled
context read also leaves attribution unverified.

A patch-artifact write failure returns its existing storage error with settled
usage intact, no claimed artifact and no automatic task replay. The healthy
OpenCode session remains available for an explicit later invocation.

Native billing identity requires both the requested route and the visible final
response URL to qualify the same provider. A redirect to a custom endpoint
keeps the model and token reports without inheriting native-provider prices.
