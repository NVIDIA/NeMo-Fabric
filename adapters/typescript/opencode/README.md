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

OpenCode 2.0.23 pins Zod 4.1.8. The NeMo Fabric source workspace overrides
compatible Zod 4 dependencies to 4.6.7 while preserving the GitLab provider's
Zod 3 requirement and its dependencies' original peer ranges. For a
published-adapter installation, apply the `@opencode/core`, `@opencode/plugin`,
`fast-uri`, and `debug` overrides from the
[workspace manifest](https://github.com/NVIDIA/NeMo-Fabric/blob/8a5549a72241dd7c3066233275266ba32011fd07/adapters/typescript/package.json)
to the consuming project's root `package.json`, use Zod 4.6.7 as its root
Zod peer, and refresh the lockfile. npm does not inherit overrides from
installed dependency packages. Verify the resulting tree with `npm ls --all`
and confirm that `npm ls fast-uri debug` resolves to `fast-uri@3.1.9` and
`debug@4.4.3`.

This update fixes early rejection for size-constrained Zod 4 collections.
Unbounded collections still require input-size and collection-size limits.
The transitive GitLab provider also retains Zod 3, which has no compatible
patched release. Do not force Zod 4 into dependencies that require Zod 3.
Track the remaining exposure and upstream fixes in
[issue #485](https://github.com/NVIDIA/NeMo-Fabric/issues/485).

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

Streaming, Relay, tool policy, NeMo Fabric subagent configuration, provider-specific model settings, and
`max_tokens` are not supported.

One Bun adapter process owns one embedded OpenCode host and session for each
NeMo Fabric runtime. Ordered invocations reuse that session. Stopping the
runtime removes the session and closes the host.

When OpenCode reports a diff for an invocation and the runtime has an artifact
root, the adapter writes it to a runtime- and invocation-scoped path beneath
`opencode/` and returns it as a `patch` artifact. Without an artifact root, it
returns no patch artifact.

## Runtime Health

OpenCode supports `live`, `active`, `ready` through `Fabric.check()`. Each level calls the owned embedded SDK `sessions.get({ sessionID })` and validates the exact native session identity and absence of an archive timestamp. This read-only session response qualifies liveness, access to the initialized session interface, and narrow readiness. Adapter closure and ownership guards reject closed or replaced sessions. Native session data is discarded.

Checks remain independent of an invocation. A busy invocation is healthy. Checks use the caller deadline and never cancel an invocation, stop the runtime, or repair a failure. Health does not establish valid model credentials, provider access, MCP connectivity, or successful model output. At most one native request remains outstanding. A check returns `unknown/probe_timeout` while an earlier native request remains pending; later checks do not reuse a delayed reply. Reports contain only the generic health contract fields.

Refer to the [SDK health-check overview](../../../docs/sdk/health-checks.mdx) for shared behavior and a comparison of native evidence across adapters.

## Token Usage

With OpenCode 2.0.23, NeMo Fabric collects provider usage for OpenCode-managed Chat Completions requests it can attribute to an invocation. Input includes cached tokens. Output includes reasoning tokens. Do not add either breakdown again. Calls made directly by shell commands or external tools are outside this scope.

The adapter proves request ownership and waits for root and automatic title work. It reports each standard token count only when every owned request supplies a valid value for that field. Attributable foreground child work is included. Harbor receives these same inclusive counts. Standard total and cache counts are available only when every request reports those fields. Provider-reported cost stays unknown; OpenCode's locally calculated cost is not a provider-reported amount.

Collected observations are also preserved in `RunResult.usage.metadata["nvidia.fabric.opencode.usage_evidence.v1"]`. An empty `gaps` list means the input and output counts cover the proven scope. Observed cache and reasoning sums can still be partial if a request omits that breakdown. `scope_unverified` means ownership or completion could not be established. `missing_usage` marks missing or invalid counts, and `unfinished_work` marks requests that did not finish. Missing counts stay absent; reported zero stays zero. Partial sums can cover different sets of requests and must not be treated as complete totals.

Background shell or child work, an unresolved title, manual session changes, an interrupted event subscription, warming, or custom OpenCode plugins can prevent proof. The adapter retains attributable observations and leaves standard totals unknown. If earlier work can resume during a later invocation, that later invocation also leaves usage unknown. Unsupported model response protocols leave token counts unknown.
