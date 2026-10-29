<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Fabric Kiro Adapter

This package provides the `nvidia.fabric.kiro` process adapter. It starts a
caller-installed Kiro CLI V3 ACP agent, creates one persistent ACP session per
NeMo Fabric runtime, and reuses that session across ordered invocations.

## Install the Adapter

Install Node.js 22.19.0 or newer, install and authenticate the Kiro CLI, then
install the adapter:

```bash
curl -fsSL https://cli.kiro.dev/install | bash
kiro-cli login
npm install nemo-fabric-adapters-kiro
```

For non-interactive authentication, set `KIRO_API_KEY` instead of using the
browser login. The adapter never installs or updates Kiro at runtime.

For source development, run these commands from the repository root:

```bash
just install-typescript-kiro
npm run build --prefix adapter-contract/typescript
npm run build --prefix adapters/typescript --workspace nemo-fabric-adapters-common
npm run build --prefix adapters/typescript --workspace nemo-fabric-adapters-kiro
```

Register the packaged descriptor at
`node_modules/nemo-fabric-adapters-kiro/kiro.fabric-adapter.json`. Source
checkouts can register `adapters/typescript/kiro/kiro.fabric-adapter.json`.

## Supported Configuration

Native mode maps configuration Kiro V3 exposes through ACP:

- The optional `default` NeMo Fabric model role selects an advertised Kiro model.
  Its provider is `kiro`; `api_key_env`, when present, must be
  `KIRO_API_KEY`.
- `harness.settings.mode_id` selects an advertised ACP session mode.
- `harness.settings.effort` selects an advertised Kiro effort level.
- `harness.settings.permission_mode` is `deny` by default. Set it to
  `allow_once` to choose only one-time allow options explicitly offered by
  Kiro. The adapter never grants persistent permission.
- NeMo Fabric MCP stdio, streamable HTTP, and SSE definitions are sent with
  `session/new`. Stdio executables must be absolute. Network MCP endpoints
  require HTTPS except for loopback development endpoints. Startup waits for
  Kiro to report every configured MCP server as connected before the first
  invocation.

Kiro's advertised modes, models, effort values, and MCP transport capabilities
are authoritative. Startup fails instead of silently ignoring an unsupported
selection. NeMo Fabric system instructions, skill paths, normalized built-in tool
policy, telemetry, and streaming are not advertised. Model endpoints are
available only through the experimental bridge described below.

See the [Kiro integration guide](../../../docs/integrations/harness/kiro.mdx)
for a complete configuration example.

## Experimental OpenAI-Compatible Bridge

The adapter can run Kiro CLI **2.34.2** against an explicit OpenAI Chat
Completions endpoint. This adapter-owned bridge uses Kiro's private service
protocol; it is experimental and does not claim official Kiro BYOM support.
Other CLI versions are rejected for this mode. Native `provider="kiro"`
authentication and session selection remain separate.

Opt in with `harness.settings.experimental_byom=true`, select the `openai`
provider, and supply `base_url`. The model ID is forwarded in full, including
namespaces such as `nvidia/`. For example:

```json
{
  "harness": {
    "adapter_id": "nvidia.fabric.kiro",
    "settings": {"experimental_byom": true, "permission_mode": "allow_once"}
  },
  "models": {
    "default": {
      "provider": "openai",
      "model": "nvidia/nemotron-model",
      "base_url": "http://127.0.0.1:8000/v1",
      "api_key_env": "MODEL_API_KEY"
    }
  }
}
```

`api_key_env` is optional for an endpoint that requires no authentication.
When supplied, the configured runtime environment takes priority over the
parent environment. A missing or empty configured credential fails startup.
The adapter uses an isolated HOME and owns the bridge lifecycle; credentials
are not supplied through model IDs or URLs. URL credentials, queries, and
fragments are rejected.

HTTPS and loopback HTTP are accepted. Other HTTP model endpoints require
explicit trust through `harness.settings.insecure_http_model_hosts`, containing
hostnames or IP addresses without schemes, ports, paths, or wildcards.
The bridge sums input and output tokens across every forwarded model request in
one invocation. OpenAI prompt counts include cache reads; those tokens are not
added twice. Missing or failed-request usage leaves the complete totals unknown.
Background shell or subagent work also leaves totals unknown for that session.
Requests outside an active invocation are rejected.

The experimental model advertisement uses a 131,072-token input budget and a
16,384-token output budget. The endpoint must support those budgets. This is a
text and tool-call bridge; image and document inputs are rejected. Both Kiro
telemetry and content sharing are disabled through its CLI settings in the
isolated home. This does not establish operating-system network containment.

## Runtime health

`Fabric.check()` supports `live`. The adapter observes the selected ACP child
process identifier, exit status, and signal status through the shared process
runner. A retained session identifier is insufficient. A busy process remains
live, and a stopped or exited process fails the liveness observation.

`active` and `ready` are unsupported. Checks do not send prompts, change session settings, or cancel invocations.

Refer to the [SDK health-check overview](../../../docs/sdk/health-checks.mdx) for shared behavior and a comparison of native evidence across adapters.
