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

The initial adapter maps only configuration Kiro V3 exposes through ACP:

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
policy, arbitrary model endpoints, telemetry, and streaming are not advertised
by this initial adapter.

See the [Kiro integration guide](../../../docs/integrations/harness/kiro.mdx)
for a complete configuration example.

## Runtime health

`Fabric.check()` supports `live`. The adapter observes the selected ACP child
process identifier, exit status, and signal status through the shared process
runner. A retained session identifier is insufficient. A busy process remains
live, and a stopped or exited process fails the liveness observation.

`active` and `ready` are unsupported. Checks do not send prompts, change session settings, or cancel invocations.

Refer to the [SDK health-check overview](../../../docs/sdk/health-checks.mdx) for shared behavior and a comparison of native evidence across adapters.
