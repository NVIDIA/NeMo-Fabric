<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Fabric OpenClaw Adapter

This adapter starts an isolated local OpenClaw Gateway for each NeMo Fabric
runtime and communicates with its OpenAI-compatible Chat Completions endpoint.

## Install

Install OpenClaw separately through npm, then install the adapter:

```bash
# Requires Node.js >=24.16.0 <25 or >=26.1.0, and npm 11.16+.
npm install --global openclaw@2026.9.4 --allow-scripts=openclaw
pip install "nemo-fabric[openclaw]"
```

OpenClaw is not a Python dependency of the adapter. If `openclaw` is not in
`PATH`, set `harness.settings.openclaw_command` to the absolute path of the executable. A bare command is resolved through `PATH`; a relative path containing a directory is resolved from the NeMo Fabric base directory.

## Configuration

The adapter maps the selected model, replacement system instruction, workspace,
skills, tool policy, and MCP servers into a generated `openclaw.json` before
starting the Gateway. A non-empty `tools.enabled` list maps to `tools.allow`, an
empty list maps to a wildcard `tools.deny` policy, and `tools.blocked` maps to
`tools.deny`. It uses a generated one-time token, loopback binding, and an
isolated temporary OpenClaw state directory. The Gateway is stopped and its
temporary configuration is removed when the NeMo Fabric runtime stops.

The adapter runs the Gateway in an isolated process group, continuously checks
its readiness endpoint, and forwards `SIGINT` and `SIGTERM` on POSIX systems.
On Linux, the adapter uses `setpriv --pdeathsig SIGTERM` when `setpriv` is
available so that the kernel signals the Gateway after abrupt adapter
termination. Windows uses a kill-on-close Job Object for the Gateway process
tree. Without `setpriv`, Linux has the same catchable-signal protection as
macOS but cannot handle `SIGKILL`.

The following harness settings are supported:

| Setting | Default | Description |
| --- | --- | --- |
| `openclaw_command` | `openclaw` | Executable name or path. |
| `port_range` | Random available range | Inclusive consumer-allocated range with `start` and `end` fields; it must span at least 111 ports. |
| `startup_timeout_seconds` | `30` | Gateway readiness timeout. |
| `shutdown_timeout_seconds` | `10` | Graceful shutdown timeout. |
| `connect_timeout_seconds` | `10` | Local HTTP connection timeout. |
| `read_timeout_seconds` | `600` | Local HTTP response timeout. |

OpenClaw's [derived-port mapping](https://docs.openclaw.ai/gateway/multiple-gateways#port-mapping-derived) uses `base_port + 2` for browser control and allocates browser CDP ports from `base_port + 11` through `base_port + 110`. Set `port_range` when the consumer reserves a port range before configuring Fabric; the adapter selects a base whose complete derived footprint fits inside that range. The adapter verifies only that `base_port` and `base_port + 2` are available before startup. OpenClaw allocates the CDP ports on demand instead of reserving the complete range, so the adapter cannot guarantee that those ports will still be available when OpenClaw needs them. Keep the configured range reserved for the lifetime of the runtime when using OpenClaw browser features.

For custom OpenAI-compatible providers, set `models.<role>.base_url`; the
adapter generates an OpenClaw custom provider using the
`openai-completions` API adapter. Set `api_key_env` when that provider requires
an API key.

Relay telemetry and native OpenTelemetry are not supported.
