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
# Requires Node.js >=24.16.0 <25 or >=26.1.0.
npm install --global openclaw@latest --allow-scripts=openclaw
pip install "nemo-fabric[openclaw]"
```

OpenClaw is not a Python dependency of the adapter. If `openclaw` is not in
`PATH`, set `harness.settings.openclaw_command` to the absolute path of the executable. A bare command is resolved through `PATH`; a relative path containing a directory is resolved from the NeMo Fabric base directory.

## Configuration

The adapter maps the selected model, replacement system instruction, workspace,
skills, tool policy, and MCP servers into a generated `openclaw.json` before
starting the Gateway. `tools.enabled` maps to `tools.allow` and
`tools.blocked` maps to `tools.deny`. It uses a generated one-time token,
loopback binding, and an isolated temporary OpenClaw state directory. The
Gateway is stopped and its temporary configuration is removed when the NeMo
Fabric runtime stops.

The adapter runs the Gateway in an isolated process group and forwards
`SIGINT` and `SIGTERM` on POSIX systems. Linux adds a parent-death supervisor,
and Windows uses a kill-on-close Job Object, so abrupt adapter termination also
stops the Gateway process tree. macOS signal forwarding covers catchable
termination signals but cannot handle `SIGKILL`.

The following harness settings are supported:

| Setting | Default | Description |
| --- | --- | --- |
| `openclaw_command` | `openclaw` | Executable name or path. |
| `port` | Random available range | Gateway base port. |
| `startup_timeout_seconds` | `30` | Gateway readiness timeout. |
| `shutdown_timeout_seconds` | `10` | Graceful shutdown timeout. |
| `connect_timeout_seconds` | `10` | Local HTTP connection timeout. |
| `read_timeout_seconds` | `600` | Local HTTP response timeout. |

For custom OpenAI-compatible providers, set `models.<role>.base_url`; the
adapter generates an OpenClaw custom provider using the
`openai-completions` API adapter. Set `api_key_env` when that provider requires
an API key.

Remote MCP servers support `authentication.type: oauth2` with dynamic client
registration. The adapter maps `scopes` and `redirect_uri` to OpenClaw OAuth
configuration. It rejects service accounts and OAuth fields that OpenClaw
cannot represent, including pre-registered client credentials and custom
authorization timeouts.

Relay telemetry and native OpenTelemetry are not supported.
