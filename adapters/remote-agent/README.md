<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Fabric Remote Agent Adapter

Use the `nvidia.fabric.remote-agent` adapter to invoke a remote agent through
an OpenAI Responses, OpenAI Chat Completions, or Anthropic Messages HTTP API.

## Install

| Installation | Runtime | Adapter | Relay gateway |
| --- | --- | --- | --- |
| `pip install "nemo-fabric[remote-agent]"` | Yes | Yes | Yes |
| `pip install "nemo-fabric-adapters-remote-agent[full]"` | No | Yes | Yes |
| `pip install nemo-fabric-adapters-remote-agent` | No | Yes | No |

The bare package includes `httpx` and can communicate directly with a remote
service. The `full` extra additionally installs the supported NeMo Relay CLI
for gateway-mode telemetry.

## Configuration

Configure the API root in `HarnessConfig.settings`. `base_url` is required and
includes `/v1`; `api_type` defaults to `openai-responses`.

| Setting | Accepted values |
| --- | --- |
| `base_url` | HTTP(S) API root, such as `https://agent.example.com/v1` |
| `api_type` | `openai-responses`, `openai-completions`, or `anthropic-messages` |

The adapter accepts `models`, `models.temperature`, and `instructions.system`.
Set `models.default.api_key_env` when the service requires a credential. For
Anthropic Messages, optionally set `models.default.settings.max_tokens`; it
otherwise uses `4096`.

This adapter does not expose MCP, skills, tool policy, streaming, or subagents.
It retains the completed user/assistant transcript for ordered invocations in
one runtime.

## Relay gateway mode

When Relay telemetry is enabled, the adapter starts a runtime-owned
`nemo-relay` gateway. The gateway forwards OpenAI requests to `base_url`; for
Anthropic Messages it uses the same host with the `/v1` suffix removed. This
requires the package's `full` extra and the configured remote service must be
reachable from the adapter process.
