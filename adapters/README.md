<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Fabric Agent Harness Adapters

NeMo Fabric adapters translate the normalized NeMo Fabric contract into
harness-native models, tools, sessions, and telemetry. Use this reference to
compare the bundled adapters and then open the linked package guide for
installation, authentication, and configuration details.

The adapter descriptor selected in `RunPlan` is authoritative for normalized
configuration and telemetry support.

## Descriptor Discovery

As a stopgap until NeMo Fabric has a provider-backed adapter registry, the
Python SDK discovers descriptors in three locations. Later locations take
precedence:

1. descriptors bundled in the NeMo Fabric source repository;
2. `<sysconfig data>/share/nemo-fabric/adapters`, populated by adapter wheels
   and queried from `ADAPTER_PYTHON` when set, otherwise from the current Python;
3. `<base_dir>/adapters`, for agent-local and development overrides.

Fabric resolves multi-component relative `ADAPTER_PYTHON` paths from
`<base_dir>`. It resolves bare command names through `PATH`.

This scan only discovers installed metadata. It is not the final registry
contract for resolving or installing third-party adapters. Installed and
agent-local descriptors both currently report `source: local`; a registry
provider should expose more precise provenance.

## Bundled Adapter Packages

| Agent Harness | Adapter ID | Python Package | Supported Python |
| --- | --- | --- | --- |
| [Claude](claude/README.md) | `nvidia.fabric.claude` | `nemo-fabric-adapters-claude` | 3.11+ |
| [Codex](codex/README.md) | `nvidia.fabric.codex` | `nemo-fabric-adapters-codex` | 3.11+ |
| [LangChain Deep Agents](deepagents/README.md) | `nvidia.fabric.langchain.deepagents` | `nemo-fabric-adapters-deepagents` | 3.11+ |
| [Hermes Agent](hermes/README.md) | `nvidia.fabric.hermes` | `nemo-fabric-adapters-hermes` | 3.11-3.13 |

## Configuration Compatibility

| Agent Harness | Models | Tools / Blocked Tools | MCP | Skills | Subagents |
| --- | --- | --- | --- | --- | --- |
| [Claude](claude/README.md) | Anthropic and NVIDIA-hosted Anthropic Messages-compatible models | `allowed_tools` adapter setting / normalized `tools.blocked` | Normalized: stdio, HTTP, streamable HTTP, and SSE | Normalized `skills.paths` | Not exposed |
| [Codex](codex/README.md) | OpenAI; NVIDIA Responses-compatible models without Relay | Codex-native tools / configuring `tools.blocked` is unsupported and raises `UnsupportedToolsPolicy` | Normalized: stdio, HTTP, and streamable HTTP | Normalized `SKILL.md` directories | Not exposed |
| [LangChain Deep Agents](deepagents/README.md) | LangChain model providers | Built-ins and MCP / normalized middleware block list | Normalized through `langchain-mcp-adapters` | Normalized | Constrained declarative local delegation |
| [Hermes Agent](hermes/README.md) | Normalized provider, model, and base URL | Toolsets / normalized disabled toolsets | Normalized | Normalized | Not exposed |

"Normalized" means that the adapter accepts the corresponding `FabricConfig`
field. "Not exposed" does not mean that the underlying harness lacks the
feature; it means that NeMo Fabric does not provide a portable configuration surface
for it. NeMo Fabric normalizes a blocked-tool list, not a portable tool-definition
catalog. Deep Agents subagents are limited to declarative local subagents that
inherit the parent agent's capabilities.

## Runtime and Observability Compatibility

All bundled adapters use one persistent Python adapter host with an ordered
`start` → `invoke*` → `stop` protocol.

NeMo Relay records raw events in Agent Trajectory Observability Format (ATOF)
and produces normalized trajectories in Agent Trajectory Interchange Format
(ATIF).

| Agent Harness | State Retained Across Turns | Relay Integration | Per-Turn Behavior | Stop Behavior | Remote Service |
| --- | --- | --- | --- | --- | --- |
| [Claude](claude/README.md) | `ClaudeSDKClient` and Claude session ID | Runtime-owned Relay CLI gateway and generated Claude hooks | Calls `client.query()`, validates the session ID, and collects ATOF and ATIF | Disconnects the client, stops the gateway, and removes the generated plugin | Not implemented |
| [Codex](codex/README.md) | `AsyncCodex` app-server client and SDK thread | Runtime-owned Relay CLI gateway and Codex SDK hooks | Reuses the SDK thread and persists its thread ID | Closes the SDK client and app server, then stops the gateway | Not implemented |
| [LangChain Deep Agents](deepagents/README.md) | Compiled LangGraph agent, checkpointer, and thread ID | NeMo Relay Python SDK integration added when the agent is compiled | Creates a fresh Relay request scope and callback for each invocation | Closes the checkpointer; no gateway process | Not implemented |
| [Hermes Agent](hermes/README.md) | `AIAgent`, `SessionDB`, and conversation history | Hermes Agent NeMo Relay plugin context | Finalizes and flushes Relay after each invocation | Closes the agent and database, then exits the plugin context | Not implemented |

Telemetry output names use the descriptor contract values. Claude, Codex, and
Hermes Agent can emit NeMo Relay ATIF, OpenTelemetry, and OpenInference output. Deep
Agents supports the same Relay outputs plus native OpenTelemetry and
OpenInference; Codex also supports native OpenTelemetry.

Shared lifecycle, Relay gateway, hook, and payload helpers are documented in
the [adapter utilities guide](common/README.md).
