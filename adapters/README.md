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

| Agent Harness | Models | Tool Policy | MCP | Skills | Subagents |
| --- | --- | --- | --- | --- | --- |
| [Claude](claude/README.md) | Native Anthropic or a configured Anthropic Messages-compatible provider | `tools.blocked` maps to Claude `disallowed_tools`; toolsets unsupported | Normalized: stdio, HTTP, streamable HTTP, and SSE | Normalized `skills.paths` | Not exposed |
| [Codex](codex/README.md) | Native OpenAI or a configured Responses-compatible provider | Per-tool blocking and toolsets unsupported | Normalized: stdio, HTTP, and streamable HTTP | Normalized `SKILL.md` directories | Not exposed |
| [LangChain Deep Agents](deepagents/README.md) | LangChain model providers | `tools.blocked` middleware covers built-ins, MCP, and local subagents; toolsets unsupported | Normalized through `langchain-mcp-adapters` | Normalized | Constrained declarative local delegation |
| [Hermes Agent](hermes/README.md) | Configurable provider, model, and base URL | `tools.toolsets` maps to Hermes toolset controls; per-tool blocking unsupported | Normalized | Normalized | Not exposed |

"Normalized" means that the adapter accepts the corresponding `FabricConfig`
field. "Not exposed" does not mean that the underlying harness lacks the
feature; it means that NeMo Fabric does not provide a portable configuration surface
for it. Individual tools and harness-defined toolsets are separate normalized
policies. Planning fails when the selected adapter cannot enforce a configured
policy; NeMo Fabric does not reinterpret one as the other. Deep Agents subagents
are limited to declarative local subagents that inherit the parent agent's
capabilities.

### Complete `FabricConfig` Support

`Core` means NeMo Fabric owns the behavior and applies it uniformly before or around
adapter execution. `Yes` means the adapter translates the normalized field into
its harness. `No` means an explicitly configured value fails planning instead
of being ignored. Provider-specific Relay subfields and additive extension maps
are grouped because their support does not vary by adapter.

| `FabricConfig` field | Claude | Codex | Deep Agents | Hermes Agent |
| --- | --- | --- | --- | --- |
| `schema_version` | Core | Core | Core | Core |
| `metadata.name`, `.description` | Core | Core | Core | Core |
| `harness.adapter_id`, `.resolution` | Core | Core | Core | Core |
| `harness.settings` | Adapter-owned escape hatch | Adapter-owned escape hatch | Adapter-owned escape hatch | Adapter-owned escape hatch |
| `models.<role>.provider` | `anthropic` uses native auth; custom names require an Anthropic Messages-compatible `base_url` and `api_key_env` | `openai` uses native auth; custom names require a Responses-compatible `base_url` and `api_key_env` | Dynamic LangChain provider; custom OpenAI-compatible endpoints require `base_url` and `api_key_env` | Dynamic Hermes provider |
| `models.<role>.model` | Yes | Yes | Yes | Yes |
| `models.<role>.api_key_env` | Yes | Yes | Yes | Yes |
| `models.<role>.base_url` | Yes | Yes | Yes | Yes |
| `models.<role>.temperature` | No | No | Yes | Yes |
| `models.<role>.settings.<key>` | No keys declared | No keys declared | No keys declared | No keys declared |
| `system_prompt` | Yes | Yes; base instructions | Yes | Yes |
| `max_turns` | Yes | No | No | Yes; iteration limit |
| `runtime.input_schema`, `.output_schema` | Core | Core | Core | Core |
| `runtime.artifacts`, `.timeout_seconds` | Core | Core | Core | Core |
| `environment.provider`, `.control_location`, `.ownership` | Core | Core | Core | Core |
| `environment.workspace`, `.artifacts`, `.env` | Core | Core | Core | Core |
| `environment.connection`, `.metadata`, `.settings` | Environment-provider-owned | Environment-provider-owned | Environment-provider-owned | Environment-provider-owned |
| `tools.blocked` | Yes | No | Yes | No |
| `tools.toolsets.enabled`, `.blocked` | No | No | No | Yes |
| `skills.paths` | Yes | Yes | Yes | Yes |
| `mcp.servers.<name>.transport`, `.url` with `harness_native` exposure | Yes | Yes | Yes | Yes |
| `mcp.servers.<name>.exposure = "fabric_managed"` | No; not implemented | No; not implemented | No; not implemented | No; not implemented |
| `telemetry.providers.relay` | Yes | Yes | Yes | Yes |
| `telemetry.providers.native` | No | Yes; OpenTelemetry | Yes; OpenTelemetry and OpenInference | No |
| `telemetry.providers.<provider>.config` | Declared-provider pass-through | Declared-provider pass-through | Declared-provider pass-through | Declared-provider pass-through |
| `relay.project`, `.output_dir`, `.observability` | Yes | Yes | Yes | Yes |
| `relay.components`, `.policy` | Yes | Yes | Yes | Yes |
| Additive `extensions` on typed config objects | Preserved; no portable adapter semantics | Preserved; no portable adapter semantics | Preserved; no portable adapter semantics | Preserved; no portable adapter semantics |

The selected model role is `default`, or the sole configured role when no
`default` exists. More than one role without `default` fails planning.

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
