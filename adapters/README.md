<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Fabric Adapters

NeMo Fabric adapters translate the normalized NeMo Fabric contract into the
native models, tools, sessions, and telemetry of an agent harness or custom
agent. Use this reference to compare the bundled harness adapters, then refer
to the custom-agent examples when building a shared framework adapter or a
dedicated adapter.

The adapter descriptor selected in `RunPlan` is authoritative for normalized
configuration, its adapter-owned settings schema, and telemetry support.

## Repository Layout

Language-specific adapter packages and shared runtime utilities live under a
language directory. TypeScript packages are under `adapters/typescript/`.
Existing Python packages remain at the adapter root until they move together
in a separate repository-organization change.

## Descriptor Discovery

The Python SDK builds one descriptor registry from three sources:

1. records bundled with NeMo Fabric;
2. records installed recursively below
   `<sysconfig data>/share/nemo-fabric`, queried from `ADAPTER_PYTHON` when set;
3. files and directories listed in `FabricConfig.discovery.local_paths`.

NeMo Fabric resolves multi-component relative `ADAPTER_PYTHON` paths from
`<base_dir>`. It resolves bare command names through `PATH`.

Identical records with the same ID are deduplicated and retain every source as
provenance. Different records with the same ID fail as ambiguous; discovery
does not merge fields or choose an override. Workflow entry points and settings
schemas come from independently registered Adapter Target Descriptors.

## Bundled Python Adapter Packages

| Agent Harness | Adapter ID | Python Package | Supported Python |
| --- | --- | --- | --- |
| [Claude](claude/README.md) | `nvidia.fabric.claude` | `nemo-fabric-adapters-claude` | 3.11+ |
| [Codex](codex/README.md) | `nvidia.fabric.codex` | `nemo-fabric-adapters-codex` | 3.11+ |
| [LangChain Deep Agents](deepagents/README.md) | `nvidia.fabric.langchain.deepagents` | `nemo-fabric-adapters-deepagents` | 3.11+ |
| [Hermes Agent](hermes/README.md) | `nvidia.fabric.hermes` | `nemo-fabric-adapters-hermes` | 3.11-3.13 |
| [mini-SWE-agent](mini-swe-agent/README.md) | `nvidia.fabric.mini-swe-agent` | `nemo-fabric-adapters-mini-swe-agent` | 3.11+ |
| [Remote Agent](remote-agent/README.md) | `nvidia.fabric.remote-agent` | `nemo-fabric-adapters-remote-agent` | 3.11+ |

## TypeScript Adapter Packages

| Agent Harness | Adapter ID | npm Package | Supported Node.js |
| --- | --- | --- | --- |
| [Pi](typescript/pi/README.md) | `nvidia.fabric.pi` | `nemo-fabric-adapters-pi` | 22.19+ |

Shared TypeScript lifecycle utilities live under
[`typescript/common`](typescript/common/README.md).
Refer to the [TypeScript adapter workspace guide](typescript/README.md) for
build commands and dependency policy.

## Custom-Agent Adapter References

Custom-agent support uses the same adapter contract as the bundled harnesses:

- The [NeMo Agent Toolkit reference adapter](../external/nat/README.md) shows
  one shared framework adapter with separately registered calculator and
  email-phishing targets.
- The [LangGraph custom-agent example](../examples/langgraph_custom_agent/README.md)
  shows a dedicated adapter beside an application-owned graph.

Start with the
[adapter contract overview](../docs/adapter-contract/README.md) to choose an
integration shape and implement the minimum lifecycle.

## Configuration Compatibility

| Agent Harness | Models | Tool Policy | MCP | Skills | Subagents |
| --- | --- | --- | --- | --- | --- |
| [Claude](claude/README.md) | Native Anthropic or a configured Anthropic Messages-compatible provider | `tools.enabled` selects built-ins; a pre-tool hook enforces enabled and blocked names across built-in, MCP, and plugin tools | Normalized: stdio, HTTP, streamable HTTP, and SSE | Normalized `skills.paths` | Not exposed |
| [Codex](codex/README.md) | Native OpenAI or a configured Responses-compatible provider | `tools.enabled` and `tools.blocked` unsupported | Normalized: stdio, HTTP, and streamable HTTP | Normalized `SKILL.md` directories | Not exposed |
| [LangChain Deep Agents](deepagents/README.md) | LangChain model providers | Middleware enforces `tools.enabled` and `tools.blocked` across built-ins, MCP, and local delegation | Normalized through `langchain-mcp-adapters` | Normalized | Built-in, declarative, and Agent Protocol |
| [Hermes Agent](hermes/README.md) | Configurable provider, model, and base URL | `tools.enabled` and `tools.blocked` map to Hermes native toolset selectors | Normalized | Normalized | Not exposed |
| [mini-SWE-agent](mini-swe-agent/README.md) | Configured provider and model | Not exposed | Not exposed | Not exposed | Not exposed |
| [Pi](typescript/pi/README.md) | One Pi-catalog provider and model with an optional base URL override | `tools.definitions`, `tools.enabled`, and `tools.blocked` cover built-ins, trusted local modules, and explicit extension tools | Not exposed | Normalized `skills.paths` | Not exposed |
| [Remote Agent](remote-agent/README.md) | Configured remote HTTP API and model | Not exposed | Not exposed | Not exposed | Not exposed |

"Normalized" means that the adapter accepts the corresponding `FabricConfig`
field. "Not exposed" does not mean that the underlying harness lacks the
feature; it means that NeMo Fabric does not provide a portable configuration
surface for it. Tool values are adapter-native selectors; NeMo Fabric does not
define a cross-harness tool-name catalog. Planning fails when the selected
adapter cannot enforce a configured policy. Deep Agents supports its built-in
subagent and descriptor-validated caller-defined subagents. A configured tools
policy permits only local declarative subagents because the adapter cannot gate
remote Agent Protocol tools.

`RunPlan.capability_plan.routes` records execution ownership, not network
routing. `harness_native` assigns a capability to the selected adapter,
`fabric_managed` assigns it to NeMo Fabric, and `unsupported` means neither can
execute it. Scalar fields are validated separately against
`adapter_descriptor.config.accepts`.

### Complete FabricConfig Support

`Core` means NeMo Fabric owns the behavior and applies it uniformly before or around
adapter execution. `Yes` means the adapter translates the normalized field into
its harness. `No` means an explicitly configured value fails planning instead
of being ignored. The following table groups provider-specific Relay subfields
and additive extension maps because their support does not vary by adapter:

| `FabricConfig` Field | Claude | Codex | Deep Agents | Hermes Agent | mini-SWE-agent | Pi | Remote Agent |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `schema_version` | Core | Core | Core | Core | Core | Core | Core |
| `metadata.name`, `.description` | Core | Core | Core | Core | Core | Core | Core |
| `harness.adapter_id`, `.resolution` | Core | Core | Core | Core | Core | Core | Core |
| `harness.settings` | Closed adapter schema | Closed adapter schema | Closed adapter schema | Closed adapter schema | Closed `timeout` schema | Closed local-extension schema | Closed `base_url`, `api_type`, and transport-timeout schema |
| `workflow.target_id`, `.settings` | No | No | No | No | No | No | No |
| `models.<role>.provider` | `anthropic` uses native auth; custom names require an Anthropic Messages-compatible `base_url` and `api_key_env` | `openai` uses native auth; custom names require a Responses-compatible `base_url` and `api_key_env` | Dynamic LangChain provider; custom OpenAI-compatible endpoints require `base_url` and `api_key_env` | Dynamic Hermes provider | Configured provider | Pi catalog provider | Configured provider |
| `models.<role>.model` | Yes | Yes | Yes | Yes | Yes | Yes; must exist in the Pi catalog | Yes |
| `models.<role>.api_key_env` | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| `models.<role>.base_url` | Yes | Yes | Yes | Yes | Yes | Yes; known catalog models only | No; use `harness.settings.base_url` |
| `models.<role>.temperature` | No | No | Yes | Yes | Yes | No | Yes |
| `models.<role>.settings.<key>` | No keys declared | No keys declared | No keys declared | No keys declared | No keys declared | No keys declared | `max_tokens` for Anthropic Messages |
| `instructions.system` | `replace`, `append` | `replace`; base instructions | `replace` | `replace` | `replace` | `replace`; Pi base instructions | `replace` |
| `runtime.input_schema`, `.output_schema` | Core | Core | Core | Core | Core | Core | Core |
| `runtime.artifacts`, `.timeout_seconds` | Core | Core | Core | Core | Core | Core | Core |
| `runtime.max_turns` | Yes | No | Yes; maps to LangGraph supersteps | Yes; iteration limit | Yes | No | No |
| `environment.provider`, `.control_location`, `.ownership` | Core | Core | Core | Core | Core | Core | Core |
| `environment.workspace`, `.artifacts`, `.env` | Core | Core | Core | Core | Core | Core | Core |
| `environment.connection`, `.metadata`, `.settings` | Environment-provider-owned | Environment-provider-owned | Environment-provider-owned | Environment-provider-owned | Environment-provider-owned | Environment-provider-owned | Environment-provider-owned |
| `tools.definitions` | No | No | No | No | No | Yes; trusted local module factories | No |
| `tools.enabled`, `.blocked` | Yes | No | Yes | Yes; native selectors are Hermes toolset names | No | Yes | No |
| `skills.paths` | Yes | Yes | Yes | Yes | No | Yes | No |
| `mcp.servers.<name>.transport`, `.url` with `harness_native` exposure | Yes | Yes | Yes | Yes | No | No | No |
| `mcp.servers.<name>.exposure = "fabric_managed"` | No; not implemented | No; not implemented | No; not implemented | No; not implemented | No | No | No |
| `telemetry.providers.relay` | Yes | Yes | Yes | Yes | Yes | No | No |
| `telemetry.providers.native` | No | Yes; OpenTelemetry | Yes; OpenTelemetry and OpenInference | No | No | No | No |
| `telemetry.providers.<provider>.config` | Declared-provider pass-through | Declared-provider pass-through | Declared-provider pass-through | Declared-provider pass-through | Declared-provider pass-through | No | No |
| `relay.project`, `.output_dir`, `.observability` | Yes | Yes | Yes | Yes | Yes | No | No |
| `relay.components`, `.policy` | Yes | Yes | Yes | Yes | Yes | No | No |
| Additive `extensions` on typed config objects | Preserved; no portable adapter semantics | Preserved; no portable adapter semantics | Preserved; no portable adapter semantics | Preserved; no portable adapter semantics | Preserved; no portable adapter semantics | Not accepted unless declared by the Pi descriptor | Preserved; no portable adapter semantics |

The selected model role is `default`, or the sole configured role when no
`default` exists. More than one role without `default` fails planning.
Claude and Codex publish a descriptor-owned `model_schema` for every configured
model role. Their native providers (`anthropic` and `openai`, respectively)
keep the existing authentication path. Other providers remain valid only with
an explicit `base_url` and `api_key_env`. The same schema rejects undeclared
`ModelConfig.settings` during planning and reports each issue through
`doctor(...)` before adapter startup.
`runtime.max_turns` is optional; omitting it preserves adapter-native defaults
without creating a compatibility requirement.

## Runtime and Observability Compatibility

All bundled adapters use a language-specific persistent adapter host with an
ordered `start` → `invoke*` → `stop` protocol.

NeMo Relay records raw events in Agent Trajectory Observability Format (ATOF)
and produces normalized trajectories in Agent Trajectory Interchange Format
(ATIF).

| Agent Harness | State Retained Across Turns | Relay Integration | Per-Turn Behavior | Stop Behavior | Remote Service |
| --- | --- | --- | --- | --- | --- |
| [Claude](claude/README.md) | `ClaudeSDKClient` and Claude session ID | Runtime-owned Relay CLI gateway and generated Claude hooks | Calls `client.query()`, validates the session ID, and collects ATOF and ATIF | Disconnects the client, stops the gateway, and removes the generated plugin | Not implemented |
| [Codex](codex/README.md) | `AsyncCodex` app-server client and SDK thread | Runtime-owned Relay CLI gateway and Codex SDK hooks | Reuses the SDK thread and persists its thread ID | Closes the SDK client and app server, then stops the gateway | Not implemented |
| [LangChain Deep Agents](deepagents/README.md) | Compiled LangGraph agent, checkpointer, and thread ID | NeMo Relay Python SDK integration added when the agent is compiled | Creates a fresh Relay request scope and callback for each invocation | Closes the checkpointer; no gateway process | Not implemented |
| [Hermes Agent](hermes/README.md) | `AIAgent`, `SessionDB`, and conversation history | Hermes Agent NeMo Relay plugin context | Finalizes and flushes Relay after each invocation | Closes the agent and database, then exits the plugin context | Not implemented |
| [mini-SWE-agent](mini-swe-agent/README.md) | Conversation history | Adapter-owned subclass with NeMo Relay Python SDK scopes | Creates a fresh Relay plugin and request scope, emits step, model, and bash-action telemetry, and collects artifacts | Clears the agent and Relay state | Not implemented |
| [Pi](typescript/pi/README.md) | In-memory Pi `AgentSession` | Not supported | Reuses the session and calls `prompt()` for ordered text input | Aborts work, emits extension shutdown, and disposes the session | Not implemented |
| [Remote Agent](remote-agent/README.md) | `httpx.AsyncClient` and user/assistant transcript | Not supported | Sends one HTTP request and retains the completed transcript | Closes the HTTP client | Not implemented |

Telemetry output names use the descriptor contract values. Claude, Codex,
Hermes Agent, and mini-SWE-agent can emit NeMo Relay ATIF, OpenTelemetry, and
OpenInference output. Deep Agents supports the same Relay outputs plus native
OpenTelemetry and OpenInference; Codex also supports native OpenTelemetry.

Shared lifecycle, Relay gateway, hook, and payload helpers are documented in
the [adapter utilities guide](common/README.md).
