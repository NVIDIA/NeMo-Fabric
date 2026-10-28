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

Language-specific adapter packages and shared runtime utilities live under
`adapters/python/` and `adapters/typescript/`. Package names, imports, and
installed descriptor locations do not include the source language directory.

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
| [Claude](python/claude/README.md) | `nvidia.fabric.claude` | `nemo-fabric-adapters-claude` | 3.11+ |
| [Codex](python/codex/README.md) | `nvidia.fabric.codex` | `nemo-fabric-adapters-codex` | 3.11+ |
| [LangChain Deep Agents](python/deepagents/README.md) | `nvidia.fabric.langchain.deepagents` | `nemo-fabric-adapters-deepagents` | 3.11+ |
| [Hermes Agent](python/hermes/README.md) | `nvidia.fabric.hermes` | `nemo-fabric-adapters-hermes` | 3.11-3.14 |
| [mini-SWE-agent](python/mini-swe-agent/README.md) | `nvidia.fabric.mini-swe-agent` | `nemo-fabric-adapters-mini-swe-agent` | 3.11+ |
| [NOOA](python/nooa/README.md) | `nvidia.fabric.nooa`, `nvidia.fabric.nooa.bench-agent` | `nemo-fabric-adapters-nooa` | 3.12-3.13 |
| [OpenClaw](python/openclaw/README.md) | `nvidia.fabric.openclaw` | `nemo-fabric-adapters-openclaw` | 3.11+ |
| [OpenHands](python/openhands/README.md) | `nvidia.fabric.openhands` | `nemo-fabric-adapters-openhands` | 3.11+; harness requires 3.12+ |
| [Remote Agent](python/remote-agent/README.md) | `nvidia.fabric.remote-agent` | `nemo-fabric-adapters-remote-agent` | 3.11+ |

Refer to the [Python adapter package guide](python/README.md) for source-layout,
build, and packaging conventions.

## TypeScript Adapter Packages

| Agent Harness | Adapter ID | npm Package | Supported Runtime |
| --- | --- | --- | --- |
| [Cline](typescript/cline/README.md) | `nvidia.fabric.cline` | `nemo-fabric-adapters-cline` | Node.js 22.19+ |
| [Factory Droid](typescript/droid/README.md) | `nvidia.fabric.droid` | `nemo-fabric-adapters-droid` | Node.js 22.19+ |
| [Kilo Code](typescript/kilo/README.md) | `nvidia.fabric.kilo` | `nemo-fabric-adapters-kilo` | Node.js 22.19+ |
| [Kiro](typescript/kiro/README.md) | `nvidia.fabric.kiro` | `nemo-fabric-adapters-kiro` | Node.js 22.19+ |
| [AWS Strands](typescript/strands/README.md) | `nvidia.fabric.strands` | `nemo-fabric-adapters-strands` | Node.js 22.19+ |
| [OpenCode](typescript/opencode/README.md) | `nvidia.fabric.opencode` | `nemo-fabric-adapters-opencode` | Bun 1.4.2+ |
| [Pi](typescript/pi/README.md) | `nvidia.fabric.pi` | `nemo-fabric-adapters-pi` | Node.js 22.19+ |
| [Qwen Code](typescript/qwen/README.md) | `nvidia.fabric.qwen` | `nemo-fabric-adapters-qwen` | Node.js 22.19+ |

OpenCode, Qwen Code, and Kilo Code require HTTPS for remote model endpoints by default. To use a trusted Docker or Kubernetes HTTP gateway, configure `harness.settings.insecure_http_model_hosts` as described in their adapter READMEs. Kilo Code also supports its existing private IPv4 opt-in.

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
| [Claude](python/claude/README.md) | Native Anthropic or a configured Anthropic Messages-compatible provider | `tools.enabled` selects built-ins; a pre-tool hook enforces enabled and blocked names across built-in, MCP, and plugin tools | Normalized: stdio, HTTP, streamable HTTP, and SSE | Normalized `skills.paths` | Not exposed |
| [Codex](python/codex/README.md) | Native OpenAI or a configured Responses-compatible provider | `tools.enabled` and `tools.blocked` unsupported | Normalized: stdio, HTTP, and streamable HTTP | Normalized `SKILL.md` directories | Not exposed |
| [LangChain Deep Agents](python/deepagents/README.md) | LangChain model providers | Middleware enforces `tools.enabled` and `tools.blocked` across built-ins, MCP, and local delegation | Normalized through `langchain-mcp-adapters` | Normalized | Built-in, declarative, and Agent Protocol |
| [Hermes Agent](python/hermes/README.md) | Configurable provider, model, and base URL | `tools.enabled` and `tools.blocked` map to Hermes native toolset selectors | Normalized | Normalized | Not exposed |
| [mini-SWE-agent](python/mini-swe-agent/README.md) | Configured provider and model | Not exposed | Not exposed | Not exposed | Not exposed |
| [NOOA](python/nooa/README.md) | Configured provider and model with optional base URL and temperature | Not exposed | InteractiveAgent: normalized whole servers; BenchAgent: not exposed | InteractiveAgent: normalized `skills.paths`; BenchAgent: not exposed | Not exposed |
| [OpenClaw](python/openclaw/README.md) | Native OpenClaw providers or a configured OpenAI Chat Completions-compatible provider | OpenClaw native policy only | Normalized: stdio, HTTP, streamable HTTP, and SSE without normalized authentication | Normalized `skills.paths` | OpenClaw native behavior |
| [OpenHands](python/openhands/README.md) | Configured LiteLLM provider and model | `tools.enabled` and `tools.blocked` select terminal and file-editor built-ins | Normalized: stdio, HTTP, streamable HTTP, and SSE without authentication or per-server tool filters | Normalized `skills.paths` | Not exposed |
| [Cline](typescript/cline/README.md) | Configured Cline provider and model with an optional compatible base URL | `tools.enabled` and `tools.blocked` map to Cline built-ins | Normalized unauthenticated stdio, streamable HTTP, and SSE without per-server tool filters | Normalized through Cline Agent Plugins v1 | Disabled |
| [Factory Droid](typescript/droid/README.md) | Factory-managed model ID; custom endpoints remain Droid-owned settings | Exact built-in allow/block policy through session tool IDs | Normalized unauthenticated stdio, streamable HTTP, and SSE without per-server tool filters | Normalized `skills.paths` through an isolated personal compatibility directory | Not exposed |
| [Kilo Code](typescript/kilo/README.md) | Local Kilo server and one Kilo session | Not supported | Reuses the session and submits one SDK prompt for ordered text input | Deletes the session, terminates the server, and removes the isolated profile | Adapter-owned loopback service |
| [Kilo Code](typescript/kilo/README.md) | Configured Kilo provider and model with an optional OpenAI-compatible base URL | `tools.enabled` and `tools.blocked` map to native Kilo tool names | Normalized: stdio, streamable HTTP, and SSE | Normalized `skills.paths` | Not exposed |
| [Kiro](typescript/kiro/README.md) | Optional model selected from the Kiro ACP session catalog | ACP permission requests use the closed adapter setting; normalized tool policy is not exposed | Normalized ACP session inputs: stdio, streamable HTTP, and SSE | Not exposed | Kiro native behavior |
| [Strands](typescript/strands/README.md) | Bedrock or explicit OpenAI-compatible Chat Completions endpoint | Built-in tool selection; stock harness permissions | Stdio, streamable HTTP, and SSE | Not exposed | Not exposed |
| [OpenCode](typescript/opencode/README.md) | Configured OpenCode provider and model with an optional OpenAI-compatible base URL | Not exposed | Normalized: stdio and streamable HTTP | Normalized `skills.paths` | Not exposed |
| [Pi](typescript/pi/README.md) | One Pi-catalog provider and model with an optional base URL override | `tools.definitions`, `tools.enabled`, and `tools.blocked` cover built-ins, trusted local modules, and explicit extension tools | Not exposed | Normalized `skills.paths` | Not exposed |
| [Qwen Code](typescript/qwen/README.md) | OpenAI-compatible provider and model | Native `tools.blocked` deny list | Normalized: stdio and streamable HTTP with per-server tool filters | Explicit `skills.paths` | Not exposed |
| [Remote Agent](python/remote-agent/README.md) | Configured remote HTTP API and model | Not exposed | Not exposed | Not exposed | Not exposed |

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

### Environment Values And Names

`environment.env` sets literal values and saves those values in the configuration and run plan. `environment.env_names` lists existing variables to forward from the NeMo Fabric process environment and saves only their names. NeMo Fabric looks up the values and supplies them to the adapter through `RuntimeContext.environment.env`.

For example, use `env={"LOG_LEVEL": "debug"}` for a literal setting and `env_names=["GITHUB_TOKEN"]` to forward a token already available to the NeMo Fabric process. `environment.env` does not expand `$GITHUB_TOKEN` or `${GITHUB_TOKEN}`, and neither field loads a `.env` file. Refer to [Set Environment Values Or Forward Them By Name](../docs/sdk/python.mdx#set-environment-values-or-forward-them-by-name) for examples and runtime refresh behavior.

### Complete FabricConfig Support

`Core` means NeMo Fabric owns the behavior and applies it uniformly before or around
adapter execution. `Yes` means the adapter translates the normalized field into
its harness. `No` means an explicitly configured value fails planning instead
of being ignored. The following table groups provider-specific Relay subfields
and additive extension maps because their support does not vary by adapter:

| `FabricConfig` Field | Claude | Codex | Deep Agents | Hermes Agent | mini-SWE-agent | NOOA | OpenClaw | OpenHands | Cline | Factory Droid | OpenCode | Pi | Qwen Code | Kilo Code | Kiro | Strands | Remote Agent |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `schema_version` | Core | Core | Core | Core | Core | Core | Core | Core | Core | Core | Core | Core | Core | Core | Core | Core | Core |
| `metadata.name`, `.description` | Core | Core | Core | Core | Core | Core | Core | Core | Core | Core | Core | Core | Core | Core | Core | Core | Core |
| `harness.adapter_id`, `.resolution` | Core | Core | Core | Core | Core | Core | Core | Core | Core | Core | Core | Core | Core | Core | Core | Core | Core |
| `harness.settings` | Closed adapter schema | Closed adapter schema | Closed adapter schema | Closed adapter schema | Closed `timeout` schema | Closed schema | Closed command, port-range, and timeout schema | Closed `condenser` schema | Closed schema with no properties | Closed schema with no properties | No settings declared | Closed local-extension and Relay-extension schema | Closed `permission_mode` schema | Closed schema | Closed mode, effort, and permission schema | Closed tool-selection, invocation-limit, and HTTP model-host schema | Closed `base_url`, `api_type`, transport-timeout, and `relay_streaming` schema |
| `workflow.target_id`, `.settings` | No | No | No | No | No | No | No | No | No | No | No | No | No | No | No | No | No |
| `models.<role>.provider` | `anthropic` uses native auth; custom names require an Anthropic Messages-compatible `base_url` and `api_key_env` | `openai` uses native auth; custom names require a Responses-compatible `base_url` and `api_key_env` | Dynamic LangChain provider; custom OpenAI-compatible endpoints require `base_url` and `api_key_env` | Dynamic Hermes provider | Configured provider | Configured provider | OpenClaw provider; custom providers require an OpenAI Chat Completions-compatible `base_url` | Prefixes unqualified model names for LiteLLM | Cline provider | `factory` | OpenCode provider | Pi catalog provider | `openai` | Kilo provider | `kiro` | `bedrock`; `openai` or `nvidia` with an explicit Chat Completions `base_url` | Configured provider |
| `models.<role>.model` | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes; Factory-managed model ID | Yes; passed to OpenCode | Yes; must exist in the Pi catalog | Yes | Yes | Yes; must be advertised by Kiro | Yes; `default` role only | Yes |
| `models.<role>.api_key_env` | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | `KIRO_API_KEY` only | Yes; Chat Completions endpoints only; omit for keyless | Yes |
| `models.<role>.base_url` | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes; compatible endpoint | No; use external Droid settings | Yes; OpenAI-compatible Chat Completions endpoint | Yes; known catalog models only | Yes; OpenAI-compatible Chat Completions endpoint | Yes; OpenAI-compatible endpoint | No | Yes; Chat Completions endpoints only | No; use `harness.settings.base_url` |
| `models.<role>.temperature` | No | No | Yes | Yes | Yes | Yes | Yes | Yes | No | No | Yes: OpenAI-compatible `base_url` only | No | Yes | Yes | No | No | Yes |
| `models.<role>.settings.<key>` | No keys declared | No keys declared | No keys declared | No keys declared | No keys declared | `client_type` | No keys declared | No keys declared | No keys declared | No keys declared | No keys declared | No keys declared | No keys declared | No keys declared | No keys declared | No keys declared | `max_tokens` for Anthropic Messages |
| `models.<role>.top_p` | No | No | Yes | Yes | Yes: passed through LiteLLM | No | Yes | Yes | No | No | Yes: OpenAI-compatible `base_url` only | No | Yes | Yes | No | No | Yes: translated to the selected API protocol |
| `models.<role>.max_tokens` | No | No | Yes | Yes | Yes: passed through LiteLLM | No | Yes | Yes | No | No | No | No | No | No | No | No | Yes: translated to the selected API protocol |
| `instructions.system` | `replace`, `append` | `replace` with base instructions | `replace` | `replace` | `replace` | `replace` | `replace` | `replace`, `append` | `replace` with Cline override prompt | `replace`, `append` | `replace` with OpenCode base instructions | `replace` with Pi base instructions | `replace`, `append` | `replace` | No | `append` | `replace` |
| `runtime.input_schema`, `.output_schema` | Core | Core | Core | Core | Core | Core | Core | Core | Core | Core | Core | Core | Core | Core | Core | Core | Core |
| `runtime.artifacts`, `.timeout_seconds` | Core | Core | Core | Core | Core | Core | Core | Core | Core | Core | Core | Core | Core | Core | Core | Core | Core |
| `runtime.max_turns` | Yes | No | Yes; maps to LangGraph supersteps | Yes; iteration limit | Yes | No | No | Yes; per invocation | No | No | No | No | No | Yes; maps to build-agent steps | No | No | No |
| `environment.provider`, `.control_location`, `.ownership` | Core | Core | Core | Core | Core | Core | Core | Core | Core | Core | Core | Core | Core | Core | Core | Core | Core |
| `environment.workspace`, `.artifacts`, `.env`, `.env_names` | Core | Core | Core | Core | Core | Core | Core | Core | Core | Core | Core | Core | Core | Core | Core | Core | Core |
| `environment.connection`, `.metadata`, `.settings` | Environment-provider-owned | Environment-provider-owned | Environment-provider-owned | Environment-provider-owned | Environment-provider-owned | Environment-provider-owned | Environment-provider-owned | Environment-provider-owned | Environment-provider-owned | Environment-provider-owned | Environment-provider-owned | Environment-provider-owned | Environment-provider-owned | Environment-provider-owned | Environment-provider-owned | Environment-provider-owned | Environment-provider-owned |
| `tools.definitions` | No | No | No | No | No | No | No | No | No | No | No | Yes; trusted local module factories | No | No | No | No | No |
| `tools.enabled`, `.blocked` | Yes | No | Yes | Yes; native selectors are Hermes toolset names | No | No | Yes | Yes; terminal and file editor | Yes; Cline built-in names | Yes; Droid built-in IDs | No | Yes | `tools.blocked`: Yes; `tools.enabled`: No | Yes; native Kilo tool names | No | No | No |
| `skills.paths` | Yes | Yes | Yes | Yes | No | InteractiveAgent: Yes, BenchAgent: No | Yes | Yes | Yes; Agent Plugins v1 | Yes; isolated personal compatibility directory | Yes | Yes | Yes | Yes | No | No | No |
| `mcp.servers.<name>.transport`, `.url` with `harness_native` exposure | Yes | Yes | Yes | Yes | No | InteractiveAgent: Yes, BenchAgent: No | Yes | Yes; no authentication or per-server tool filters | Yes; unauthenticated only | Yes; unauthenticated stdio, streamable HTTP, and SSE | Yes: stdio `args` and `env`, streamable HTTP `custom_headers` | No | Yes: stdio and streamable HTTP with per-server tool filters | Yes: stdio, streamable HTTP, and SSE | Yes: stdio, streamable HTTP, and SSE without authentication or tool filters | Yes; stdio, streamable HTTP, and SSE; no normalized authentication or tool filters | No |
| `mcp.servers.<name>.exposure = "fabric_managed"` | No; not implemented | No; not implemented | No; not implemented | No; not implemented | No | No; not implemented | No; not implemented | No | No | No | No | No | No | No | No | No | No |
| `telemetry.providers.relay` | Yes | Yes | Yes | Yes | Yes | Yes | No | No | No | No | No | Yes, supports embedded collector-backed Agent Trajectory Observability Format (ATOF) streaming | No | No | No | No | Yes, supports collector-backed ATOF streaming |
| `telemetry.providers.native` | No | Yes; OpenTelemetry | Yes; OpenTelemetry and OpenInference | No | No | No | No | No | No | No | No | No | No | No | No | No | No |
| `telemetry.providers.<provider>.config` | Declared-provider pass-through | Declared-provider pass-through | Declared-provider pass-through | Declared-provider pass-through | Declared-provider pass-through | Declared-provider pass-through | No | No | No | No | No | Declared-provider pass-through | No | No | No | No | No |
| `relay.project`, `.output_dir`, `.observability` | Yes | Yes | Yes | Yes | Yes | Yes | No | No | No | No | No | Yes | No | No | No | No | Uses the named external collector sink when selected; config is not sent to the remote service |
| `relay.components`, `.policy` | Yes | Yes | Yes | Yes | Yes | Yes | No | No | No | No | No | Yes | No | No | No | No | Not sent to the remote service |
| Other additive `extensions` on typed config objects | Rejected unless declared by the descriptor | Rejected unless declared by the descriptor | Rejected unless declared by the descriptor | Rejected unless declared by the descriptor | Rejected unless declared by the descriptor | Rejected unless declared by the descriptor | Rejected unless declared by the descriptor | Rejected unless declared by the descriptor | Rejected unless declared by the descriptor | Rejected unless declared by the descriptor | Rejected unless declared by the descriptor | Rejected unless declared by the descriptor | Rejected unless declared by the descriptor | Rejected unless declared by the descriptor | Rejected unless declared by the descriptor | Rejected unless declared by the descriptor | Rejected unless declared by the descriptor |

The selected model role is `default`, or the sole configured role when no
`default` exists. More than one role without `default` fails planning. Kiro and Strands accept only the `default` role.
All bundled adapters except Hermes publish a descriptor-owned `model_schema`
for every configured model role. The Claude and Codex native providers
(`anthropic` and `openai`, respectively) keep the existing authentication path.
Other Claude and Codex providers remain valid only with an explicit `base_url`
and `api_key_env`. Deep Agents keeps dynamic LangChain provider selection. Each
published schema rejects undeclared `ModelConfig.settings` during planning and
reports each issue through `doctor(...)` before adapter startup. Existing
configurations that supplied `top_p` and `max_tokens` as flattened model
extensions retain the same wire shape when loaded as normalized fields. Legacy
adapter descriptors that accept either name through `extension_schemas.model`
continue receiving it in `AgentModelConfig.extensions` while adapters that advertise
the normalized capability receive the typed field instead.
`runtime.max_turns` is optional; omitting it preserves adapter-native defaults
without creating a compatibility requirement.

## Runtime and Observability Compatibility

All bundled adapters use a language-specific persistent adapter host with an
ordered `start` → `invoke*` → `stop` protocol.

NeMo Relay records raw events in ATOF and produces normalized trajectories in
Agent Trajectory Interchange Format (ATIF).

| Agent Harness | State Retained Across Turns | Relay Integration | Per-Turn Behavior | Stop Behavior | Remote Service |
| --- | --- | --- | --- | --- | --- |
| [Claude](python/claude/README.md) | `ClaudeSDKClient` and Claude session ID | Runtime-owned Relay CLI gateway and generated Claude hooks | Calls `client.query()`, validates the session ID, and collects ATOF, ATIF, and OTLP | Disconnects the client, stops the gateway, and removes the generated plugin | Not implemented |
| [Codex](python/codex/README.md) | `AsyncCodex` app-server client and SDK thread | Runtime-owned Relay CLI gateway and Codex SDK hooks | Reuses the SDK thread and persists its thread ID | Closes the SDK client and app server, then stops the gateway | Not implemented |
| [LangChain Deep Agents](python/deepagents/README.md) | Compiled LangGraph agent, checkpointer, and thread ID | NeMo Relay Python SDK integration added when the agent is compiled | Creates a fresh Relay request scope and callback for each invocation | Closes the checkpointer; no gateway process | Not implemented |
| [Hermes Agent](python/hermes/README.md) | `AIAgent`, `SessionDB`, and conversation history | Hermes Agent NeMo Relay plugin context | Finalizes and flushes Relay after each invocation | Closes the agent and database, then exits the plugin context | Not implemented |
| [mini-SWE-agent](python/mini-swe-agent/README.md) | Conversation history | Adapter-owned subclass with NeMo Relay Python SDK scopes | Creates a fresh Relay plugin and request scope, emits step, model, and bash-action telemetry, and collects artifacts | Clears the agent and Relay state | Not implemented |
| [NOOA](python/nooa/README.md) | InteractiveAgent queue dispatcher or BenchAgent task state | Adapter-owned Relay middleware and generated Relay configuration | InteractiveAgent dispatches queued requests; BenchAgent evaluates one task | Closes agent resources and Relay state | Not implemented |
| [OpenClaw](python/openclaw/README.md) | OpenClaw Gateway session selected by Fabric runtime ID | Not supported | Sends a terminal Chat Completions request to the isolated loopback Gateway | Terminates the Gateway process tree and removes its temporary config and state | Adapter-owned loopback service |
| [OpenHands](python/openhands/README.md) | OpenHands `Conversation` and event history | Not supported | Sends one message and runs the existing conversation to its next terminal state | Closes the conversation | Not implemented |
| [Cline](typescript/cline/README.md) | Local `ClineCore` and interactive Cline session | Not supported | Initializes without a prompt at runtime startup, then calls `send()` with the retained session ID | Stops the session, disposes the core, and removes runtime-scoped plugin and data directories | Not implemented |
| [Factory Droid](typescript/droid/README.md) | TypeScript SDK `DroidSession` | Not supported | Calls `stream()` for each ordered prompt on the retained session | Closes the session and its caller-installed CLI subprocess | Not implemented |
| [OpenCode](typescript/opencode/README.md) | Embedded OpenCode host and session | Not supported | Reuses the session and calls `prompt()`, `wait()`, and `context()` for ordered text input | Removes the session and closes the host | Not implemented |
| [Pi](typescript/pi/README.md) | In-memory Pi `AgentSession` | Runtime-owned Relay 0.9 or 0.10 CLI gateway and explicit Pi extension | Reuses the session and calls `prompt()` for ordered text input; with `streaming=True`, routes per-invocation model-turn ATOF for successful redirects through the embedded collector; startup `model_redirect` marks remain in configured Relay ATOF artifacts and are not included in `invoke_stream()`; `relay_artifacts` does not include local ATIF | Aborts work, emits extension shutdown so local ATIF finalizes on disk, disposes the session, and then stops the gateway | Not implemented |
| [Qwen Code](typescript/qwen/README.md) | SDK query and bundled Qwen CLI child process | Not supported | Reuses one live SDK query across ordered text turns | Closes the query and removes isolated temporary settings | Not implemented |
| [Kiro](typescript/kiro/README.md) | Kiro CLI V3 child process and ACP session | Not supported | Reuses one ACP session and submits ordered text prompts | Closes the ACP session and terminates the Kiro child process | Not implemented |
| [Strands](typescript/strands/README.md) | Strands CLI ACP child process and session | Not supported | Ordered text prompts retain one bounded conversation | Terminates the child process and removes temporary CLI state | Not implemented |
| [Remote Agent](python/remote-agent/README.md) | `httpx.AsyncClient` and user/assistant transcript | Remote Relay publishes to a shared ATOF collector | Registers the request ID, maps it into body metadata, sends one HTTP request, and retains the completed transcript | Closes the HTTP client | Implemented over HTTP(S) |

Telemetry output names use the descriptor contract values. Claude, Codex, Hermes Agent, mini-SWE-agent, and Pi can emit NeMo Relay ATIF, OpenTelemetry, and
OpenInference output. Deep Agents supports the same Relay outputs plus native
OpenTelemetry and OpenInference; Codex also supports native OpenTelemetry. The
Remote Agent adapter maps the request ID into the remote API body. The
independently instrumented service publishes correlated ATOF to the collector
base URL shared by its Relay configuration and `FabricConfig`. The collector can
serve multiple runtimes when each request ID is unique.

Refer to the [SDK health-check support table](../docs/sdk/health-checks.mdx#native-evidence-by-adapter) for supported levels, native evidence, and adapters without runtime health support.

Local Cline runtimes support `Fabric.check()` at the `live`, `active`, and `ready` levels before and during the first invocation. Checks verify native session residency and closure, request-interface state, and a fresh session snapshot. Refer to the [Cline health contract](typescript/cline/README.md#runtime-health).

Local Pi runtimes also support `Fabric.check()` at the `live`, `active`, and
`ready` levels. These observations can run during an invocation and inspect
native session and local setup state. Refer to the [Pi health contract](typescript/pi/README.md#runtime-health).

OpenHands runtimes support `live` and `active` through native conversation lifecycle fields. Before native lazy initialization completes during the first invocation, `active` returns `unknown/capabilities_unverified`; `ready` remains unsupported. Refer to the [OpenHands health contract](python/openhands/README.md#runtime-health).

Prepared and attached OpenClaw Gateways support service-scoped `Fabric.check()` calls before any runtime exists. OpenClaw runtimes also support `live`, `active`, and `ready`; runtime `active` verifies startup, selected-agent registration, and the enabled Chat Completions interface. Both scopes use native `/readyz`, including configured channels. Refer to the [Gateway health contract](python/openclaw/README.md#gateway-health) and [runtime health contract](python/openclaw/README.md#runtime-health).

Shared lifecycle, Relay gateway, hook, and payload helpers are documented in
the [adapter utilities guide](python/common/README.md).

OpenCode runtime health supports `live`, `active`, `ready`. Refer to the [OpenCode health guide](typescript/opencode/README.md#runtime-health) for the qualified evidence and limits.

Qwen Code runtime health supports `live`, `active`, `ready`. Refer to the [Qwen Code health guide](typescript/qwen/README.md#runtime-health) for the qualified evidence and limits.

Strands runtime health supports `live`. Refer to the [Strands health guide](typescript/strands/README.md#runtime-health) for the qualified evidence and limits.
