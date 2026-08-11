<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Fabric LangChain Deep Agents Adapter

Runs a [LangChain Deep Agents](https://github.com/langchain-ai/deepagents) agent
inside NeMo Fabric's persistent Python adapter host. One started runtime retains the
compiled graph, checkpointer, and LangGraph thread across ordered invocations.

## Install

The following table shows which components each installation provides:

| Installation | Runtime | Adapter | Harness | NeMo Relay Python Package |
| --- | --- | --- | --- | --- |
| `pip install "nemo-fabric[deepagents]"` | Yes | Yes | Yes | No |
| `pip install "nemo-fabric[deepagents,relay]"` | Yes | Yes | Yes | Yes |
| `pip install "nemo-fabric-adapters-deepagents[harness]"` | No | Yes | Yes | No |
| `pip install "nemo-fabric-adapters-deepagents[full]"` | No | Yes | Yes | Yes |
| `pip install "nemo-fabric-adapters-deepagents[relay]"` | No | Yes | No | Yes |
| `pip install nemo-fabric-adapters-deepagents` | No | Yes | No | No |

For an environment-managed stack, use `deepagents>=0.6.12,<0.7.0`,
`langchain>=1.3,<2.0`, and `langgraph>=1.2,<2.0`. For split runtime and adapter
environments, configure `ADAPTER_PYTHON` and use matching NeMo Fabric release
versions. Refer to the
[installation guide](https://docs.nvidia.com/nemo/fabric/getting-started/install#install-an-adapter-and-harness-without-the-runtime).

## Model and Authentication

The adapter builds a LangChain chat model from the selected NeMo Fabric model
role: `models.default`, or the sole configured role when `default` is absent.
The `openai`, `nvidia`, and `openai-compatible` providers use `ChatOpenAI`;
`nvidia` and `openai-compatible` require an explicit compatible `base_url`.
Any other provider is constructed through
`langchain.chat_models.init_chat_model`, so LangChain-supported backends do not
require adapter-specific branches.

`models.<role>.api_key_env` names the environment variable holding the API key,
and defaults to `OPENAI_API_KEY` only for the native `openai` provider. Every
other provider must set `api_key_env` explicitly (a missing one is a normalized
configuration failure), so a key is never sent to the wrong endpoint.

Because `models.<role>.api_key_env` is provider-specific, the adapter declares no
static env requirement; a runtime **preflight** verifies that the `deepagents`
package is importable and the configured credential is set. A failed preflight
fails runtime start with a stable lifecycle error.

NeMo Fabric maps the following into the harness:

- The selected `models` role supplies `model`, `provider`, `api_key_env`,
  `base_url`, and `temperature`.
- `instructions.system` becomes the Deep Agents `system_prompt`.
- `runtime.timeout_seconds` sets the NeMo Fabric invocation deadline.
- `environment.workspace` roots the Deep Agents filesystem backend
  (`FilesystemBackend(root_dir=..., virtual_mode=True)`). `virtual_mode`
  confines the agent to the workspace: absolute paths and `..` cannot escape
  `root_dir`.
- Routed `skills` (`native.skill_paths`) become the Deep Agents `skills` sources.
- Configured MCP servers are loaded as Deep Agents tools via
  `langchain-mcp-adapters`. A misconfigured server (non-mapping, empty target,
  unsupported transport) is a normalized configuration failure, not a silent drop.
- `tools.enabled` and `tools.blocked` are enforced by middleware across the full
  tool surface: Deep Agents built-ins (including `task`), MCP tools, and
  **delegated subagents** alike. Use Deep Agents-native tool names.
- `harness.settings.deepagents` accepts the JSON-serializable Deep Agents
  `interrupt_on` and `subagents` options. The descriptor schema rejects unknown
  settings and fields before runtime start.

### Harness Settings

Use `harness.settings.deepagents` for Deep Agents-native controls that do not
have a normalized NeMo Fabric field:

```python
from nemo_fabric import HarnessConfig

harness = HarnessConfig(
    adapter_id="nvidia.fabric.langchain.deepagents",
    settings={
        "deepagents": {
            "interrupt_on": {
                "write_file": {
                    "allowed_decisions": ["approve", "edit", "reject"],
                    "description": "Review this file write.",
                }
            },
            "subagents": [
                {
                    "name": "researcher",
                    "description": "Researches the workspace before implementation.",
                    "system_prompt": "Investigate the request and return concise findings.",
                }
            ],
        }
    },
)
```

The `deepagents` object is closed and supports the following properties:

- `interrupt_on` maps a Deep Agents tool name to a boolean or an object with
  required `allowed_decisions`. Decisions are `approve`, `edit`, `reject`, or
  `respond`. The object can also contain a static `description` and an
  `args_schema` JSON Schema. An omitted map defaults to no caller-defined
  interrupts. Callable descriptions and `when` predicates cannot cross the
  JSON configuration boundary.
- `subagents` defaults to no caller-defined subagents and accepts declarative
  synchronous or Agent Protocol asynchronous subagents. A declarative subagent
  requires `name`, `description`, and `system_prompt`; it can also contain
  a `provider:model` override, its own `interrupt_on` map, skill source paths,
  and a JSON `response_format`. An asynchronous subagent requires `name`,
  `description`, and `graph_id`; it can also contain `url` and string-valued
  `headers`.

Python middleware, `FilesystemPermission` objects, Python tool objects, and
precompiled `runnable` subagents are not exposed through `harness.settings`.
When `tools.enabled` or `tools.blocked` is configured, NeMo Fabric applies the
policy to declarative subagents and rejects asynchronous subagents because
their remote tools cannot be gated locally.

### Subagents

Deep Agents can delegate through its built-in `task` tool. The built-in
subagent **inherits** the parent run's model, tools, skills, workspace,
telemetry, and permissions. When a normalized tools policy is configured,
NeMo Fabric supplies an explicitly gated `general-purpose` subagent so
delegation cannot broaden capabilities beyond the parent. Caller-defined
declarative subagents run through the same local graph. Agent Protocol
subagents run asynchronously on their configured server. Precompiled
subagents are not exposed through the public NeMo Fabric SDK because their
`runnable` objects cannot cross the JSON configuration boundary.

The normalized result includes the final response, buffered messages and
per-step events, LangGraph thread id, token usage (and cost when the provider
reports it), and errors. Usage aggregates the current turn across the main agent
and any delegated subagents (streamed with `subgraphs=True`). Configuration and
preflight failures (a missing credential, an absent `deepagents` package, or an
invalid MCP server) fail runtime start before an invocation is accepted.

## Runtime Lifecycle

NeMo Fabric starts one local adapter host for every runtime. During runtime start,
the host compiles one Deep Agents graph, opens its async LangGraph checkpointer,
and creates one thread ID. Every invocation reuses those native objects; later
turns report `resumed` as `true`. The checkpointer lives under
the NeMo Fabric artifact root, scoped by runtime ID, and is closed during
runtime stop. The live host owns the thread identity, and LangGraph owns the
transcript.

`Fabric.run(...)` is a convenience over that same lifecycle: it starts the
runtime, invokes it once, and stops it. It does not use a separate adapter
entrypoint or execution path.

The `deepagents_config()` builder in `examples/code_review_agent` is the SDK
example. Run it from the CLI with
`python -m examples.code_review_agent --variant deepagents --input "..."`, or
drive the SDK directly:

```python
from examples.code_review_agent import BASE_DIR, deepagents_config
from nemo_fabric import Fabric

config = deepagents_config()
client = Fabric()

# Single invocation through the standard runtime lifecycle.
result = await client.run(
    config, base_dir=BASE_DIR, input="Review the workspace changes."
)
print(result["output"]["response"])

# Multi-turn: one started runtime keeps the LangGraph thread across turns.
async with await client.start_runtime(config, base_dir=BASE_DIR) as runtime:
    await runtime.invoke(input="Remember the value 42.")
    reply = await runtime.invoke(input="What value did I ask you to remember?")
    # reply["output"]["resumed"] is True and the response recalls "42".
    print(reply["output"]["resumed"], reply["output"]["response"])
```

## Telemetry

NeMo Relay is Deep Agents' single, SDK-native observability path — the adapter
does not expose gateway, CLI, or plugin launch modes for this harness. Relay is
**optional**: `nemo_relay` is imported lazily and only when telemetry is enabled,
so the core install stays Relay-neutral at import time. Relay telemetry and
`Runtime.invoke_stream()` require one of the installations in the table that
includes the NeMo Relay Python package.

- **Relay** (`telemetry.providers.relay`): the SDK-native integration attaches
  three complementary pieces around `create_deep_agent`, applied uniformly to
  single-invocation, multi-turn, and subagent-enabled runs:
  - `nemo_relay.integrations.deepagents.add_nemo_relay_integration(...)` injects
    Deep Agents-aware **middleware** that routes model and tool calls through
    Relay and emits skill/subagent configuration marks.
  - The top-level invocation runs inside a
    `nemo_relay.scope.scope("deepagents-request", nemo_relay.ScopeType.Agent)`
    scope, so the whole NeMo Fabric turn is captured under one Agent scope.
  - `NemoRelayDeepAgentsCallbackHandler()` is added to the LangGraph run config
    (without dropping consumer-provided callbacks) to capture LangGraph scopes
    and human-in-the-loop interrupt/resume marks.

  Runs emit ATOF/ATIF artifacts to the configured output directory, referenced in
  the normalized result's `relay_artifacts` (and the `RunResult` `ArtifactManifest`).
  OTel/OpenInference export is available through the relay plugin config; the
  example provides `with_relay_otel(...)` and
  `with_relay_openinference(...)` variants.

  Telemetry is a separate failure domain from the agent turn. After the agent has
  been invoked, no telemetry fault — a failed scope close, a failed export flush,
  or a failed artifact scan — changes the functional outcome: it is reported in the
  `telemetry` block instead, as `telemetry.degraded: true` plus a `telemetry.error`
  message. A turn the agent completed therefore stays `completed`, and a turn the
  agent failed stays failed with its own `error`; the telemetry fault never
  overwrites either. Faults from more than one stage are joined into that one
  message rather than the first one winning. Both keys are absent on a clean run.

  `telemetry.degraded` is the machine-readable signal to branch on. After a scope
  or flush fault the run is degraded but `relay_artifacts` is still populated,
  because a partial trajectory is usually worth reading — treat it as untrusted
  rather than absent. When artifact collection itself is what failed there is
  nothing to reference, so `relay_artifacts` is absent entirely.

  A telemetry failure that happens *before* the agent runs leaves no functional
  outcome to preserve, so it is reported as an invocation `error` as well.

  Relay's scope stack lives in the process and outlives a single invocation, so a
  fault that leaves a scope current poisons the runtime rather than just the turn.
  When that happens the runtime is quarantined: every later turn keeps running and
  stays `completed`, but is no longer wrapped in a request scope, reports
  `telemetry.degraded: true` with a sticky message, and references no
  `relay_artifacts` of its own — the artifacts on disk belong to the earlier turns.
  This contains the damage rather than repairing it: the Relay middleware attached
  to the agent at start still emits, and those events nest under the stale scope, so
  a quarantined runtime's trajectory is untrustworthy rather than empty. The
  quarantine deliberately survives `stop()`/`start()`, because restarting the
  runtime does not clean the process's scope stack.

  On the turn the fault happened, `telemetry.error` carries it verbatim. On the
  turns that inherit the quarantine it appears as `telemetry.quarantine_cause`
  instead, so a consumer counting or matching per-turn errors does not see the same
  fault reported once per remaining turn.
- **Native** (`telemetry.providers.native.config`): the provider config
  OpenTelemetry/OpenInference exporter is applied and spans export directly to
  the configured collector, without writing ATOF/ATIF relay artifacts.

**Subagent boundary.** The built-in and caller-defined declarative subagents
are instrumented with the same Relay middleware, so their model and tool calls
appear under the same trajectory. Agent Protocol subagents execute on their
configured server and are outside the local adapter's Relay instrumentation.
Precompiled subagents are not exposed through the public NeMo Fabric
configuration.

### Typed Relay configuration

Enable Relay on a `FabricConfig` with the typed helpers — no gateway process or
CLI flags are involved:

```python
from nemo_fabric import (
    RelayAtifConfig,
    RelayAtofConfig,
    RelayAtofFileSinkConfig,
    RelayObservabilityConfig,
)
from examples.code_review_agent import deepagents_config

# Start from a complete Deep Agents configuration, then enable typed Relay telemetry.
config = deepagents_config()
config.enable_relay(
    output_dir="./artifacts/relay",
    observability=RelayObservabilityConfig(
        atof=RelayAtofConfig(
            enabled=True,
            sinks=[
                RelayAtofFileSinkConfig(
                    output_directory="./artifacts/relay",
                    filename="events.atof.jsonl",
                    mode="overwrite",
                )
            ],
        ),
        atif=RelayAtifConfig(
            enabled=True,
            output_directory="./artifacts/relay",
            filename_template="trajectory-{session_id}.atif.json",
            agent_name="deepagents-agent",
        ),
    ),
)
```
