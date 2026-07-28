<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Fabric LangChain Deep Agents Adapter

Runs a [LangChain Deep Agents](https://github.com/langchain-ai/deepagents) agent
inside NeMo Fabric's persistent Python adapter host. One started runtime retains the
compiled graph, checkpointer, and LangGraph thread across ordered invocations.

To install just the Deep Agents adapter by itself:

```bash
pip install "nemo-fabric[deepagents]"
```

To install just the Deep Agents adapter along with the NeMo Fabric Runtime:

```bash
pip install "nemo-fabric[deepagents, runtime]"
```

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
fails runtime start with a stable lifecycle error. `fabric doctor` validates
adapter resolution.

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
- The current Deep Agents descriptor does not declare `settings_schema`, so
  `harness.settings` must remain empty. Planning rejects non-empty settings.
  Caller-defined `subagents` and `interrupt_on` controls remain unavailable
  through the public NeMo Fabric configuration until a follow-up descriptor
  schema declares them.

### Subagents

Deep Agents can delegate through its built-in `task` tool. The built-in
subagent **inherits** the parent run's model, tools, skills, workspace,
telemetry, and permissions. When a normalized tools policy is configured,
NeMo Fabric supplies an explicitly gated `general-purpose` subagent so
delegation cannot broaden capabilities beyond the parent. Caller-defined,
remote, and precompiled subagents are not exposed through the public NeMo Fabric
SDK.

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
so the core install stays Relay-neutral at import time. Install it through Relay's
own `deepagents` integration extra:

```bash
pip install "nemo-fabric-adapters-deepagents[relay]"   # -> nemo-relay[deepagents]
```

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
- **Native** (`telemetry.providers.native.config`): the provider config
  OpenTelemetry/OpenInference exporter is applied and spans export directly to
  the configured collector, without writing ATOF/ATIF relay artifacts.

**Subagent boundary.** The built-in subagent is instrumented with the same
Relay middleware, so its model and tool calls appear under the same trajectory.
Caller-defined, remote, and precompiled subagents are not currently exposed
through the public NeMo Fabric configuration.

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
