<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Fabric LangChain Deep Agents Adapter

Runs a [LangChain Deep Agents](https://github.com/langchain-ai/deepagents) agent
through Fabric's inline Python adapter lifecycle. The same adapter supports
one-shot, multi-turn, and resumed execution.

To install just the Deep Agents adapter by itself:

```bash
pip install "nemo-fabric[deepagents]"
```

To install just the Deep Agents adapter along with the NeMo Fabric Runtime:

```bash
pip install "nemo-fabric[deepagents, runtime]"
```

## Model and Authentication

The adapter builds a LangChain chat model from Fabric's `models.default` config.
For `nvidia` (or an unspecified provider) it targets NVIDIA-hosted,
OpenAI-compatible endpoints (`https://integrate.api.nvidia.com/v1`) via
`ChatOpenAI`; `openai` and `openai-compatible` also use `ChatOpenAI` with the
provider's own default endpoint. Any other provider is constructed through
`langchain.chat_models.init_chat_model`, so additional backends can be added
without changing the adapter.

`models.default.api_key_env` names the environment variable holding the API key,
and defaults **per provider** — `NVIDIA_API_KEY` for `nvidia` (or an unspecified
provider) and `OPENAI_API_KEY` for `openai`. Every other provider — including
`openai-compatible` and any `init_chat_model` backend — must set `api_key_env`
explicitly (a missing one is a normalized configuration failure), so a key is
never sent to the wrong endpoint.

Because `models.default.api_key_env` is provider-specific, the adapter declares no
static env requirement; a runtime **preflight** verifies that the `deepagents`
package is importable and the configured credential is set, and returns a
normalized failure otherwise. `fabric doctor` validates adapter resolution.

Fabric maps the following into the harness:

- `models.default.model` / `harness.settings.model_name` selects the model.
- `models.default.provider` selects the client (`nvidia`/`openai` → OpenAI-compatible).
- `models.default.temperature` / `harness.settings.temperature` sets sampling.
- `harness.settings.base_url` overrides the model endpoint.
- `harness.settings.system_prompt` becomes the Deep Agents `system_prompt`.
- `environment.workspace` roots the Deep Agents filesystem backend
  (`FilesystemBackend(root_dir=..., virtual_mode=True)`). `virtual_mode`
  confines the agent to the workspace: absolute paths and `..` cannot escape
  `root_dir`.
- Routed `skills` (`native.skill_paths`) become the Deep Agents `skills` sources.
- Configured MCP servers are loaded as Deep Agents tools via
  `langchain-mcp-adapters`. A misconfigured server (non-mapping, empty target,
  unsupported transport) is a normalized configuration failure, not a silent drop.
- `tools.blocked` is enforced by middleware across the full tool surface — Deep
  Agents built-ins (including `task`), MCP tools, and **delegated subagents**
  alike. Use Deep Agents/native tool names in the blocked list.
- `harness.settings.deepagents` forwards a small set of **documented,
  JSON-serializable** `create_deep_agent` options (currently `subagents` and
  `interrupt_on`). It is not a general Python-object escape hatch: the SDK config
  round-trips through JSON and Rust planning, so `AgentMiddleware`, `BaseTool`
  instances, and Python callables cannot cross the boundary. Fabric-owned
  arguments (`model`, `tools`, `backend`, `skills`, `system_prompt`, `middleware`,
  `checkpointer`) cannot be overridden through this passthrough, and an unknown or
  unsupported key is a normalized configuration failure rather than a silently
  dropped setting.

### Subagents

Deep Agents can delegate to subagents through its built-in `task` tool. Subagents
**inherit** the parent run's model, tools, skills, workspace, telemetry, and
permissions. When `tools.blocked` is configured, Fabric supplies an explicitly
gated `general-purpose` subagent and gates every declarative local subagent, so
delegation cannot broaden capabilities beyond the parent. Remote and precompiled
subagents are rejected in that case because their execution cannot be governed by
the local middleware. Independently configured subagent tools, skills, models,
MCP servers, middleware, or permissions are **not** exposed through the Fabric SDK
yet; a `subagents` definition here only carries JSON-shaped fields.

The normalized result includes the final response, buffered messages and
per-step events, LangGraph thread id, token usage (and cost when the provider
reports it), and errors. Usage aggregates the current turn across the main agent
and any delegated subagents (streamed with `subgraphs=True`). Configuration and
preflight failures (a missing credential, an absent `deepagents` package, an
invalid MCP server, or a passthrough option) are returned as a
normalized failure result rather than a raw traceback.

## Runtime Modes

A one-shot `run` starts a local adapter host, compiles one Deep Agents graph,
opens its async LangGraph checkpointer, invokes the graph once with `astream`,
and then closes the checkpointer and host. The result contains the final agent
message, buffered messages and per-step events, usage, and the LangGraph thread
ID. Each one-shot run gets a fresh Fabric `runtime_id`, so `resumed` is `false`.

Multi-turn and resume are keyed by the Fabric `runtime_id`, which is stable
across `invoke` calls in a started runtime (`start_runtime`) and fresh for each
one-shot run. The host compiles the graph and opens the checkpointer once during
runtime start. Every turn reuses both native objects and the same LangGraph
thread ID; later turns report `resumed` as `true`. The checkpointer lives under
`harness.settings.state_dir` (default the runtime artifacts directory) and is
closed during runtime stop. Fabric owns the runtime-to-thread correlation
record; LangGraph owns the transcript.

The `deepagents_config()` builder in `examples/code_review_agent` is the SDK
example. Run it from the CLI with
`python -m examples.code_review_agent --variant deepagents --input "..."`, or
drive the SDK directly:

```python
from examples.code_review_agent import BASE_DIR, deepagents_config
from nemo_fabric import Fabric

config = deepagents_config()
client = Fabric()

# One-shot: each run gets a fresh runtime, so `resumed` is False.
result = await client.run(
    config, base_dir=BASE_DIR, input="Review the workspace changes."
)
print(result["output"]["response"])

# Multi-turn + resume: one started runtime keeps the LangGraph thread across turns.
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
  one-shot, multi-turn, resumed, and subagent-enabled runs:
  - `nemo_relay.integrations.deepagents.add_nemo_relay_integration(...)` injects
    Deep Agents-aware **middleware** that routes model and tool calls through
    Relay and emits skill/subagent configuration marks.
  - The top-level invocation runs inside a
    `nemo_relay.scope.scope("deepagents-request", nemo_relay.ScopeType.Agent)`
    scope, so the whole Fabric turn is captured under one Agent scope.
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

**Subagent boundary.** In-process, dictionary-style subagents are instrumented
with the same Relay middleware, so their model/tool calls appear under the same
trajectory. Remote and precompiled subagents (those defined with `graph_id` or
`url`) are **out of scope**: their internals execute in a separate runtime and
must be instrumented there with their own Relay integration.

### Typed Relay configuration

Enable Relay on a `FabricConfig` with the typed helpers — no gateway process or
CLI flags are involved:

```python
from nemo_fabric import (
    RelayAtifConfig,
    RelayAtofConfig,
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
            output_directory="./artifacts/relay",
            filename="events.atof.jsonl",
            mode="overwrite",
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
