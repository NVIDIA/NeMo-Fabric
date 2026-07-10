<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# LangChain Deep Agents Adapter

Runs a [LangChain Deep Agents](https://github.com/langchain-ai/deepagents) agent
through Fabric's inline Python adapter lifecycle. The same adapter supports
one-shot, multi-turn, and resumed execution.

Install Fabric with the adapter dependency before running it:

```bash
python3 -m pip install -e ".[deepagents]"
```

## Model and Authentication

The adapter builds a LangChain chat model from Fabric's `models.default` config.
For `nvidia` (or an unspecified provider) it targets NVIDIA-hosted,
OpenAI-compatible endpoints (`https://integrate.api.nvidia.com/v1`) via
`ChatOpenAI`; a plain `openai` provider uses ChatOpenAI's own default endpoint.
Set `models.default.api_key_env` (default `NVIDIA_API_KEY`) to the environment
variable holding the key. Providers other than `nvidia`/`openai` are constructed
through `langchain.chat_models.init_chat_model` so additional backends can be
added without changing the adapter.

Because `models.default.api_key_env` is provider-specific, the adapter declares no
static env requirement; a runtime **preflight** verifies that the `deepagents`
package is importable and the configured credential is set, and returns a
normalized failure otherwise. `fabric doctor` validates adapter resolution.

`api_key_env` defaults per provider — `NVIDIA_API_KEY` for `nvidia` (or an
unspecified provider) and `OPENAI_API_KEY` for `openai`. Any other provider must
set `api_key_env` explicitly so a key is never sent to the wrong endpoint.

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
- `tools` (Fabric's `config.tools` allow-list) is enforced by a gating middleware
  across the full tool surface — Deep Agents built-ins (including `task`), MCP
  tools, and **delegated subagents** alike; tool calls whose name is not on the
  list are blocked, so tools routed through the `task` tool cannot run ungated. A
  non-list `tools` value is a normalized configuration failure rather than a
  silently disabled allow-list.
- `harness.settings.deepagents` is an escape hatch passed through to
  `create_deep_agent` (e.g. `subagents`, `interrupt_on`).

The normalized result includes the final response, buffered messages and
per-step events, LangGraph thread id, token usage (and cost when the provider
reports it), and errors. Usage aggregates the current turn across the main agent
and any delegated subagents (streamed with `subgraphs=True`). Configuration and
preflight failures (a missing credential, an absent `deepagents` package, an
invalid allow-list or MCP server) are returned as a normalized failure result
rather than a raw traceback.

## Runtime Modes

A one-shot `run` streams the agent with `astream` (buffering `updates` events and
`values` snapshots) and returns the final agent message, buffered messages and
per-step events, usage, and the LangGraph thread ID in the normalized Fabric
result. Each one-shot run gets a fresh Fabric `runtime_id`, so `resumed` is
`false`.

Multi-turn and resume are keyed by the Fabric `runtime_id`, which is stable
across `invoke` calls in a started runtime (`start_runtime`) and fresh for each
one-shot run. On the first turn the adapter generates a LangGraph thread ID and
records it against the runtime; later turns of the same runtime reuse that thread
ID and a persistent LangGraph SQLite checkpointer to resume (`resumed` is `true`).
The checkpointer lives under `harness.settings.state_dir` (default the runtime
artifacts directory). Fabric owns the runtime-to-thread correlation record;
LangGraph owns the transcript.

The `deepagents_config()` builder in `examples/code_review_agent` is the SDK
example; the `deepagents` profile under
`tests/fixtures/file-config-agent/profiles/` covers file-based resolution. Run it
from the CLI with `python -m examples.code_review_agent --variant deepagents
--input "..."`, or drive the SDK directly:

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

- **Relay** (`telemetry.provider: relay`): the agent is wrapped with
  `nemo_relay.integrations.deepagents.add_nemo_relay_integration`, emitting
  ATOF/ATIF artifacts referenced in the `ArtifactManifest`. OTel/OpenInference
  export is available through the relay plugin config (see the `relay-otel` and
  `relay-openinference` profiles).
- **Native** (`telemetry.provider: native`): the `telemetry.config`
  OpenTelemetry/OpenInference exporter is applied and spans export directly to
  the configured collector, without writing ATOF/ATIF relay artifacts.
