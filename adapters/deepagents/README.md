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

Fabric maps the following into the harness:

- `models.default.model` / `harness.settings.model_name` selects the model.
- `models.default.provider` selects the client (`nvidia`/`openai` → OpenAI-compatible).
- `models.default.temperature` / `harness.settings.temperature` sets sampling.
- `harness.settings.base_url` overrides the model endpoint.
- `harness.settings.system_prompt` becomes the Deep Agents `system_prompt`.
- `environment.workspace` roots the Deep Agents filesystem backend
  (`FilesystemBackend(root_dir=...)`).
- Routed `skills` (`native.skill_paths`) become the Deep Agents `skills` sources.
- Configured MCP servers are loaded as Deep Agents tools via
  `langchain-mcp-adapters`.
- `tools` (Fabric's `config.tools` allow-list) is enforced by a gating middleware
  across the full tool surface (Deep Agents built-ins and MCP tools); tool calls
  whose name is not on the list are blocked.
- `harness.settings.deepagents` is an escape hatch passed through to
  `create_deep_agent` (e.g. `subagents`, `interrupt_on`).

The normalized result includes the final response, buffered messages and
per-step events, LangGraph thread id, token usage (and cost when the provider
reports it), and errors.

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
`tests/fixtures/file-config-agent/profiles/` covers file-based resolution.

## Telemetry

- **Relay** (`telemetry.provider: relay`): the agent is wrapped with
  `nemo_relay.integrations.deepagents.add_nemo_relay_integration`, emitting
  ATOF/ATIF artifacts referenced in the `ArtifactManifest`. OTel/OpenInference
  export is available through the relay plugin config (see the `relay-otel` and
  `relay-openinference` profiles).
- **Native** (`telemetry.provider: native`): the `telemetry.config`
  OpenTelemetry/OpenInference exporter is applied and spans export directly to
  the configured collector, without writing ATOF/ATIF relay artifacts.
