<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Fabric Deep Agents Code Adapter

This package provides the `nvidia.fabric.langchain.deepagents-code` process
adapter. It runs LangChain's
[Deep Agents Code](https://github.com/langchain-ai/deepagents/tree/main/libs/code)
(`dcode`) as an ACP agent: one `dcode --acp` child process and one ACP session
per NeMo Fabric runtime, reused across ordered invocations.

## Install the Adapter

Install Node.js 22.19.0 or newer and the adapter:

```bash
npm install nemo-fabric-adapters-deepagents-code
```

Install `dcode` separately. It requires Python 3.14 and pins exact dependency
versions, so give it its own environment:

```bash
uv tool install deepagents-code==0.2.2
```

The adapter runs `dcode` from `PATH` and never installs or updates it. The
adapter was tested with `deepagents-code` 0.2.2.

For source development, run these commands from the repository root:

```bash
npm run build --prefix adapter-contract/typescript
npm run build --prefix adapters/typescript --workspace nemo-fabric-adapters-common
npm run build --prefix adapters/typescript --workspace nemo-fabric-adapters-deepagents-code
```

Register the packaged descriptor at
`node_modules/nemo-fabric-adapters-deepagents-code/deepagents-code.fabric-adapter.json`.
Source checkouts can register
`adapters/typescript/deepagents-code/deepagents-code.fabric-adapter.json`.

## Supported Configuration

- **Model.** The `default` model role, or the only role, uses provider
  `openai`. With `base_url`, `dcode` calls that OpenAI-compatible endpoint
  through Chat Completions; without it, OpenAI's own endpoint. The key comes
  from the variable named by `api_key_env`, `OPENAI_API_KEY` by default. HTTP
  endpoints must be loopback or listed in
  `harness.settings.insecure_http_model_hosts`. The adapter writes the variable's
  name, never its value, to `dcode`'s `config.toml`.
- **`approval_mode`** (required). No person can answer an approval during an
  invocation, so choose `dcode`'s unattended behavior explicitly:
  - `auto` runs `dcode --acp --auto-approve`. `dcode`'s classifier, using the
    default model, reviews gated actions, and the adapter denies any it
    escalates.
  - `yolo` allows every gated action, such as shell commands and file writes.

The ACP session has 60 seconds to start. NeMo Fabric system instructions,
skills, MCP servers, tool policy, telemetry, usage reporting, streaming and
cancellation aren't mapped yet. `dcode` reports no token usage over ACP.

## State and Outbound Requests

Each runtime keeps `dcode`'s state, including `config.toml`, sessions and
memory, in its own `DEEPAGENTS_HOME`:
`<artifact root>/.fabric/deepagents-code/runtimes/<runtime-id>/deepagents`. The
artifact root is the runtime context's artifact directory, or
`<base_dir>/artifacts`. The configuration is regenerated at each start.

The adapter passes `dcode` only a fixed set of process variables, such as
`HOME`, `PATH`, proxy and certificate settings, the runtime environment's
variables, and the model key's variable. It runs `dcode` with `--no-tracing`
and sets `DEEPAGENTS_CODE_NO_UPDATE_CHECK=1`, `DEEPAGENTS_CODE_AUTO_UPDATE=0`,
`DEEPAGENTS_CODE_PRICES_AUTO_UPDATE=0` and
`DEEPAGENTS_CODE_PLUGIN_AUTO_UPDATE=0` to turn off update checks, automatic
updates, and price and plugin refreshes. It also sets
`DEEPAGENTS_CODE_READ_PROJECT_DOTENV=0`, so a `.env` file in the workspace
can't add variables: configuration comes from `FabricConfig`.

In a run of `deepagents-code` 0.2.2 against a loopback model, invocations
contacted only the model endpoint, apart from what the agent's own tools
request.

## Runtime Health

`Fabric.check()` supports `live`. `dcode` runs its agent inside the ACP process
and keeps the runtime's session open in it, so `live` observes that the ACP
child process is still running; `active` and `ready` are unsupported. A busy process passes. A
stopped or exited process fails. Checks don't send prompts.

Refer to the [SDK health-check overview](../../../docs/sdk/health-checks.mdx)
for shared behavior.
