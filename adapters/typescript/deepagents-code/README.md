<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Fabric Deep Agents Code Adapter

This package provides the `nvidia.fabric.langchain.deepagents-code` process
adapter. It runs LangChain's
[Deep Agents Code](https://github.com/langchain-ai/deepagents/tree/main/libs/code)
(`dcode`) as an ACP agent: one `dcode --acp` child process and one ACP session
per NeMo Fabric runtime, reused across ordered invocations. It can also write a
launcher for `dcode`'s terminal UI that uses the runtime's configuration.

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
- **`interfaces.terminal`**. See [Open the Terminal UI](#open-the-terminal-ui).

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
can't add variables: configuration comes from `FabricConfig`. The terminal
launcher sets the same variables.

In a run of `deepagents-code` 0.2.2 against a loopback model, invocations
contacted only the model endpoint, apart from what the agent's own tools
request.

## Open the Terminal UI

With `harness.settings.interfaces.terminal` present and `enabled` not `false`,
the adapter writes an executable launcher to
`<artifact root>/.fabric/deepagents-code/runtimes/<runtime-id>/interfaces/terminal`.
In an OpenShell sandbox, a person runs it with:

```bash
openshell sandbox exec -n SANDBOX --tty -- <launcher path>
```

`openshell sandbox connect` attaches to the sandbox's main process, not a
shell, so it can't run the launcher. On a `noexec` mount, run it as
`sh <launcher path>`.

The launcher opens `dcode`'s terminal UI in a **separate session** with the
runtime's `DEEPAGENTS_HOME`, configuration and working directory. `dcode` can't
attach to another process's agent, so the terminal doesn't see invocations
live. It can list the runtime's thread, whose ID is in
`interfaces/terminal.running`, with `<launcher path> threads list`, but can't
open it: the ACP session holds that thread while the runtime runs, and `dcode`
refuses to resume a thread open elsewhere.

- The launcher holds paths and switches, never secrets. The terminal UI reads
  the model key from the launching environment under the configured variable
  name, unless the endpoint needs no key.
- The terminal UI uses `dcode`'s own approval behavior, not `approval_mode`.
- After the ACP session starts, the adapter writes `interfaces/terminal.running`;
  `stop` removes it before stopping `dcode`. The launcher refuses to start
  without it. Terminal sessions already open when the runtime stops keep
  running. If the adapter host is killed, the marker can remain until the
  sandbox is replaced.
- The agent runs as the same user and can modify the launcher.
- The terminal UI starts its own `dcode` agent server on `127.0.0.1` without
  authentication, as `dcode` always does; its `THREAT_MODEL.md` lists this as
  T6. Invocations use ACP over stdio and open no listener.
- At each start, that `langgraph dev` agent server sends LangGraph CLI's usage
  record, with the operating system and its version, the Python and CLI
  versions, and the command, to `kzrlppojinpcyyaipxnb.supabase.co`. To opt out,
  set `LANGGRAPH_CLI_NO_ANALYTICS=1` in the environment that runs the launcher.
- Without `rg` on `PATH`, `dcode` downloads ripgrep from GitHub into its own
  installation. Install ripgrep in the environment to avoid the download.

## Runtime Health

`Fabric.check()` supports `live`, `active` and `ready`. `dcode` runs its agent
inside the ACP process and keeps the runtime's session open in it, so each level
observes that the ACP child process is still running. A busy process passes. A
stopped or exited process fails. Checks don't send prompts.

Refer to the [SDK health-check overview](../../../docs/sdk/health-checks.mdx)
for shared behavior.
