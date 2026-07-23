<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Fabric Codex Adapter

The `nvidia.fabric.codex` adapter uses the official Codex Python SDK behind
NeMo Fabric's normalized invocation contract. It does not resolve or execute a
separately installed `codex` command. The SDK package owns its pinned
app-server runtime and typed JSON-RPC protocol.

## Install

To install just the Codex adapter by itself:

```bash
pip install "nemo-fabric[codex]"
```

To install just the Codex adapter along with the NeMo Fabric Runtime:

```bash
pip install "nemo-fabric[codex, runtime]"
```

## Authentication

NeMo Fabric reuses the authentication state that Codex stores under `CODEX_HOME`
(default: `~/.codex`). NeMo Fabric does not perform an interactive login, copy
credentials, or mutate the user's Codex configuration.

Codex supports two OpenAI authentication modes:

- **ChatGPT login:** Sign in through Codex with a ChatGPT plan. NeMo Fabric can then
  run without `OPENAI_API_KEY` while that cached login remains valid.
- **API key login:** Provision the same Codex credential store with an OpenAI
  API key. This mode uses OpenAI Platform billing rather than ChatGPT plan
  credits.

For a nondefault credential store, set `CODEX_HOME` before both login and the
NeMo Fabric invocation. Treat `CODEX_HOME/auth.json` as a secret when Codex uses
file-based credential storage. Refer to the
[Codex authentication documentation](https://developers.openai.com/codex/auth/)
for login, headless setup, and credential-storage options.

The adapter forwards `OPENAI_API_KEY` and a selected model's `api_key_env` to
the SDK runtime. The current real-agent acceptance path validates an existing
Codex login; it does not yet claim a raw environment variable as a complete
login flow.

When `models.default.provider` is `nvidia`, the adapter defines a Codex model
provider for the configured NVIDIA Responses endpoint. `Fabric.run(...)` owns
that provider for one invocation, while `Fabric.start_runtime(...)` fixes it for
the lifetime of the persistent runtime. The adapter reads the credential from
`api_key_env` (default: `NVIDIA_API_KEY`) and isolates Codex state under the
NeMo Fabric artifact root, so execution does not depend on or modify a user's Codex
login. Set the endpoint in
`models.default.settings.base_url` or `NVIDIA_FRONTIER_BASE_URL`; the adapter
does not assume a default frontier endpoint.

The adapter depends on the Codex SDK, which installs and selects its matching
app-server runtime. NeMo Fabric does not declare the runtime package directly or
treat it as a user-installed command or adapter descriptor requirement.

A `codex` command on `PATH` is not selected implicitly. To override the
SDK-selected runtime intentionally, set
`harness.settings.codex_bin` to an app-server path that is absolute or relative
to the explicit `base_dir`. NeMo Fabric passes the resolved path through
`CodexConfig.codex_bin`; the SDK remains the execution driver.

## Execution Model

Each NeMo Fabric runtime currently starts one local adapter host and retains one
`AsyncCodex` client and one Codex thread. The Codex starts and controls its
pinned local `codex app-server` subprocess over JSON-RPC. Ordered
`Runtime.invoke(...)` calls reuse that client and thread directly; the adapter
closes the SDK client and app-server transport during `Runtime.stop()`. Codex
owns the transcript; NeMo Fabric owns runtime-to-thread correlation, timeout,
cancellation, and cleanup.

The result includes the SDK's typed terminal response, turn status, token
usage, timing, and completed thread items. It does not expose CLI commands,
return codes, stdout, or stderr.

## Configuration

Use normalized `FabricConfig` fields for portable configuration:

- `models` selects the Codex model. The adapter supports the built-in `openai`
  provider and NVIDIA-hosted Responses-compatible models through the `nvidia`
  provider.
- `environment.workspace` sets the working directory.
- `mcp` maps stdio, HTTP, and streamable HTTP servers into the Codex thread's
  `mcp_servers` configuration. For stdio, NeMo Fabric parses `url` as a command plus
  arguments.
- `skills.paths` names skill directories that contain `SKILL.md`. The adapter
  registers each directory as a process-scoped Codex skill root so Codex can
  select matching skills through its normal discovery behavior.
- `telemetry` enables native OpenTelemetry or NeMo Relay observability.

The Codex adapter declares `tools.blocked` support and translates these names
to Codex-native registration or filtering controls:

- `shell`, `browser`, `web_search`, `apps`, `plugins`, `image_generation`,
  `multi_agent`, `tool_suggest`, and `request_user_input` disable the
  corresponding Codex tool or toolset.
- `mcp` disables every configured MCP server.
- `mcp:<server>:<tool>` adds the raw MCP tool name to that Fabric MCP server's
  `disabled_tools` list.
- `app:<connector>:<tool>` disables the raw app tool name for that connector.

The adapter applies this policy after `config_overrides`, so Codex-specific
overrides cannot re-enable a blocked tool. Unsupported, malformed, or unknown
MCP server names fail before the Codex SDK runtime starts. This includes Codex
built-ins without an enforceable per-tool control, such as `apply_patch`, and
dynamic tool names that the supported high-level SDK does not expose.

Blocked-tool policy requires the app-server runtime pinned by the adapter.
NeMo Fabric rejects a configuration that combines `tools.blocked` with
`harness.settings.codex_bin` because it cannot establish the same controls for
an arbitrary runtime override.

Codex-specific controls belong in `harness.settings`:

- `sandbox`: `read-only`, `workspace-write`, or `danger-full-access`
- `approval_mode`: `auto_review` or `deny_all`
- `base_instructions` and `developer_instructions`
- `personality`, `reasoning_effort`, `service_name`, and `service_tier`
- `output_schema` for SDK-native structured output
- `codex_bin` for an explicit Codex app-server runtime override
- `config_overrides` as dotted Codex configuration keys applied when the SDK
  runtime starts, such as Codex-only MCP timeout or required-server options
- `timeout_seconds`, defaulting to 1800
- `env` for variables explicitly forwarded to the Codex runtime
- `nemo_relay_command` for the optional external Relay gateway executable

Set model selection through `models` and the working directory through
`environment.workspace`.

For `Fabric.start_runtime(...)`, the model provider, MCP configuration, skill
roots, blocked tools, and `config_overrides` are fixed when the runtime starts
and cannot vary between `Runtime.invoke(...)` calls. Start a new runtime to
change them.
`Fabric.run(...)` starts the same runtime, invokes it once, and stops it, so the
same settings are scoped to that single invocation.

The adapter filters the inherited environment. It retains portable OS and
Codex state variables, the selected model's `api_key_env`, and explicit
`settings.env` values while clearing unrelated parent-process secrets.

## Relay Integration

Relay-enabled runs also require the external `nemo-relay` CLI. Refer to the
[NeMo Relay CLI](https://docs.nvidia.com/nemo/fabric/getting-started/install#nemo-relay-cli) install guide for instructions on installing the CLI tool.
