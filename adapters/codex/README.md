<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Fabric Codex Adapter

The `nvidia.fabric.codex` adapter uses the official Codex Python SDK behind
Fabric's normalized invocation contract. It does not resolve or execute a
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

Fabric reuses the authentication state that Codex stores under `CODEX_HOME`
(default: `~/.codex`). Fabric does not perform an interactive login, copy
credentials, or mutate the user's Codex configuration.

Codex supports two OpenAI authentication modes:

- **ChatGPT login:** Sign in through Codex with a ChatGPT plan. Fabric can then
  run without `OPENAI_API_KEY` while that cached login remains valid.
- **API key login:** Provision the same Codex credential store with an OpenAI
  API key. This mode uses OpenAI Platform billing rather than ChatGPT plan
  credits.

For a nondefault credential store, set `CODEX_HOME` before both login and the
Fabric invocation. Treat `CODEX_HOME/auth.json` as a secret when Codex uses
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
Fabric artifact root, so execution does not depend on or modify a user's Codex
login. Set the endpoint in
`models.default.settings.base_url` or `NVIDIA_FRONTIER_BASE_URL`; the adapter
does not assume a default frontier endpoint.

The adapter depends on the Codex SDK, which installs and selects its matching
app-server runtime. Fabric does not declare the runtime package directly or
treat it as a user-installed command or adapter descriptor requirement.

A `codex` command on `PATH` is not selected implicitly. To override the
SDK-selected runtime intentionally, set
`harness.settings.codex_bin` to an app-server path that is absolute or relative
to the explicit `base_dir`. Fabric passes the resolved path through
`CodexConfig.codex_bin`; the SDK remains the execution driver.

## Execution Model

Each Fabric runtime currently starts one local adapter host and retains one
`AsyncCodex` client and one Codex thread. The Codex SDK starts and controls its
pinned local `codex app-server` subprocess over JSON-RPC. Ordered
`Runtime.invoke(...)` calls reuse that client and thread directly; the adapter
closes the SDK client and app-server transport during `Runtime.stop()`. The
first successful invocation also persists the thread ID under the Fabric
artifact root for the adapter's direct per-invocation compatibility path. The
persistent host does not resume the thread between turns. Codex owns the
transcript; Fabric owns runtime-to-thread correlation, timeout, cancellation,
and cleanup.

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
  `mcp_servers` configuration. For stdio, Fabric parses `url` as a command plus
  arguments.
- `skills.paths` names skill directories that contain `SKILL.md`. The adapter
  registers each directory as a process-scoped Codex skill root so Codex can
  select matching skills through its normal discovery behavior.
- `telemetry` enables native OpenTelemetry or NeMo Relay observability.

The Codex adapter does not declare `tools.blocked` support. The current Codex
runtime has per-MCP-server tool filters, but it does not provide one complete
deny boundary for built-in, local, MCP, and hosted tools. Fabric therefore
routes normalized blocked-tool policy as unsupported instead of applying a
partial policy.

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
roots, and `config_overrides` are fixed when the runtime starts and cannot vary
between `Runtime.invoke(...)` calls. Start a new runtime to change them.
`Fabric.run(...)` creates a fresh one-shot runtime, so the same settings are
scoped to that invocation.

The adapter filters the inherited environment. It retains portable OS and
Codex state variables, the selected model's `api_key_env`, and explicit
`settings.env` values while clearing unrelated parent-process secrets.

## Relay Observability

Enable Relay through Fabric's normalized telemetry configuration. For each
Relay-enabled Fabric runtime, the adapter:

1. Resolves one external `nemo-relay` executable.
2. Generates runtime-scoped gateway and plugin configuration.
3. Starts and health-checks `nemo-relay --config ... --bind ...`.
4. Redirects the built-in OpenAI provider with runtime-scoped
   `openai_base_url` and passes Relay hooks through the Codex SDK's `config`
   argument.
5. Reuses the SDK client and gateway across turns, interrupting a timed-out turn.
6. Closes the SDK runtime and stops the gateway during runtime shutdown.

The SDK remains the Codex execution driver. Relay is a supervised sidecar and
hook forwarder; the adapter never invokes a `nemo-relay codex` wrapper. The
result reports the gateway config, URL, log, and collected Relay artifacts.

Fabric deliberately keeps Codex on its reserved built-in `openai` provider.
Defining Relay as a custom model-provider alias breaks the Python SDK's
ChatGPT-authenticated request path. Redirecting only `openai_base_url` preserves
the SDK's supported authentication and host metadata while allowing Relay to
capture Responses traffic. Fabric does not spoof the Codex CLI identity or fall
back to CLI execution. Relay routes and observes requests; it does not provide
OpenAI credentials or change the selected Codex authentication mode.

Relay-enabled Codex runs require `models.default.provider: openai`. The custom
`nvidia` provider is not supported with Relay because its configured NVIDIA
Responses base URL bypasses the built-in `openai_base_url` redirect. Run the
`nvidia` provider without Relay and supply its credential through `api_key_env`
(default: `NVIDIA_API_KEY`) and its endpoint through
`models.default.settings.base_url` or `NVIDIA_FRONTIER_BASE_URL`.

Relay-enabled runs require the external `nemo-relay` CLI in addition to the
Python package dependencies. Fabric accepts CLI versions `>=0.6.0,<0.7.0`.
Until the request-decoding fix is released, install the tested PR revision:

```bash
git clone https://github.com/NVIDIA/NeMo-Relay.git nemo-relay
git -C nemo-relay fetch origin pull/452/head
git -C nemo-relay checkout --detach 0b02e01ac10d7d678da28830feba0ebf6743a7c0
cargo install --locked --path nemo-relay/crates/cli
```

Removal of this temporary source installation is tracked in
[TODO.md](https://github.com/NVIDIA/NeMo-Fabric/blob/main/TODO.md#nemo-relay-06x-request-decoding-release).

The `nemo-relay` Python package does not install this executable. Refer to the
[NeMo Relay installation guide](https://docs.nvidia.com/nemo/relay/getting-started/installation)
for other supported installation methods.

Relay owns HTTP content decoding at the gateway boundary; Fabric does not
configure Codex request compression. The Relay `0.6.0-alpha.20260716` tag cannot
recover semantic fields from zstd-compressed SDK requests. Until a later `0.6.x`
release contains the fix, use the pinned
[NeMo Relay PR #452](https://github.com/NVIDIA/NeMo-Relay/pull/452) revision
above.

For Phoenix, native Codex OpenTelemetry targets the OTLP collector at
`http://localhost:4318/v1/traces` and provides low-level app-server spans.
Relay OpenInference provides the semantic chain, LLM, and tool hierarchy with
decoded prompt, response, and token attributes. Prefer Relay OpenInference for
agent-turn inspection.
