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

When `models.default.provider` is `nvidia`, the adapter defines a request-scoped
Codex model provider for the configured NVIDIA Responses endpoint. It reads the
credential from `api_key_env` (default: `NVIDIA_API_KEY`) and isolates Codex
state under the NeMo Fabric artifact root, so the invocation does not depend on or
modify a user's Codex login. Set the endpoint in
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

Each NeMo Fabric invocation starts a fresh SDK client and closes its app-server
transport before returning. The first invocation creates a Codex thread and
persists its ID under the NeMo Fabric artifact root. Later invocations for the same
NeMo Fabric runtime resume that exact thread. Codex owns the transcript; NeMo Fabric owns
runtime-to-thread correlation, timeout, cancellation, and cleanup.

The result includes the SDK's typed terminal response, turn status, token
usage, timing, and completed thread items. It does not expose CLI commands,
return codes, stdout, or stderr.

## Configuration

Use normalized `FabricConfig` fields for portable configuration:

- `models` selects the Codex model. The adapter supports the built-in `openai`
  provider and NVIDIA-hosted Responses-compatible models through the `nvidia`
  provider.
- `environment.workspace` sets the working directory.
- `mcp` maps stdio, HTTP, and streamable HTTP servers into request-scoped Codex
  `mcp_servers` configuration. For stdio, NeMo Fabric parses `url` as a command plus
  arguments.
- `skills.paths` names skill directories that contain `SKILL.md`. The adapter
  registers each directory as a process-scoped Codex skill root so Codex can
  select matching skills through its normal discovery behavior.
- `telemetry` enables native OpenTelemetry or NeMo Relay observability.

The Codex adapter does not declare `tools.blocked` support. The current Codex
runtime has per-MCP-server tool filters, but it does not provide one complete
deny boundary for built-in, local, MCP, and hosted tools. NeMo Fabric therefore
routes normalized blocked-tool policy as unsupported instead of applying a
partial policy.

Codex-specific controls belong in `harness.settings`:

- `sandbox`: `read-only`, `workspace-write`, or `danger-full-access`
- `approval_mode`: `auto_review` or `deny_all`
- `base_instructions` and `developer_instructions`
- `personality`, `reasoning_effort`, `service_name`, and `service_tier`
- `output_schema` for SDK-native structured output
- `codex_bin` for an explicit Codex app-server runtime override
- `config_overrides` as dotted request-scoped Codex configuration keys, such as
  Codex-only MCP timeout or required-server options
- `timeout_seconds`, defaulting to 1800
- `env` for variables explicitly forwarded to the Codex runtime
- `nemo_relay_command` for the optional external Relay gateway executable

Set model selection through `models` and the working directory through
`environment.workspace`.

The adapter filters the inherited environment. It retains portable OS and
Codex state variables, the selected model's `api_key_env`, and explicit
`settings.env` values while clearing unrelated parent-process secrets.

## Relay Integration

Relay-enabled runs also require the external `nemo-relay` CLI. Refer to the
[NeMo Relay CLI](https://docs.nvidia.com/nemo/fabric/getting-started/install#nemo-relay-cli) install guide for instructions on installing the CLI tool.
