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

The dependency graph includes `openai-codex-cli-bin`. The Codex SDK owns this
pinned app-server distribution; Fabric does not treat it as a user-installed
command or an adapter descriptor requirement.

This adapter pins `openai-codex==0.1.0b3`, which pins
`openai-codex-cli-bin==0.137.0a4`. A newer `codex` command on `PATH` is not used
implicitly. When testing a newer compatible runtime, set
`harness.settings.codex_bin` to an app-server path that is absolute or relative
to the Fabric config root. Fabric passes the resolved path through
`CodexConfig.codex_bin`; the SDK remains the execution driver.

## Execution Model

Each Fabric invocation starts a fresh SDK client and closes its app-server
transport before returning. The first invocation creates a Codex thread and
persists its ID under the Fabric artifact root. Later invocations for the same
Fabric runtime resume that exact thread. Codex owns the transcript; Fabric owns
runtime-to-thread correlation, timeout, cancellation, and cleanup.

The result includes the SDK's typed terminal response, turn status, token
usage, timing, and completed thread items. It does not expose CLI commands,
return codes, stdout, or stderr.

## Configuration

Use normalized `FabricConfig` fields for portable configuration:

- `models` selects the Codex model. The adapter requires and explicitly selects
  the built-in `openai` provider.
- `environment.workspace` sets the working directory.
- `telemetry` enables native OpenTelemetry or NeMo Relay observability.

Codex-specific controls belong in `harness.settings`:

- `sandbox`: `read-only`, `workspace-write`, or `danger-full-access`
- `approval_mode`: `auto_review` or `deny_all`
- `base_instructions` and `developer_instructions`
- `personality`, `reasoning_effort`, `service_name`, and `service_tier`
- `output_schema` for SDK-native structured output
- `codex_bin` for an explicit Codex app-server runtime override
- `config_overrides` as dotted request-scoped Codex configuration keys
- `timeout_seconds`, defaulting to 1800
- `env` for variables explicitly forwarded to the Codex runtime
- `nemo_relay_command` for the optional external Relay gateway executable

The removed CLI settings `codex_command`, `codex_args`, `codex_profile`,
`codex_state_dir`, and `skip_git_repo_check` are errors. `model_name` and `cwd`
must use the normalized model and environment fields.

The adapter filters the inherited environment. It retains portable OS and
Codex state variables, the selected model's `api_key_env`, and explicit
`settings.env` values while clearing unrelated parent-process secrets.

## Relay Integration

Relay-enabled runs also require the external `nemo-relay` CLI. Refer to the
[NeMo Relay CLI](https://docs.nvidia.com/nemo/fabric/getting-started/install#nemo-relay-cli) install guide for instructions on installing the CLI tool.

