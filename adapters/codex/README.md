<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Codex Adapter

The `nvidia.fabric.codex` adapter uses the official Codex Python SDK behind
Fabric's normalized invocation contract. It does not resolve or execute a
separately installed `codex` command. The SDK package owns its pinned
app-server runtime and typed JSON-RPC protocol.

## Install

Build the local wheels and install the Codex adapter:

```bash
just wheels
python -m pip install --find-links dist "nemo-fabric[codex]"
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

## Relay Observability

Enable Relay through Fabric's normalized telemetry configuration. For each
Relay-enabled invocation, Fabric:

1. Resolves one external `nemo-relay` executable.
2. Generates invocation-scoped gateway and plugin configuration.
3. Starts and health-checks `nemo-relay --config ... --bind ...`.
4. Redirects the built-in OpenAI provider with request-scoped
   `openai_base_url` and passes Relay hooks through the Codex SDK's `config`
   argument.
5. Interrupts timed-out turns, closes the SDK runtime, and stops the gateway.

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
[TODO.md](../../TODO.md#nemo-relay-06x-request-decoding-release).

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

## Local Validation

Run the unit and opt-in real SDK tests separately:

```bash
uv run pytest tests/adapters/test_codex_adapter.py -q
RUN_FABRIC_CODEX_INTEGRATION=1 uv run pytest tests/e2e/test_codex.py -q
RUN_FABRIC_CODEX_RELAY_INTEGRATION=1 \
  FABRIC_TEST_NEMO_RELAY_COMMAND=/path/to/nemo-relay \
  uv run pytest tests/e2e/test_codex.py -q
```

Set `FABRIC_TEST_CODEX_BIN=/path/to/codex` on either opt-in command to validate
an explicit app-server override instead of the SDK-pinned runtime.

The SDK test uses the current Codex authentication state and exercises both a
one-shot invocation and multi-turn thread resume. The Relay test additionally
requires an external gateway binary and verifies one-shot and resumed model
responses, stable thread identity, ATOF, and ATIF; gateway startup alone is not
a passing result. The semantic regression also requires decoded LLM request
content, a model, token usage, and the expected agent response in ATIF.
