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

The following table shows which components each installation provides:

| Installation | Runtime | Adapter | Harness | NeMo Relay CLI |
| --- | --- | --- | --- | --- |
| `pip install "nemo-fabric[codex]"` | Yes | Yes | Yes | No |
| `pip install "nemo-fabric-adapters-codex[harness]"` | No | Yes | Yes | No |
| `pip install nemo-fabric-adapters-codex` | No | Yes | No | No |

For an environment-managed SDK, use `openai-codex==0.144.4`. For split runtime
and adapter environments, configure `ADAPTER_PYTHON` and use matching NeMo
Fabric release versions. Refer to the
[installation guide](https://docs.nvidia.com/nemo/fabric/getting-started/install#install-an-adapter-and-harness-without-the-runtime).

The `full` extra is equivalent to `harness`. Relay is optional for ordinary
runs. Relay telemetry and `Runtime.invoke_stream()` require the external CLI
described under [Relay Integration](#relay-integration).

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

The native `openai` provider retains Codex authentication and endpoint
discovery. For another provider name, configure both
`models.<role>.api_key_env` and `models.<role>.base_url`. The endpoint must
implement the OpenAI Responses protocol. The adapter defines a runtime-scoped
Codex model provider with that name and isolates its Codex state under the
NeMo Fabric artifact root, so execution does not depend on or modify a user's
Codex login. Provider names identify configuration; the adapter does not
maintain a provider allowlist.

The adapter uses the Codex SDK, which installs and selects its matching
app-server runtime. NeMo Fabric does not declare the runtime package directly or
treat it as a user-installed command or adapter descriptor requirement.

A `codex` command on `PATH` is not selected implicitly.

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

- `models` selects the Codex model. The native `openai` provider retains Codex
  authentication and endpoint discovery. Any other provider name must configure
  a Responses-compatible `base_url` and `api_key_env`.
- `instructions.system` maps to Codex base instructions.
- `runtime.timeout_seconds` sets the NeMo Fabric invocation deadline.
- `environment.workspace` sets the working directory, and `environment.env`
  supplies explicit harness-visible variables.
- `mcp` maps stdio, HTTP, and streamable HTTP servers into the Codex thread's
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

Only Codex-specific controls belong in `harness.settings`:

| Setting | Type | Required | Static Default |
| --- | --- | --- | --- |
| `sandbox` | One of `read-only`, `workspace-write`, or `danger-full-access` | No | `read-only` |
| `approval_mode` | One of `auto_review` or `deny_all` | No | `auto_review` |
| `developer_instructions` | Nonempty string | No | No default |
| `personality` | One of `none`, `friendly`, or `pragmatic` | No | No default |
| `reasoning_effort` | One of `none`, `minimal`, `low`, `medium`, `high`, or `xhigh` | No | No default |
| `service_tier` | Nonempty string | No | No default |
| `output_schema` | JSON Schema object for the final assistant message | No | No default |
| `config_overrides` | Object that maps nonempty dotted Codex configuration keys to JSON-compatible values | No | `{}` |

Planning validates these settings against the schema in the resolved Codex
descriptor. Unknown keys, empty dotted-key segments, invalid types, and invalid
enum values fail before the adapter starts. Schema defaults are documentation
only; planning preserves the supplied settings without adding defaults.
`config_overrides` is the intentional adapter-specific escape hatch for Codex
configuration that has no normalized NeMo Fabric field.

Set model selection and endpoints through `models`, system instructions through
`instructions.system`, the invocation deadline through
`runtime.timeout_seconds`, and the working directory and explicit environment
through `environment`. In particular, the SDK's `base_instructions` value comes
from `instructions.system`, not `harness.settings`.

For `Fabric.start_runtime(...)`, the model provider, MCP configuration, skill
roots, and `config_overrides` are fixed when the runtime starts and cannot vary
between `Runtime.invoke(...)` calls. Start a new runtime to change them.
`Fabric.run(...)` starts the same runtime, invokes it once, and stops it.

The adapter filters the inherited environment. It retains portable OS and
Codex state variables, the selected model's `api_key_env`, and explicit
`environment.env` values while clearing unrelated parent-process secrets.

## Relay Integration

Relay requires a separately installed NeMo Relay 0.6.x CLI on `PATH`; the Python
`nemo-relay` package does not provide the executable. Follow the
[NeMo Relay installation instructions](https://docs.nvidia.com/nemo/fabric/getting-started/install#install-nemo-relay).

Enable Relay with `FabricConfig.enable_relay(...)`. The adapter starts the
installed `nemo-relay` CLI as a supervised sidecar; do not start the gateway
separately.
NeMo Fabric routes the selected Responses-compatible provider through the
gateway and passes its explicit `base_url` to Relay as the upstream endpoint.


## Testing

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

The SDK test uses the current Codex authentication state and exercises both the
single-invocation convenience API and multiple turns against one started
runtime. The Relay test additionally requires an external gateway binary and
verifies model responses, stable thread identity across turns, Agent Trajectory
Observability Format (ATOF), and Agent Trajectory Interchange Format (ATIF);
gateway startup alone is not a passing result. The semantic regression requires
the LLM request content to be decoded. It also requires ATIF to contain the
model, token usage, and expected agent response.
