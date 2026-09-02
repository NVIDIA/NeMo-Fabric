<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Fabric Claude Adapter

The `nvidia.fabric.claude` adapter uses the official Claude Agent SDK for
Python behind NeMo Fabric's normalized invocation contract. The SDK is an
implementation detail; consumers select the Claude harness by adapter ID.

The `harness` extra pins `claude-agent-sdk==0.2.120`; use the same version for an
environment-managed SDK. The SDK supplies a compatible Claude Code runtime.

## Install

The following table shows which components each installation provides:

| Installation | Runtime | Adapter | Harness | NeMo Relay CLI |
| --- | --- | --- | --- | --- |
| `pip install "nemo-fabric[claude]"` | Yes | Yes | Yes | Yes |
| `pip install "nemo-fabric-adapters-claude[harness]"` | No | Yes | Yes | Yes |
| `pip install nemo-fabric-adapters-claude` | No | Yes | No | No |

For split runtime and adapter environments, configure `ADAPTER_PYTHON` and use
matching NeMo Fabric release versions. Refer to the
[installation guide](https://docs.nvidia.com/nemo/fabric/getting-started/install#install-an-adapter-and-harness-without-the-runtime).

The `full` extra is equivalent to `harness`. Both install the supported NeMo
Relay CLI so Relay telemetry and `Runtime.invoke_stream()` work without a
separate CLI installation.

## Authentication

NeMo Fabric preserves Claude's native credential resolution. Use an existing Claude
Code login for local development, `ANTHROPIC_AUTH_TOKEN` for a gateway or proxy
bearer credential, `ANTHROPIC_API_KEY` for a static API credential, or Anthropic
Workload Identity Federation (WIF) for production and CI workloads that should
not store a long-lived API key.

The native `anthropic` provider can use any Claude authentication mode above
without an explicit endpoint. For another provider name, configure both
`models.<role>.api_key_env` and `models.<role>.base_url`. The endpoint must
implement the Anthropic Messages protocol; the adapter maps the named
credential and endpoint into the environment expected by Claude Code. Provider
names identify configuration; the adapter does not maintain a provider
allowlist. The runtime-scoped mapping does not change the parent environment.

The adapter forwards the Anthropic profile and federation environment variables
that Claude Code and the Claude Agent SDK consume. This includes
`ANTHROPIC_CONFIG_DIR`, `ANTHROPIC_PROFILE`, the direct federation identifiers,
and `ANTHROPIC_IDENTITY_TOKEN` or `ANTHROPIC_IDENTITY_TOKEN_FILE`. NeMo Fabric reads
selected environment values and forwards them to the Claude runtime, but it
does not persist or log them in configuration or artifacts. Authentication is
validated when the Claude runtime starts.

Unset unused `ANTHROPIC_API_KEY` and `ANTHROPIC_AUTH_TOKEN` variables before
using WIF. Anthropic credential resolution treats an empty variable as selected,
so an empty API credential prevents fallback to a federation profile.

Refer to the [Claude adapter authentication guide](https://docs.nvidia.com/nemo/fabric/integrations/harness-integrations/claude-code)
for mode selection, required WIF variables, and the Relay boundary. Package
installation is verified by the adapter wheel and module-entrypoint tests.

## Execution Model

The Claude adapter implements NeMo Fabric's persistent local-host wire protocol.
`Fabric.start_runtime(...)` launches one adapter host, creates one
`ClaudeSDKClient`, and connects it once. Every `Runtime.invoke(...)` reuses that
client and its event loop; `Runtime.stop()` disconnects the client and exits the
host. `Fabric.run(...)` uses the same lifecycle around one invocation.

One NeMo Fabric runtime maps to one live Claude session. The adapter records the
terminal Claude session ID under the NeMo Fabric artifact root for correlation, but
does not silently recreate a crashed host or replay an invocation. Start a new
NeMo Fabric runtime when the host or SDK connection becomes unusable. Runtime
hosting is adapter-declared; consumers do not configure a runtime strategy in
`FabricConfig` or `harness.settings`.

## Configuration

Configure portable capabilities through the normalized `FabricConfig` fields:

- `models` selects the Claude model. The native `anthropic` provider retains
  Claude authentication and endpoint discovery. Any other provider name must
  configure an Anthropic Messages-compatible `base_url` and `api_key_env`.
- `instructions.system` supports `replace` and `append`. `replace` supplies the
  complete Claude system prompt. `append` preserves the `claude_code` preset and
  adds the configured content after it.
- `runtime.max_turns` sets the Claude turn limit.
- `runtime.timeout_seconds` sets the NeMo Fabric invocation deadline.
- `environment.workspace` sets the Claude working directory, and
  `environment.env` supplies explicit harness-visible variables.
- `tools.enabled` selects Claude built-in tools. `None` preserves the Claude
  default, while an empty list disables every tool. With `permission_mode` set
  to `dontAsk`, explicitly enabled tools are also pre-approved so headless runs
  can invoke them.
- `tools.blocked` maps to Claude `disallowed_tools`. A pre-tool hook enforces
  both lists across built-in, MCP, and plugin tools.
- `mcp` configures stdio, HTTP, streamable HTTP, or SSE servers. For stdio,
  set `url` to the executable and pass each command-line argument as a separate
  `args` element.
- `skills.paths` names skill directories that contain `SKILL.md`. The adapter
  stages these directories as a local Claude plugin for the runtime.

Only Claude-specific controls belong in `harness.settings`:

| Setting | Type | Required | Static default |
| --- | --- | --- | --- |
| `permission_mode` | One of `default`, `acceptEdits`, `bypassPermissions`, `plan`, `dontAsk`, or `auto` | No | No default |
| `max_budget_usd` | Number greater than `0` | No | No default |
| `setting_sources` | Array containing `user`, `project`, or `local` | No | `[]` |

Planning validates these settings against the schema in the resolved Claude
descriptor. Unknown keys and invalid values fail before the adapter starts.
Schema defaults are documentation only; planning preserves the supplied settings
without adding `setting_sources`.

The adapter filters the inherited environment before launching Claude Code.
It retains portable OS/config variables, the selected model's `api_key_env`,
and explicitly configured `environment.env` values. Raw Claude stderr is consumed
by the SDK and is not persisted as a NeMo Fabric artifact.

## Relay Observability

Relay requires a NeMo Relay CLI in the `>=0.7.2,<0.8` range on `PATH`. The
Claude adapter does not provide a separate `relay` extra; its `harness` and
`full` extras install the compatible CLI through `nemo-relay-cli-bin`. The root
`nemo-fabric[relay]` extra installs only the Relay Python package.

Enable Relay through the normalized NeMo Fabric configuration:

```python
config.enable_relay(
    project="fabric-review",
    output_dir="./artifacts/relay",
)
```

For each Relay-enabled Claude runtime, NeMo Fabric starts one `nemo-relay` gateway,
waits for its health endpoint, and stops it with the runtime. NeMo Fabric passes the
gateway URL to the connected Claude Code process through `ANTHROPIC_BASE_URL`
and `NEMO_RELAY_GATEWAY_URL`, and passes the selected explicit model endpoint to
the gateway as its Anthropic upstream. It also stages a runtime-scoped Claude
plugin that forwards lifecycle hooks with `nemo-relay hook-forward claude`.
`Fabric.run(...)` starts the same runtime, invokes it once, and stops it, so the
gateway has the same lifecycle as that single invocation.

The NeMo Fabric result includes `relay_runtime.gateway_config_path`,
`relay_runtime.gateway_log_path`, and the collected `relay_artifacts`. Relay
startup failures return a stable adapter error and retain the gateway log for
diagnosis.

## Typed Configuration

Build the agent configuration with the typed SDK models before invoking
NeMo Fabric:

```python
from pathlib import Path

from nemo_fabric import (
    EnvironmentConfig,
    Fabric,
    FabricConfig,
    HarnessConfig,
    InstructionConfig,
    InstructionsConfig,
    McpConfig,
    McpServerConfig,
    MetadataConfig,
    ModelConfig,
    RuntimeConfig,
    SkillConfig,
    ToolsConfig,
)

base_dir = Path("/workspace/review-agent")
config = FabricConfig(
    metadata=MetadataConfig(name="claude-review-agent"),
    harness=HarnessConfig(
        adapter_id="nvidia.fabric.claude",
        resolution="preinstalled",
        settings={
            "permission_mode": "dontAsk",
        },
    ),
    models={
        "default": ModelConfig(
            provider="anthropic",
            model="your-claude-model",
            api_key_env="ANTHROPIC_API_KEY",
        )
    },
    instructions=InstructionsConfig(
        system=InstructionConfig(
            content="Review changes for correctness and regressions.",
            mode="replace",
        )
    ),
    runtime=RuntimeConfig(
        artifacts="./artifacts",
        timeout_seconds=600,
        max_turns=8,
    ),
    environment=EnvironmentConfig(provider="local", workspace="."),
    tools=ToolsConfig(
        enabled=["Read", "Edit", "Bash"],
        blocked=["WebFetch"],
    ),
    mcp=McpConfig(
        servers={
            "repo": McpServerConfig(
                transport="stdio",
                url="repo-mcp",
                args=["--root", "."],
                exposure="harness_native",
            )
        }
    ),
    skills=SkillConfig(paths=["./skills/code-review"]),
)

fabric = Fabric()
```

## Single Invocation

```python
result = await fabric.run(
    config,
    base_dir=base_dir,
    input="Inspect the repository",
)

print(result.output["response"])
print(result.output["session_id"])
```

## Multi-Turn Runtime

```python
async with await fabric.start_runtime(config, base_dir=base_dir) as runtime:
    first = await runtime.invoke(input="Inspect the repository")
    second = await runtime.invoke(input="Now review the latest patch")

assert first.runtime_id == second.runtime_id
assert first.output["session_id"] == second.output["session_id"]
```

The runtime must remain on the same local host for its lifetime. A persisted
NeMo Fabric-to-Claude correlation record is not an attach token and cannot recover a
stopped or crashed local host.


## Testing

The default suite uses deterministic mock Claude Code and NeMo Relay CLIs and
requires no credentials. Test a current `nemo-relay` CLI with the mock Claude
client, or run the live integrations on an authenticated developer host:

```bash
FABRIC_NEMO_RELAY_COMMAND="$(command -v nemo-relay)" uv run --no-sync pytest tests/e2e/test_claude.py -q -k real_relay_gateway
RUN_FABRIC_CLAUDE_INTEGRATION=1 uv run --no-sync pytest tests/e2e/test_claude.py -q -k live
RUN_FABRIC_CLAUDE_RELAY_INTEGRATION=1 uv run --no-sync pytest tests/e2e/test_claude.py -q -k live_claude_relay
```

Set `FABRIC_TEST_CLAUDE_MODEL` to override the default live-test model,
`claude-sonnet-4-5`.

The live NeMo Relay test applies the same semantic artifact contract as Codex: ATOF
must contain structured LLM requests and token usage, and ATIF must contain the
expected agent response.
