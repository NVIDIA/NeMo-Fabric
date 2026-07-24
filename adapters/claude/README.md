<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Fabric Claude Adapter

The `nvidia.fabric.claude` adapter uses the official Claude Agent SDK for
Python behind NeMo Fabric's normalized invocation contract. The SDK is an
implementation detail; consumers select the Claude harness by adapter ID.

This adapter pins `claude-agent-sdk==0.2.120`. The SDK supplies and selects its
compatible Claude Code runtime.

## Install

To install just the Claude adapter by itself:

```bash
pip install "nemo-fabric[claude]"
```

To install just the Claude adapter along with the NeMo Fabric Runtime:

```bash
pip install "nemo-fabric[claude, runtime]"
```

## Authentication

NeMo Fabric preserves Claude's native credential resolution. Use an existing Claude
Code login for local development, `ANTHROPIC_AUTH_TOKEN` for a gateway or proxy
bearer credential, `ANTHROPIC_API_KEY` for a static API credential, or Anthropic
Workload Identity Federation (WIF) for production and CI workloads that should
not store a long-lived API key.

When the selected model provider is `nvidia`, the adapter reads the selected
model's credential from `api_key_env` (default: `NVIDIA_API_KEY`) and translates
the configured NVIDIA `/v1` endpoint into the host URL expected by Claude Code.
`models.<role>.base_url` overrides the public NVIDIA API Catalog endpoint. Set
it explicitly for a frontier or self-hosted endpoint. This request-scoped
mapping does not change the parent environment.

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

Refer to the [Claude adapter authentication guide](https://nvidia-nemo-fabric.docs.buildwithfern.com/nemo/fabric/integrations/claude)
for mode selection, required WIF variables, and the Relay boundary. Package
installation is verified by the adapter wheel and module-entrypoint tests.

Relay-enabled runs also require the external `nemo-relay` CLI. Refer to the [NeMo Relay CLI](https://docs.nvidia.com/nemo/fabric/getting-started/install#nemo-relay-cli) install guide for instructions on installing the CLI tool.

The Python `nemo-relay` package does not install this executable. Refer to the
[NeMo Relay installation guide](https://docs.nvidia.com/nemo/relay/getting-started/installation)
for other supported installation methods.

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

- `models` selects the Claude model. The adapter accepts the native `anthropic`
  provider and NVIDIA-hosted Anthropic Messages-compatible models through the
  `nvidia` provider. `models.<role>.base_url` selects an explicit endpoint.
- `system_prompt` supplies the Claude system instructions.
- `max_turns` sets the Claude turn limit.
- `runtime.timeout_seconds` sets the Fabric invocation deadline.
- `environment.workspace` sets the Claude working directory, and
  `environment.env` supplies explicit harness-visible variables.
- `tools.blocked` maps to Claude `disallowed_tools` using Claude-native tool
  names.
- `mcp` configures stdio, HTTP, streamable HTTP, or SSE servers. For stdio,
  NeMo Fabric parses `url` as a command plus arguments.
- `skills.paths` names skill directories that contain `SKILL.md`. The adapter
  stages these directories as a local Claude plugin for the runtime.

Only Claude-specific controls belong in `harness.settings`:

- `allowed_tools` and `permission_mode`
- `max_budget_usd`
- `setting_sources` (defaults to `[]` for deterministic isolation)

The adapter filters the inherited environment before launching Claude Code.
It retains portable OS/config variables, the selected model's `api_key_env`,
and explicitly configured `environment.env` values. Raw Claude stderr is consumed
by the SDK and is not persisted as a NeMo Fabric artifact.

## Relay Observability

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
and `NEMO_RELAY_GATEWAY_URL`. It also stages a runtime-scoped Claude plugin that
forwards lifecycle hooks with `nemo-relay hook-forward claude`.
`Fabric.run(...)` starts the same runtime, invokes it once, and stops it, so the
gateway has the same lifecycle as that single invocation.

The NeMo Fabric result includes `relay_runtime.gateway_config_path`,
`relay_runtime.gateway_log_path`, and the collected `relay_artifacts`. Relay
startup failures return a stable adapter error and retain the gateway log for
diagnosis. The default Claude Agent SDK dependency bundles a compatible Claude
Code executable.

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
    system_prompt="Review changes for correctness and regressions.",
    max_turns=8,
    runtime=RuntimeConfig(artifacts="./artifacts", timeout_seconds=600),
    environment=EnvironmentConfig(provider="local", workspace="."),
    tools=ToolsConfig(blocked=["WebFetch"]),
    mcp=McpConfig(
        servers={
            "repo": McpServerConfig(
                transport="stdio",
                url="repo-mcp --root .",
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
