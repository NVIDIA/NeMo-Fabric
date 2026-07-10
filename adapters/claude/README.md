<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Claude Adapter

The `nvidia.fabric.claude` adapter uses the official Claude Agent SDK for
Python behind Fabric's normalized invocation contract. The SDK is an
implementation detail; consumers select the Claude harness by adapter ID.

## Install

```bash
just wheels
python -m pip install --find-links dist "nemo-fabric[claude]"
```

Claude Code authentication can come from an existing cached login or from
`ANTHROPIC_API_KEY`. Package installation is verified by the adapter wheel and
module-entrypoint tests. Authentication is validated when Claude starts the
invocation.

## Execution Model

Each `invoke` starts a fresh adapter process. The adapter persists the terminal
Claude session ID under the Fabric artifact root, keyed by `runtime_id`, and
passes it as `ClaudeAgentOptions.resume` on the next invocation. One Fabric
runtime therefore maps to one Claude session even though no adapter process
stays resident.

## Configuration

Configure portable capabilities through the normalized `FabricConfig` fields:

- `models` selects the Claude model. A configured model must use
  `provider="anthropic"`; normalized hosted/custom provider resolution is
  tracked in [FABRIC-64](https://linear.app/nvidia/issue/FABRIC-64/add-normalized-model-provider-resolution-and-harness-compatibility).
- `environment.workspace` sets the Claude working directory.
- `tools` sets the base Claude tool list.
- `mcp` configures stdio, HTTP, streamable HTTP, or SSE servers. For stdio,
  Fabric parses `url` as a command plus arguments.
- `skills.paths` names skill directories that contain `SKILL.md`. The adapter
  stages these directories as a local Claude plugin for the invocation.

Only Claude-specific controls belong in `harness.settings`:

- `system_prompt`, `allowed_tools`, `disallowed_tools`, and `permission_mode`
- `max_turns`, `max_budget_usd`, and `timeout_seconds`
- `setting_sources` (defaults to `[]` for deterministic isolation)
- `cli_path` for testing or an explicitly installed Claude Code executable
- `env` for variables explicitly forwarded to Claude Code

Putting `model_name`, `cwd`, `tools`, `mcp_servers`, or `skills` in
`harness.settings` is an error. Use the corresponding normalized field so the
same consumer configuration can compose with other adapters.

The adapter filters the inherited environment before launching Claude Code.
It retains portable OS/config variables, the selected model's `api_key_env`,
and explicitly configured `settings.env` values. Raw Claude stderr is consumed
by the SDK and is not persisted as a Fabric artifact.

## Typed Configuration

Build the agent configuration with the typed SDK models before invoking
Fabric:

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
)

base_dir = Path("/workspace/review-agent")
config = FabricConfig(
    metadata=MetadataConfig(name="claude-review-agent"),
    harness=HarnessConfig(
        adapter_id="nvidia.fabric.claude",
        resolution="preinstalled",
        settings={
            "system_prompt": "Review changes for correctness and regressions.",
            "permission_mode": "dontAsk",
            "max_turns": 8,
        },
    ),
    models={
        "default": ModelConfig(
            provider="anthropic",
            model="your-claude-model",
            api_key_env="ANTHROPIC_API_KEY",
        )
    },
    runtime=RuntimeConfig(artifacts="./artifacts"),
    environment=EnvironmentConfig(provider="local", workspace="."),
    tools=["Read", "Glob", "Grep"],
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

## One-Shot Run

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

Resume requires the same workspace and Claude state directory on the same host.
The Fabric-to-Claude correlation record alone is insufficient if Claude's
underlying transcript store is removed.

## Tests

The default suite uses a deterministic mock Claude Code CLI and requires no
credentials. Run the real integration only on an authenticated developer host:

```bash
RUN_FABRIC_CLAUDE_INTEGRATION=1 uv run --no-sync pytest tests/e2e/test_claude.py -q -k live
```
