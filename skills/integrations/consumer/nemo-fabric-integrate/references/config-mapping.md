<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Mapping Consumer Config To FabricConfig

Translate the consumer's own application, job, or deployment object into a typed
`FabricConfig` in memory. The consumer keeps owning its configuration model;
NeMo Fabric only receives the validated slice it needs.

## Public Config Models

Import these from the top-level `nemo_fabric` package:

| Model | Purpose |
| --- | --- |
| `FabricConfig` | Root config passed to every `Fabric` call. |
| `MetadataConfig` | Agent name and description. |
| `HarnessConfig` | `adapter_id`, `resolution`, and adapter-owned `settings`. |
| `ModelConfig` | Provider, model, credentials (`api_key_env`), endpoint, and sampling. |
| `InstructionsConfig` / `InstructionConfig` | Portable agent instructions and replacement mode. |
| `RuntimeConfig` | Input/output labels, artifact location, invocation timeout, and harness turn limit. |
| `EnvironmentConfig` | Execution environment, workspace, and harness-visible variables. |
| `ToolsConfig` | Adapter-native tool selection and blocking policy. |
| `McpConfig` / `McpServerConfig` | MCP transport, URL or command, process arguments, environment, and exposure. |
| `SkillConfig` | Skill directories. |
| `TelemetryConfig` | Telemetry providers. |
| `RelayConfig` and `Relay*Config` | NVIDIA NeMo Relay observability under the top-level `relay` block. |

The [models reference](https://github.com/NVIDIA/NeMo-Fabric/blob/main/docs/reference/api/python-library-reference/nemo_fabric.models.md)
indexes the public config models. The generated pages omit constructor fields and
defaults, so read the installed `nemo_fabric` models (they ship `py.typed`) for
exact field names and defaults.

## Build And Shape

Construct the nested config directly, then adjust capabilities with helper
methods that edit the typed config in place and return it:

- `add_skill_path(path)` / `remove_skill_path(path)`
- `add_mcp_server(name, *, transport, url, args, env, exposure, ...)` / `remove_mcp_server(name)`
- `enable_relay(...)` for NVIDIA NeMo Relay observability in the `relay` block
- `ToolsConfig(enabled=..., blocked=...)` for tool policy
- `block_tools(...)` for additive deny policy

```python
config = FabricConfig(
    metadata=MetadataConfig(name=job.name),
    harness=HarnessConfig(adapter_id=job.adapter_id, resolution="preinstalled"),
    models={"default": ModelConfig(provider=job.provider, model=job.model, api_key_env=job.api_key_env, base_url=job.base_url)},
    instructions=(
        InstructionsConfig(
            system=InstructionConfig(content=job.system_instruction),
        )
        if job.system_instruction is not None
        else None
    ),
    runtime=RuntimeConfig(
        input_schema="chat",
        output_schema="message",
        timeout_seconds=job.timeout_seconds,
        max_turns=job.max_turns,
    ),
)
config.add_skill_path(job.skill_dir)
```

## Variants Without Files

Create deployment or evaluation variants with deep copies and plain functions.
Each copy resolves, plans, and runs independently.

```python
def with_relay(base: FabricConfig) -> FabricConfig:
    config = base.model_copy(deep=True)
    config.enable_relay(output_dir="./artifacts/relay")
    return config
```

Use this function-and-copy pattern for every variant; keep all variation in
ordinary Python.

For ATOF, author the NeMo Relay 0.6 sink model directly. Put
`RelayAtofFileSinkConfig` and `RelayAtofStreamSinkConfig` instances in
`RelayAtofConfig.sinks`, and set `RelayAtofConfig.enabled=True`.

## Relative Paths

If the config uses relative paths for skills, workspaces, or artifacts, pass
`base_dir=...` to `plan(...)`, `doctor(...)`, `run(...)`, or
`start_runtime(...)`. The base directory anchors those paths to the consumer's
package or job layout, so nothing depends on the process working directory.

## Adapter-Owned And Caller-Owned Data

- Use normalized fields for portable behavior: models, instructions, turn
  limit, runtime, environment, tools, skills, MCP, and telemetry.
- Supply request context through `RunRequest.context` for each invocation;
  request context is not part of `FabricConfig`.
- Use `harness.settings` for adapter-owned configuration declared by the
  selected descriptor. Each bundled adapter declares a closed settings schema.
  Executable paths, state directories, and Relay command discovery are runtime
  implementation details, not adapter settings. Adapter settings are not
  portable. Planning and `doctor(...)` validate them against the schema embedded
  in the exact resolved adapter descriptor; a descriptor without a schema
  accepts only an empty settings map.
- Use `metadata` and extension fields for caller-owned annotations NeMo Fabric carries
  but does not interpret. Config `metadata` is not echoed into
  `RunResult.metadata`: the name surfaces as `RunResult.agent_name`, and for
  caller-owned correlation on a specific invocation set `RunRequest.request_id`,
  which is returned as `RunResult.request_id`.

## Stays Hidden Behind The Boundary

Do not surface these mechanics in the consumer-facing integration:

- Replacing direct SDK calls with a private serialized transport
  (`to_mapping()` is for inspection and private process boundaries; redact
  credentials, headers, metadata, and other sensitive or user-provided fields
  before logging, and never emit an unredacted mapping to logs).
- Importing `nemo_fabric._native` or adapter-internal modules.
- Reimplementing harness start, invoke, or stop logic, or managing adapter
  threads, sessions, or processes.
