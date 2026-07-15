<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Mapping Consumer Config To FabricConfig

Translate the consumer's own application, job, or deployment object into a typed
`FabricConfig` in memory. The consumer keeps owning its configuration model;
Fabric only receives the validated slice it needs.

## Public Config Models

Import these from the top-level `nemo_fabric` package:

| Model | Purpose |
| --- | --- |
| `FabricConfig` | Root config passed to every `Fabric` call. |
| `MetadataConfig` | Agent name and description. |
| `HarnessConfig` | `adapter_id`, `resolution`, and adapter-owned `settings`. |
| `ModelConfig` | Provider, model, credentials (`api_key_env`), and sampling. |
| `RuntimeConfig` | `input_schema`, `output_schema`, and artifact locations. |
| `EnvironmentConfig` | Execution environment (`local`, sandbox, control location). |
| `McpConfig` / `McpServerConfig` | MCP servers and exposure. |
| `SkillConfig` | Skill directories. |
| `TelemetryConfig` | Telemetry providers. |
| `RelayConfig` and `Relay*Config` | Relay observability under the top-level `relay` block. |

Generated references remain the source of truth for exact fields and defaults:
the [models reference](https://github.com/NVIDIA/NeMo-Fabric/blob/main/docs/reference/api/python-library-reference/nemo_fabric.models.md).

## Build And Shape

Construct the nested config directly, then adjust capabilities with helper
methods that edit the typed config in place and return it:

- `add_skill_path(path)` / `remove_skill_path(path)`
- `add_mcp_server(name, *, transport, url, exposure, ...)` / `remove_mcp_server(name)`
- `enable_relay(...)` for Relay observability in the `relay` block

```python
config = FabricConfig(
    metadata=MetadataConfig(name=job.name),
    harness=HarnessConfig(adapter_id=job.adapter_id, resolution="preinstalled"),
    models={"default": ModelConfig(provider=job.provider, model=job.model, api_key_env=job.api_key_env)},
    runtime=RuntimeConfig(input_schema="chat", output_schema="message"),
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

Use this function-and-copy pattern for every variant. Profiles — file-backed or
in-memory — are not part of the consumer-facing API; keep all variation in
ordinary Python.

## Relative Paths

If the config uses relative paths for skills, workspaces, or artifacts, pass
`base_dir=...` to `resolve(...)`, `plan(...)`, `doctor(...)`, `run(...)`, or
`start_runtime(...)`. The base directory anchors those paths to the consumer's
package or job layout, so nothing depends on the process working directory.

## Adapter-Owned And Caller-Owned Data

- Use normalized fields for portable behavior: models, runtime, environment,
  skills, MCP, telemetry, and request context.
- Use `harness.settings` for adapter-owned configuration the selected adapter
  understands (for example Hermes launch options or Codex CLI flags). Adapter
  settings are not portable, and `doctor(...)` does not validate their contents —
  an unknown or misspelled key still passes and is silently ignored unless the
  adapter reads it. Validate settings against the adapter's docs and your
  integration tests.
- Use `metadata` for caller-owned annotations Fabric should preserve and echo
  back but not interpret.

## Stays Hidden Behind The Boundary

Do not surface these mechanics in the consumer-facing integration:

- Writing, reading, or materializing `agent.yaml` or portable agent packages.
- Serializing `FabricConfig` to disk as the integration path (`to_mapping()` is
  for inspection and logging, not a required file step).
- Profiles of any kind: file-backed profiles, profile-by-name resolution, or
  in-memory `FabricProfileConfig` overlays.
- Importing `nemo_fabric._native`, `nemo_fabric._config_sources`, or
  adapter-internal modules.
- Reimplementing harness start, invoke, or stop logic, or managing adapter
  threads, sessions, or processes.
