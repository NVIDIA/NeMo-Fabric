<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Fabric JSON Schemas

This directory contains committed JSON Schema snapshots for the public NeMo Fabric
contract. The files are generated from the Rust core types, not edited by hand.

The Python SDK exposes Pydantic authoring models for application callers. Those
models are hand-maintained against these Rust-generated schemas for now. When a
schema-backed Rust type changes, update the matching Pydantic model and its
schema-alignment tests in the same change.

## Exported Schemas

The core schema generator exports the current public typed contract.

### Config and Planning

- `agent`: complete typed `FabricConfig`.
- `agent-config`: typed configuration projected southbound to one adapter
  target. Adapter-owned additions are carried only through explicit
  `extensions` blocks and validated against schemas declared by the selected
  adapter descriptor.
- `agent-run-request`: invocation input and caller context projected southbound
  after Fabric resolves consumer-owned request fields.
- `agent-run-result`: terminal status, output, errors, and artifact references
  returned by an adapter target before Fabric adds northbound identity,
  telemetry, and lifecycle data.
- `adapter-descriptor`: minimal adapter descriptor consumed by NeMo Fabric. Each
  descriptor declares a `contract_version`; NeMo Fabric rejects descriptors for
  unsupported adapter contracts during planning. A descriptor can embed the
  JSON Schemas for its adapter-owned `harness.settings`, optional
  `FabricConfig.workflow`, and normalized tool definitions, plus schemas for
  explicit adapter-owned extension points in the southbound contract;
  malformed schemas fail descriptor loading. `config.input=agent_config` opts
  the adapter into the projected southbound config; omitting it preserves the
  legacy northbound `FabricConfig` payload. A missing settings schema rejects
  non-empty settings, and a configured workflow requires a workflow schema.
  The `process` and `python` adapter kinds use NeMo Fabric's persistent
  local-host wire protocol.
- `run-plan`: executable plan containing the canonical northbound config, its
  projected southbound `AgentConfig`, the selected adapter, and derived
  execution metadata.

### Adapter Invocation

- `adapter-invocation`: per-turn payload sent to an initialized persistent
  local adapter host. It contains only `runtime_context` and `request`; NeMo Fabric
  sends configuration and capability planning data during lifecycle start.
- `runtime-context`: per-run/per-invocation context included in adapter
  invocations.
- `run-request`: per-invocation request/input.

### Runtime Lifecycle

- `environment-handle`: prepared execution environment context.
- `runtime-handle`: active harness runtime identity and opaque adapter binding.
- `invocation-handle`: one request/turn sent to a runtime.

### Results, Artifacts, And Diagnostics

- `run-result`: normalized invocation result.
- `artifact-manifest`: normalized artifact references.
- `error-info`: structured runtime or adapter error metadata.
- `fabric-event`: NeMo Fabric lifecycle/progress event.

### Deferred Core Objects

The MVP core object pass intentionally defers normalized trajectory structures
and policy hooks for auditability. They are not separate NeMo Fabric schemas yet.
When NeMo Fabric owns those contracts directly, add them as first-class Rust types
and export them here.

## How To Maintain

Use the core generator to regenerate them after intentional contract changes:

```bash
cargo run -p nemo-fabric-core --example generate-schemas -- schemas
```

To add a new schema-backed typed model:

1. Define the public Rust type in `crates/fabric-core`.
2. Derive `Serialize`, `Deserialize`, and `JsonSchema`.
3. Add a `SchemaName` variant in `crates/fabric-core/src/schema.rs`.
4. Add the variant to `SchemaName::ALL`, `as_str()`, `parse()`, and
   `generate_schema()`.
5. Regenerate schemas with the command above.
6. Add the new schema to the exported list above.

Run `cargo test` after regenerating schemas. The snapshot tests compare the
committed files against the schemas generated from the current Rust types and
fail on accidental drift.
