<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Fabric JSON Schemas

This directory contains committed JSON Schema snapshots for the public Fabric
contract. The files are generated from the Rust core types, not edited by hand.

The Python SDK exposes Pydantic authoring models for application callers. Those
models are hand-maintained against these Rust-generated schemas for now. When a
schema-backed Rust type changes, update the matching Pydantic model and its
schema-alignment tests in the same change.

## Exported Schemas

The core schema generator exports the current public typed contract.

### Config and Planning

- `agent`: complete typed `FabricConfig`.
- `adapter-descriptor`: minimal adapter descriptor consumed by Fabric. Each
  descriptor declares a `contract_version`; Fabric rejects descriptors for
  unsupported adapter contracts during planning. The `process` and `python`
  adapter kinds use Fabric's persistent local-host wire protocol.
- `run-plan`: executable plan containing the canonical typed config, absolute
  base directory, selected adapter, and derived execution metadata.

### Adapter Invocation

- `adapter-invocation`: per-turn payload sent to an initialized persistent
  local adapter host. It contains only `runtime_context` and `request`; Fabric
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
- `fabric-event`: Fabric lifecycle/progress event.

### Deferred Core Objects

The MVP core object pass intentionally defers normalized trajectory structures
and policy hooks for auditability. They are not separate Fabric schemas yet.
When Fabric owns those contracts directly, add them as first-class Rust types
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
