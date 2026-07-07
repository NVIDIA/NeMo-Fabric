<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Fabric JSON Schemas

This directory contains committed JSON Schema snapshots for the public Fabric
contract. The files are generated from the Rust core types, not edited by hand.

## Exported Schemas

`fabric schema` exports the current public typed contract.

### Config And Planning

- `agent`: portable base `agent.yaml` config.
- `profile`: profile config applied over an agent config.
- `adapter-descriptor`: minimal adapter descriptor consumed by Fabric. Each
  descriptor declares a `contract_version`; Fabric rejects descriptors for
  unsupported adapter contracts during planning.
- `effective-config`: merged config after profile resolution.
- `run-plan`: executable plan derived from effective config.

### Adapter Invocation

- `adapter-invocation`: adapter-facing payload sent to inline or process
  adapters.
- `runtime-context`: per-run/per-invocation context included in adapter
  invocations.
- `run-request`: per-invocation request/input.

### Runtime Lifecycle

- `environment-handle`: prepared execution environment context.
- `runtime-handle`: lower-level active or resumable harness runtime binding.
- `session-handle`: caller-facing handle for one live or resumable agent
  session.
- `started-session`: native session start result containing both the public
  session handle and lower-level runtime handle.
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

Use the Fabric CLI to regenerate them after intentional contract changes:

```bash
cargo run -p fabric-cli -- schema --output-dir schemas
```

Use the CLI to inspect one schema:

```bash
cargo run -p fabric-cli -- schema --name agent
```

To add a new schema-backed typed model:

1. Define the public Rust type in `crates/fabric-core`.
2. Derive `Serialize`, `Deserialize`, and `JsonSchema`.
3. Add a `SchemaName` variant in `crates/fabric-core/src/schema.rs`.
4. Add the variant to `SchemaName::ALL`, `as_str()`, `parse()`, and
   `generate_schema()`.
5. Regenerate schemas with `cargo run -p fabric-cli -- schema --output-dir
   schemas`.
6. Add the new schema to the exported list above.

Run `cargo test` after regenerating schemas. The snapshot tests compare the
committed files against the schemas generated from the current Rust types and
fail on accidental drift.
