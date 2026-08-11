<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Fabric JSON Schemas

This directory contains committed JSON Schema snapshots for the public NeMo Fabric
contract. The files are generated from the Rust core types, not edited by hand.

The Python adapter-contract package exposes dependency-free dataclasses with
optional Pydantic interoperability. Those models are hand-maintained against
these Rust-generated schemas. The TypeScript adapter-contract package generates
compile-time declarations from the committed schema snapshots. When a
schema-backed Rust type changes, update each applicable language binding and
its parity tests in the same change.

## Directory Layout

The core schema generator separates the southbound adapter contract from
consumer and Fabric-runtime schemas:

```text
schemas/
├── adapter-contract/          # Public adapter-facing contract
│   ├── adapter-descriptor.schema.json
│   ├── agent-config.schema.json
│   ├── agent-run-request.schema.json
│   ├── agent-run-result.schema.json
│   ├── runtime-context.schema.json
│   └── legacy/
│       ├── adapter-invocation.schema.json
│       ├── openai-stream-invocation.schema.json
│       └── openai-stream-record.schema.json
└── *.schema.json              # Northbound and Fabric-runtime contracts
```

An adapter author can treat `adapter-contract/` as the complete schema entry
point. The `legacy/` subdirectory contains transitional local-host payloads
used while first-party adapters migrate to typed execution types.

The language bindings preserve this boundary:

- Python adapters use `nemo-fabric-adapter-contract` for dependency-free
  dataclasses and optional Pydantic models.
- TypeScript adapters use `nemo-fabric-adapter-contract` for the descriptor,
  configuration, runtime-context, request, and result types, matching the
  Python package's single model namespace. Request and result types retain
  their documented preview status until the typed invocation transport is
  negotiated. The package also includes these canonical schemas for runtime
  validation without selecting a validation-library dependency.

`FabricConfig` is the northbound source of consumer intent. Planning produces
the `CapabilityPlan` as routed evidence and projects the fields accepted by the
selected descriptor into `AgentConfig`, the authoritative southbound adapter
input. The generated schemas and projection tests must change together so
these related representations do not drift.

## Adapter Contract

- `adapter-contract/adapter-descriptor`: adapter identity, runner,
  requirements, accepted normalized fields, schemas, telemetry support, and
  runtime capability claims.
- `adapter-contract/agent-config`: typed configuration projected southbound to
  one adapter target. Adapter-owned additions are carried only through explicit
  `extensions` blocks and validated against schemas declared by the selected
  descriptor.
- `adapter-contract/agent-run-request`: invocation input and caller context
  projected southbound after Fabric resolves consumer-owned request fields.
- `adapter-contract/agent-run-result`: terminal status, output, errors, usage,
  and artifact references returned before Fabric adds northbound identity,
  telemetry, and lifecycle data.
- `adapter-contract/runtime-context`: Fabric-generated runtime, invocation,
  environment, artifact, and telemetry context passed southbound.

Tool definitions fail closed. An adapter must both accept
`tools.definitions` and publish `tool_definition_schema`; planning rejects the
configuration otherwise. The base normalized fields and adapter-owned
extensions are validated separately against the generated contract and the
descriptor schema.

### Legacy Adapter Transport

- `adapter-contract/legacy/adapter-invocation`: current per-turn payload sent to
  an initialized persistent local adapter host. It contains `runtime_context`
  and the northbound `run-request`. It will be removed after adapters consume
  the typed southbound request directly.
- `adapter-contract/legacy/openai-stream-invocation`: current native OpenAI
  stream payload sent to an initialized persistent local adapter host. It adds
  a Fabric-owned authenticated loopback stream sink to the per-turn runtime
  context and request. The common host validates and removes the sink before
  calling `invoke_openai_stream(payload, emit)`.
- `adapter-contract/legacy/openai-stream-record`: correlated chunk and explicit
  end records carried as chunked NDJSON. The chunk variant freezes the
  `openai.chat_completions.chunk/v1` profile accepted by the SDK listener.

## Fabric Consumer and Runtime Contracts

### Config and Planning

- `agent`: complete typed northbound `FabricConfig`.
- `run-plan`: executable plan containing the canonical northbound config, its
  projected southbound `AgentConfig`, the selected adapter, and derived
  execution metadata.
- `run-request`: northbound per-invocation request and input.

### Runtime Lifecycle

- `environment-handle`: prepared execution environment context.
- `runtime-handle`: active harness runtime identity and opaque adapter binding.
- `invocation-handle`: one request or turn sent to a runtime.

### Results, Artifacts, and Diagnostics

- `run-result`: normalized consumer-facing invocation result.
- `artifact-manifest`: normalized artifact references.
- `error-info`: structured runtime or adapter error metadata.
- `fabric-event`: NeMo Fabric lifecycle or progress event.

### Deferred Core Objects

The MVP core object pass intentionally defers normalized trajectory structures
and policy hooks for auditability. They are not separate NeMo Fabric schemas yet.
When NeMo Fabric owns those contracts directly, add them as first-class Rust types
and export them here.

## How To Maintain

Use the core generator to regenerate them after intentional contract changes:

```bash
just schemas
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

Regenerate the TypeScript projection after an intentional adapter-contract
schema change:

```bash
just generate-typescript-contract
```

Run `just test-typescript` to check generated-file drift, strict compile-time
fixtures, package contents, and clean-consumer imports.
