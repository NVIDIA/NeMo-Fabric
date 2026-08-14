{/*
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
*/}

# NVIDIA NeMo Fabric Adapter Contract

The adapter contract is the versioned boundary between NeMo Fabric and an
adapter target. An adapter target can be an opinionated agent harness, a
lower-level framework that runs custom agents, or a remote service.

Consumers author northbound `FabricConfig`, `RunRequest`, and `RunResult`
objects. NeMo Fabric resolves those objects and projects the adapter-owned
parts into the southbound contract:

| Northbound Consumer Type | Southbound Adapter Type | Purpose |
| --- | --- | --- |
| `FabricConfig` | `AgentConfig` | Resolved configuration the selected adapter can apply. |
| `RunRequest` | `AgentRunRequest` (preview) | Future typed adapter-target invocation. |
| `RunResult` | `AgentRunResult` (preview) | Future typed adapter-target outcome before NeMo Fabric adds its own context. |

NeMo Fabric owns planning, environment preparation, adapter selection,
correlation, and consumer-facing results. The adapter owns translation into
the target, target lifecycle state, request execution, and target cleanup.

## Minimum Surface

A conforming local adapter provides:

1. A discoverable `*.fabric-adapter.json` descriptor.
2. One `start` operation that initializes an isolated runtime.
3. Zero or more ordered `invoke` operations.
4. One `stop` operation that attempts to release all runtime resources.

The minimum profile does not require adapter-managed queues, concurrent turns,
native streaming, cancellation, or live configuration updates. Consumers can
run independent Fabric runtimes concurrently.

Relay-backed `Runtime.invoke_stream()` does not expand this minimum. NeMo
Fabric invokes the ordinary adapter operation while NeMo Relay sends
invocation-correlated ATOF records directly to the consumer-side stream.

## Versioning

The current contract version is `fabric.adapter/v1alpha2`. The adapter
descriptor declares this value in `contract_version`. That version covers the
descriptor, `AgentConfig`, and the current lifecycle and runtime-context
binding. Published preview types are outside the negotiated contract until the
runtime enforces them; promoting them requires another contract-version bump.

An adapter package release version identifies an implementation release. It is
not the adapter contract version.

## Contract Status

Adapters receive `AgentConfig`; `FabricConfig` is never sent across the
southbound boundary. `AgentRunRequest` and `AgentRunResult` are preview
schemas only: the current local-host transport still sends `RunRequest` inside
the invocation payload and accepts JSON-compatible adapter output. Do not emit
`AgentRunResult` as the current host protocol; keep request and result
translation isolated for the future typed boundary.

The current local-host transport still wraps invocation input in a generic
payload. A future contract version can promote the typed request and result
schemas without changing the normalized configuration boundary.

## Continue Reading

Use these pages for the detailed contract:

- [Adapter Descriptor](adapter-descriptor.md)
- [Normalized Configuration](normalized-configuration.md)
- [Execution](execution.md)
- [Results and Telemetry](results.md)
- [Registration and Discovery](registration-and-discovery.md)
- [Custom Agents](custom-agents.md)
- [Conformance](conformance.md)

Canonical adapter-facing JSON Schemas are published in the repository
[`schemas/adapter-contract/` directory](https://github.com/NVIDIA/NeMo-Fabric/tree/main/schemas/adapter-contract).
Python adapters can validate the southbound models with
`nemo-fabric-adapter-contract` without depending on the NeMo Fabric runtime.
TypeScript adapters can import the descriptor, configuration, runtime-context,
request, and result types from `nemo-fabric-adapter-contract`, matching the
Python package's single model namespace. The request and result types retain
their documented preview status until the typed invocation transport is
negotiated. The TypeScript declarations provide compile-time structure, not
runtime JSON validation; validate untrusted data against the packaged JSON
Schemas.
