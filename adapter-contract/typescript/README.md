<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Fabric Adapter Contract

Dependency-free TypeScript types for implementing adapters against the NVIDIA
NeMo Fabric adapter contract. The package is generated from the versioned JSON
Schemas maintained in the NeMo Fabric repository.

## Install

Install the package from the public npm registry:

```bash
npm install nemo-fabric-adapter-contract
```

Use Node.js 20.18.3 or later and TypeScript 5.6 or later. The TypeScript floor
ensures NodeNext consumers can import the bundled JSON Schemas. Use a compatible
TypeScript module pair: `module: "NodeNext"` with
`moduleResolution: "NodeNext"`, or `module: "ESNext"` with
`moduleResolution: "bundler"`. Enable `resolveJsonModule` when importing the
bundled JSON Schemas.

The package is ESM-only and does not support CommonJS `require`. NodeNext
consumers must set `"type": "module"` in their `package.json`.

## Stable v1alpha2 Contract

The root entry point contains the negotiated descriptor, southbound agent
configuration, and runtime context types:

```typescript
import { ADAPTER_CONTRACT_VERSION } from "nemo-fabric-adapter-contract";
import type {
  AdapterDescriptor,
  AgentConfig,
  RuntimeContext,
} from "nemo-fabric-adapter-contract";

const descriptor: AdapterDescriptor = {
  contract_version: ADAPTER_CONTRACT_VERSION,
  adapter_id: "pi",
  harness: "pi",
  adapter_kind: "process",
};
```

Property names intentionally match the JSON wire format and remain
`snake_case`. Optional properties are distinct from properties whose value may
be `null`.

## Invocation Types

The package root also exports the same request and result models as the Python
adapter-contract package:

```typescript
import type {
  AgentRunRequest,
  AgentRunResult,
} from "nemo-fabric-adapter-contract";
```

These types remain documented as preview because the current local-host
transport does not yet enforce them. Keep request and result translation at the
adapter edge until the typed invocation boundary is negotiated. In particular,
token counts originate from JSON Schema `uint64` values but are represented as
JavaScript `number`; values greater than `Number.MAX_SAFE_INTEGER` cannot be
represented exactly.

## JSON Schemas

The package bundles byte-identical copies of the canonical schemas. Consumers
that need runtime validation can use their validator of choice, for example:

```typescript
import agentConfigSchema from "nemo-fabric-adapter-contract/schemas/agent-config" with { type: "json" };
```

The TypeScript declarations provide compile-time checking only. They do not
apply schema defaults or enforce runtime constraints such as string patterns,
numeric ranges, or relative paths.

## Development

From this directory:

```bash
npm ci
npm run generate:check
npm test
```

Run `npm run generate` after the canonical schemas change. Generated source and
schema copies are committed so drift is reviewable.

### Build Dependencies

`json-schema-to-typescript` generates declarations from the canonical JSON
Schemas. A hand-maintained declaration hierarchy was rejected because it would
create a second contract authority; a custom generator would duplicate an
existing focused build tool. `typescript` compiles the package and strict
positive and negative fixtures; transpilers cannot replace its type checker.
Both dependencies are exact-pinned build inputs and neither is present in the
published production dependency graph.

The resolved development graph includes `argparse@2.0.1` under `Python-2.0`.
That build-only license remains an explicit dependency-approver review item.
