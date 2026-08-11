<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Fabric Adapter Contract for TypeScript

Dependency-free TypeScript types for implementing adapters against the NVIDIA
NeMo Fabric adapter contract. The package is generated from the versioned JSON
Schemas maintained in the NeMo Fabric repository.

## Install

```bash
npm install @nvidia/nemo-fabric-adapter-contract
```

Use Node.js 20.18.3 or later and TypeScript 5.0 or later. Configure TypeScript
with `node16`, `nodenext`, or `bundler` module resolution so package export
subpaths resolve correctly.

## Stable v1alpha2 contract

The root entry point contains the negotiated descriptor, southbound agent
configuration, and runtime context types:

```typescript
import { ADAPTER_CONTRACT_VERSION } from "@nvidia/nemo-fabric-adapter-contract";
import type {
  AdapterDescriptor,
  AgentConfig,
  RuntimeContext,
} from "@nvidia/nemo-fabric-adapter-contract";

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

## Preview invocation types

Request and result types are not part of the negotiated v1alpha2 lifecycle
transport. Import them through the explicit preview entry point:

```typescript
import type {
  AgentRunRequest,
  AgentRunResult,
} from "@nvidia/nemo-fabric-adapter-contract/preview";
```

Do not depend on preview types as a stable transport contract. In particular,
token counts originate from JSON Schema `uint64` values but are represented as
JavaScript `number`; values greater than `Number.MAX_SAFE_INTEGER` cannot be
represented exactly.

## JSON Schemas

The package bundles byte-identical copies of the canonical schemas. Consumers
that need runtime validation can use their validator of choice, for example:

```typescript
import agentConfigSchema from "@nvidia/nemo-fabric-adapter-contract/schemas/agent-config" with { type: "json" };
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
