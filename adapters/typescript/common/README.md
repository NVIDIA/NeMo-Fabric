<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Fabric TypeScript Adapter Utilities

This package provides the persistent process lifecycle host shared by
TypeScript adapters. It validates southbound configuration, requests, runtime
context, and terminal results against the schemas bundled with
`nemo-fabric-adapter-contract`.

The host owns JSONL framing, ordered `start`/`invoke`/`stop` dispatch, runtime
identity checks, safe lifecycle failures, and cleanup after partial startup or
end of input. Adapter implementations own only target translation and target
state.

The following example starts and serves a `MyAdapterRuntime` instance:

```typescript
import { serve } from "nemo-fabric-adapters-common";

await serve(() => new MyAdapterRuntime());
```

The factory may return a runtime directly or resolve one asynchronously. The
host begins reading lifecycle input before it awaits asynchronous adapter setup.

This package is intended to be published as the shared runtime dependency for
TypeScript adapters. Its public API will be versioned independently from the
adapters that use it.

## Dependency Rationale

`ajv` enforces the canonical JSON Schema contracts at the process boundary;
hand-written validators were rejected because they could drift from those
schemas. `nemo-fabric-adapter-contract` supplies the shared types and packaged
schemas; copying them into this package would create another contract authority.

`typescript` and `@types/node` are exact-pinned build inputs. They provide the
compiler and Node.js declarations without entering the published production
dependency graph. The private TypeScript workspace uses a local
`nemo-fabric-adapter-contract` file link so source builds test the checked-out
contract. Published package manifests use the registry version instead.

## Model Endpoint Policy

`insecureHttpModelHosts(settings, errorCode)` validates and normalizes the explicit `insecure_http_model_hosts` setting. `modelEndpointAllowed(url, hosts, allowPrivateHttp)` applies the shared HTTPS, loopback HTTP, and exact HTTP host policy used by OpenCode, Qwen, and Kilo. The optional private IPv4 argument preserves Kilo's existing opt-in. These utilities do not apply to MCP endpoints.
