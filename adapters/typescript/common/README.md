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

`withPackageVersion(runtime, packageJson, servedModel?)` wraps a runtime so that each result carries the `nemo_fabric.versions.v1` extension with the adapter's package version, read synchronously from `packageJson` when the runtime starts. Declare that key as `{"type": "object"}` under `extension_schemas.run_result` in the adapter descriptor. The harness name and version and the served models read `unsupported` with reason `no_source`. The optional `servedModel(input)` callback receives the start input and returns a `ServedModelFact`, or `undefined` to keep `no_source` for every role.

`ServedModelFact` is `{role: string; fact: JsonObject}`: the served-model fact for one model role. Build a fact without a value with `missingFact(state, reason, method)`, for example `missingFact("unavailable", "not_attributed", "harness_api")`; `method` defaults to `adapter_report`.

`packageVersionReport(packageVersion, modelRoles, servedModel?)` builds the report object itself. `servedModel`, a `ServedModelFact`, gives its role that fact, and every other role keeps `no_source`; a role that is not in `modelRoles` adds no entry. Like NeMo Fabric core, the report keeps at most 16 roles in sorted order and skips a role longer than 128 characters.
