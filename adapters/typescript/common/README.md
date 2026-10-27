<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Fabric TypeScript Adapter Utilities

This package provides the persistent process lifecycle host shared by
TypeScript adapters. It validates southbound configuration, requests, runtime
context, and terminal results against the schemas bundled with
`nemo-fabric-adapter-contract`. It also provides an ACP client runner for
adapters whose harness exposes the Agent Client Protocol over stdio.

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

`startAcpSession` owns ACP process launch, capability negotiation, one
persistent session, prompt exchange, and cleanup. An adapter remains
responsible for translating normalized NeMo Fabric configuration into its harness's
ACP inputs and for normalizing ACP outcomes into NeMo Fabric results.

Refer to the [ACP runner guide](./acp-runner.md) for the process boundary,
invocation flow, lifecycle ownership, readiness handling, and shutdown behavior.

This package is intended to be published as the shared runtime dependency for
TypeScript adapters. Its public API will be versioned independently from the
adapters that use it.

## Runtime Health

An optional `AdapterRuntime.check(level)` method provides native health evidence
through the shared `AdapterCheckResult` contract. The lifecycle host exposes
checks through a separate authenticated loopback channel while an invocation
owns the ordered lifecycle pipe. Endpoint credentials are generated per process
and stored in a private file under `FABRIC_HOME`. HMAC-SHA256 signs requests
and responses with a fresh client challenge. The host verifies each response against that challenge and invalidates in-flight observations when shutdown begins.

Refer to the [SDK health transport diagram](https://github.com/NVIDIA/NeMo-Fabric/blob/main/docs/sdk/python.mdx#runtime-health) for lifecycle and health request routing.

## Dependency Rationale

`ajv` enforces the canonical JSON Schema contracts at the process boundary;
hand-written validators were rejected because they could drift from those
schemas. `nemo-fabric-adapter-contract` supplies the shared types and packaged
schemas; copying them into this package would create another contract authority.

`@agentclientprotocol/sdk` provides the stable ACP v1 client and schema types.
`zod` satisfies its runtime peer dependency.

`typescript` and `@types/node` are exact-pinned build inputs. They provide the
compiler and Node.js declarations without entering the published production
dependency graph. The private TypeScript workspace uses a local
`nemo-fabric-adapter-contract` file link so source builds test the checked-out
contract. Published package manifests use the registry version instead.

## Model Endpoint Policy

`insecureHttpModelHosts(settings, errorCode)` validates and normalizes the explicit `insecure_http_model_hosts` setting. `modelEndpointAllowed(url, hosts, allowPrivateHttp)` applies the shared HTTPS, loopback HTTP, and exact HTTP host policy used by OpenCode, Qwen, and Kilo. The optional private IPv4 argument preserves Kilo's existing opt-in. These utilities do not apply to MCP endpoints.

Lifecycle hosts collect selected software evidence at initialization and retain its original capture times across invocations. Supply `LifecycleHostOptions.software` with the adapter's own package URL, selected packages and harness name. No executable or service probe is issued. The start payload's `provenance.enabled=false` skips collection. Evidence travels beside the lifecycle outcome; `AgentRunResult` remains unchanged.
