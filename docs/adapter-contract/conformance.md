{/*
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
*/}

# NVIDIA NeMo Fabric Adapter Conformance

The automated adapter conformance suite is planned for a follow-up release.
Until then, use this checklist to state the exact contract surface an adapter
implements. Passing the required profile does not imply that NVIDIA has
reviewed, trusted, or verified the adapter.

## Required Profile

The required profile covers the minimum adapter contract:

- The package installs one self-contained `fabric-adapter.json` descriptor.
- `contract_version` matches the installed NeMo Fabric adapter contract.
- The descriptor can be discovered without importing adapter code.
- Every embedded schema compiles and rejects undeclared fields where the
  adapter does not intentionally expose an open compatibility surface.
- `config.input` is `agent_config`, and the implementation validates
  `AgentConfig` before target translation.
- Planning rejects unsupported normalized fields, settings, workflows, tool
  definitions, and extensions.
- One `start`, zero or more ordered `invoke` operations, and one `stop` work for
  an isolated runtime.
- Startup failure and EOF both attempt cleanup.
- Runtime, invocation, and request IDs are preserved as opaque values.
- Target failures are normalized without logging credentials, environment
  secrets, or arbitrary user values.
- Independent Fabric runtimes do not share mutable target state accidentally.

## Claimed Capabilities

Test each descriptor claim separately:

| Capability | Evidence |
| --- | --- |
| Normalized config field | One accepted case and one unsupported or invalid case. |
| Settings, workflow, tool-definition, or extension schema | Valid and invalid examples exercised through planning. |
| Requirements | `doctor(...)` reports both satisfied and missing states. |
| Telemetry output | Output is produced and correlated to the correct invocation. |
| Relay-backed stream | Ordinary invoke completes while correlated ATOF reaches `Runtime.invoke_stream()`. |
| Cancellation, updates, or native streaming | Do not claim until the installed Fabric runtime exposes and tests the corresponding adapter operation. |

## Minimum Test Matrix

Run this minimum test matrix before publishing an adapter:

1. Descriptor discovery from an installed wheel.
2. Descriptor discovery from `<base_dir>/adapters` for development.
3. `Fabric().plan(...)` with the smallest valid config.
4. Planning rejection for one unsupported normalized field.
5. Planning rejection for invalid adapter settings and each published schema.
6. `doctor(...)` with missing and satisfied requirements.
7. Start, successful invoke, failed invoke, second invoke when permitted, and
   stop.
8. Start failure, invoke transport failure, malformed output, and cleanup on
   EOF.
9. Two independent runtimes to check state isolation.
10. Secret-redaction checks for logs and persisted diagnostic payloads.

Record unsupported optional capabilities explicitly rather than omitting them
from release notes. Link test results to the exact adapter package and contract
versions used.

See [Registration and Discovery](registration-and-discovery.md) for installed
descriptor verification and [Execution](execution.md) for lifecycle semantics.
