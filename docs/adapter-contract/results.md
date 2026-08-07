<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Results and Telemetry

`AgentRunResult` is the terminal adapter-facing result. It is deliberately
smaller than consumer-facing `RunResult`, which also contains Fabric-owned
identity, correlation, lifecycle events, collected artifacts, and telemetry
references.

## AgentRunResult

| Field | Requirement | Description |
| --- | --- | --- |
| `status` | Required | `succeeded`, `failed`, or `cancelled`. |
| `output` | Required | Primary JSON-compatible output; it may be `null`. |
| `error` | Required for `failed` | Stable code, safe message, retry guidance, and declared extensions. |
| `usage` | Optional | Input, output, and total tokens plus cost in US dollars when known. |
| `artifacts` | Optional | Target-produced artifact references relative to the runtime artifact root. |
| `extensions` | Optional | Adapter-owned result data validated by the descriptor. |

Use the generated
[`AgentRunResult` JSON Schema](https://github.com/NVIDIA/NeMo-Fabric/blob/main/schemas/adapter-contract/agent-run-result.schema.json)
for exact fields and constraints.

A failed result must contain `error`. A succeeded result must not contain
`error`. Do not infer status from arbitrary fields in `output`. Exactly one
terminal result is produced for an invocation, and its status is immutable
once returned.

## Failure Classes

- A lifecycle failure means the adapter could not satisfy `start`, `invoke`, or
  `stop`. It is surfaced at the relevant Fabric error stage and may invalidate
  the runtime.
- A terminal invocation failure means the target completed the invocation with
  a failed or cancelled outcome. It remains a normalized result rather than a
  lifecycle transport error.

Set `retryable` only when retrying at the consumer boundary is safe. NeMo
Fabric propagates retry guidance but does not automatically retry adapter
operations.

## Fabric Enrichment

NeMo Fabric combines the adapter outcome with Fabric-owned context:

| Fabric adds | Source |
| --- | --- |
| Agent, harness, adapter, and runtime identity | Resolved plan and runtime handle |
| Runtime, invocation, and request correlation | `RuntimeContext` |
| Fabric lifecycle and progress events | Runtime orchestration |
| Collected artifact manifest | Fabric and adapter artifact declarations |
| Telemetry reference | Resolved telemetry plan and runtime telemetry context |
| Error stage | The lifecycle boundary where a failure surfaced |

Adapter extensions become namespaced adapter metadata only after validation.
Do not duplicate Fabric-owned IDs or telemetry references inside arbitrary
output.

## Streaming Results

For Relay-backed streaming, raw ATOF records and the terminal result describe
the same invocation. The result is authoritative and is returned separately
from the event stream. Stream exhaustion does not imply success, and stopping
stream consumption does not change the terminal status.

## Telemetry Ownership

Telemetry configuration and result references are Fabric-owned. The descriptor
declares what the adapter can produce or forward. At invocation time the
adapter receives the resolved `RuntimeTelemetryContext`, including whether
Relay is enabled, an optional generated config path, environment values, and
metadata.

An adapter may initialize target-native telemetry or forward Fabric-provided
Relay configuration, but it must not reinterpret correlation IDs or claim
outputs it did not produce. Never log unredacted telemetry environment values.

The current Python host accepts JSON-compatible invoke output. Enforced
`AgentRunResult` decoding is a contract transition still to be wired into that
transport; adapters should normalize target outcomes in one dedicated function
today.
