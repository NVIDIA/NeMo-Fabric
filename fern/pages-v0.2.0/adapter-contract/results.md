{/*
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
*/}

# Results and Telemetry

**Preview status:** `AgentRunResult` is not part of the negotiated adapter
contract yet. The current local-host transport treats it as ordinary JSON and
does not interpret `status: failed`; adapters must use the current host error
mechanism. The rest of this page defines the intended typed result boundary.

`AgentRunResult` is deliberately smaller than consumer-facing `RunResult`,
which also contains NeMo Fabric-owned identity, correlation, lifecycle events,
collected artifacts, and telemetry references.

## AgentRunResult

The result contains these adapter-facing fields:

| Field | Requirement | Description |
| --- | --- | --- |
| `status` | Required | `succeeded`, `failed`, or `cancelled`. |
| `output` | Required | Primary JSON-compatible output; it can be `null`. |
| `error` | Required for `failed` | Stable code, safe message, retry guidance, and declared extensions. |
| `usage` | Optional | Input, output, and total tokens plus cost in US dollars when known. |
| `artifacts` | Optional | Target-produced artifact references relative to the runtime artifact root. |
| `extensions` | Optional | Adapter-owned result data validated by the descriptor. |

Use the generated
[`AgentRunResult` JSON Schema](https://github.com/NVIDIA/NeMo-Fabric/blob/0.2.0-rc.1/schemas/adapter-contract/agent-run-result.schema.json)
for exact fields and constraints.

A failed result must contain `error`. A succeeded result can omit `error` or
set it to `null`; it must not contain a non-null error. Do not infer status
from arbitrary fields in `output`. Exactly one terminal result is produced for
an invocation, and its status is immutable once returned.

## Failure Classes

Failures fall into two classes:

- A lifecycle failure means the adapter could not satisfy `start`, `invoke`, or
  `stop`. It is surfaced at the relevant NeMo Fabric error stage and can
  invalidate the runtime.
- A terminal invocation failure means the target completed the invocation with
  a failed or cancelled outcome. It remains a normalized result rather than a
  lifecycle transport error.

Set `retryable` only when retrying at the consumer boundary is safe. NeMo
Fabric propagates retry guidance but does not automatically retry adapter
operations.

## NeMo Fabric Enrichment

NeMo Fabric combines the adapter outcome with NeMo Fabric-owned context:

| NeMo Fabric Adds | Source |
| --- | --- |
| Agent, harness, adapter, and runtime identity | Resolved plan and runtime handle |
| Runtime, invocation, and request correlation | `RuntimeContext` |
| NeMo Fabric lifecycle and progress events | Runtime orchestration |
| Collected artifact manifest | NeMo Fabric and adapter artifact declarations |
| Telemetry reference | Resolved telemetry plan and runtime telemetry context |
| Error stage | The lifecycle boundary where a failure surfaced |

Adapter extensions become namespaced adapter metadata only after validation.
Do not duplicate NeMo Fabric-owned IDs or telemetry references inside arbitrary
output.

## Streaming Results

For Relay-backed streaming, raw ATOF records and the terminal result describe
the same invocation. The result is authoritative and is returned separately
from the event stream. Stream exhaustion does not imply success, and stopping
stream consumption does not change the terminal status.

## Telemetry Ownership

Telemetry configuration and result references are NeMo Fabric-owned. The
descriptor declares what the adapter can produce or forward. At invocation
time the adapter receives the resolved `RuntimeTelemetryContext`, including
whether Relay is enabled, an optional generated config path, environment
values, and metadata.

An adapter can initialize target-native telemetry or forward NeMo
Fabric-provided Relay configuration, but it must not reinterpret correlation
IDs or claim outputs it did not produce. Never log unredacted telemetry
environment values.

The typed result will be promoted in a future contract version when the host
decodes and validates it. Until then, adapters should normalize target outcomes
in one dedicated function without returning this preview envelope directly.
