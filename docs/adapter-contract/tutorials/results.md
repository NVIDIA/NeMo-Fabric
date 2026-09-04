{/*
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
*/}

# Stage 4: Normalize Results and Telemetry

Every target invocation that completes through the adapter boundary produces
one terminal outcome. A lifecycle or transport failure can terminate the
operation before an `AgentRunResult` exists. Keep target-specific parsing
inside the adapter so consumers receive a stable NeMo Fabric `RunResult` and
do not need to understand the target's native response objects.

## Prerequisites

Before you start, complete the following:

1. Complete [Stage 3: Implement execution](execution.md) so the `invoke`
   operation can return a terminal outcome.
2. Identify the target's native response, usage, and error objects you will
   translate.
3. Keep the canonical
   [`agent-run-result.schema.json`](https://github.com/NVIDIA/NeMo-Fabric/blob/main/schemas/adapter-contract/agent-run-result.schema.json)
   open for the exact result shape.

## Concepts Overview

NeMo Fabric combines `AgentRunResult` with runtime-owned information when it
constructs the consumer-facing `RunResult`:

| NeMo Fabric Adds | Source |
| --- | --- |
| Adapter and runtime identity | Resolved plan and runtime handle. |
| Runtime, invocation, and request correlation | `RuntimeContext`. |
| Lifecycle stage and events | Runtime orchestration. |
| Collected artifact manifest | NeMo Fabric and adapter artifact declarations. |
| Telemetry reference | Resolved telemetry plan and runtime telemetry context. |

Do not duplicate NeMo Fabric-owned IDs, lifecycle events, or telemetry
references inside arbitrary adapter output or extensions.

## To Normalize Results and Telemetry

Work through each step in order, verifying your progress at each checkpoint.

### 1. Return AgentRunResult

`invoke` returns one typed `AgentRunResult`. Translate the target's native
outcome into the normalized status, output, usage, errors, and artifacts that
apply:

```python
return AgentRunResult(
    status=AgentRunStatus.SUCCEEDED,
    output={"response": native.final_text},
    usage=AgentUsage(
        input_tokens=native.input_tokens,
        output_tokens=native.output_tokens,
    ),
)
```

| Field | Requirement | Purpose |
| --- | --- | --- |
| `status` | Required | Reports `succeeded`, `failed`, or `cancelled`. |
| `output` | Required | Carries the primary JSON-compatible output and can be `null`. |
| `error` | Required for `failed` | Carries a stable code, safe message, retry guidance, and declared extensions. |
| `usage` | Optional | Carries normalized input, output, and total token counts plus cost when known. |
| `artifacts` | Optional | Carries target-produced artifact references relative to the runtime artifact root. |
| `extensions` | Optional | Carries adapter-owned result data validated by the descriptor. |

Use the canonical
[`agent-run-result.schema.json`](https://github.com/NVIDIA/NeMo-Fabric/blob/main/schemas/adapter-contract/agent-run-result.schema.json)
for the exact shape. A failed result contains an error; a successful result
does not contain a non-null error. Status is explicit and is not inferred from
arbitrary output fields.

**Success Check**: `invoke` returns one typed `AgentRunResult` whose explicit
status matches the presence or absence of an error.

### 2. Separate Failure Classes

Use these failure classes consistently:

- A **lifecycle failure** means the adapter could not satisfy `start`,
  `invoke`, or `stop`. It is reported at the relevant NeMo Fabric error stage
  and can invalidate the runtime.
- A **terminal target failure** means the target completed the invocation with
  a failed outcome. Return `AgentRunResult` with
  `status=AgentRunStatus.FAILED` and a safe, structured error.

Set retry guidance only when retrying at the consumer boundary is safe. NeMo
Fabric propagates failure information but does not automatically retry adapter
operations.

**Success Check**: A target failure returns a `FAILED` result while a lifecycle
failure is reported at its NeMo Fabric error stage.

### 3. Keep Artifacts Inside the Runtime Root

Write target artifacts below the artifact root supplied through
`RuntimeContext.environment`. Return relative artifact references; do not
return arbitrary host filesystem paths. NeMo Fabric combines adapter-declared
artifacts with its collected artifact manifest.

**Success Check**: Returned artifact references are relative to the runtime
artifact root, with no absolute host paths.

### 4. Integrate Telemetry Without Changing the Result

Telemetry configuration, correlation, and result references are NeMo
Fabric-owned. The Adapter Descriptor declares which outputs the adapter can
produce or forward. `RuntimeContext.telemetry` supplies the resolved
invocation-level context, including generated Relay configuration when
enabled.

An adapter can initialize target-native telemetry or forward NeMo
Fabric-provided Relay configuration. It must not reinterpret correlation IDs,
claim outputs it did not produce, or log unredacted telemetry environment
values.

Relay-backed ATOF records and the terminal result describe the same invocation
but remain separate. Stream exhaustion does not imply success, and stopping
stream consumption does not change the terminal outcome.

**Success Check**: Telemetry integration reuses NeMo Fabric correlation without
altering the terminal result or logging unredacted telemetry values.

## Summary

In this tutorial, you have:

- Returned one typed `AgentRunResult` with an explicit, error-consistent status.
- Separated lifecycle failures from terminal target failures.
- Kept artifacts inside the runtime artifact root as relative references.
- Integrated telemetry without changing the terminal result or leaking secrets.

## Next Steps

With outcomes safe and stable, continue through the adapter authoring stages:

<CardGroup cols={2}>

<Card title="Register and discover the adapter" href="registration-and-discovery.md">

Continue to Stage 5 and package and register the adapter.
</Card>

<Card title="AgentRunResult schema" href="https://github.com/NVIDIA/NeMo-Fabric/blob/main/schemas/adapter-contract/agent-run-result.schema.json">

Review the canonical schema for the exact result shape.
</Card>

</CardGroup>
