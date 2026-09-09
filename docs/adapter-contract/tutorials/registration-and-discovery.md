{/*
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
*/}

# Stage 5: Register and Discover the Adapter

Registration makes descriptor metadata available to NVIDIA NeMo Fabric.
Discovery reads that metadata without importing adapter code, installing
dependencies, or starting a runtime. Runtime loading begins only after
selection and planning succeed.

## Prerequisites

Before you start, complete the following:

1. Complete [Stage 4: Normalize results and telemetry](results.md) so the
   adapter produces stable terminal outcomes.
2. Have the Adapter Descriptor, and any Adapter Target Descriptor records, from
   [Stage 1: Describe the adapter](adapter-descriptor.md).
3. Decide whether the package installs registered targets that require a
   `*.fabric-target.json` record.

## Concepts Overview

Registration and selection depend on the integration type:

| Integration | Selected By | Required Records |
| --- | --- | --- |
| Harness or dedicated custom-agent adapter | `harness.adapter_id` | Adapter Descriptor only |
| Custom agent using a shared framework adapter | `workflow.target_id` | Adapter Target Descriptor plus the selected Adapter Descriptor |

Register an Adapter Target only when multiple independently installed targets
share one adapter. A harness or dedicated custom-agent adapter is already
identified by its Adapter Descriptor and does not publish a separate target
record.

NeMo Fabric builds one registry from these sources in deterministic order:

1. Descriptor records bundled with NeMo Fabric.
2. Records installed recursively below
   `<sysconfig data>/share/nemo-fabric`. When `ADAPTER_PYTHON` is set, NeMo
   Fabric queries that Python environment instead of the current one.
3. Files or directories listed in `FabricConfig.discovery.local_paths`.
   Relative paths resolve from `base_dir`.

The order above controls discovery, not precedence. There is no implicit
`<base_dir>/adapters` scan and no local override rule.

Semantically identical records with the same ID are deduplicated and retain
all provenance. Different records with the same ID are ambiguous and fail
planning. Explicit paths that do not exist, files with an unrecognized suffix,
and malformed records fail when selection depends on them.

The v1alpha2 registry resolves adapters and targets by exact ID. It does not
provide a human-facing catalog or presentation metadata.

Planning performs these steps before target code starts:

1. Discover and validate descriptor records.
2. Resolve `workflow.target_id` when present.
3. Select the target's adapter, or select `harness.adapter_id` for direct use.
4. Cross-check an optional dual selector and the adapter's supported target
   type.
5. Validate harness settings, workflow settings, normalized configuration, and
   declared schemas.
6. Project `AgentConfig` and retain the resolved records in `RunPlan`.
7. Load the adapter runner only when the runtime starts.

`RunPlan.adapter_descriptor` and `RunPlan.adapter_target_descriptor` retain the
resolved records and discovery provenance. `doctor(...)` reports both records
for registered-target plans.

## To Register and Discover the Adapter

Work through each step in order, verifying your progress at each checkpoint.

### 1. Publish Package Records

An adapter package publishes one `*.fabric-adapter.json` record. A package that
installs registered targets publishes one `*.fabric-target.json` record per
target. A shared adapter and its target packages can be distributed
independently.

```text
acme-fabric-package/
├── acme.fabric-adapter.json
├── email-phishing.fabric-target.json
└── src/acme_fabric_adapter/
```

Python wheels install records below the common data root. Directory names are
organizational; descriptor IDs are authoritative:

```toml
[tool.setuptools.data-files]
"share/nemo-fabric/adapters/acme" = ["acme.fabric-adapter.json"]
"share/nemo-fabric/targets/acme" = ["email-phishing.fabric-target.json"]
```

An adapter that consumes typed southbound configuration depends on
`nemo-fabric-adapter-contract`. Add
`nemo-fabric-adapters-common` only when the adapter uses its optional lifecycle
or Relay helpers. A bare adapter package does not need the NeMo Fabric runtime
as a dependency.

**Success Check**: Installing the package places the `*.fabric-adapter.json`
record, and any `*.fabric-target.json` records, below the common data root.

### 2. Use Explicit Paths During Development

Bundled and installed descriptors require no discovery configuration. Point to
local files or directories only for source examples and adapter development:

```python
FabricConfig(
    discovery=DiscoveryConfig(
        local_paths=[
            "./adapter-metadata/acme.fabric-adapter.json",
            "./targets/email-phishing.fabric-target.json",
        ]
    ),
    workflow=WorkflowConfig(
        target_id="com.acme.email-phishing",
        settings={"llm_name": "default"},
    ),
)
```

Pass the directory that owns these relative paths as `base_dir` when planning
or starting the runtime.

**Success Check**: A local record listed in `discovery.local_paths` resolves
relative to `base_dir` without being installed.

### 3. Select a Harness Adapter Directly

A harness or dedicated custom-agent adapter is selected directly by
`harness.adapter_id`:

```python
FabricConfig(
    harness=HarnessConfig(
        adapter_id="com.acme.fabric.example",
    ),
)
```

The selected Adapter Descriptor supplies the runtime binding, supported
configuration, schemas, capabilities, requirements, and telemetry claims.

**Success Check**: `harness.adapter_id` resolves to exactly one Adapter
Descriptor without a separate target record.

### 4. Select a Registered Target

A shared framework target is selected by `workflow.target_id`:

```python
FabricConfig(
    workflow=WorkflowConfig(
        target_id="nvidia.examples.nat.email-phishing-analyzer",
        settings={"llm_name": "default"},
    ),
    models={"default": ModelConfig(...)},
)
```

The selected Adapter Target Descriptor supplies `adapter_id`, the
adapter-scoped workflow entry point, and the workflow settings schema. The
consumer does not repeat the entry point or need to know the shared adapter ID.

`harness` can also be present when adapter-wide settings are required. In that
case `harness.adapter_id` must match the adapter selected by the target.

**Success Check**: `workflow.target_id` resolves to one target that names its
adapter, without the consumer repeating the entry point.

## Summary

In this tutorial, you have:

- Published the `*.fabric-adapter.json` and any `*.fabric-target.json` package
  records.
- Resolved local records through `discovery.local_paths` during development.
- Selected a harness adapter directly by `harness.adapter_id`.
- Selected a registered target by `workflow.target_id`.

## Next Steps

With installed and explicit discovery both working, continue through the
adapter authoring stages:

<CardGroup cols={2}>

<Card title="Verify the adapter" href="conformance.md">

Continue to Stage 6 and verify every descriptor claim.
</Card>

<Card title="Adapter Target Descriptor schema" href="https://github.com/NVIDIA/NeMo-Fabric/blob/main/schemas/adapter-contract/adapter-target-descriptor.schema.json">

Review the canonical schema for the complete target record.
</Card>

</CardGroup>
