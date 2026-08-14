{/*
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
*/}

# Registration and Discovery

Registration exposes descriptor metadata to NVIDIA NeMo Fabric without
importing adapter code. Runtime loading begins only after discovery, selection,
and planning succeed.

## Package Records

An adapter package publishes one `*.fabric-adapter.json` record. A package that
installs registered targets publishes one `*.fabric-target.json` record per
target. A shared adapter and its targets can be distributed independently.

```text
acme-fabric-package/
├── acme.fabric-adapter.json
├── email-phishing.fabric-target.json
└── src/acme_fabric_adapter/
```

Python wheels install records below the common data root. The directory names
are organizational; descriptor IDs are authoritative.

```toml
[tool.setuptools.data-files]
"share/nemo-fabric/adapters/acme" = ["acme.fabric-adapter.json"]
"share/nemo-fabric/targets/acme" = ["email-phishing.fabric-target.json"]
```

An adapter that consumes typed southbound configuration depends on
`nemo-fabric-adapter-contract`. The optional lifecycle and Relay helpers remain
in `nemo-fabric-adapters-common`.

## Discovery Sources

NeMo Fabric builds one registry from these sources, in deterministic order:

1. Descriptor records bundled with NeMo Fabric.
2. Records installed recursively below
   `<sysconfig data>/share/nemo-fabric`. When `ADAPTER_PYTHON` is set, NeMo Fabric
   queries that Python environment instead of the current one.
3. Files or directories listed explicitly in
   `FabricConfig.discovery.local_paths`. Relative paths resolve from
   `base_dir`.

There is no implicit `<base_dir>/adapters` scan and no source override rule.
Semantically identical records with the same ID are deduplicated and retain all
provenance. Different records with the same ID are ambiguous and fail
planning. A malformed record fails when selection depends on it.

The current registry resolves an adapter or target by exact ID. It does not
enumerate a human-facing catalog or attach presentation metadata to records.

```python
config = FabricConfig(
    metadata=MetadataConfig(name="local-workflow"),
    discovery=DiscoveryConfig(
        local_paths=["./adapter-metadata", "./targets/email.fabric-target.json"]
    ),
    workflow=WorkflowConfig(
        target_id="com.acme.email-phishing",
        settings={"llm_name": "default"},
    ),
)
```

Explicit files must use a recognized suffix. Explicit paths that do not exist
fail planning rather than being ignored.

## Selection

Harness use selects an adapter directly:

```python
FabricConfig(
    metadata=MetadataConfig(name="claude-review"),
    harness=HarnessConfig(adapter_id="nvidia.fabric.claude"),
)
```

Registered-target use selects the target; the target selects its adapter and
supplies the entry point:

```python
FabricConfig(
    metadata=MetadataConfig(name="phishing-review"),
    workflow=WorkflowConfig(
        target_id="nvidia.examples.nat.email-phishing-analyzer",
        settings={"llm_name": "default"},
    ),
)
```

Both selectors can be present when harness-wide settings are also needed. In
that case `harness.adapter_id` must equal the adapter selected by the target.

## Resolution Stages

Planning performs these steps:

1. Discover and validate descriptor records.
2. Resolve `workflow.target_id`, when present.
3. Select the target's adapter, or `harness.adapter_id` for direct harness use.
4. Validate an optional dual selector and the adapter's supported target type.
5. Validate harness settings, workflow settings, normalized configuration, and
   declared compatibility.
6. Project `AgentConfig`. Load the runner only when the runtime starts.

`RunPlan.adapter_descriptor` and `RunPlan.adapter_target_descriptor` retain the
resolved records and all discovery provenance. `doctor(...)` reports both
records for workflow plans.
