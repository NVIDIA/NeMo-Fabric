<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Registration and Discovery

Registration makes descriptor metadata discoverable without importing or
executing adapter code. Adapter implementation loading occurs only after
descriptor resolution and validation, when the runtime starts.

## Package Layout

A publishable Python adapter normally contains:

```text
acme-fabric-adapter/
├── pyproject.toml
├── fabric-adapter.json
└── src/
    └── acme_fabric_adapter/
        ├── __init__.py
        └── adapter.py
```

Install the descriptor into the shared data directory with package metadata:

```toml
[tool.setuptools.data-files]
"share/nemo-fabric/adapters/acme" = ["fabric-adapter.json"]
```

The adapter distribution owns adapter code and adapter-target runtime
dependencies. A bare adapter need not install the NeMo Fabric runtime. Python
adapters that use typed southbound configuration depend on
`nemo-fabric-adapter-contract`; depend on
`nemo-fabric-adapters-common` only when using its optional lifecycle or Relay
helpers.

## Current Discovery Order

Until a provider-backed registry is introduced, the Python SDK scans these
locations. Later locations take precedence:

1. Descriptors bundled in the NeMo Fabric source repository.
2. `<sysconfig data>/share/nemo-fabric/adapters` from `ADAPTER_PYTHON` when set,
   otherwise from the current Python environment.
3. `<base_dir>/adapters` for agent-local and development overrides.

The selected descriptor is atomic: runner metadata and schemas are never
merged across sources. Current discovery uses replacement precedence for a
duplicate adapter ID; treat duplicates as an intentional override and inspect
`RunPlan.adapter_descriptor` to confirm the winning source and path.

NeMo Fabric resolves multi-component relative `ADAPTER_PYTHON` paths from
`base_dir` and bare command names through `PATH`.

## Resolution Stages

1. Scan descriptor metadata without executing adapter code.
2. Select the complete descriptor for `harness.adapter_id`.
3. Validate descriptor shape, contract version, and embedded schemas.
4. Validate the effective config and declared requirements.
5. Load the runner only when the runtime starts.

Registration does not imply installation, trust, or conformance. Local and
preinstalled adapters work without a central registry. An installation policy,
when supported, is selected explicitly through `HarnessConfig.resolution`.

## Verify Discovery

Create a minimal `FabricConfig` that selects the adapter and call `plan` before
starting it:

```python
from nemo_fabric import Fabric

plan = Fabric().plan(config, base_dir=project_root)
print(plan.adapter_descriptor.path)
print(plan.adapter_descriptor.descriptor.adapter_id)
```

Confirm the adapter ID, descriptor location, `config.input`, accepted fields,
schemas, and resolved capabilities. Then run `doctor(...)` in the target
environment to validate declared requirements.

See [Adapter Descriptor](adapter-descriptor.md) for descriptor fields and
[Conformance](conformance.md) for the release checklist.
