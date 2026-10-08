<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Fabric Adapter Catalog

`nemo-fabric-adapter-catalog` provides release snapshots of the bundled Python and TypeScript adapter descriptors and their registered targets. It has no runtime dependencies and imports no adapter code or harness SDKs.

## Install And Inspect

Install the matching catalog release:

```bash
pip install nemo-fabric-adapter-catalog
```

Look up a descriptor by its exact ID:

```python
from nemo_fabric_adapter_catalog import get_adapter_descriptor, get_target_descriptor

adapter = get_adapter_descriptor("nvidia.fabric.codex")
print(adapter["config"]["accepts"])
target = get_target_descriptor("nvidia.nooa.arc-solver")
```

Each call returns a fresh dictionary containing the canonical descriptor object. Unknown IDs raise `KeyError`; lookup does not guess an adapter or fall back to another version. Catalog-format errors and unreadable or malformed resources propagate as errors.

## Supported API

Consumers should use `get_adapter_descriptor()` and `get_target_descriptor()`. Returned descriptors follow the [Fabric adapter descriptor contract](https://github.com/NVIDIA/NeMo-Fabric/blob/main/docs/adapter-contract/README.md). The package's resource files, including `catalog.json`, their layout, and their metadata fields are internal implementation details and can change between releases. Do not read these resources directly; the lookup functions keep consumers independent of the bundle's storage format.

## Metadata Is Not Execution

The bundle is a package resource, not an installed descriptor under `share/nemo-fabric`. Installing it does not register executable adapters or change NeMo Fabric runner selection, including when it is co-installed with adapter packages. Task execution continues to discover and validate task-owned descriptors, runners, and harness dependencies through the existing runtime APIs.

A snapshot claim is not proof of the task environment's capabilities. Pin compatible releases and compare the snapshot with the actual task descriptor before relying on it. A provider declaring ATIF support does not establish that it is enabled or that an artifact was produced. Capability normalization, host/task compatibility enforcement, and runtime-observed provenance are separate concerns, not implemented by this package.

The runtime's public [host inspection API](https://github.com/NVIDIA/NeMo-Fabric/blob/main/docs/sdk/python.mdx#inspect-metadata-on-a-separate-host)
can validate supplied metadata and derive standalone admission claims. Pass its
descriptor fingerprint to task-side execution to reject mismatched metadata.
The catalog itself remains dependency-free and does not perform runtime inspection.
