<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Fabric Software Evidence

`nemo-fabric-evidence` provides shared Python software-evidence types, validation, and
JSON-compatible serialization for the NeMo Fabric SDK and adapter contract.
It uses only the Python standard library and has no runtime dependencies.

```bash
pip install nemo-fabric-evidence
```

```python
from nemo_fabric_evidence.evidence import ExecutionEvidence

report = ExecutionEvidence.gap("caller_sdk", "report_not_collected")
assert report.to_mapping()["software_environments"][0]["complete"] is False
```

`ExecutionEvidence`, `SoftwareEnvironment`, `SoftwareEntry`, and
`VersionObservation` describe selected software observations. Infrastructure
reports link to the same result-local environment references. Validation checks
availability, original UTC capture times, references, and report bounds.
`ExecutionEvidence.receive()` turns invalid optional evidence into an explicit
gap so it cannot replace an invocation's outcome.

The SDK and `nemo-fabric-adapter-contract` depend on the same release version of
this package. The SDK retains its public `nemo_fabric` result facade. Existing
`nemo_fabric_adapter_contract.evidence` and `.codec` imports re-export the shared
objects for compatibility. Adapter configuration and lifecycle types remain in
`nemo-fabric-adapter-contract`.

The [Rust-generated JSON Schemas](https://github.com/NVIDIA/NeMo-Fabric/tree/main/schemas)
remain the wire-contract authority. This package contains Python bindings; it
does not collect metadata, load adapters, start processes, or implement a
transport.
