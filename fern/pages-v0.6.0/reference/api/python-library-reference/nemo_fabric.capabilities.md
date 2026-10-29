---
title: "Adapter Capability Inspection"
slug: "/reference/api/python-library-reference/capabilities"
description: "Inspect standalone adapter metadata and verify host/task descriptor correspondence."
---
{/* SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0 */}

# <kbd>module</kbd> `nemo_fabric.capabilities`

Metadata-only adapter inspection for host-side admission.


---


## <kbd>function</kbd> `inspect_adapter`

```python
inspect_adapter(
    config: 'FabricConfig',
    descriptor: 'Mapping[str, Any] | None',
    target_descriptor: 'Mapping[str, Any] | None' = None
) → AdapterCapabilityProfile
```

Inspect supplied adapter and optional workflow metadata without loading its SDK.

Uses the existing native planner and validation contract. A temporary metadata copy is used only for inspection; it is never an execution origin. Task-local discovery paths are not accessed on the host. Callers supply official catalog metadata or an external adapter's canonical descriptor.

Missing metadata permits basic execution with false optional-feature claims; requested skills, MCP, or telemetry fail rather than being silently dropped. Configured registered workflows require their canonical target metadata as well as adapter metadata. Their fingerprint covers both descriptors.


---


## <kbd>class</kbd> `AdapterCapabilityProfile`

Declared runtime support, not observed execution provenance.

A missing descriptor produces conservative false claims and no fingerprint. The fingerprint covers the core-normalized descriptor, not discovery paths. Registered workflows require both adapter and target metadata; their fingerprint covers both normalized descriptors. Attached services require separate route-specific admission.


### <kbd>method</kbd> `__init__`

```python
def __init__(
    adapter_id: str,
    descriptor_sha256: str | None = None,
    skills: bool = False,
    mcp: bool = False,
    atif: bool = False,
) -> None
```











---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
