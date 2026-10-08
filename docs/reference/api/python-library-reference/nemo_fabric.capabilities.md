---
title: "Adapter Capability Inspection"
slug: "/reference/api/python-library-reference/capabilities"
description: "Inspect standalone adapter metadata and verify host/task descriptor correspondence."
---
<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0 -->

# <kbd>module</kbd> `nemo_fabric.capabilities`

Metadata-only adapter inspection for host-side admission.


---


## <kbd>function</kbd> `inspect_adapter`

```python
inspect_adapter(
    config: 'FabricConfig',
    descriptor: 'Mapping[str, Any] | None'
) → AdapterCapabilityProfile
```

Inspect a standalone runtime using descriptor metadata without loading its SDK.

Uses the existing native planner and validation contract. A temporary metadata copy is used only for inspection; it is never an execution origin. Task-local discovery paths are not accessed on the host. Callers supply official catalog metadata or an external adapter's canonical descriptor.

Missing metadata permits basic execution with false optional-feature claims; requested skills, MCP, or telemetry fail rather than being silently dropped. Configured registered workflows require their target metadata and are excluded.


---


## <kbd>class</kbd> `AdapterCapabilityProfile`

Declared support for a standalone runtime, not observed execution provenance.

A missing descriptor produces conservative false claims and no fingerprint. The fingerprint covers the core-normalized descriptor, not discovery paths. Attached services and registered workflow targets need route-specific admission and are not qualified by this standalone profile.


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
