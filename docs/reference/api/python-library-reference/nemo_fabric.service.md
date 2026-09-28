---
title: "Service"
slug: "/reference/api/python-library-reference/service"
description: "Prepare, attach, share, and release long-lived adapter services."
---
<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0 -->

# <kbd>module</kbd> `nemo_fabric.service`

Long-lived service lifecycle support for the NeMo Fabric Python SDK.



---


## <kbd>class</kbd> `ServiceStatus`

Lifecycle state of a prepared or attached service.






### Inheritance

Direct bases: `str`, `Enum`.

### Values

The enum defines the following values:

| Name | Value |
| --- | --- |
| `ACTIVE` | `active` |
| `RELEASING` | `releasing` |
| `RELEASED` | `released` |
| `FAILED` | `failed` |

---


## <kbd>class</kbd> `Service`

One prepared or attached long-lived adapter service.

Create services with ``Fabric.prepare_service()`` or ``Fabric.attach_service()``. Use the object as an asynchronous context manager to guarantee release or detach.


---

### <kbd>property</kbd> handle

Return a detached snapshot of the service handle.

---

### <kbd>property</kbd> service_id

Return the unique identifier for this process-local service.

---

### <kbd>property</kbd> status

Return the current service lifecycle state.



---


### <kbd>method</kbd> `release`

```python
async def release() -> None
```

Stop an owned service or detach from a caller-owned service.

Release fails while active runtimes still reference the service. Repeated calls after a successful release are no-ops.




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
