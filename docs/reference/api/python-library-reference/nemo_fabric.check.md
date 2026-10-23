---
title: "Runtime Health"
slug: "/reference/api/python-library-reference/check"
description: "Observe bounded server and agent health without changing runtime state."
---
<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0 -->

# <kbd>module</kbd> `nemo_fabric.check`

Health requests and immutable health reports.



---


## <kbd>class</kbd> `CheckRequest`

Select server or runtime health in a separately supplied environment.

Runtime checks default to ``live``. One positive, finite timeout covers all requested levels. Server checks reject runtime IDs and levels, including explicit nulls.



### Fields

The model defines the following fields:

| Field | Type | Required | Default | Constraints | Description |
| --- | --- | --- | --- | --- | --- |
| `schema_version` | `Literal['fabric.check/v1alpha1']` | No | `'fabric.check/v1alpha1'` | — | — |
| `scope` | `Literal['server', 'runtime']` | Yes | — | — | — |
| `runtime_id` | `str \| None` | No | `None` | — | — |
| `level` | `CheckLevel \| None` | No | `None` | — | — |
| `timeout_seconds` | `float` | No | `3.0` | `Gt(gt=0)` | — |

---

### <kbd>property</kbd> model_extra

Get extra fields set during validation.



**Returns:**
  A dictionary of extra fields, or `None` if `config.extra` is not set to `"allow"`.

---

### <kbd>property</kbd> model_fields_set

Returns the set of fields that have been explicitly set on this model instance.



**Returns:**
  A set of strings representing the fields that have been set,  i.e. that were not filled from defaults.



---


### <kbd>method</kbd> `to_mapping`

```python
def to_mapping() -> dict[str, Any]
```

Return a detached request using the defaulted level and timeout.


---


## <kbd>class</kbd> `CheckReport`

Immutable health check report.

``to_mapping()`` returns detached JSON data.



### Fields

The mapping exposes the following typed fields:

| Field | Type |
| --- | --- |
| `schema_version` | `str` |
| `scope` | `str` |
| `runtime_id` | `str \| None` |
| `requested_level` | `str \| None` |
| `outcome` | `str` |
| `reason_code` | `str \| None` |
| `completed_at` | `str` |
| `elapsed_ms` | `float` |
| `server` | `Mapping[str, Any] \| None` |
| `runtime` | `Mapping[str, Any] \| None` |
| `checks` | `list[Mapping[str, Any]]` |

---

### <kbd>property</kbd> extra_fields

Return an immutable view of preserved extension fields.






---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
