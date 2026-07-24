---
title: "Types"
slug: "/reference/api/python-library-reference/types"
description: "Typed config, request, plan, result, artifact, telemetry, and runtime contracts."
---
{/* SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0 */}

# <kbd>module</kbd> `nemo_fabric.types`
Public data contracts for the NeMo Fabric Python SDK.



---


## <kbd>class</kbd> `AdapterInfo`
Resolved adapter identity attached to a run plan.



**Attributes:**

 - <b>`adapter_id`</b>:  Stable identifier of the Fabric adapter implementation.
 - <b>`harness`</b>:  Stable machine-readable harness identifier.
 - <b>`adapter_kind`</b>:  Execution mechanism used by the adapter.
 - <b>`metadata`</b>:  Adapter-specific, JSON-compatible metadata.



### Fields

The mapping exposes the following typed fields:

| Field | Type |
| --- | --- |
| `adapter_id` | `str` |
| `harness` | `str` |
| `adapter_kind` | `str` |
| `metadata` | `Mapping[str, Any]` |

### <kbd>method</kbd> `__init__`

```text
__init__(mapping: Mapping[str, Any]) -> None
```






---

### <kbd>property</kbd> extra_fields

Return an immutable view of preserved extension fields.



---


### <kbd>classmethod</kbd> `from_mapping`

```text
from_mapping(mapping: Mapping[str, Any]) -> Self
```

Validate and copy a mapping into the requested typed model.

---


### <kbd>method</kbd> `to_dict`

```text
to_dict() -> dict[str, Any]
```

Return the same detached representation as ``to_mapping()``.

---


### <kbd>method</kbd> `to_mapping`

```text
to_mapping() -> dict[str, Any]
```

Return a detached, JSON-compatible mapping for serialization.


---


## <kbd>class</kbd> `RuntimeCapabilities`
Operations declared by the resolved runtime and adapter.

Capabilities describe what the selected runtime can support; callers should still expect a capability-specific error when a transport is modeled but not implemented.



**Attributes:**

 - <b>`service`</b>:  Whether long-lived service handles are supported.
 - <b>`streaming`</b>:  Whether event streaming is supported.
 - <b>`updates`</b>:  Whether runtime configuration updates are supported.
 - <b>`cancellation`</b>:  Whether in-flight cancellation is supported.
 - <b>`metadata`</b>:  Additional capability details.



### Fields

The mapping exposes the following typed fields:

| Field | Type |
| --- | --- |
| `service` | `bool` |
| `streaming` | `bool` |
| `updates` | `bool` |
| `cancellation` | `bool` |
| `metadata` | `Mapping[str, Any]` |

### <kbd>method</kbd> `__init__`

```text
__init__(mapping: Mapping[str, Any]) -> None
```






---

### <kbd>property</kbd> extra_fields

Return an immutable view of preserved extension fields.



---


### <kbd>classmethod</kbd> `from_mapping`

```text
from_mapping(mapping: Mapping[str, Any]) -> Self
```

Validate and copy a mapping into the requested typed model.

---


### <kbd>method</kbd> `to_dict`

```text
to_dict() -> dict[str, Any]
```

Return the same detached representation as ``to_mapping()``.

---


### <kbd>method</kbd> `to_mapping`

```text
to_mapping() -> dict[str, Any]
```

Return a detached, JSON-compatible mapping for serialization.


---


## <kbd>class</kbd> `RunPlan`
Immutable execution plan produced before a runtime is started.



**Attributes:**

 - <b>`agent_name`</b>:  Resolved agent name.
 - <b>`base_dir`</b>:  Base directory used to resolve relative paths.
 - <b>`config`</b>:  Typed configuration snapshot.
 - <b>`adapter`</b>:  Resolved adapter identity.
 - <b>`capabilities`</b>:  Operations declared by the resolved runtime.



### Fields

The mapping exposes the following typed fields:

| Field | Type |
| --- | --- |
| `agent_name` | `str` |
| `base_dir` | `Path` |
| `config` | `_FabricConfigSnapshot` |
| `adapter` | `AdapterInfo` |
| `capabilities` | `RuntimeCapabilities` |

### <kbd>method</kbd> `__init__`

```text
__init__(mapping: Mapping[str, Any]) -> None
```






---

### <kbd>property</kbd> extra_fields

Return an immutable view of preserved extension fields.



---


### <kbd>classmethod</kbd> `from_mapping`

```text
from_mapping(mapping: Mapping[str, Any]) -> Self
```

Validate and copy a mapping into the requested typed model.

---


### <kbd>method</kbd> `to_dict`

```text
to_dict() -> dict[str, Any]
```

Return the same detached representation as ``to_mapping()``.

---


### <kbd>method</kbd> `to_mapping`

```text
to_mapping() -> dict[str, Any]
```

Return a detached, JSON-compatible mapping for serialization.


---


## <kbd>class</kbd> `DoctorCheck`
One diagnostic check in a ``DoctorReport``.



**Attributes:**

 - <b>`name`</b>:  Stable check identifier.
 - <b>`status`</b>:  Check outcome: ``pass``, ``warn``, or ``fail``.
 - <b>`message`</b>:  Human-readable result.
 - <b>`metadata`</b>:  Structured check details.



### Fields

The mapping exposes the following typed fields:

| Field | Type |
| --- | --- |
| `name` | `str` |
| `status` | `str` |
| `message` | `str` |
| `metadata` | `Mapping[str, Any]` |

### <kbd>method</kbd> `__init__`

```text
__init__(mapping: Mapping[str, Any]) -> None
```






---

### <kbd>property</kbd> extra_fields

Return an immutable view of preserved extension fields.



---


### <kbd>classmethod</kbd> `from_mapping`

```text
from_mapping(mapping: Mapping[str, Any]) -> Self
```

Validate and copy a mapping into the requested typed model.

---


### <kbd>method</kbd> `to_dict`

```text
to_dict() -> dict[str, Any]
```

Return the same detached representation as ``to_mapping()``.

---


### <kbd>method</kbd> `to_mapping`

```text
to_mapping() -> dict[str, Any]
```

Return a detached, JSON-compatible mapping for serialization.


---


## <kbd>class</kbd> `DoctorReport`
Aggregate preflight diagnostics for a resolved run plan.



**Attributes:**

 - <b>`agent_name`</b>:  Resolved agent name.
 - <b>`status`</b>:  Aggregate outcome: ``pass``, ``warn``, or ``fail``.
 - <b>`checks`</b>:  Ordered ``DoctorCheck`` results.



### Fields

The mapping exposes the following typed fields:

| Field | Type |
| --- | --- |
| `agent_name` | `str` |
| `status` | `str` |
| `checks` | `Sequence[DoctorCheck]` |

### <kbd>method</kbd> `__init__`

```text
__init__(mapping: Mapping[str, Any]) -> None
```






---

### <kbd>property</kbd> extra_fields

Return an immutable view of preserved extension fields.



---


### <kbd>classmethod</kbd> `from_mapping`

```text
from_mapping(mapping: Mapping[str, Any]) -> Self
```

Validate and copy a mapping into the requested typed model.

---


### <kbd>method</kbd> `to_dict`

```text
to_dict() -> dict[str, Any]
```

Return the same detached representation as ``to_mapping()``.

---


### <kbd>method</kbd> `to_mapping`

```text
to_mapping() -> dict[str, Any]
```

Return a detached, JSON-compatible mapping for serialization.


---


## <kbd>class</kbd> `ErrorInfo`
Structured failure returned inside a normalized ``RunResult``.



**Attributes:**

 - <b>`stage`</b>:  Lifecycle stage that failed.
 - <b>`code`</b>:  Stable machine-readable error code.
 - <b>`message`</b>:  Human-readable failure description.
 - <b>`retryable`</b>:  Whether retrying may succeed without changing the request.
 - <b>`metadata`</b>:  Adapter- or runtime-specific details.



### Fields

The mapping exposes the following typed fields:

| Field | Type |
| --- | --- |
| `stage` | `str` |
| `code` | `str` |
| `message` | `str` |
| `retryable` | `bool` |
| `metadata` | `Mapping[str, Any]` |

### <kbd>method</kbd> `__init__`

```text
__init__(mapping: Mapping[str, Any]) -> None
```






---

### <kbd>property</kbd> extra_fields

Return an immutable view of preserved extension fields.



---


### <kbd>classmethod</kbd> `from_mapping`

```text
from_mapping(mapping: Mapping[str, Any]) -> Self
```

Validate and copy a mapping into the requested typed model.

---


### <kbd>method</kbd> `to_dict`

```text
to_dict() -> dict[str, Any]
```

Return the same detached representation as ``to_mapping()``.

---


### <kbd>method</kbd> `to_mapping`

```text
to_mapping() -> dict[str, Any]
```

Return a detached, JSON-compatible mapping for serialization.


---


## <kbd>class</kbd> `ArtifactRef`
Reference to one artifact produced by a run.



**Attributes:**

 - <b>`name`</b>:  Stable artifact name.
 - <b>`kind`</b>:  Artifact category.
 - <b>`path`</b>:  Artifact path under the manifest root or workspace.
 - <b>`media_type`</b>:  Optional MIME type.
 - <b>`metadata`</b>:  Artifact-specific details.



### Fields

The mapping exposes the following typed fields:

| Field | Type |
| --- | --- |
| `name` | `str` |
| `kind` | `str` |
| `path` | `Path` |
| `media_type` | `str \| None` |
| `metadata` | `Mapping[str, Any]` |

### <kbd>method</kbd> `__init__`

```text
__init__(mapping: Mapping[str, Any]) -> None
```






---

### <kbd>property</kbd> extra_fields

Return an immutable view of preserved extension fields.



---


### <kbd>classmethod</kbd> `from_mapping`

```text
from_mapping(mapping: Mapping[str, Any]) -> Self
```

Validate and copy a mapping into the requested typed model.

---


### <kbd>method</kbd> `to_dict`

```text
to_dict() -> dict[str, Any]
```

Return the same detached representation as ``to_mapping()``.

---


### <kbd>method</kbd> `to_mapping`

```text
to_mapping() -> dict[str, Any]
```

Return a detached, JSON-compatible mapping for serialization.


---


## <kbd>class</kbd> `ArtifactManifest`
Normalized collection of artifacts produced by a run.



**Attributes:**

 - <b>`root`</b>:  Optional common artifact root.
 - <b>`artifacts`</b>:  Ordered ``ArtifactRef`` entries.



### Fields

The mapping exposes the following typed fields:

| Field | Type |
| --- | --- |
| `root` | `Path \| None` |
| `artifacts` | `Sequence[ArtifactRef]` |

### <kbd>method</kbd> `__init__`

```text
__init__(mapping: Mapping[str, Any]) -> None
```






---

### <kbd>property</kbd> extra_fields

Return an immutable view of preserved extension fields.



---


### <kbd>classmethod</kbd> `from_mapping`

```text
from_mapping(mapping: Mapping[str, Any]) -> Self
```

Validate and copy a mapping into the requested typed model.

---


### <kbd>method</kbd> `to_dict`

```text
to_dict() -> dict[str, Any]
```

Return the same detached representation as ``to_mapping()``.

---


### <kbd>method</kbd> `to_mapping`

```text
to_mapping() -> dict[str, Any]
```

Return a detached, JSON-compatible mapping for serialization.


---


## <kbd>class</kbd> `TelemetryRef`
Reference to external or persisted telemetry for a run.



**Attributes:**

 - <b>`provider`</b>:  Telemetry provider, such as Relay.
 - <b>`kind`</b>:  Reference kind, such as ``trace``.
 - <b>`uri`</b>:  Optional location of persisted telemetry.
 - <b>`trace_id`</b>:  Optional provider trace identifier.
 - <b>`metadata`</b>:  Provider-specific details.



### Fields

The mapping exposes the following typed fields:

| Field | Type |
| --- | --- |
| `provider` | `str` |
| `kind` | `str` |
| `uri` | `str \| None` |
| `trace_id` | `str \| None` |
| `metadata` | `Mapping[str, Any]` |

### <kbd>method</kbd> `__init__`

```text
__init__(mapping: Mapping[str, Any]) -> None
```






---

### <kbd>property</kbd> extra_fields

Return an immutable view of preserved extension fields.



---


### <kbd>classmethod</kbd> `from_mapping`

```text
from_mapping(mapping: Mapping[str, Any]) -> Self
```

Validate and copy a mapping into the requested typed model.

---


### <kbd>method</kbd> `to_dict`

```text
to_dict() -> dict[str, Any]
```

Return the same detached representation as ``to_mapping()``.

---


### <kbd>method</kbd> `to_mapping`

```text
to_mapping() -> dict[str, Any]
```

Return a detached, JSON-compatible mapping for serialization.


---


## <kbd>class</kbd> `FabricEvent`
One normalized lifecycle or invocation event.



**Attributes:**

 - <b>`event_id`</b>:  Stable event identifier.
 - <b>`timestamp_millis`</b>:  Event time as Unix epoch milliseconds.
 - <b>`kind`</b>:  Machine-readable event kind.
 - <b>`message`</b>:  Human-readable event description.
 - <b>`metadata`</b>:  Event-specific structured details.



### Fields

The mapping exposes the following typed fields:

| Field | Type |
| --- | --- |
| `event_id` | `str` |
| `timestamp_millis` | `int` |
| `kind` | `str` |
| `message` | `str` |
| `metadata` | `Mapping[str, Any]` |

### <kbd>method</kbd> `__init__`

```text
__init__(mapping: Mapping[str, Any]) -> None
```






---

### <kbd>property</kbd> extra_fields

Return an immutable view of preserved extension fields.



---


### <kbd>classmethod</kbd> `from_mapping`

```text
from_mapping(mapping: Mapping[str, Any]) -> Self
```

Validate and copy a mapping into the requested typed model.

---


### <kbd>method</kbd> `to_dict`

```text
to_dict() -> dict[str, Any]
```

Return the same detached representation as ``to_mapping()``.

---


### <kbd>method</kbd> `to_mapping`

```text
to_mapping() -> dict[str, Any]
```

Return a detached, JSON-compatible mapping for serialization.


---


## <kbd>class</kbd> `RuntimeHandle`
Opaque identity and binding for one started runtime.

Applications should treat ``runtime_binding`` as opaque. Fabric validates the handle against the run plan before invocation or shutdown.



**Attributes:**

 - <b>`runtime_id`</b>:  Unique identifier for this runtime lifecycle.
 - <b>`runtime_binding`</b>:  Opaque integrity-bound runtime binding.
 - <b>`agent_name`</b>:  Resolved agent name.
 - <b>`harness`</b>:  Stable harness identifier.
 - <b>`adapter_kind`</b>:  Adapter execution mechanism.
 - <b>`adapter_id`</b>:  Optional Fabric adapter identifier.
 - <b>`environment`</b>:  Prepared environment snapshot.



### Fields

The mapping exposes the following typed fields:

| Field | Type |
| --- | --- |
| `runtime_id` | `str` |
| `runtime_binding` | `str` |
| `agent_name` | `str` |
| `harness` | `str` |
| `adapter_kind` | `str` |
| `adapter_id` | `str \| None` |
| `environment` | `Mapping[str, Any]` |

### <kbd>method</kbd> `__init__`

```text
__init__(mapping: Mapping[str, Any]) -> None
```






---

### <kbd>property</kbd> extra_fields

Return an immutable view of preserved extension fields.



---


### <kbd>classmethod</kbd> `from_mapping`

```text
from_mapping(mapping: Mapping[str, Any]) -> Self
```

Validate and copy a mapping into the requested typed model.

---


### <kbd>method</kbd> `to_dict`

```text
to_dict() -> dict[str, Any]
```

Return the same detached representation as ``to_mapping()``.

---


### <kbd>method</kbd> `to_mapping`

```text
to_mapping() -> dict[str, Any]
```

Return a detached, JSON-compatible mapping for serialization.


---


## <kbd>class</kbd> `RunOutput`
Normalized adapter output.

``response`` is a known adapter response field whose value follows the core Fabric JSON contract. Other keys are adapter-specific extensions.



### Fields

The mapping exposes the following typed fields:

| Field | Type |
| --- | --- |
| `response` | `JSONValue \| None` |

### <kbd>method</kbd> `__init__`

```text
__init__(mapping: Mapping[str, Any]) -> None
```






---

### <kbd>property</kbd> extra_fields

Return an immutable view of preserved extension fields.

---

### <kbd>property</kbd> response

Return the raw ``response`` JSON value, or ``None`` when absent.



---


### <kbd>classmethod</kbd> `from_mapping`

```text
from_mapping(mapping: Mapping[str, Any]) -> Self
```

Validate and copy a mapping into the requested typed model.

---


### <kbd>method</kbd> `to_dict`

```text
to_dict() -> dict[str, Any]
```

Return the same detached representation as ``to_mapping()``.

---


### <kbd>method</kbd> `to_mapping`

```text
to_mapping() -> dict[str, Any]
```

Return a detached, JSON-compatible mapping for serialization.


---


## <kbd>class</kbd> `RunResult`
Normalized terminal result from one Fabric invocation.

The model is both attribute-accessible and mapping-compatible. A harness failure can be represented by ``status`` and ``error`` without raising when the adapter successfully returns a normalized result.



**Attributes:**

 - <b>`agent_name`</b>:  Resolved agent name.
 - <b>`harness`</b>:  Stable harness identifier.
 - <b>`adapter_kind`</b>:  Adapter execution mechanism.
 - <b>`adapter_id`</b>:  Fabric adapter identifier.
 - <b>`runtime_id`</b>:  Runtime lifecycle identifier.
 - <b>`invocation_id`</b>:  Identifier for this invocation.
 - <b>`request_id`</b>:  Correlated request identifier.
 - <b>`status`</b>:  Terminal invocation status.
 - <b>`output`</b>:  Object-shaped adapter output as ``RunOutput``; non-object values  are preserved as-is.
 - <b>`error`</b>:  Structured failure, or ``None`` on success.
 - <b>`artifacts`</b>:  Normalized artifact manifest.
 - <b>`telemetry`</b>:  Ordered telemetry references.
 - <b>`events`</b>:  Ordered lifecycle and invocation events.
 - <b>`metadata`</b>:  Result-specific structured details.



### Fields

The mapping exposes the following typed fields:

| Field | Type |
| --- | --- |
| `agent_name` | `str` |
| `harness` | `str` |
| `adapter_kind` | `str` |
| `adapter_id` | `str` |
| `runtime_id` | `str` |
| `invocation_id` | `str` |
| `request_id` | `str` |
| `status` | `str` |
| `output` | `RunOutput \| JSONValue` |
| `error` | `ErrorInfo \| None` |
| `artifacts` | `ArtifactManifest` |
| `telemetry` | `Sequence[TelemetryRef]` |
| `events` | `Sequence[FabricEvent]` |
| `metadata` | `Mapping[str, Any]` |

### <kbd>method</kbd> `__init__`

```text
__init__(mapping: Mapping[str, Any]) -> None
```






---

### <kbd>property</kbd> extra_fields

Return an immutable view of preserved extension fields.



---


### <kbd>classmethod</kbd> `from_mapping`

```text
from_mapping(mapping: Mapping[str, Any]) -> Self
```

Validate and copy a mapping into the requested typed model.

---


### <kbd>method</kbd> `to_dict`

```text
to_dict() -> dict[str, Any]
```

Return the same detached representation as ``to_mapping()``.

---


### <kbd>method</kbd> `to_mapping`

```text
to_mapping() -> dict[str, Any]
```

Return a detached, JSON-compatible mapping for serialization.




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
