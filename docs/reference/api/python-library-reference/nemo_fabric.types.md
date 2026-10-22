---
title: "Types"
slug: "/reference/api/python-library-reference/types"
description: "Typed config, request, plan, result, artifact, telemetry, and runtime contracts."
---
<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0 -->

# <kbd>module</kbd> `nemo_fabric.types`

Public data contracts for the NeMo Fabric Python SDK.



---

## <kbd>class</kbd> `AdapterInfo`

Resolved adapter identity attached to a run plan.



**Attributes:**

 - <b>`adapter_id`</b>:  Stable identifier of the NeMo Fabric adapter implementation.
 - <b>`adapter_kind`</b>:  Execution mechanism used by the adapter.
 - <b>`metadata`</b>:  Adapter-specific, JSON-compatible metadata.


### Fields

The mapping exposes the following typed fields:

| Field | Type |
| --- | --- |
| `adapter_id` | `str` |
| `adapter_kind` | `str` |
| `metadata` | `Mapping[str, Any]` |

### <kbd>method</kbd> `__init__`

```python
def __init__(mapping: Mapping[str, Any]) -> None
```






---

### <kbd>property</kbd> extra_fields

Return an immutable view of preserved extension fields.



---

### <kbd>classmethod</kbd> `from_mapping`

```python
def from_mapping(mapping: Mapping[str, Any]) -> Self
```

Validate and copy a mapping into the requested typed model.

---

### <kbd>method</kbd> `to_dict`

```python
def to_dict() -> dict[str, Any]
```

Return the same detached representation as ``to_mapping()``.

---

### <kbd>method</kbd> `to_mapping`

```python
def to_mapping() -> dict[str, Any]
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

```python
def __init__(mapping: Mapping[str, Any]) -> None
```






---

### <kbd>property</kbd> extra_fields

Return an immutable view of preserved extension fields.



---

### <kbd>classmethod</kbd> `from_mapping`

```python
def from_mapping(mapping: Mapping[str, Any]) -> Self
```

Validate and copy a mapping into the requested typed model.

---

### <kbd>method</kbd> `to_dict`

```python
def to_dict() -> dict[str, Any]
```

Return the same detached representation as ``to_mapping()``.

---

### <kbd>method</kbd> `to_mapping`

```python
def to_mapping() -> dict[str, Any]
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

```python
def __init__(mapping: Mapping[str, Any]) -> None
```






---

### <kbd>property</kbd> extra_fields

Return an immutable view of preserved extension fields.



---

### <kbd>classmethod</kbd> `from_mapping`

```python
def from_mapping(mapping: Mapping[str, Any]) -> Self
```

Validate and copy a mapping into the requested typed model.

---

### <kbd>method</kbd> `to_dict`

```python
def to_dict() -> dict[str, Any]
```

Return the same detached representation as ``to_mapping()``.

---

### <kbd>method</kbd> `to_mapping`

```python
def to_mapping() -> dict[str, Any]
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

```python
def __init__(mapping: Mapping[str, Any]) -> None
```






---

### <kbd>property</kbd> extra_fields

Return an immutable view of preserved extension fields.



---

### <kbd>classmethod</kbd> `from_mapping`

```python
def from_mapping(mapping: Mapping[str, Any]) -> Self
```

Validate and copy a mapping into the requested typed model.

---

### <kbd>method</kbd> `to_dict`

```python
def to_dict() -> dict[str, Any]
```

Return the same detached representation as ``to_mapping()``.

---

### <kbd>method</kbd> `to_mapping`

```python
def to_mapping() -> dict[str, Any]
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

```python
def __init__(mapping: Mapping[str, Any]) -> None
```






---

### <kbd>property</kbd> extra_fields

Return an immutable view of preserved extension fields.



---

### <kbd>classmethod</kbd> `from_mapping`

```python
def from_mapping(mapping: Mapping[str, Any]) -> Self
```

Validate and copy a mapping into the requested typed model.

---

### <kbd>method</kbd> `to_dict`

```python
def to_dict() -> dict[str, Any]
```

Return the same detached representation as ``to_mapping()``.

---

### <kbd>method</kbd> `to_mapping`

```python
def to_mapping() -> dict[str, Any]
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

```python
def __init__(mapping: Mapping[str, Any]) -> None
```






---

### <kbd>property</kbd> extra_fields

Return an immutable view of preserved extension fields.



---

### <kbd>classmethod</kbd> `from_mapping`

```python
def from_mapping(mapping: Mapping[str, Any]) -> Self
```

Validate and copy a mapping into the requested typed model.

---

### <kbd>method</kbd> `to_dict`

```python
def to_dict() -> dict[str, Any]
```

Return the same detached representation as ``to_mapping()``.

---

### <kbd>method</kbd> `to_mapping`

```python
def to_mapping() -> dict[str, Any]
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

```python
def __init__(mapping: Mapping[str, Any]) -> None
```






---

### <kbd>property</kbd> extra_fields

Return an immutable view of preserved extension fields.



---

### <kbd>classmethod</kbd> `from_mapping`

```python
def from_mapping(mapping: Mapping[str, Any]) -> Self
```

Validate and copy a mapping into the requested typed model.

---

### <kbd>method</kbd> `to_dict`

```python
def to_dict() -> dict[str, Any]
```

Return the same detached representation as ``to_mapping()``.

---

### <kbd>method</kbd> `to_mapping`

```python
def to_mapping() -> dict[str, Any]
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

```python
def __init__(mapping: Mapping[str, Any]) -> None
```






---

### <kbd>property</kbd> extra_fields

Return an immutable view of preserved extension fields.



---

### <kbd>classmethod</kbd> `from_mapping`

```python
def from_mapping(mapping: Mapping[str, Any]) -> Self
```

Validate and copy a mapping into the requested typed model.

---

### <kbd>method</kbd> `to_dict`

```python
def to_dict() -> dict[str, Any]
```

Return the same detached representation as ``to_mapping()``.

---

### <kbd>method</kbd> `to_mapping`

```python
def to_mapping() -> dict[str, Any]
```

Return a detached, JSON-compatible mapping for serialization.


---

## <kbd>class</kbd> `TelemetryRef`

Reference to external or persisted telemetry for a run.



**Attributes:**

 - <b>`provider`</b>:  Telemetry provider, such as NVIDIA NeMo Relay.
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

```python
def __init__(mapping: Mapping[str, Any]) -> None
```






---

### <kbd>property</kbd> extra_fields

Return an immutable view of preserved extension fields.



---

### <kbd>classmethod</kbd> `from_mapping`

```python
def from_mapping(mapping: Mapping[str, Any]) -> Self
```

Validate and copy a mapping into the requested typed model.

---

### <kbd>method</kbd> `to_dict`

```python
def to_dict() -> dict[str, Any]
```

Return the same detached representation as ``to_mapping()``.

---

### <kbd>method</kbd> `to_mapping`

```python
def to_mapping() -> dict[str, Any]
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

```python
def __init__(mapping: Mapping[str, Any]) -> None
```






---

### <kbd>property</kbd> extra_fields

Return an immutable view of preserved extension fields.



---

### <kbd>classmethod</kbd> `from_mapping`

```python
def from_mapping(mapping: Mapping[str, Any]) -> Self
```

Validate and copy a mapping into the requested typed model.

---

### <kbd>method</kbd> `to_dict`

```python
def to_dict() -> dict[str, Any]
```

Return the same detached representation as ``to_mapping()``.

---

### <kbd>method</kbd> `to_mapping`

```python
def to_mapping() -> dict[str, Any]
```

Return a detached, JSON-compatible mapping for serialization.


---

## <kbd>class</kbd> `EnvironmentReference`

Provider-specific identity for an existing execution environment.

Pass this reference to ``Fabric.attach_environment()``. The provider verifies the resource before returning a durable ``EnvironmentHandle`` and never gains deletion authority over the caller-owned resource.



**Attributes:**

 - <b>`provider`</b>:  Stable environment-provider identifier.
 - <b>`resource`</b>:  Provider-specific resource identity.


### Fields

The mapping exposes the following typed fields:

| Field | Type |
| --- | --- |
| `provider` | `str` |
| `resource` | `Mapping[str, Any]` |

### <kbd>method</kbd> `__init__`

```python
def __init__(mapping: Mapping[str, Any]) -> None
```






---

### <kbd>property</kbd> extra_fields

Return an immutable view of preserved extension fields.



---

### <kbd>classmethod</kbd> `from_mapping`

```python
def from_mapping(mapping: Mapping[str, Any]) -> Self
```

Validate and copy a mapping into the requested typed model.

---

### <kbd>method</kbd> `to_dict`

```python
def to_dict() -> dict[str, Any]
```

Return the same detached representation as ``to_mapping()``.

---

### <kbd>method</kbd> `to_mapping`

```python
def to_mapping() -> dict[str, Any]
```

Return a detached, JSON-compatible mapping for serialization.


---

## <kbd>class</kbd> `EnvironmentHandle`

Durable identity and provider binding for a prepared or attached environment.

Environment handles are independent of runtime sessions. Runtime and reuse limits are provider-specific. Release the environment separately after stopping its runtime and collecting results.



**Attributes:**

 - <b>`environment_id`</b>:  Unique identifier for the prepared environment.
 - <b>`provider`</b>:  Stable environment-provider identifier.
 - <b>`control_location`</b>:  Whether Fabric control runs outside or inside the environment.
 - <b>`workspace`</b>:  Optional workspace visible to the harness runtime.
 - <b>`artifacts`</b>:  Optional artifact root visible to the harness runtime.
 - <b>`env`</b>:  Environment variables visible to the harness and its tools.
 - <b>`ownership`</b>:  Whether the caller or Fabric owns the environment resource.
 - <b>`connection`</b>:  Provider connection metadata.
 - <b>`metadata`</b>:  Provider-specific metadata.


### Fields

The mapping exposes the following typed fields:

| Field | Type |
| --- | --- |
| `environment_id` | `str` |
| `provider` | `str` |
| `control_location` | `str` |
| `workspace` | `str \| Path \| None` |
| `artifacts` | `str \| Path \| None` |
| `env` | `Mapping[str, str]` |
| `ownership` | `str` |
| `connection` | `Mapping[str, Any]` |
| `metadata` | `Mapping[str, Any]` |

### <kbd>method</kbd> `__init__`

```python
def __init__(mapping: Mapping[str, Any]) -> None
```






---

### <kbd>property</kbd> extra_fields

Return an immutable view of preserved extension fields.



---

### <kbd>classmethod</kbd> `from_mapping`

```python
def from_mapping(mapping: Mapping[str, Any]) -> Self
```

Validate and copy a mapping into the requested typed model.

---

### <kbd>method</kbd> `to_dict`

```python
def to_dict() -> dict[str, Any]
```

Return the same detached representation as ``to_mapping()``.

---

### <kbd>method</kbd> `to_mapping`

```python
def to_mapping() -> dict[str, Any]
```

Return a detached, JSON-compatible mapping for serialization.


---

## <kbd>class</kbd> `RuntimeHandle`

Opaque identity and binding for one started runtime.

Applications should treat ``runtime_binding`` as opaque. NeMo Fabric validates the handle against the run plan before invocation or shutdown.



**Attributes:**

 - <b>`runtime_id`</b>:  Unique identifier for this runtime lifecycle.
 - <b>`runtime_binding`</b>:  Opaque integrity-bound runtime binding.
 - <b>`agent_name`</b>:  Resolved agent name.
 - <b>`harness`</b>:  Stable harness identifier.
 - <b>`adapter_kind`</b>:  Adapter execution mechanism.
 - <b>`adapter_id`</b>:  Optional NeMo Fabric adapter identifier.
 - <b>`service_id`</b>:  Optional service used by this runtime.
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
| `service_id` | `str \| None` |
| `environment` | `Mapping[str, Any]` |

### <kbd>method</kbd> `__init__`

```python
def __init__(mapping: Mapping[str, Any]) -> None
```






---

### <kbd>property</kbd> extra_fields

Return an immutable view of preserved extension fields.



---

### <kbd>classmethod</kbd> `from_mapping`

```python
def from_mapping(mapping: Mapping[str, Any]) -> Self
```

Validate and copy a mapping into the requested typed model.

---

### <kbd>method</kbd> `to_dict`

```python
def to_dict() -> dict[str, Any]
```

Return the same detached representation as ``to_mapping()``.

---

### <kbd>method</kbd> `to_mapping`

```python
def to_mapping() -> dict[str, Any]
```

Return a detached, JSON-compatible mapping for serialization.


---

## <kbd>class</kbd> `ServiceReference`

Reference to an already-running caller-owned service.

``adapter_id`` selects the adapter that understands the service type and connection fields.

``connection`` contains endpoints and credential references such as an environment-variable name. It must not contain credential values.


### Fields

The mapping exposes the following typed fields:

| Field | Type |
| --- | --- |
| `adapter_id` | `str` |
| `service_type` | `str` |
| `connection` | `Mapping[str, Any]` |
| `metadata` | `Mapping[str, Any]` |

### <kbd>method</kbd> `__init__`

```python
def __init__(mapping: Mapping[str, Any]) -> None
```






---

### <kbd>property</kbd> extra_fields

Return an immutable view of preserved extension fields.



---

### <kbd>classmethod</kbd> `from_mapping`

```python
def from_mapping(mapping: Mapping[str, Any]) -> Self
```

Validate and copy a mapping into the requested typed model.

---

### <kbd>method</kbd> `to_dict`

```python
def to_dict() -> dict[str, Any]
```

Return the same detached representation as ``to_mapping()``.

---

### <kbd>method</kbd> `to_mapping`

```python
def to_mapping() -> dict[str, Any]
```

Return a detached, JSON-compatible mapping for serialization.


---

## <kbd>class</kbd> `ServiceHandle`

Opaque identity and binding for one prepared or attached service.

``service_id`` identifies the process-local Fabric service lifecycle, not the underlying remote deployment. ``service_binding`` is opaque and must be returned unchanged when the handle is used. ``adapter_id`` identifies the adapter that created or attached to the service, and ``service_type`` is that adapter's stable service kind. ``ownership`` is ``fabric_owned`` or ``caller_owned``. ``connection`` is a sanitized summary that never contains credential values, while ``metadata`` contains adapter-reported version and readiness information.


### Fields

The mapping exposes the following typed fields:

| Field | Type |
| --- | --- |
| `service_id` | `str` |
| `service_binding` | `str` |
| `adapter_id` | `str` |
| `service_type` | `str` |
| `ownership` | `str` |
| `connection` | `Mapping[str, Any]` |
| `metadata` | `Mapping[str, Any]` |

### <kbd>method</kbd> `__init__`

```python
def __init__(mapping: Mapping[str, Any]) -> None
```






---

### <kbd>property</kbd> extra_fields

Return an immutable view of preserved extension fields.



---

### <kbd>classmethod</kbd> `from_mapping`

```python
def from_mapping(mapping: Mapping[str, Any]) -> Self
```

Validate and copy a mapping into the requested typed model.

---

### <kbd>method</kbd> `to_dict`

```python
def to_dict() -> dict[str, Any]
```

Return the same detached representation as ``to_mapping()``.

---

### <kbd>method</kbd> `to_mapping`

```python
def to_mapping() -> dict[str, Any]
```

Return a detached, JSON-compatible mapping for serialization.


---

## <kbd>class</kbd> `RunOutput`

Normalized adapter output.

``response`` is a known adapter response field whose value follows the core NeMo Fabric JSON contract. Other keys are adapter-specific extensions.


### Fields

The mapping exposes the following typed fields:

| Field | Type |
| --- | --- |
| `response` | `JSONValue \| None` |

### <kbd>method</kbd> `__init__`

```python
def __init__(mapping: Mapping[str, Any]) -> None
```






---

### <kbd>property</kbd> extra_fields

Return an immutable view of preserved extension fields.

---

### <kbd>property</kbd> response

Return the raw ``response`` JSON value, or ``None`` when absent.



---

### <kbd>classmethod</kbd> `from_mapping`

```python
def from_mapping(mapping: Mapping[str, Any]) -> Self
```

Validate and copy a mapping into the requested typed model.

---

### <kbd>method</kbd> `to_dict`

```python
def to_dict() -> dict[str, Any]
```

Return the same detached representation as ``to_mapping()``.

---

### <kbd>method</kbd> `to_mapping`

```python
def to_mapping() -> dict[str, Any]
```

Return a detached, JSON-compatible mapping for serialization.


---

## <kbd>class</kbd> `RunUsage`

Normalized invocation-local usage reported by an adapter target.

Missing counters and cost are unknown, not zero. ``cached_input_tokens`` records cache usage; ``input_tokens_include_cache`` declares whether input already includes that count. An absent flag means the semantics are unknown. Descriptor-declared detailed usage is preserved in ``metadata``.


### Fields

The mapping exposes the following typed fields:

| Field | Type |
| --- | --- |
| `input_tokens` | `int \| None` |
| `cached_input_tokens` | `int \| None` |
| `input_tokens_include_cache` | `bool \| None` |
| `output_tokens` | `int \| None` |
| `total_tokens` | `int \| None` |
| `cost_usd` | `float \| None` |
| `metadata` | `Mapping[str, Any]` |

### <kbd>method</kbd> `__init__`

```python
def __init__(mapping: Mapping[str, Any]) -> None
```






---

### <kbd>property</kbd> extra_fields

Return an immutable view of preserved extension fields.



---

### <kbd>classmethod</kbd> `from_mapping`

```python
def from_mapping(mapping: Mapping[str, Any]) -> Self
```

Validate and copy a mapping into the requested typed model.

---

### <kbd>method</kbd> `to_dict`

```python
def to_dict() -> dict[str, Any]
```

Return the same detached representation as ``to_mapping()``.

---

### <kbd>method</kbd> `to_mapping`

```python
def to_mapping() -> dict[str, Any]
```

Return a detached, JSON-compatible mapping for serialization.


---

## <kbd>class</kbd> `RuntimeResourceSnapshot`

Timestamped resource usage of one runtime's local adapter host process tree.

Memory, process count, and CPU time cover the adapter host process and its live descendants, not a shared service process. Values that cannot be read raise an error instead of reading as zero.



**Attributes:**

 - <b>`runtime_id`</b>:  Runtime this snapshot describes.
 - <b>`started_at`</b>:  When NeMo Fabric started the adapter host, as RFC 3339 UTC.
 - <b>`observed_at`</b>:  When NeMo Fabric read these values, as RFC 3339 UTC.
 - <b>`resident_memory_bytes`</b>:  Sum of the live processes' resident set sizes.  Pages shared between processes count once per process.
 - <b>`process_count`</b>:  Live processes in the tree, including the adapter host.
 - <b>`cpu_time_seconds`</b>:  CPU time used since the adapter host started,  including exited descendants. It can drop when a descendant  leaves the tree.


### Fields

The mapping exposes the following typed fields:

| Field | Type |
| --- | --- |
| `runtime_id` | `str` |
| `started_at` | `str` |
| `observed_at` | `str` |
| `resident_memory_bytes` | `int` |
| `process_count` | `int` |
| `cpu_time_seconds` | `float` |

### <kbd>method</kbd> `__init__`

```python
def __init__(mapping: Mapping[str, Any]) -> None
```






---

### <kbd>property</kbd> extra_fields

Return an immutable view of preserved extension fields.



---

### <kbd>classmethod</kbd> `from_mapping`

```python
def from_mapping(mapping: Mapping[str, Any]) -> Self
```

Validate and copy a mapping into the requested typed model.

---

### <kbd>method</kbd> `to_dict`

```python
def to_dict() -> dict[str, Any]
```

Return the same detached representation as ``to_mapping()``.

---

### <kbd>method</kbd> `to_mapping`

```python
def to_mapping() -> dict[str, Any]
```

Return a detached, JSON-compatible mapping for serialization.


---

## <kbd>class</kbd> `ResourceObservation`

Adapter host process-tree reading, or an explicit collection failure.



**Attributes:**

 - <b>`status`</b>:  ``available``, ``unavailable``, ``unsupported``, or ``not_collected``.
 - <b>`snapshot`</b>:  Present only when available; a ``RuntimeResourceSnapshot``.
 - <b>`reason`</b>:  Present only when unavailable or unsupported.


### Fields

The mapping exposes the following typed fields:

| Field | Type |
| --- | --- |
| `status` | `str` |

### <kbd>method</kbd> `__init__`

```python
def __init__(mapping: Mapping[str, Any]) -> None
```






---

### <kbd>property</kbd> extra_fields

Return an immutable view of preserved extension fields.

---

### <kbd>property</kbd> reason

Return the collection failure reason, otherwise None.

---

### <kbd>property</kbd> snapshot

Return the reading when available, otherwise None.



---

### <kbd>classmethod</kbd> `from_mapping`

```python
def from_mapping(mapping: Mapping[str, Any]) -> Self
```

Validate and copy a mapping into the requested typed model.

---

### <kbd>method</kbd> `to_dict`

```python
def to_dict() -> dict[str, Any]
```

Return the same detached representation as ``to_mapping()``.

---

### <kbd>method</kbd> `to_mapping`

```python
def to_mapping() -> dict[str, Any]
```

Return a detached, JSON-compatible mapping for serialization.


---

## <kbd>class</kbd> `Measurement`

Measured numeric value, an unlimited limit, or explicit missing evidence.



**Attributes:**

 - <b>`status`</b>:  available, unlimited, unavailable, unsupported, or not_collected.
 - <b>`value`</b>:  Numeric reading when available, otherwise None.
 - <b>`reason`</b>:  Collection failure/capability reason, otherwise None.


### Fields

The mapping exposes the following typed fields:

| Field | Type |
| --- | --- |
| `status` | `str` |

### <kbd>method</kbd> `__init__`

```python
def __init__(mapping: Mapping[str, Any]) -> None
```






---

### <kbd>property</kbd> extra_fields

Return an immutable view of preserved extension fields.

---

### <kbd>property</kbd> reason

Return the reason for missing evidence, otherwise None.

---

### <kbd>property</kbd> value

Return an available reading, otherwise None.



---

### <kbd>classmethod</kbd> `from_mapping`

```python
def from_mapping(mapping: Mapping[str, Any]) -> Self
```

Validate and copy a mapping into the requested typed model.

---

### <kbd>method</kbd> `to_dict`

```python
def to_dict() -> dict[str, Any]
```

Return the same detached representation as ``to_mapping()``.

---

### <kbd>method</kbd> `to_mapping`

```python
def to_mapping() -> dict[str, Any]
```

Return a detached, JSON-compatible mapping for serialization.


---

## <kbd>class</kbd> `CgroupSnapshot`

Native adapter-host cgroup measurements, separate from process RSS.

Counters cover this cgroup and its descendants, potentially including other workloads. Limits are visible ancestor bounds; hidden ancestors and actual exclusive capacity are unknown. memory_peak_bytes covers cgroup lifetime or its last external reset, never this invocation's peak.


### Fields

The mapping exposes the following typed fields:

| Field | Type |
| --- | --- |
| `source` | `str` |
| `scope` | `str` |
| `target` | `str` |
| `observed_at` | `str` |
| `visible_cpu_limit_cores` | `Measurement` |
| `visible_memory_limit_bytes` | `Measurement` |
| `memory_current_bytes` | `Measurement` |
| `memory_peak_bytes` | `Measurement` |
| `swap_current_bytes` | `Measurement` |
| `cpu_usage_microseconds` | `Measurement` |
| `cpu_throttled_microseconds` | `Measurement` |
| `oom_kills` | `Measurement` |
| `pressure_stall_microseconds` | `Mapping[str, Measurement]` |

### <kbd>method</kbd> `__init__`

```python
def __init__(mapping: Mapping[str, Any]) -> None
```






---

### <kbd>property</kbd> extra_fields

Return an immutable view of preserved extension fields.



---

### <kbd>classmethod</kbd> `from_mapping`

```python
def from_mapping(mapping: Mapping[str, Any]) -> Self
```

Validate and copy a mapping into the requested typed model.

---

### <kbd>method</kbd> `to_dict`

```python
def to_dict() -> dict[str, Any]
```

Return the same detached representation as ``to_mapping()``.

---

### <kbd>method</kbd> `to_mapping`

```python
def to_mapping() -> dict[str, Any]
```

Return a detached, JSON-compatible mapping for serialization.


---

## <kbd>class</kbd> `CgroupObservation`

Native cgroup snapshot or explicit unavailable/unsupported reason.


### Fields

The mapping exposes the following typed fields:

| Field | Type |
| --- | --- |
| `status` | `str` |

### <kbd>method</kbd> `__init__`

```python
def __init__(mapping: Mapping[str, Any]) -> None
```






---

### <kbd>property</kbd> extra_fields

Return an immutable view of preserved extension fields.

---

### <kbd>property</kbd> reason

Return the missing-evidence reason, otherwise None.

---

### <kbd>property</kbd> snapshot

Return the cgroup reading, otherwise None.



---

### <kbd>classmethod</kbd> `from_mapping`

```python
def from_mapping(mapping: Mapping[str, Any]) -> Self
```

Validate and copy a mapping into the requested typed model.

---

### <kbd>method</kbd> `to_dict`

```python
def to_dict() -> dict[str, Any]
```

Return the same detached representation as ``to_mapping()``.

---

### <kbd>method</kbd> `to_mapping`

```python
def to_mapping() -> dict[str, Any]
```

Return a detached, JSON-compatible mapping for serialization.


---

## <kbd>class</kbd> `CgroupStats`

Cgroup endpoint observations and counter deltas over an invocation.


### Fields

The mapping exposes the following typed fields:

| Field | Type |
| --- | --- |
| `before` | `CgroupObservation` |
| `after` | `CgroupObservation` |
| `cpu_usage_microseconds` | `Measurement` |
| `cpu_throttled_microseconds` | `Measurement` |
| `oom_kills` | `Measurement` |

### <kbd>method</kbd> `__init__`

```python
def __init__(mapping: Mapping[str, Any]) -> None
```






---

### <kbd>property</kbd> extra_fields

Return an immutable view of preserved extension fields.



---

### <kbd>classmethod</kbd> `from_mapping`

```python
def from_mapping(mapping: Mapping[str, Any]) -> Self
```

Validate and copy a mapping into the requested typed model.

---

### <kbd>method</kbd> `to_dict`

```python
def to_dict() -> dict[str, Any]
```

Return the same detached representation as ``to_mapping()``.

---

### <kbd>method</kbd> `to_mapping`

```python
def to_mapping() -> dict[str, Any]
```

Return a detached, JSON-compatible mapping for serialization.


---

## <kbd>class</kbd> `GaugeSummary`

Sampled gauge extrema, hold-last-value mean, and covered time.

Sampled peak is not true high-water. Means exclude missing readings and gaps longer than twice the requested cadence. covered_fraction discloses how much of the invocation contributes to the mean.


### Fields

The mapping exposes the following typed fields:

| Field | Type |
| --- | --- |
| `unit` | `str` |
| `final_value` | `int \| None` |
| `min` | `int \| None` |
| `sampled_peak` | `int \| None` |
| `time_weighted_mean` | `float \| None` |
| `valid_samples` | `int` |
| `covered_seconds` | `float` |
| `covered_fraction` | `float` |

### <kbd>method</kbd> `__init__`

```python
def __init__(mapping: Mapping[str, Any]) -> None
```






---

### <kbd>property</kbd> extra_fields

Return an immutable view of preserved extension fields.



---

### <kbd>classmethod</kbd> `from_mapping`

```python
def from_mapping(mapping: Mapping[str, Any]) -> Self
```

Validate and copy a mapping into the requested typed model.

---

### <kbd>method</kbd> `to_dict`

```python
def to_dict() -> dict[str, Any]
```

Return the same detached representation as ``to_mapping()``.

---

### <kbd>method</kbd> `to_mapping`

```python
def to_mapping() -> dict[str, Any]
```

Return a detached, JSON-compatible mapping for serialization.


---

## <kbd>class</kbd> `SampleSummary`

Retained native series, reductions, and collection-quality diagnostics.


### Fields

The mapping exposes the following typed fields:

| Field | Type |
| --- | --- |
| `target_interval_ms` | `int` |
| `max_samples` | `int` |
| `sample_count` | `int` |
| `dropped_samples` | `int` |
| `largest_gap_seconds` | `float` |
| `collection_seconds` | `float` |
| `truncated` | `bool` |
| `diagnostics` | `Sequence[str]` |
| `samples` | `ArtifactRef \| None` |
| `cgroup_target_continuous` | `bool` |
| `resident_memory_bytes` | `GaugeSummary \| None` |
| `process_count` | `GaugeSummary \| None` |
| `cpu_time_seconds_estimate` | `Measurement` |
| `cpu_percent_one_core` | `Measurement` |

### <kbd>method</kbd> `__init__`

```python
def __init__(mapping: Mapping[str, Any]) -> None
```






---

### <kbd>property</kbd> extra_fields

Return an immutable view of preserved extension fields.



---

### <kbd>classmethod</kbd> `from_mapping`

```python
def from_mapping(mapping: Mapping[str, Any]) -> Self
```

Validate and copy a mapping into the requested typed model.

---

### <kbd>method</kbd> `to_dict`

```python
def to_dict() -> dict[str, Any]
```

Return the same detached representation as ``to_mapping()``.

---

### <kbd>method</kbd> `to_mapping`

```python
def to_mapping() -> dict[str, Any]
```

Return a detached, JSON-compatible mapping for serialization.


---

## <kbd>class</kbd> `InvocationSystemStats`

System measurements for one invocation in its executing runner.

Observations cover the adapter host process tree, excluding the orchestrator and remote services. Endpoints are neither peaks nor time averages.



**Attributes:**

 - <b>`elapsed_seconds`</b>:  Monotonic adapter exchange duration, excluding startup,  endpoint collection, artifact preparation, and teardown.
 - <b>`before`</b>:  Observation immediately before the adapter exchange.
 - <b>`after`</b>:  Observation immediately after the exchange, under the same lock.
 - <b>`cgroup`</b>:  Native adapter-host cgroup readings and counter deltas, or None.
 - <b>`sampled`</b>:  Optional retained series, gauge reductions, and collection quality.
 - <b>`cpu_time_seconds_estimate`</b>:  Nonnegative CPU counter difference, or None  if an endpoint is missing, identities differ, or the counter fell.  Process reaping and reparenting can make this underestimate work.


### Fields

The mapping exposes the following typed fields:

| Field | Type |
| --- | --- |
| `cgroup` | `CgroupStats \| None` |
| `sampled` | `SampleSummary \| None` |
| `collected_groups` | `Sequence[str]` |
| `elapsed_seconds` | `float` |
| `before` | `ResourceObservation` |
| `after` | `ResourceObservation` |

### <kbd>method</kbd> `__init__`

```python
def __init__(mapping: Mapping[str, Any]) -> None
```






---

### <kbd>property</kbd> cpu_time_seconds_estimate

Return the CPU counter difference estimate, or None when unknown.

---

### <kbd>property</kbd> extra_fields

Return an immutable view of preserved extension fields.



---

### <kbd>classmethod</kbd> `from_mapping`

```python
def from_mapping(mapping: Mapping[str, Any]) -> Self
```

Validate and copy a mapping into the requested typed model.

---

### <kbd>method</kbd> `to_dict`

```python
def to_dict() -> dict[str, Any]
```

Return the same detached representation as ``to_mapping()``.

---

### <kbd>method</kbd> `to_mapping`

```python
def to_mapping() -> dict[str, Any]
```

Return a detached, JSON-compatible mapping for serialization.


---

## <kbd>class</kbd> `PricingBasis`

Validated comparison basis for persisted catalog estimates.


### Fields

The mapping exposes the following typed fields:

| Field | Type |
| --- | --- |
| `basis` | `str` |
| `currency` | `str` |
| `normalization_version` | `str` |
| `assumption_policy` | `str` |
| `bundled_catalog_sha256` | `str` |
| `user_catalog_sha256` | `str \| None` |

### <kbd>method</kbd> `__init__`

```python
def __init__(mapping: Mapping[str, Any]) -> None
```






---

### <kbd>property</kbd> extra_fields

Return an immutable view of preserved extension fields.



---

### <kbd>classmethod</kbd> `from_mapping`

```python
def from_mapping(mapping: Mapping[str, Any]) -> Self
```

Validate and copy a mapping into the requested typed model.

---

### <kbd>method</kbd> `to_dict`

```python
def to_dict() -> dict[str, Any]
```

Return the same detached representation as ``to_mapping()``.

---

### <kbd>method</kbd> `to_mapping`

```python
def to_mapping() -> dict[str, Any]
```

Return a detached, JSON-compatible mapping for serialization.


---

## <kbd>class</kbd> `AppliedPricing`

Selected price entry and integer nano-USD rates per million tokens.


### Fields

The mapping exposes the following typed fields:

| Field | Type |
| --- | --- |
| `catalog_sha256` | `str` |
| `provider` | `str` |
| `model_id` | `str` |
| `pricing_as_of` | `str` |
| `pricing_source` | `str` |
| `rates` | `dict[str, int \| None]` |

### <kbd>method</kbd> `__init__`

```python
def __init__(mapping: Mapping[str, Any]) -> None
```






---

### <kbd>property</kbd> extra_fields

Return an immutable view of preserved extension fields.



---

### <kbd>classmethod</kbd> `from_mapping`

```python
def from_mapping(mapping: Mapping[str, Any]) -> Self
```

Validate and copy a mapping into the requested typed model.

---

### <kbd>method</kbd> `to_dict`

```python
def to_dict() -> dict[str, Any]
```

Return the same detached representation as ``to_mapping()``.

---

### <kbd>method</kbd> `to_mapping`

```python
def to_mapping() -> dict[str, Any]
```

Return a detached, JSON-compatible mapping for serialization.


---

## <kbd>class</kbd> `TokenCounts`

Independently nullable tokens in one fixed invocation/model scope.


### Fields

The mapping exposes the following typed fields:

| Field | Type |
| --- | --- |
| `input_tokens` | `int \| None` |
| `cached_input_tokens` | `int \| None` |
| `cache_write_input_tokens` | `int \| None` |
| `output_tokens` | `int \| None` |
| `total_tokens` | `int \| None` |

### <kbd>method</kbd> `__init__`

```python
def __init__(mapping: Mapping[str, Any]) -> None
```






---

### <kbd>property</kbd> extra_fields

Return an immutable view of preserved extension fields.



---

### <kbd>classmethod</kbd> `from_mapping`

```python
def from_mapping(mapping: Mapping[str, Any]) -> Self
```

Validate and copy a mapping into the requested typed model.

---

### <kbd>method</kbd> `to_dict`

```python
def to_dict() -> dict[str, Any]
```

Return the same detached representation as ``to_mapping()``.

---

### <kbd>method</kbd> `to_mapping`

```python
def to_mapping() -> dict[str, Any]
```

Return a detached, JSON-compatible mapping for serialization.


---

## <kbd>class</kbd> `UsageGap`

Dimension-specific incompleteness with an optional stable diagnostic.


### Fields

The mapping exposes the following typed fields:

| Field | Type |
| --- | --- |
| `dimensions` | `tuple[str, ...]` |
| `reason` | `str` |
| `code` | `str \| None` |

### <kbd>method</kbd> `__init__`

```python
def __init__(mapping: Mapping[str, Any]) -> None
```






---

### <kbd>property</kbd> extra_fields

Return an immutable view of preserved extension fields.



---

### <kbd>classmethod</kbd> `from_mapping`

```python
def from_mapping(mapping: Mapping[str, Any]) -> Self
```

Validate and copy a mapping into the requested typed model.

---

### <kbd>method</kbd> `to_dict`

```python
def to_dict() -> dict[str, Any]
```

Return the same detached representation as ``to_mapping()``.

---

### <kbd>method</kbd> `to_mapping`

```python
def to_mapping() -> dict[str, Any]
```

Return a detached, JSON-compatible mapping for serialization.


---

## <kbd>class</kbd> `CostAmount`

Complete and observed amounts in integer nano-USD, never an invoice guarantee.


### Fields

The mapping exposes the following typed fields:

| Field | Type |
| --- | --- |
| `amount_nano_usd` | `int \| None` |
| `observed_nano_usd` | `int \| None` |

### <kbd>method</kbd> `__init__`

```python
def __init__(mapping: Mapping[str, Any]) -> None
```






---

### <kbd>property</kbd> extra_fields

Return an immutable view of preserved extension fields.



---

### <kbd>classmethod</kbd> `from_mapping`

```python
def from_mapping(mapping: Mapping[str, Any]) -> Self
```

Validate and copy a mapping into the requested typed model.

---

### <kbd>method</kbd> `to_dict`

```python
def to_dict() -> dict[str, Any]
```

Return the same detached representation as ``to_mapping()``.

---

### <kbd>method</kbd> `to_mapping`

```python
def to_mapping() -> dict[str, Any]
```

Return a detached, JSON-compatible mapping for serialization.


---

## <kbd>class</kbd> `TokenCost`

Provider-reported amounts and separate catalog estimates with their basis.


### Fields

The mapping exposes the following typed fields:

| Field | Type |
| --- | --- |
| `reported` | `CostAmount` |
| `estimated` | `CostAmount` |
| `pricing` | `PricingBasis \| None` |
| `applied_rates` | `tuple[AppliedPricing, ...]` |

### <kbd>method</kbd> `__init__`

```python
def __init__(mapping: Mapping[str, Any]) -> None
```






---

### <kbd>property</kbd> extra_fields

Return an immutable view of preserved extension fields.



---

### <kbd>classmethod</kbd> `from_mapping`

```python
def from_mapping(mapping: Mapping[str, Any]) -> Self
```

Validate and copy a mapping into the requested typed model.

---

### <kbd>method</kbd> `to_dict`

```python
def to_dict() -> dict[str, Any]
```

Return the same detached representation as ``to_mapping()``.

---

### <kbd>method</kbd> `to_mapping`

```python
def to_mapping() -> dict[str, Any]
```

Return a detached, JSON-compatible mapping for serialization.


---

## <kbd>class</kbd> `PerModelUsage`

Normalized counts for a source-established accounting model/provider.

A configured identity requires source proof that the included calls used it. An outer requested alias alone does not establish accounting attribution. This identity does not replace served-model provenance.


### Fields

The mapping exposes the following typed fields:

| Field | Type |
| --- | --- |
| `model` | `str` |
| `provider` | `str \| None` |
| `input_tokens` | `int \| None` |
| `cached_input_tokens` | `int \| None` |
| `cache_write_input_tokens` | `int \| None` |
| `output_tokens` | `int \| None` |
| `total_tokens` | `int \| None` |
| `observed` | `TokenCounts` |
| `gaps` | `tuple[UsageGap, ...]` |
| `cost` | `TokenCost` |

### <kbd>method</kbd> `__init__`

```python
def __init__(mapping: Mapping[str, Any]) -> None
```






---

### <kbd>property</kbd> extra_fields

Return an immutable view of preserved extension fields.



---

### <kbd>classmethod</kbd> `from_mapping`

```python
def from_mapping(mapping: Mapping[str, Any]) -> Self
```

Validate and copy a mapping into the requested typed model.

---

### <kbd>method</kbd> `to_dict`

```python
def to_dict() -> dict[str, Any]
```

Return the same detached representation as ``to_mapping()``.

---

### <kbd>method</kbd> `to_mapping`

```python
def to_mapping() -> dict[str, Any]
```

Return a detached, JSON-compatible mapping for serialization.


---

## <kbd>class</kbd> `ModelUsage`

Complete normalized tokens, observations, gaps and separately typed costs.

Input includes cache and output includes reasoning. All counts belong to this invocation, including descendants. Late work never moves to another invocation. Observations are not guaranteed mathematical lower bounds.


### Fields

The mapping exposes the following typed fields:

| Field | Type |
| --- | --- |
| `input_tokens` | `int \| None` |
| `cached_input_tokens` | `int \| None` |
| `cache_write_input_tokens` | `int \| None` |
| `output_tokens` | `int \| None` |
| `total_tokens` | `int \| None` |
| `observed` | `TokenCounts` |
| `gaps` | `tuple[UsageGap, ...]` |
| `models_exhaustive` | `bool` |
| `models` | `tuple[PerModelUsage, ...]` |
| `cost` | `TokenCost` |

### <kbd>method</kbd> `__init__`

```python
def __init__(mapping: Mapping[str, Any]) -> None
```






---

### <kbd>property</kbd> extra_fields

Return an immutable view of preserved extension fields.



---

### <kbd>classmethod</kbd> `from_mapping`

```python
def from_mapping(mapping: Mapping[str, Any]) -> Self
```

Validate and copy a mapping into the requested typed model.

---

### <kbd>method</kbd> `to_dict`

```python
def to_dict() -> dict[str, Any]
```

Return the same detached representation as ``to_mapping()``.

---

### <kbd>method</kbd> `to_mapping`

```python
def to_mapping() -> dict[str, Any]
```

Return a detached, JSON-compatible mapping for serialization.


---

## <kbd>class</kbd> `RunResult`

Normalized terminal result from one NeMo Fabric invocation.

The model is both attribute-accessible and mapping-compatible. A harness failure can be represented by ``status`` and ``error`` without raising when the adapter successfully returns a normalized result.



**Attributes:**

 - <b>`agent_name`</b>:  Resolved agent name.
 - <b>`harness`</b>:  Stable harness identifier.
 - <b>`adapter_kind`</b>:  Adapter execution mechanism.
 - <b>`adapter_id`</b>:  NeMo Fabric adapter identifier.
 - <b>`runtime_id`</b>:  Runtime lifecycle identifier.
 - <b>`invocation_id`</b>:  Identifier for this invocation.
 - <b>`request_id`</b>:  Correlated request identifier.
 - <b>`status`</b>:  Terminal invocation status.
 - <b>`output`</b>:  Object-shaped adapter output as ``RunOutput``; non-object values  are preserved as-is.
 - <b>`error`</b>:  Structured failure, or ``None`` on success.
 - <b>`usage`</b>:  Normalized invocation usage, or ``None`` when unavailable.
 - <b>`model_usage`</b>:  Complete counts, observations and accounting gaps, or  ``None`` when supported accounting evidence is unavailable.
 - <b>`resource_snapshot`</b>:  Runtime resource snapshot read once after the  adapter returned this result, or ``None`` when the runtime's  resources could not be observed.
 - <b>`system_stats`</b>:  Invocation duration and endpoint observations, or None  for older results or execution backends without collection.
 - <b>`artifacts`</b>:  Normalized artifact manifest.
 - <b>`telemetry`</b>:  Ordered telemetry references.
 - <b>`events`</b>:  Ordered lifecycle and invocation events.
 - <b>`metadata`</b>:  Result-specific structured details.
 - <b>`what_ran`</b>:  What executed for this invocation, schema  ``nemo.fabric.what_ran`` version ``v1alpha1``, as a JSON mapping.  Each fact holds ``value``, or ``state`` and ``reason``, plus  ``source``, ``collected_by`` and ``method``. A minimal object  that the byte budget left holds only ``schema``, ``version`` and  ``what_ran_truncated``, and reads as present. ``None`` when the  producing core predates it, or when the value is not a  ``what_ran`` object of that schema and version with every present  section and fact well formed, as the Rust core reads it.


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
| `usage` | `RunUsage \| None` |
| `model_usage` | `ModelUsage \| None` |
| `system_stats` | `InvocationSystemStats \| None` |
| `resource_snapshot` | `RuntimeResourceSnapshot \| None` |
| `artifacts` | `ArtifactManifest` |
| `telemetry` | `Sequence[TelemetryRef]` |
| `events` | `Sequence[FabricEvent]` |
| `metadata` | `Mapping[str, Any]` |
| `what_ran` | `Mapping[str, Any] \| None` |

### <kbd>method</kbd> `__init__`

```python
def __init__(mapping: Mapping[str, Any]) -> None
```






---

### <kbd>property</kbd> extra_fields

Return an immutable view of preserved extension fields.



---

### <kbd>classmethod</kbd> `from_mapping`

```python
def from_mapping(mapping: Mapping[str, Any]) -> Self
```

Validate and copy a mapping into the requested typed model.

---

### <kbd>method</kbd> `to_dict`

```python
def to_dict() -> dict[str, Any]
```

Return the same detached representation as ``to_mapping()``.

---

### <kbd>method</kbd> `to_mapping`

```python
def to_mapping() -> dict[str, Any]
```

Return a detached, JSON-compatible mapping for serialization.




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
