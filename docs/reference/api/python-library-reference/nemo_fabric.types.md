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


## <kbd>class</kbd> `MetadataConfig`
Agent identity and human-readable metadata.



**Attributes:**

 - <b>`name`</b>:  Stable, non-empty agent name.
 - <b>`description`</b>:  Optional human-readable description.
 - <b>`extra_fields`</b>:  Preserved extension fields not recognized by this SDK.


### <kbd>method</kbd> `__init__`

```python
__init__(
    name: 'str',
    description: 'str | None' = None,
    extra_fields: 'Mapping[str, Any] | None' = None
) → None
```






---

#### <kbd>property</kbd> extra_fields

Return preserved schema-extension fields as a deep copy.



---


### <kbd>classmethod</kbd> `from_mapping`

```python
from_mapping(value: 'Mapping[str, Any]') → 'MetadataConfig'
```

Validate a metadata mapping and preserve unknown extension fields.

---


### <kbd>method</kbd> `to_mapping`

```python
to_mapping() → dict[str, Any]
```

Return a detached, JSON-compatible mapping for serialization.


---


## <kbd>class</kbd> `HarnessConfig`
Harness adapter selection and adapter-owned settings.



**Attributes:**

 - <b>`adapter_id`</b>:  Stable identifier of the Fabric adapter to resolve.
 - <b>`resolution`</b>:  Optional adapter resolution strategy.
 - <b>`settings`</b>:  JSON-compatible settings owned by the selected adapter.
 - <b>`extra_fields`</b>:  Preserved extension fields not recognized by this SDK.


### <kbd>method</kbd> `__init__`

```python
__init__(
    adapter_id: 'str',
    resolution: 'str | None' = None,
    settings: 'Mapping[str, Any] | None' = None,
    extra_fields: 'Mapping[str, Any] | None' = None
) → None
```






---

#### <kbd>property</kbd> extra_fields

Return preserved schema-extension fields as a deep copy.



---


### <kbd>classmethod</kbd> `from_mapping`

```python
from_mapping(value: 'Mapping[str, Any]') → 'HarnessConfig'
```

Validate a harness mapping and preserve unknown extension fields.

---


### <kbd>method</kbd> `to_mapping`

```python
to_mapping() → dict[str, Any]
```

Return a detached, JSON-compatible mapping for serialization.


---


## <kbd>class</kbd> `RuntimeConfig`
Runtime input/output contract.



**Attributes:**

 - <b>`input_schema`</b>:  Optional logical input contract identifier.
 - <b>`output_schema`</b>:  Optional logical output contract identifier.
 - <b>`artifacts`</b>:  Optional artifact-root path.
 - <b>`extra_fields`</b>:  Preserved extension fields not recognized by this SDK.


### <kbd>method</kbd> `__init__`

```python
__init__(
    input_schema: 'str | None' = None,
    output_schema: 'str | None' = None,
    artifacts: 'str | Path | None' = None,
    extra_fields: 'Mapping[str, Any] | None' = None
) → None
```






---

#### <kbd>property</kbd> extra_fields

Return preserved schema-extension fields as a deep copy.



---


### <kbd>classmethod</kbd> `from_mapping`

```python
from_mapping(value: 'Mapping[str, Any]') → 'RuntimeConfig'
```

Validate a runtime mapping and apply stable constructor defaults.

---


### <kbd>method</kbd> `to_mapping`

```python
to_mapping() → dict[str, Any]
```

Return a detached, JSON-compatible mapping for serialization.


---


## <kbd>class</kbd> `EnvironmentConfig`
Execution environment configuration.



**Attributes:**

 - <b>`provider`</b>:  Environment provider identifier; defaults to ``local``.
 - <b>`workspace`</b>:  Optional workspace path visible to the harness.
 - <b>`artifacts`</b>:  Optional environment-specific artifact path.
 - <b>`settings`</b>:  JSON-compatible provider settings.
 - <b>`metadata`</b>:  JSON-compatible caller metadata.
 - <b>`extra_fields`</b>:  Preserved extension fields not recognized by this SDK.


### <kbd>method</kbd> `__init__`

```python
__init__(
    provider: 'str' = 'local',
    workspace: 'str | Path | None' = None,
    artifacts: 'str | Path | None' = None,
    settings: 'Mapping[str, Any] | None' = None,
    metadata: 'Mapping[str, Any] | None' = None,
    extra_fields: 'Mapping[str, Any] | None' = None
) → None
```






---

#### <kbd>property</kbd> extra_fields

Return preserved schema-extension fields as a deep copy.



---


### <kbd>classmethod</kbd> `from_mapping`

```python
from_mapping(value: 'Mapping[str, Any]') → 'EnvironmentConfig'
```

Validate an environment mapping and preserve extension fields.

---


### <kbd>method</kbd> `to_mapping`

```python
to_mapping() → dict[str, Any]
```

Return a detached, JSON-compatible mapping for serialization.


---


## <kbd>class</kbd> `SkillConfig`
Skill capability configuration.

The shape matches the ``skills`` section in ``agent.yaml`` while providing small authoring helpers for Python callers.


### <kbd>method</kbd> `__init__`

```python
__init__(
    paths: 'Sequence[str | Path] | None' = None,
    extra_fields: 'Mapping[str, Any] | None' = None
) → None
```






---

#### <kbd>property</kbd> extra_fields

Return preserved schema-extension fields as a deep copy.



---


### <kbd>method</kbd> `add_path`

```python
add_path(path: 'str | Path') → 'SkillConfig'
```

Add a skill path to this config if it is not already present.

---


### <kbd>classmethod</kbd> `from_mapping`

```python
from_mapping(value: 'Mapping[str, Any]') → 'SkillConfig'
```

Validate a skill mapping and preserve extension fields.

---


### <kbd>method</kbd> `remove_path`

```python
remove_path(path: 'str | Path') → 'SkillConfig'
```

Remove a skill path from this config if present.

---


### <kbd>method</kbd> `to_mapping`

```python
to_mapping() → dict[str, Any]
```

Return a detached, JSON-compatible mapping for serialization.


---


## <kbd>class</kbd> `McpConfig`
MCP capability configuration with authoring helpers.


### <kbd>method</kbd> `__init__`

```python
__init__(
    servers: 'Mapping[str, Any] | None' = None,
    extra_fields: 'Mapping[str, Any] | None' = None
) → None
```






---

#### <kbd>property</kbd> extra_fields

Return preserved schema-extension fields as a deep copy.



---


### <kbd>method</kbd> `add_server`

```python
add_server(
    name: 'str',
    transport: 'str',
    url: 'str',
    exposure: 'str' = 'harness_native',
    extra_fields: 'Mapping[str, Any] | None' = None
) → 'McpConfig'
```

Add or replace a named MCP server.

---


### <kbd>classmethod</kbd> `from_mapping`

```python
from_mapping(value: 'Mapping[str, Any]') → 'McpConfig'
```

Validate an MCP mapping and preserve extension fields.

---


### <kbd>method</kbd> `remove_server`

```python
remove_server(name: 'str') → 'McpConfig'
```

Remove a named MCP server if present.

---


### <kbd>method</kbd> `to_mapping`

```python
to_mapping() → dict[str, Any]
```

Return a detached, JSON-compatible mapping for serialization.


---


## <kbd>class</kbd> `TelemetryConfig`
Telemetry configuration with authoring helpers.


### <kbd>method</kbd> `__init__`

```python
__init__(
    enabled: 'bool' = False,
    provider: 'str | None' = None,
    project: 'str | None' = None,
    output_dir: 'str | Path | None' = None,
    config: 'Mapping[str, Any] | None' = None,
    extra_fields: 'Mapping[str, Any] | None' = None
) → None
```






---

#### <kbd>property</kbd> extra_fields

Return preserved schema-extension fields as a deep copy.



---


### <kbd>method</kbd> `disable`

```python
disable() → 'TelemetryConfig'
```

Disable telemetry for subsequently started runtimes.

---


### <kbd>method</kbd> `enable_native`

```python
enable_native() → 'TelemetryConfig'
```

Let the selected harness adapter handle telemetry natively.

---


### <kbd>method</kbd> `enable_relay`

```python
enable_relay(
    project: 'str | None' = None,
    output_dir: 'str | Path | None' = None,
    config: 'Mapping[str, Any] | None' = None
) → 'TelemetryConfig'
```

Enable NeMo Relay telemetry for subsequently started runtimes.

---


### <kbd>classmethod</kbd> `from_mapping`

```python
from_mapping(value: 'Mapping[str, Any]') → 'TelemetryConfig'
```

Validate a telemetry mapping and preserve extension fields.

---


### <kbd>method</kbd> `to_mapping`

```python
to_mapping() → dict[str, Any]
```

Return a detached, JSON-compatible mapping for serialization.


---


## <kbd>class</kbd> `FabricConfig`
Mutable typed representation of a Fabric agent configuration.

The object follows the same schema as ``agent.yaml``. It is mutable while callers compose a job, then copied into immutable resolution and plan snapshots. Unknown fields survive round trips through ``extra_fields``.



**Attributes:**

 - <b>`schema_version`</b>:  Agent schema identifier.
 - <b>`metadata`</b>:  Required ``MetadataConfig`` agent identity.
 - <b>`harness`</b>:  Required ``HarnessConfig`` adapter selection.
 - <b>`runtime`</b>:  Runtime input/output configuration.
 - <b>`environment`</b>:  Optional execution environment configuration.
 - <b>`models`</b>:  Named, JSON-compatible model configurations.
 - <b>`mcp`</b>:  Optional MCP configuration.
 - <b>`skills`</b>:  Optional skill configuration.
 - <b>`telemetry`</b>:  Optional telemetry configuration.
 - <b>`profiles`</b>:  Optional profile-discovery configuration.
 - <b>`tools`</b>:  Optional harness-neutral tool configuration.
 - <b>`extra_fields`</b>:  Preserved extension fields not recognized by this SDK.


### <kbd>method</kbd> `__init__`

```python
__init__(
    metadata: 'MetadataConfig | Mapping[str, Any]',
    harness: 'HarnessConfig | Mapping[str, Any]',
    runtime: 'RuntimeConfig | Mapping[str, Any] | None' = None,
    schema_version: 'str' = 'fabric.agent/v1alpha1',
    environment: 'EnvironmentConfig | Mapping[str, Any] | None' = None,
    models: 'Mapping[str, Any] | None' = None,
    mcp: 'Mapping[str, Any] | None' = None,
    skills: 'Mapping[str, Any] | None' = None,
    telemetry: 'Mapping[str, Any] | None' = None,
    profiles: 'Mapping[str, Any] | None' = None,
    tools: 'Any' = None,
    extra_fields: 'Mapping[str, Any] | None' = None
) → None
```






---

#### <kbd>property</kbd> extra_fields

Return preserved schema-extension fields as a deep copy.

---

#### <kbd>property</kbd> mcp

Mutable MCP capability config, created on first access.

---

#### <kbd>property</kbd> skills

Mutable skill capability config, created on first access.

---

#### <kbd>property</kbd> telemetry

Mutable telemetry config, created on first access.



---


### <kbd>method</kbd> `add_mcp_server`

```python
add_mcp_server(
    name: 'str',
    transport: 'str',
    url: 'str',
    exposure: 'str' = 'harness_native',
    extra_fields: 'Mapping[str, Any] | None' = None
) → 'FabricConfig'
```

Add or replace a named MCP server and return this config.

---


### <kbd>method</kbd> `add_skill_path`

```python
add_skill_path(path: 'str | Path') → 'FabricConfig'
```

Add a skill path and return this config.

---


### <kbd>method</kbd> `enable_relay`

```python
enable_relay(
    project: 'str | None' = None,
    output_dir: 'str | Path | None' = None,
    config: 'Mapping[str, Any] | None' = None
) → 'FabricConfig'
```

Enable NeMo Relay telemetry and return this config.

---


### <kbd>classmethod</kbd> `from_mapping`

```python
from_mapping(value: 'Mapping[str, Any]') → 'FabricConfig'
```

Build a typed agent config from the ``agent.yaml`` mapping shape.

---


### <kbd>method</kbd> `to_mapping`

```python
to_mapping() → dict[str, Any]
```

Return a detached, JSON-compatible mapping for serialization.


---


## <kbd>class</kbd> `AdapterInfo`
Resolved adapter identity attached to a run plan.



**Attributes:**

 - <b>`adapter_id`</b>:  Stable identifier of the Fabric adapter implementation.
 - <b>`harness`</b>:  Stable machine-readable harness identifier.
 - <b>`adapter_kind`</b>:  Execution mechanism used by the adapter.
 - <b>`metadata`</b>:  Adapter-specific, JSON-compatible metadata.


### <kbd>method</kbd> `__init__`

```python
__init__(mapping: 'Mapping[str, Any]') → None
```






---

#### <kbd>property</kbd> extra_fields

Return an immutable view of preserved extension fields.



---


### <kbd>classmethod</kbd> `from_mapping`

```python
from_mapping(mapping: 'Mapping[str, Any]') → 'FabricMapping'
```

Validate and copy a mapping into the requested typed model.

---


### <kbd>method</kbd> `to_dict`

```python
to_dict() → dict[str, Any]
```

Return the same detached representation as ``to_mapping()``.

---


### <kbd>method</kbd> `to_mapping`

```python
to_mapping() → dict[str, Any]
```

Return a detached, JSON-compatible mapping for serialization.


---


## <kbd>class</kbd> `RuntimeCapabilities`
Operations declared by the resolved runtime and adapter.

Capabilities describe what the selected runtime can support; callers should still expect a capability-specific error when a transport is modeled but not implemented.



**Attributes:**

 - <b>`session`</b>:  Whether stateful multi-turn sessions are supported.
 - <b>`service`</b>:  Whether long-lived service handles are supported.
 - <b>`streaming`</b>:  Whether event streaming is supported.
 - <b>`updates`</b>:  Whether runtime configuration updates are supported.
 - <b>`cancellation`</b>:  Whether in-flight cancellation is supported.
 - <b>`concurrent_invocations`</b>:  Whether invocations may overlap safely.
 - <b>`metadata`</b>:  Additional capability details.


### <kbd>method</kbd> `__init__`

```python
__init__(mapping: 'Mapping[str, Any]') → None
```






---

#### <kbd>property</kbd> extra_fields

Return an immutable view of preserved extension fields.



---


### <kbd>classmethod</kbd> `from_mapping`

```python
from_mapping(mapping: 'Mapping[str, Any]') → 'FabricMapping'
```

Validate and copy a mapping into the requested typed model.

---


### <kbd>method</kbd> `to_dict`

```python
to_dict() → dict[str, Any]
```

Return the same detached representation as ``to_mapping()``.

---


### <kbd>method</kbd> `to_mapping`

```python
to_mapping() → dict[str, Any]
```

Return a detached, JSON-compatible mapping for serialization.


---


## <kbd>class</kbd> `EffectiveConfig`
Immutable result of config loading and ordered profile application.



**Attributes:**

 - <b>`agent_name`</b>:  Resolved agent name.
 - <b>`profiles`</b>:  Applied profile names in caller order.
 - <b>`agent_root`</b>:  Root directory of the path-backed agent source.
 - <b>`config_path`</b>:  Source config path, or ``None`` for typed configs.
 - <b>`config_root`</b>:  Base directory used to resolve relative paths.
 - <b>`config`</b>:  Fully resolved typed ``FabricConfig``.


### <kbd>method</kbd> `__init__`

```python
__init__(mapping: 'Mapping[str, Any]') → None
```






---

#### <kbd>property</kbd> extra_fields

Return an immutable view of preserved extension fields.



---


### <kbd>classmethod</kbd> `from_mapping`

```python
from_mapping(mapping: 'Mapping[str, Any]') → 'FabricMapping'
```

Validate and copy a mapping into the requested typed model.

---


### <kbd>method</kbd> `to_dict`

```python
to_dict() → dict[str, Any]
```

Return the same detached representation as ``to_mapping()``.

---


### <kbd>method</kbd> `to_mapping`

```python
to_mapping() → dict[str, Any]
```

Return a detached, JSON-compatible mapping for serialization.


---


## <kbd>class</kbd> `RunPlan`
Immutable execution plan produced before a runtime is started.



**Attributes:**

 - <b>`effective_config`</b>:  Resolved configuration snapshot.
 - <b>`agent_name`</b>:  Resolved agent name.
 - <b>`profiles`</b>:  Applied profile names in caller order.
 - <b>`adapter`</b>:  Resolved adapter identity.
 - <b>`capabilities`</b>:  Operations declared by the resolved runtime.


### <kbd>method</kbd> `__init__`

```python
__init__(mapping: 'Mapping[str, Any]') → None
```






---

#### <kbd>property</kbd> extra_fields

Return an immutable view of preserved extension fields.



---


### <kbd>classmethod</kbd> `from_mapping`

```python
from_mapping(mapping: 'Mapping[str, Any]') → 'FabricMapping'
```

Validate and copy a mapping into the requested typed model.

---


### <kbd>method</kbd> `to_dict`

```python
to_dict() → dict[str, Any]
```

Return the same detached representation as ``to_mapping()``.

---


### <kbd>method</kbd> `to_mapping`

```python
to_mapping() → dict[str, Any]
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


### <kbd>method</kbd> `__init__`

```python
__init__(mapping: 'Mapping[str, Any]') → None
```






---

#### <kbd>property</kbd> extra_fields

Return an immutable view of preserved extension fields.



---


### <kbd>classmethod</kbd> `from_mapping`

```python
from_mapping(mapping: 'Mapping[str, Any]') → 'FabricMapping'
```

Validate and copy a mapping into the requested typed model.

---


### <kbd>method</kbd> `to_dict`

```python
to_dict() → dict[str, Any]
```

Return the same detached representation as ``to_mapping()``.

---


### <kbd>method</kbd> `to_mapping`

```python
to_mapping() → dict[str, Any]
```

Return a detached, JSON-compatible mapping for serialization.


---


## <kbd>class</kbd> `DoctorReport`
Aggregate preflight diagnostics for a resolved run plan.



**Attributes:**

 - <b>`agent_name`</b>:  Resolved agent name.
 - <b>`profiles`</b>:  Applied profile names in caller order.
 - <b>`status`</b>:  Aggregate outcome: ``pass``, ``warn``, or ``fail``.
 - <b>`checks`</b>:  Ordered ``DoctorCheck`` results.


### <kbd>method</kbd> `__init__`

```python
__init__(mapping: 'Mapping[str, Any]') → None
```






---

#### <kbd>property</kbd> extra_fields

Return an immutable view of preserved extension fields.



---


### <kbd>classmethod</kbd> `from_mapping`

```python
from_mapping(mapping: 'Mapping[str, Any]') → 'FabricMapping'
```

Validate and copy a mapping into the requested typed model.

---


### <kbd>method</kbd> `to_dict`

```python
to_dict() → dict[str, Any]
```

Return the same detached representation as ``to_mapping()``.

---


### <kbd>method</kbd> `to_mapping`

```python
to_mapping() → dict[str, Any]
```

Return a detached, JSON-compatible mapping for serialization.


---


## <kbd>class</kbd> `RunRequest`
One normalized invocation request.

``input`` and all mapping fields must be JSON-compatible. Fabric generates a request identifier when callers omit one and preserves unknown mapping fields for forward compatibility.



**Attributes:**

 - <b>`input`</b>:  Harness input; defaults to an empty string.
 - <b>`request_id`</b>:  Caller-provided or generated request identifier.
 - <b>`context`</b>:  Caller-owned metadata propagated with the invocation.
 - <b>`overrides`</b>:  Optional invocation-scoped config overrides.
 - <b>`extra_fields`</b>:  Preserved extension fields not recognized by this SDK.


### <kbd>method</kbd> `__init__`

```python
__init__(
    input: 'Any' = ...,
    request_id: 'str | None' = None,
    context: 'Mapping[str, Any] | None' = None,
    overrides: 'Mapping[str, Any] | None' = None,
    extra_fields: 'Mapping[str, Any] | None' = None
) → None
```






---

#### <kbd>property</kbd> extra_fields

Return an immutable view of preserved extension fields.



---


### <kbd>classmethod</kbd> `from_mapping`

```python
from_mapping(value: 'Mapping[str, Any]') → 'RunRequest'
```





---


### <kbd>method</kbd> `to_dict`

```python
to_dict() → dict[str, Any]
```

Return the same detached representation as ``to_mapping()``.

---


### <kbd>method</kbd> `to_mapping`

```python
to_mapping() → dict[str, Any]
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


### <kbd>method</kbd> `__init__`

```python
__init__(mapping: 'Mapping[str, Any]') → None
```






---

#### <kbd>property</kbd> extra_fields

Return an immutable view of preserved extension fields.



---


### <kbd>classmethod</kbd> `from_mapping`

```python
from_mapping(mapping: 'Mapping[str, Any]') → 'FabricMapping'
```

Validate and copy a mapping into the requested typed model.

---


### <kbd>method</kbd> `to_dict`

```python
to_dict() → dict[str, Any]
```

Return the same detached representation as ``to_mapping()``.

---


### <kbd>method</kbd> `to_mapping`

```python
to_mapping() → dict[str, Any]
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


### <kbd>method</kbd> `__init__`

```python
__init__(mapping: 'Mapping[str, Any]') → None
```






---

#### <kbd>property</kbd> extra_fields

Return an immutable view of preserved extension fields.



---


### <kbd>classmethod</kbd> `from_mapping`

```python
from_mapping(mapping: 'Mapping[str, Any]') → 'FabricMapping'
```

Validate and copy a mapping into the requested typed model.

---


### <kbd>method</kbd> `to_dict`

```python
to_dict() → dict[str, Any]
```

Return the same detached representation as ``to_mapping()``.

---


### <kbd>method</kbd> `to_mapping`

```python
to_mapping() → dict[str, Any]
```

Return a detached, JSON-compatible mapping for serialization.


---


## <kbd>class</kbd> `ArtifactManifest`
Normalized collection of artifacts produced by a run.



**Attributes:**

 - <b>`root`</b>:  Optional common artifact root.
 - <b>`artifacts`</b>:  Ordered ``ArtifactRef`` entries.


### <kbd>method</kbd> `__init__`

```python
__init__(mapping: 'Mapping[str, Any]') → None
```






---

#### <kbd>property</kbd> extra_fields

Return an immutable view of preserved extension fields.



---


### <kbd>classmethod</kbd> `from_mapping`

```python
from_mapping(mapping: 'Mapping[str, Any]') → 'FabricMapping'
```

Validate and copy a mapping into the requested typed model.

---


### <kbd>method</kbd> `to_dict`

```python
to_dict() → dict[str, Any]
```

Return the same detached representation as ``to_mapping()``.

---


### <kbd>method</kbd> `to_mapping`

```python
to_mapping() → dict[str, Any]
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


### <kbd>method</kbd> `__init__`

```python
__init__(mapping: 'Mapping[str, Any]') → None
```






---

#### <kbd>property</kbd> extra_fields

Return an immutable view of preserved extension fields.



---


### <kbd>classmethod</kbd> `from_mapping`

```python
from_mapping(mapping: 'Mapping[str, Any]') → 'FabricMapping'
```

Validate and copy a mapping into the requested typed model.

---


### <kbd>method</kbd> `to_dict`

```python
to_dict() → dict[str, Any]
```

Return the same detached representation as ``to_mapping()``.

---


### <kbd>method</kbd> `to_mapping`

```python
to_mapping() → dict[str, Any]
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


### <kbd>method</kbd> `__init__`

```python
__init__(mapping: 'Mapping[str, Any]') → None
```






---

#### <kbd>property</kbd> extra_fields

Return an immutable view of preserved extension fields.



---


### <kbd>classmethod</kbd> `from_mapping`

```python
from_mapping(mapping: 'Mapping[str, Any]') → 'FabricMapping'
```

Validate and copy a mapping into the requested typed model.

---


### <kbd>method</kbd> `to_dict`

```python
to_dict() → dict[str, Any]
```

Return the same detached representation as ``to_mapping()``.

---


### <kbd>method</kbd> `to_mapping`

```python
to_mapping() → dict[str, Any]
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


### <kbd>method</kbd> `__init__`

```python
__init__(mapping: 'Mapping[str, Any]') → None
```






---

#### <kbd>property</kbd> extra_fields

Return an immutable view of preserved extension fields.



---


### <kbd>classmethod</kbd> `from_mapping`

```python
from_mapping(mapping: 'Mapping[str, Any]') → 'FabricMapping'
```

Validate and copy a mapping into the requested typed model.

---


### <kbd>method</kbd> `to_dict`

```python
to_dict() → dict[str, Any]
```

Return the same detached representation as ``to_mapping()``.

---


### <kbd>method</kbd> `to_mapping`

```python
to_mapping() → dict[str, Any]
```

Return a detached, JSON-compatible mapping for serialization.


---


## <kbd>class</kbd> `RunResult`
Normalized terminal result from one Fabric invocation.

The model is both attribute-accessible and mapping-compatible. A harness failure can be represented by ``status`` and ``error`` without raising when the adapter successfully returns a normalized result.



**Attributes:**

 - <b>`agent_name`</b>:  Resolved agent name.
 - <b>`profiles`</b>:  Applied profile names.
 - <b>`harness`</b>:  Stable harness identifier.
 - <b>`adapter_kind`</b>:  Adapter execution mechanism.
 - <b>`adapter_id`</b>:  Fabric adapter identifier.
 - <b>`runtime_id`</b>:  Runtime lifecycle identifier.
 - <b>`invocation_id`</b>:  Identifier for this invocation.
 - <b>`request_id`</b>:  Correlated request identifier.
 - <b>`status`</b>:  Terminal invocation status.
 - <b>`output`</b>:  JSON-compatible harness output.
 - <b>`error`</b>:  Structured failure, or ``None`` on success.
 - <b>`artifacts`</b>:  Normalized artifact manifest.
 - <b>`telemetry`</b>:  Ordered telemetry references.
 - <b>`events`</b>:  Ordered lifecycle and invocation events.
 - <b>`metadata`</b>:  Result-specific structured details.


### <kbd>method</kbd> `__init__`

```python
__init__(mapping: 'Mapping[str, Any]') → None
```






---

#### <kbd>property</kbd> extra_fields

Return an immutable view of preserved extension fields.



---


### <kbd>classmethod</kbd> `from_mapping`

```python
from_mapping(mapping: 'Mapping[str, Any]') → 'FabricMapping'
```

Validate and copy a mapping into the requested typed model.

---


### <kbd>method</kbd> `to_dict`

```python
to_dict() → dict[str, Any]
```

Return the same detached representation as ``to_mapping()``.

---


### <kbd>method</kbd> `to_mapping`

```python
to_mapping() → dict[str, Any]
```

Return a detached, JSON-compatible mapping for serialization.


---


## <kbd>class</kbd> `SessionInfo`
Read-only metadata snapshot for an active or stopped session.



**Attributes:**

 - <b>`session_id`</b>:  Stable conversation identifier.
 - <b>`runtime_id`</b>:  Runtime lifecycle identifier.
 - <b>`agent_name`</b>:  Resolved agent name.
 - <b>`profiles`</b>:  Applied profile names.
 - <b>`harness`</b>:  Stable harness identifier.
 - <b>`adapter_id`</b>:  Fabric adapter identifier.
 - <b>`adapter_kind`</b>:  Adapter execution mechanism.
 - <b>`status`</b>:  Current session lifecycle state.
 - <b>`capabilities`</b>:  Operations declared by the runtime.


### <kbd>method</kbd> `__init__`

```python
__init__(mapping: 'Mapping[str, Any]') → None
```






---

#### <kbd>property</kbd> extra_fields

Return an immutable view of preserved extension fields.



---


### <kbd>classmethod</kbd> `from_mapping`

```python
from_mapping(mapping: 'Mapping[str, Any]') → 'FabricMapping'
```

Validate and copy a mapping into the requested typed model.

---


### <kbd>method</kbd> `to_dict`

```python
to_dict() → dict[str, Any]
```

Return the same detached representation as ``to_mapping()``.

---


### <kbd>method</kbd> `to_mapping`

```python
to_mapping() → dict[str, Any]
```

Return a detached, JSON-compatible mapping for serialization.


---


## <kbd>class</kbd> `RuntimeUpdate`
Capability-gated update requested for a running session.



**Attributes:**

 - <b>`overrides`</b>:  Config overrides to apply to the runtime.
 - <b>`metadata`</b>:  Caller-owned update metadata.


### <kbd>method</kbd> `__init__`

```python
__init__(mapping: 'Mapping[str, Any]') → None
```






---

#### <kbd>property</kbd> extra_fields

Return an immutable view of preserved extension fields.



---


### <kbd>classmethod</kbd> `from_mapping`

```python
from_mapping(mapping: 'Mapping[str, Any]') → 'FabricMapping'
```

Validate and copy a mapping into the requested typed model.

---


### <kbd>method</kbd> `to_dict`

```python
to_dict() → dict[str, Any]
```

Return the same detached representation as ``to_mapping()``.

---


### <kbd>method</kbd> `to_mapping`

```python
to_mapping() → dict[str, Any]
```

Return a detached, JSON-compatible mapping for serialization.


---


## <kbd>class</kbd> `RuntimeUpdateResult`
Normalized outcome of a runtime update request.



**Attributes:**

 - <b>`status`</b>:  Terminal update status.
 - <b>`applied`</b>:  Overrides accepted by the runtime.
 - <b>`rejected`</b>:  Overrides rejected by the runtime.
 - <b>`reason`</b>:  Optional explanation for partial or complete rejection.


### <kbd>method</kbd> `__init__`

```python
__init__(mapping: 'Mapping[str, Any]') → None
```






---

#### <kbd>property</kbd> extra_fields

Return an immutable view of preserved extension fields.



---


### <kbd>classmethod</kbd> `from_mapping`

```python
from_mapping(mapping: 'Mapping[str, Any]') → 'FabricMapping'
```

Validate and copy a mapping into the requested typed model.

---


### <kbd>method</kbd> `to_dict`

```python
to_dict() → dict[str, Any]
```

Return the same detached representation as ``to_mapping()``.

---


### <kbd>method</kbd> `to_mapping`

```python
to_mapping() → dict[str, Any]
```

Return a detached, JSON-compatible mapping for serialization.




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
