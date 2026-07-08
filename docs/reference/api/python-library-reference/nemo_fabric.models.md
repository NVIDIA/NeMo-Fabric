---
title: "Models"
slug: "/reference/api/python-library-reference/models"
description: "Pydantic authoring models for Fabric config and request inputs."
---
{/* SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0 */}

# <kbd>module</kbd> `nemo_fabric.models`
Pydantic SDK models for NeMo Fabric configuration and requests.

The Rust core remains the source of truth for persisted schema snapshots. These models provide the Python SDK's typed authoring surface and intentionally keep extension fields so consumers can carry adapter- or application-owned data without waiting for a schema release.



---


## <kbd>class</kbd> `FabricBaseModel`
Base class for SDK-facing Pydantic models.


---

#### <kbd>property</kbd> extra_fields

Return fields preserved by the extension point for this model.

---

#### <kbd>property</kbd> model_extra

Get extra fields set during validation.



**Returns:**
  A dictionary of extra fields, or `None` if `config.extra` is not set to `"allow"`.

---

#### <kbd>property</kbd> model_fields_set

Returns the set of fields that have been explicitly set on this model instance.



**Returns:**
  A set of strings representing the fields that have been set,  i.e. that were not filled from defaults.



---


### <kbd>classmethod</kbd> `from_mapping`

```python
from_mapping(value: 'Mapping[str, Any]') → Self
```

Validate a mapping using this Pydantic model.

---


### <kbd>method</kbd> `to_mapping`

```python
to_mapping() → dict[str, Any]
```

Return a detached JSON-compatible mapping for Rust/core calls.


---


## <kbd>class</kbd> `MetadataConfig`
Human-readable agent identity.


---

#### <kbd>property</kbd> extra_fields

Return fields preserved by the extension point for this model.

---

#### <kbd>property</kbd> model_extra

Get extra fields set during validation.



**Returns:**
  A dictionary of extra fields, or `None` if `config.extra` is not set to `"allow"`.

---

#### <kbd>property</kbd> model_fields_set

Returns the set of fields that have been explicitly set on this model instance.



**Returns:**
  A set of strings representing the fields that have been set,  i.e. that were not filled from defaults.



---


### <kbd>classmethod</kbd> `from_mapping`

```python
from_mapping(value: 'Mapping[str, Any]') → Self
```

Validate a mapping using this Pydantic model.

---


### <kbd>method</kbd> `to_mapping`

```python
to_mapping() → dict[str, Any]
```

Return a detached JSON-compatible mapping for Rust/core calls.


---


## <kbd>class</kbd> `HarnessConfig`
Harness adapter selection plus adapter-owned settings.


---

#### <kbd>property</kbd> extra_fields

Return fields preserved by the extension point for this model.

---

#### <kbd>property</kbd> model_extra

Get extra fields set during validation.



**Returns:**
  A dictionary of extra fields, or `None` if `config.extra` is not set to `"allow"`.

---

#### <kbd>property</kbd> model_fields_set

Returns the set of fields that have been explicitly set on this model instance.



**Returns:**
  A set of strings representing the fields that have been set,  i.e. that were not filled from defaults.



---


### <kbd>classmethod</kbd> `from_mapping`

```python
from_mapping(value: 'Mapping[str, Any]') → Self
```

Validate a mapping using this Pydantic model.

---


### <kbd>method</kbd> `to_mapping`

```python
to_mapping() → dict[str, Any]
```

Return a detached JSON-compatible mapping for Rust/core calls.


---


## <kbd>class</kbd> `RuntimeConfig`
Runtime input/output contract.


---

#### <kbd>property</kbd> extra_fields

Return fields preserved by the extension point for this model.

---

#### <kbd>property</kbd> model_extra

Get extra fields set during validation.



**Returns:**
  A dictionary of extra fields, or `None` if `config.extra` is not set to `"allow"`.

---

#### <kbd>property</kbd> model_fields_set

Returns the set of fields that have been explicitly set on this model instance.



**Returns:**
  A set of strings representing the fields that have been set,  i.e. that were not filled from defaults.



---


### <kbd>classmethod</kbd> `from_mapping`

```python
from_mapping(value: 'Mapping[str, Any]') → Self
```

Validate a mapping using this Pydantic model.

---


### <kbd>method</kbd> `to_mapping`

```python
to_mapping() → dict[str, Any]
```

Return a detached JSON-compatible mapping for Rust/core calls.


---


## <kbd>class</kbd> `EnvironmentConfig`
Execution environment metadata supplied by the consumer.


---

#### <kbd>property</kbd> extra_fields

Return fields preserved by the extension point for this model.

---

#### <kbd>property</kbd> model_extra

Get extra fields set during validation.



**Returns:**
  A dictionary of extra fields, or `None` if `config.extra` is not set to `"allow"`.

---

#### <kbd>property</kbd> model_fields_set

Returns the set of fields that have been explicitly set on this model instance.



**Returns:**
  A set of strings representing the fields that have been set,  i.e. that were not filled from defaults.



---


### <kbd>classmethod</kbd> `from_mapping`

```python
from_mapping(value: 'Mapping[str, Any]') → Self
```

Validate a mapping using this Pydantic model.

---


### <kbd>method</kbd> `to_mapping`

```python
to_mapping() → dict[str, Any]
```

Return a detached JSON-compatible mapping for Rust/core calls.


---


## <kbd>class</kbd> `ModelConfig`
Model alias configuration.


---

#### <kbd>property</kbd> extra_fields

Return fields preserved by the extension point for this model.

---

#### <kbd>property</kbd> model_extra

Get extra fields set during validation.



**Returns:**
  A dictionary of extra fields, or `None` if `config.extra` is not set to `"allow"`.

---

#### <kbd>property</kbd> model_fields_set

Returns the set of fields that have been explicitly set on this model instance.



**Returns:**
  A set of strings representing the fields that have been set,  i.e. that were not filled from defaults.



---


### <kbd>classmethod</kbd> `from_mapping`

```python
from_mapping(value: 'Mapping[str, Any]') → Self
```

Validate a mapping using this Pydantic model.

---


### <kbd>method</kbd> `to_mapping`

```python
to_mapping() → dict[str, Any]
```

Return a detached JSON-compatible mapping for Rust/core calls.


---


## <kbd>class</kbd> `SkillConfig`
Skill capability configuration.


---

#### <kbd>property</kbd> extra_fields

Return fields preserved by the extension point for this model.

---

#### <kbd>property</kbd> model_extra

Get extra fields set during validation.



**Returns:**
  A dictionary of extra fields, or `None` if `config.extra` is not set to `"allow"`.

---

#### <kbd>property</kbd> model_fields_set

Returns the set of fields that have been explicitly set on this model instance.



**Returns:**
  A set of strings representing the fields that have been set,  i.e. that were not filled from defaults.



---


### <kbd>method</kbd> `add_path`

```python
add_path(path: 'str | Path') → Self
```

Add a skill path if absent.

---


### <kbd>classmethod</kbd> `from_mapping`

```python
from_mapping(value: 'Mapping[str, Any]') → Self
```

Validate a mapping using this Pydantic model.

---


### <kbd>method</kbd> `remove_path`

```python
remove_path(path: 'str | Path') → Self
```

Remove a skill path if present.

---


### <kbd>method</kbd> `to_mapping`

```python
to_mapping() → dict[str, Any]
```

Return a detached JSON-compatible mapping for Rust/core calls.


---


## <kbd>class</kbd> `McpServerConfig`
MCP server configuration.


---

#### <kbd>property</kbd> extra_fields

Return fields preserved by the extension point for this model.

---

#### <kbd>property</kbd> model_extra

Get extra fields set during validation.



**Returns:**
  A dictionary of extra fields, or `None` if `config.extra` is not set to `"allow"`.

---

#### <kbd>property</kbd> model_fields_set

Returns the set of fields that have been explicitly set on this model instance.



**Returns:**
  A set of strings representing the fields that have been set,  i.e. that were not filled from defaults.



---


### <kbd>classmethod</kbd> `from_mapping`

```python
from_mapping(value: 'Mapping[str, Any]') → Self
```

Validate a mapping using this Pydantic model.

---


### <kbd>method</kbd> `to_mapping`

```python
to_mapping() → dict[str, Any]
```

Return a detached JSON-compatible mapping for Rust/core calls.


---


## <kbd>class</kbd> `McpConfig`
MCP capability configuration.


---

#### <kbd>property</kbd> extra_fields

Return fields preserved by the extension point for this model.

---

#### <kbd>property</kbd> model_extra

Get extra fields set during validation.



**Returns:**
  A dictionary of extra fields, or `None` if `config.extra` is not set to `"allow"`.

---

#### <kbd>property</kbd> model_fields_set

Returns the set of fields that have been explicitly set on this model instance.



**Returns:**
  A set of strings representing the fields that have been set,  i.e. that were not filled from defaults.



---


### <kbd>method</kbd> `add_server`

```python
add_server(
    name: 'str',
    transport: 'str',
    url: 'str',
    exposure: "Literal['harness_native', 'fabric_managed']" = 'harness_native',
    extra_fields: 'Mapping[str, Any] | None' = None
) → Self
```

Add or replace a named MCP server.

---


### <kbd>classmethod</kbd> `from_mapping`

```python
from_mapping(value: 'Mapping[str, Any]') → Self
```

Validate a mapping using this Pydantic model.

---


### <kbd>method</kbd> `remove_server`

```python
remove_server(name: 'str') → Self
```

Remove a named MCP server if present.

---


### <kbd>method</kbd> `to_mapping`

```python
to_mapping() → dict[str, Any]
```

Return a detached JSON-compatible mapping for Rust/core calls.


---


## <kbd>class</kbd> `TelemetryConfig`
Telemetry configuration.


---

#### <kbd>property</kbd> extra_fields

Return fields preserved by the extension point for this model.

---

#### <kbd>property</kbd> model_extra

Get extra fields set during validation.



**Returns:**
  A dictionary of extra fields, or `None` if `config.extra` is not set to `"allow"`.

---

#### <kbd>property</kbd> model_fields_set

Returns the set of fields that have been explicitly set on this model instance.



**Returns:**
  A set of strings representing the fields that have been set,  i.e. that were not filled from defaults.



---


### <kbd>method</kbd> `disable`

```python
disable() → Self
```

Disable telemetry.

---


### <kbd>method</kbd> `enable_native`

```python
enable_native() → Self
```

Let the selected adapter handle telemetry natively.

---


### <kbd>method</kbd> `enable_relay`

```python
enable_relay(
    project: 'str | None' = None,
    output_dir: 'str | Path | None' = None,
    config: 'Mapping[str, Any] | None' = None
) → Self
```

Enable NeMo Relay telemetry for subsequently started runtimes.

---


### <kbd>classmethod</kbd> `from_mapping`

```python
from_mapping(value: 'Mapping[str, Any]') → Self
```

Validate a mapping using this Pydantic model.

---


### <kbd>method</kbd> `to_mapping`

```python
to_mapping() → dict[str, Any]
```

Return a detached JSON-compatible mapping for Rust/core calls.


---


## <kbd>class</kbd> `ProfileRegistryConfig`
Profile discovery config for portable file-backed agent packages.


---

#### <kbd>property</kbd> extra_fields

Return fields preserved by the extension point for this model.

---

#### <kbd>property</kbd> model_extra

Get extra fields set during validation.



**Returns:**
  A dictionary of extra fields, or `None` if `config.extra` is not set to `"allow"`.

---

#### <kbd>property</kbd> model_fields_set

Returns the set of fields that have been explicitly set on this model instance.



**Returns:**
  A set of strings representing the fields that have been set,  i.e. that were not filled from defaults.



---


### <kbd>classmethod</kbd> `from_mapping`

```python
from_mapping(value: 'Mapping[str, Any]') → Self
```

Validate a mapping using this Pydantic model.

---


### <kbd>method</kbd> `to_mapping`

```python
to_mapping() → dict[str, Any]
```

Return a detached JSON-compatible mapping for Rust/core calls.


---


## <kbd>class</kbd> `FabricConfig`
SDK-facing typed Fabric agent configuration.


---

#### <kbd>property</kbd> extra_fields

Return fields preserved by the extension point for this model.

---

#### <kbd>property</kbd> model_extra

Get extra fields set during validation.



**Returns:**
  A dictionary of extra fields, or `None` if `config.extra` is not set to `"allow"`.

---

#### <kbd>property</kbd> model_fields_set

Returns the set of fields that have been explicitly set on this model instance.



**Returns:**
  A set of strings representing the fields that have been set,  i.e. that were not filled from defaults.



---


### <kbd>method</kbd> `add_mcp_server`

```python
add_mcp_server(
    name: 'str',
    transport: 'str',
    url: 'str',
    exposure: "Literal['harness_native', 'fabric_managed']" = 'harness_native',
    extra_fields: 'Mapping[str, Any] | None' = None
) → Self
```

Add or replace a named MCP server and return this config.

---


### <kbd>method</kbd> `add_skill_path`

```python
add_skill_path(path: 'str | Path') → Self
```

Add a skill path and return this config.

---


### <kbd>method</kbd> `enable_relay`

```python
enable_relay(
    project: 'str | None' = None,
    output_dir: 'str | Path | None' = None,
    config: 'Mapping[str, Any] | None' = None
) → Self
```

Enable NeMo Relay telemetry and return this config.

---


### <kbd>classmethod</kbd> `from_mapping`

```python
from_mapping(value: 'Mapping[str, Any]') → Self
```

Validate the public agent config mapping shape.

---


### <kbd>method</kbd> `to_mapping`

```python
to_mapping() → dict[str, Any]
```

Return a detached mapping matching the Rust ``FabricConfig`` schema.


---


## <kbd>class</kbd> `FabricProfileConfig`
Typed profile overlay used when a Python caller wants file-style overlays.


---

#### <kbd>property</kbd> extra_fields

Return fields preserved by the extension point for this model.

---

#### <kbd>property</kbd> model_extra

Get extra fields set during validation.



**Returns:**
  A dictionary of extra fields, or `None` if `config.extra` is not set to `"allow"`.

---

#### <kbd>property</kbd> model_fields_set

Returns the set of fields that have been explicitly set on this model instance.



**Returns:**
  A set of strings representing the fields that have been set,  i.e. that were not filled from defaults.



---


### <kbd>classmethod</kbd> `from_mapping`

```python
from_mapping(value: 'Mapping[str, Any]') → Self
```

Validate a mapping using this Pydantic model.

---


### <kbd>method</kbd> `to_mapping`

```python
to_mapping() → dict[str, Any]
```

Return a detached JSON-compatible mapping for Rust/core calls.


---


## <kbd>class</kbd> `RunRequest`
One validated Fabric invocation request.


---

#### <kbd>property</kbd> extra_fields

Return fields preserved by the extension point for this model.

---

#### <kbd>property</kbd> model_extra

Get extra fields set during validation.



**Returns:**
  A dictionary of extra fields, or `None` if `config.extra` is not set to `"allow"`.

---

#### <kbd>property</kbd> model_fields_set

Returns the set of fields that have been explicitly set on this model instance.



**Returns:**
  A set of strings representing the fields that have been set,  i.e. that were not filled from defaults.



---


### <kbd>classmethod</kbd> `from_mapping`

```python
from_mapping(value: 'Mapping[str, Any]') → Self
```

Validate a mapping using this Pydantic model.

---


### <kbd>method</kbd> `to_mapping`

```python
to_mapping() → dict[str, Any]
```

Return a detached request mapping for the Rust runtime.




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
