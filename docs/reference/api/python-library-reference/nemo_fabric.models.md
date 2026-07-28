---
title: "Models"
slug: "/reference/api/python-library-reference/models"
description: "Pydantic authoring models for NVIDIA NeMo Fabric config and request inputs."
---
<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0 -->

# <kbd>module</kbd> `nemo_fabric.models`

Pydantic SDK models for NVIDIA NeMo Fabric configuration and requests.

The Rust core remains the source of truth for persisted schema snapshots. These models provide the Python SDK's typed authoring surface and intentionally keep extension fields so consumers can carry adapter- or application-owned data without waiting for a schema release.



---


## <kbd>class</kbd> `FabricBaseModel`

Base class for SDK-facing Pydantic models.


---

### <kbd>property</kbd> extra_fields

Return fields preserved by the extension point for this model.

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


### <kbd>classmethod</kbd> `from_mapping`

```python
def from_mapping(value: Mapping[str, Any]) -> Self
```

Validate a mapping using this Pydantic model.

---


### <kbd>method</kbd> `to_mapping`

```python
def to_mapping() -> dict[str, Any]
```

Return a detached JSON-compatible mapping for Rust/core calls.


---


## <kbd>class</kbd> `MetadataConfig`

Human-readable agent identity.



### Fields

The model defines the following fields:

| Field | Type | Required | Default | Constraints | Description |
| --- | --- | --- | --- | --- | --- |
| `name` | `str` | Yes | — | `MinLen(min_length=1)` | — |
| `description` | `str \| None` | No | `None` | — | — |

---

### <kbd>property</kbd> extra_fields

Return fields preserved by the extension point for this model.

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


### <kbd>classmethod</kbd> `from_mapping`

```python
def from_mapping(value: Mapping[str, Any]) -> Self
```

Validate a mapping using this Pydantic model.

---


### <kbd>method</kbd> `to_mapping`

```python
def to_mapping() -> dict[str, Any]
```

Return a detached JSON-compatible mapping for Rust/core calls.


---


## <kbd>class</kbd> `HarnessConfig`

Harness adapter selection plus adapter-owned settings.



### Fields

The model defines the following fields:

| Field | Type | Required | Default | Constraints | Description |
| --- | --- | --- | --- | --- | --- |
| `adapter_id` | `str` | Yes | — | `MinLen(min_length=1)` | — |
| `resolution` | `Literal['preinstalled', 'image_provided', 'pip_uv', 'npm', 'source', 'service', 'native_plugin'] \| None` | No | `None` | — | — |
| `settings` | `dict[str, Any]` | No | `dict()` | — | — |

---

### <kbd>property</kbd> extra_fields

Return fields preserved by the extension point for this model.

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


### <kbd>classmethod</kbd> `from_mapping`

```python
def from_mapping(value: Mapping[str, Any]) -> Self
```

Validate a mapping using this Pydantic model.

---


### <kbd>method</kbd> `to_mapping`

```python
def to_mapping() -> dict[str, Any]
```

Return a detached JSON-compatible mapping for Rust/core calls.


---


## <kbd>class</kbd> `InstructionConfig`

One portable instruction value.



### Fields

The model defines the following fields:

| Field | Type | Required | Default | Constraints | Description |
| --- | --- | --- | --- | --- | --- |
| `content` | `str` | Yes | — | — | — |
| `mode` | `Literal['replace']` | No | `'replace'` | — | — |

---

### <kbd>property</kbd> extra_fields

Return fields preserved by the extension point for this model.

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


### <kbd>classmethod</kbd> `from_mapping`

```python
def from_mapping(value: Mapping[str, Any]) -> Self
```

Validate a mapping using this Pydantic model.

---


### <kbd>method</kbd> `to_mapping`

```python
def to_mapping() -> dict[str, Any]
```

Return a detached JSON-compatible mapping for Rust/core calls.


---


## <kbd>class</kbd> `InstructionsConfig`

Harness-neutral agent instructions.



### Fields

The model defines the following fields:

| Field | Type | Required | Default | Constraints | Description |
| --- | --- | --- | --- | --- | --- |
| `system` | `InstructionConfig \| None` | No | `None` | — | — |

---

### <kbd>property</kbd> extra_fields

Return fields preserved by the extension point for this model.

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


### <kbd>classmethod</kbd> `from_mapping`

```python
def from_mapping(value: Mapping[str, Any]) -> Self
```

Validate a mapping using this Pydantic model.

---


### <kbd>method</kbd> `to_mapping`

```python
def to_mapping() -> dict[str, Any]
```

Return a detached JSON-compatible mapping for Rust/core calls.


---


## <kbd>class</kbd> `RuntimeConfig`

Invocation runtime contract.



### Fields

The model defines the following fields:

| Field | Type | Required | Default | Constraints | Description |
| --- | --- | --- | --- | --- | --- |
| `input_schema` | `str \| None` | No | `None` | — | — |
| `output_schema` | `str \| None` | No | `None` | — | — |
| `artifacts` | `str \| Path \| None` | No | `None` | — | — |
| `timeout_seconds` | `float \| None` | No | `None` | `Gt(gt=0), _PydanticGeneralMetadata(allow_inf_nan=False)` | — |
| `max_turns` | `int \| None` | No | `None` | `Gt(gt=0)` | — |

---

### <kbd>property</kbd> extra_fields

Return fields preserved by the extension point for this model.

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


### <kbd>classmethod</kbd> `from_mapping`

```python
def from_mapping(value: Mapping[str, Any]) -> Self
```

Validate a mapping using this Pydantic model.

---


### <kbd>method</kbd> `to_mapping`

```python
def to_mapping() -> dict[str, Any]
```

Return a detached JSON-compatible mapping for Rust/core calls.


---


## <kbd>class</kbd> `EnvironmentConfig`

Execution environment configuration supplied by the consumer.

``provider`` selects the environment implementation. ``workspace`` is the path visible to the harness, while ``artifacts`` is the provider-specific output location. ``settings`` configures the selected provider; ``connection`` describes how NeMo Fabric reaches an existing environment; and ``metadata`` carries consumer-owned values that NeMo Fabric does not interpret. ``ownership`` identifies who tears the environment down, and ``control_location`` identifies whether NeMo Fabric control code runs inside or outside it.



### Fields

The model defines the following fields:

| Field | Type | Required | Default | Constraints | Description |
| --- | --- | --- | --- | --- | --- |
| `provider` | `str` | No | `'local'` | `MinLen(min_length=1)` | Environment provider, such as local, docker, opensandbox, or k8s. |
| `workspace` | `str \| Path \| None` | No | `None` | — | Workspace path visible to the harness. |
| `artifacts` | `str \| Path \| None` | No | `None` | — | Environment-specific artifact path. |
| `env` | `dict[str, str]` | No | `dict()` | — | Environment variables visible to the harness and its tools. Values are serialized into configuration and run plans; prefer api_key_env-style environment-variable-name indirection for credentials. |
| `settings` | `dict[str, Any]` | No | `dict()` | — | Provider-specific configuration interpreted by the environment provider. |
| `metadata` | `dict[str, Any]` | No | `dict()` | — | Consumer-owned environment metadata passed through without NeMo Fabric semantics. |
| `connection` | `dict[str, Any]` | No | `dict()` | — | Connection data for an existing environment, such as URL, namespace, or credential reference. |
| `ownership` | `Literal['caller_owned', 'fabric_owned']` | No | `'caller_owned'` | — | Whether the caller or NeMo Fabric owns environment teardown. |
| `control_location` | `Literal['external_control', 'in_env_control']` | No | `'in_env_control'` | — | Whether NeMo Fabric control code runs outside or inside the environment. |

---

### <kbd>property</kbd> extra_fields

Return fields preserved by the extension point for this model.

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


### <kbd>classmethod</kbd> `from_mapping`

```python
def from_mapping(value: Mapping[str, Any]) -> Self
```

Validate a mapping using this Pydantic model.

---


### <kbd>method</kbd> `to_mapping`

```python
def to_mapping() -> dict[str, Any]
```

Return a detached JSON-compatible mapping for Rust/core calls.


---


## <kbd>class</kbd> `ModelConfig`

Configuration for one model role.



### Fields

The model defines the following fields:

| Field | Type | Required | Default | Constraints | Description |
| --- | --- | --- | --- | --- | --- |
| `provider` | `str` | Yes | — | `MinLen(min_length=1)` | — |
| `model` | `str` | Yes | — | `MinLen(min_length=1)` | — |
| `api_key_env` | `str \| None` | No | `None` | — | — |
| `temperature` | `float \| None` | No | `None` | — | — |
| `base_url` | `str \| None` | No | `None` | `MinLen(min_length=1)` | — |
| `settings` | `dict[str, Any]` | No | `dict()` | — | — |

---

### <kbd>property</kbd> extra_fields

Return fields preserved by the extension point for this model.

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


### <kbd>classmethod</kbd> `from_mapping`

```python
def from_mapping(value: Mapping[str, Any]) -> Self
```

Validate a mapping using this Pydantic model.

---


### <kbd>method</kbd> `to_mapping`

```python
def to_mapping() -> dict[str, Any]
```

Return a detached JSON-compatible mapping for Rust/core calls.


---


## <kbd>class</kbd> `SkillConfig`

Skill capability configuration.



### Fields

The model defines the following fields:

| Field | Type | Required | Default | Constraints | Description |
| --- | --- | --- | --- | --- | --- |
| `paths` | `list[str \| Path]` | No | `list()` | — | — |

---

### <kbd>property</kbd> extra_fields

Return fields preserved by the extension point for this model.

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


### <kbd>method</kbd> `add_path`

```python
def add_path(path: str | Path) -> Self
```

Add a skill path if absent.

---


### <kbd>classmethod</kbd> `from_mapping`

```python
def from_mapping(value: Mapping[str, Any]) -> Self
```

Validate a mapping using this Pydantic model.

---


### <kbd>method</kbd> `remove_path`

```python
def remove_path(path: str | Path) -> Self
```

Remove a skill path if present.

---


### <kbd>method</kbd> `to_mapping`

```python
def to_mapping() -> dict[str, Any]
```

Return a detached JSON-compatible mapping for Rust/core calls.


---


## <kbd>class</kbd> `McpServerConfig`

MCP server configuration.



### Fields

The model defines the following fields:

| Field | Type | Required | Default | Constraints | Description |
| --- | --- | --- | --- | --- | --- |
| `transport` | `str` | Yes | — | `MinLen(min_length=1)` | — |
| `url` | `str` | Yes | — | `MinLen(min_length=1)` | — |
| `exposure` | `Literal['harness_native', 'fabric_managed']` | No | `'harness_native'` | — | — |

---

### <kbd>property</kbd> extra_fields

Return fields preserved by the extension point for this model.

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


### <kbd>classmethod</kbd> `from_mapping`

```python
def from_mapping(value: Mapping[str, Any]) -> Self
```

Validate a mapping using this Pydantic model.

---


### <kbd>method</kbd> `to_mapping`

```python
def to_mapping() -> dict[str, Any]
```

Return a detached JSON-compatible mapping for Rust/core calls.


---


## <kbd>class</kbd> `McpConfig`

MCP capability configuration.



### Fields

The model defines the following fields:

| Field | Type | Required | Default | Constraints | Description |
| --- | --- | --- | --- | --- | --- |
| `servers` | `dict[str, McpServerConfig]` | No | `dict()` | — | — |

---

### <kbd>property</kbd> extra_fields

Return fields preserved by the extension point for this model.

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


### <kbd>method</kbd> `add_server`

```python
def add_server(
    name: str,
    *,
    transport: str,
    url: str,
    exposure: Literal['harness_native', 'fabric_managed'] = 'harness_native',
    extra_fields: Mapping[str, Any] | None = None,
) -> Self
```

Add or replace a named MCP server.

---


### <kbd>classmethod</kbd> `from_mapping`

```python
def from_mapping(value: Mapping[str, Any]) -> Self
```

Validate a mapping using this Pydantic model.

---


### <kbd>method</kbd> `remove_server`

```python
def remove_server(name: str) -> Self
```

Remove a named MCP server if present.

---


### <kbd>method</kbd> `to_mapping`

```python
def to_mapping() -> dict[str, Any]
```

Return a detached JSON-compatible mapping for Rust/core calls.


---


## <kbd>class</kbd> `RelayConfigPolicy`

NVIDIA NeMo Relay config validation policy.



### Fields

The model defines the following fields:

| Field | Type | Required | Default | Constraints | Description |
| --- | --- | --- | --- | --- | --- |
| `unknown_component` | `Literal['ignore', 'warn', 'error']` | No | `'warn'` | — | — |
| `unknown_field` | `Literal['ignore', 'warn', 'error']` | No | `'warn'` | — | — |
| `unsupported_value` | `Literal['ignore', 'warn', 'error']` | No | `'error'` | — | — |

---

### <kbd>property</kbd> extra_fields

Return fields preserved by the extension point for this model.

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


### <kbd>classmethod</kbd> `from_mapping`

```python
def from_mapping(value: Mapping[str, Any]) -> Self
```

Validate a mapping using this Pydantic model.

---


### <kbd>method</kbd> `to_mapping`

```python
def to_mapping() -> dict[str, Any]
```

Return a detached JSON-compatible mapping for Rust/core calls.


---


## <kbd>class</kbd> `RelayAtofFileSinkConfig`

NeMo Relay ATOF file sink configuration.



### Fields

The model defines the following fields:

| Field | Type | Required | Default | Constraints | Description |
| --- | --- | --- | --- | --- | --- |
| `type` | `Literal['file']` | No | `'file'` | — | — |
| `output_directory` | `str \| Path \| None` | No | `None` | — | — |
| `filename` | `str \| None` | No | `None` | — | — |
| `mode` | `Literal['append', 'overwrite']` | No | `'append'` | — | — |

---

### <kbd>property</kbd> extra_fields

Return fields preserved by the extension point for this model.

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


### <kbd>classmethod</kbd> `from_mapping`

```python
def from_mapping(value: Mapping[str, Any]) -> Self
```

Validate a mapping using this Pydantic model.

---


### <kbd>method</kbd> `to_mapping`

```python
def to_mapping() -> dict[str, Any]
```

Return a detached JSON-compatible mapping for Rust/core calls.


---


## <kbd>class</kbd> `RelayAtofStreamSinkConfig`

NeMo Relay ATOF stream sink configuration.



### Fields

The model defines the following fields:

| Field | Type | Required | Default | Constraints | Description |
| --- | --- | --- | --- | --- | --- |
| `type` | `Literal['stream']` | No | `'stream'` | — | — |
| `url` | `str` | Yes | — | — | — |
| `transport` | `Literal['http_post', 'websocket', 'ndjson']` | No | `'http_post'` | — | — |
| `headers` | `dict[str, str]` | No | `dict()` | — | — |
| `header_env` | `dict[str, str]` | No | `dict()` | — | — |
| `timeout_millis` | `int` | No | `3000` | — | — |
| `field_name_policy` | `Literal['preserve', 'replace_dots']` | No | `'preserve'` | — | — |
| `name` | `str \| None` | No | `None` | — | — |

---

### <kbd>property</kbd> extra_fields

Return fields preserved by the extension point for this model.

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


### <kbd>classmethod</kbd> `from_mapping`

```python
def from_mapping(value: Mapping[str, Any]) -> Self
```

Validate a mapping using this Pydantic model.

---


### <kbd>method</kbd> `to_mapping`

```python
def to_mapping() -> dict[str, Any]
```

Return a detached JSON-compatible mapping for Rust/core calls.


---


## <kbd>class</kbd> `RelayAtofConfig`

NeMo Relay ATOF export configuration.



### Fields

The model defines the following fields:

| Field | Type | Required | Default | Constraints | Description |
| --- | --- | --- | --- | --- | --- |
| `enabled` | `bool` | No | `False` | — | — |
| `sinks` | `list[Annotated[RelayAtofFileSinkConfig \| RelayAtofStreamSinkConfig, Field(discriminator='type')] \| dict[str, Any]] \| None` | No | `None` | — | — |

---

### <kbd>property</kbd> extra_fields

Return fields preserved by the extension point for this model.

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


### <kbd>classmethod</kbd> `from_mapping`

```python
def from_mapping(value: Mapping[str, Any]) -> Self
```

Validate a mapping using this Pydantic model.

---


### <kbd>method</kbd> `to_mapping`

```python
def to_mapping() -> dict[str, Any]
```

Return a detached JSON-compatible mapping for Rust/core calls.


---


## <kbd>class</kbd> `RelayS3StorageConfig`

NeMo Relay ATIF S3 storage configuration.



### Fields

The model defines the following fields:

| Field | Type | Required | Default | Constraints | Description |
| --- | --- | --- | --- | --- | --- |
| `type` | `Literal['s3']` | No | `'s3'` | — | — |
| `bucket` | `str` | No | `''` | — | — |
| `key_prefix` | `str \| None` | No | `None` | — | — |
| `access_key_id` | `str \| None` | No | `None` | — | — |
| `secret_access_key_var` | `str \| None` | No | `None` | — | — |
| `session_token_var` | `str \| None` | No | `None` | — | — |
| `region` | `str \| None` | No | `None` | — | — |
| `endpoint_url` | `str \| None` | No | `None` | — | — |
| `allow_http` | `bool \| None` | No | `None` | — | — |

---

### <kbd>property</kbd> extra_fields

Return fields preserved by the extension point for this model.

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


### <kbd>classmethod</kbd> `from_mapping`

```python
def from_mapping(value: Mapping[str, Any]) -> Self
```

Validate a mapping using this Pydantic model.

---


### <kbd>method</kbd> `to_mapping`

```python
def to_mapping() -> dict[str, Any]
```

Return a detached JSON-compatible mapping for Rust/core calls.


---


## <kbd>class</kbd> `RelayHttpStorageConfig`

NeMo Relay ATIF HTTP storage configuration.



### Fields

The model defines the following fields:

| Field | Type | Required | Default | Constraints | Description |
| --- | --- | --- | --- | --- | --- |
| `type` | `Literal['http']` | No | `'http'` | — | — |
| `endpoint` | `str` | No | `''` | — | — |
| `headers` | `dict[str, str]` | No | `dict()` | — | — |
| `header_env` | `dict[str, str]` | No | `dict()` | — | — |
| `timeout_millis` | `int` | No | `3000` | — | — |

---

### <kbd>property</kbd> extra_fields

Return fields preserved by the extension point for this model.

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


### <kbd>classmethod</kbd> `from_mapping`

```python
def from_mapping(value: Mapping[str, Any]) -> Self
```

Validate a mapping using this Pydantic model.

---


### <kbd>method</kbd> `to_mapping`

```python
def to_mapping() -> dict[str, Any]
```

Return a detached JSON-compatible mapping for Rust/core calls.


---


## <kbd>class</kbd> `RelayAtifConfig`

NeMo Relay ATIF export configuration.



### Fields

The model defines the following fields:

| Field | Type | Required | Default | Constraints | Description |
| --- | --- | --- | --- | --- | --- |
| `enabled` | `bool` | No | `False` | — | — |
| `agent_name` | `str` | No | `'NeMo Relay'` | — | — |
| `agent_version` | `str \| None` | No | `None` | — | — |
| `model_name` | `str` | No | `'unknown'` | — | — |
| `tool_definitions` | `list[dict[str, Any]] \| None` | No | `None` | — | — |
| `extra` | `dict[str, Any] \| None` | No | `None` | — | — |
| `output_directory` | `str \| Path \| None` | No | `None` | — | — |
| `filename_template` | `str` | No | `'nemo-relay-atif-{session_id}.json'` | — | — |
| `storage` | `list[Annotated[RelayS3StorageConfig \| RelayHttpStorageConfig, Field(discriminator='type')] \| dict[str, Any]] \| None` | No | `None` | — | — |

---

### <kbd>property</kbd> extra_fields

Return fields preserved by the extension point for this model.

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


### <kbd>classmethod</kbd> `from_mapping`

```python
def from_mapping(value: Mapping[str, Any]) -> Self
```

Validate a mapping using this Pydantic model.

---


### <kbd>method</kbd> `to_mapping`

```python
def to_mapping() -> dict[str, Any]
```

Return a detached JSON-compatible mapping for Rust/core calls.


---


## <kbd>class</kbd> `RelayOtlpConfig`

NeMo Relay OTLP export configuration for OpenTelemetry/OpenInference.



### Fields

The model defines the following fields:

| Field | Type | Required | Default | Constraints | Description |
| --- | --- | --- | --- | --- | --- |
| `enabled` | `bool` | No | `False` | — | — |
| `transport` | `Literal['http_binary', 'grpc']` | No | `'http_binary'` | — | — |
| `endpoint` | `str \| None` | No | `None` | — | — |
| `headers` | `dict[str, str]` | No | `dict()` | — | — |
| `resource_attributes` | `dict[str, str]` | No | `dict()` | — | — |
| `service_name` | `str` | No | `'nemo-relay'` | — | — |
| `service_namespace` | `str \| None` | No | `None` | — | — |
| `service_version` | `str \| None` | No | `None` | — | — |
| `instrumentation_scope` | `str \| None` | No | `None` | — | — |
| `timeout_millis` | `int` | No | `3000` | — | — |

---

### <kbd>property</kbd> extra_fields

Return fields preserved by the extension point for this model.

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


### <kbd>classmethod</kbd> `from_mapping`

```python
def from_mapping(value: Mapping[str, Any]) -> Self
```

Validate a mapping using this Pydantic model.

---


### <kbd>method</kbd> `to_mapping`

```python
def to_mapping() -> dict[str, Any]
```

Return a detached JSON-compatible mapping for Rust/core calls.


---


## <kbd>class</kbd> `RelayObservabilityConfig`

NeMo Relay observability component configuration.



### Fields

The model defines the following fields:

| Field | Type | Required | Default | Constraints | Description |
| --- | --- | --- | --- | --- | --- |
| `version` | `int` | No | `2` | — | — |
| `atof` | `RelayAtofConfig \| dict[str, Any] \| None` | No | `None` | — | — |
| `atif` | `RelayAtifConfig \| dict[str, Any] \| None` | No | `None` | — | — |
| `opentelemetry` | `RelayOtlpConfig \| dict[str, Any] \| None` | No | `None` | — | — |
| `openinference` | `RelayOtlpConfig \| dict[str, Any] \| None` | No | `None` | — | — |
| `policy` | `RelayConfigPolicy \| dict[str, Any] \| None` | No | `None` | — | — |

---

### <kbd>property</kbd> extra_fields

Return fields preserved by the extension point for this model.

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


### <kbd>classmethod</kbd> `from_mapping`

```python
def from_mapping(value: Mapping[str, Any]) -> Self
```

Validate a mapping using this Pydantic model.

---


### <kbd>method</kbd> `to_mapping`

```python
def to_mapping() -> dict[str, Any]
```

Return a detached JSON-compatible mapping for Rust/core calls.


---


## <kbd>class</kbd> `RelayComponentConfig`

Generic NeMo Relay plugin component configuration.



### Fields

The model defines the following fields:

| Field | Type | Required | Default | Constraints | Description |
| --- | --- | --- | --- | --- | --- |
| `kind` | `str` | Yes | — | `MinLen(min_length=1)` | — |
| `enabled` | `bool` | No | `True` | — | — |
| `config` | `dict[str, Any]` | No | `dict()` | — | — |

---

### <kbd>property</kbd> extra_fields

Return fields preserved by the extension point for this model.

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


### <kbd>classmethod</kbd> `from_mapping`

```python
def from_mapping(value: Mapping[str, Any]) -> Self
```

Validate a mapping using this Pydantic model.

---


### <kbd>method</kbd> `to_mapping`

```python
def to_mapping() -> dict[str, Any]
```

Return a detached JSON-compatible mapping for Rust/core calls.


---


## <kbd>class</kbd> `RelayConfig`

First-class NeMo Relay integration configuration.



### Fields

The model defines the following fields:

| Field | Type | Required | Default | Constraints | Description |
| --- | --- | --- | --- | --- | --- |
| `project` | `str \| None` | No | `None` | — | — |
| `output_dir` | `str \| Path \| None` | No | `None` | — | — |
| `observability` | `RelayObservabilityConfig \| dict[str, Any] \| None` | No | `None` | — | — |
| `components` | `list[RelayComponentConfig \| dict[str, Any]]` | No | `list()` | — | — |
| `policy` | `RelayConfigPolicy \| dict[str, Any] \| None` | No | `None` | — | — |

---

### <kbd>property</kbd> extra_fields

Return fields preserved by the extension point for this model.

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


### <kbd>classmethod</kbd> `from_mapping`

```python
def from_mapping(value: Mapping[str, Any]) -> Self
```

Validate a mapping using this Pydantic model.

---


### <kbd>method</kbd> `to_mapping`

```python
def to_mapping() -> dict[str, Any]
```

Return a detached JSON-compatible mapping for Rust/core calls.


---


## <kbd>class</kbd> `TelemetryProviderConfig`

Provider-specific telemetry configuration.



### Fields

The model defines the following fields:

| Field | Type | Required | Default | Constraints | Description |
| --- | --- | --- | --- | --- | --- |
| `config` | `dict[str, Any] \| None` | No | `None` | — | — |

---

### <kbd>property</kbd> extra_fields

Return fields preserved by the extension point for this model.

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


### <kbd>classmethod</kbd> `from_mapping`

```python
def from_mapping(value: Mapping[str, Any]) -> Self
```

Validate a mapping using this Pydantic model.

---


### <kbd>method</kbd> `to_mapping`

```python
def to_mapping() -> dict[str, Any]
```

Return a detached JSON-compatible mapping for Rust/core calls.


---


## <kbd>class</kbd> `TelemetryConfig`

Telemetry configuration.



### Fields

The model defines the following fields:

| Field | Type | Required | Default | Constraints | Description |
| --- | --- | --- | --- | --- | --- |
| `providers` | `dict[Literal['relay', 'native'], TelemetryProviderConfig \| dict[str, Any]]` | No | `dict()` | — | — |

---

### <kbd>property</kbd> extra_fields

Return fields preserved by the extension point for this model.

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


### <kbd>method</kbd> `enable_native`

```python
def enable_native(*, config: Mapping[str, Any] | None = None) -> Self
```

Let the selected adapter handle telemetry natively.

---


### <kbd>method</kbd> `enable_relay`

```python
def enable_relay() -> Self
```

Enable NeMo Relay telemetry for subsequently started runtimes.

---


### <kbd>classmethod</kbd> `from_mapping`

```python
def from_mapping(value: Mapping[str, Any]) -> Self
```

Validate a mapping using this Pydantic model.

---


### <kbd>method</kbd> `remove_provider`

```python
def remove_provider(provider: Literal['relay', 'native']) -> Self
```

Remove a configured telemetry provider.

---


### <kbd>method</kbd> `to_mapping`

```python
def to_mapping() -> dict[str, Any]
```

Return a detached JSON-compatible mapping for Rust/core calls.


---


## <kbd>class</kbd> `ToolsConfig`

Harness-neutral tool capability configuration.



### Fields

The model defines the following fields:

| Field | Type | Required | Default | Constraints | Description |
| --- | --- | --- | --- | --- | --- |
| `enabled` | `list[str] \| None` | No | `None` | — | Adapter-native tools to expose. None preserves the harness default; an empty list exposes no tools. |
| `blocked` | `list[str]` | No | `list()` | — | — |

---

### <kbd>property</kbd> extra_fields

Return fields preserved by the extension point for this model.

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


### <kbd>classmethod</kbd> `from_mapping`

```python
def from_mapping(value: Mapping[str, Any]) -> Self
```

Validate a mapping using this Pydantic model.

---


### <kbd>method</kbd> `to_mapping`

```python
def to_mapping() -> dict[str, Any]
```

Return a detached JSON-compatible mapping for Rust/core calls.


---


## <kbd>class</kbd> `FabricConfig`

SDK-facing typed NeMo Fabric agent configuration.

NeMo Fabric-owned fields apply uniformly. Adapter-translated fields are checked against the selected descriptor; refer to the [normalized configuration compatibility table](../../../sdk/python.mdx#normalized-configuration-compatibility).



### Fields

The model defines the following fields:

| Field | Type | Required | Default | Constraints | Description |
| --- | --- | --- | --- | --- | --- |
| `schema_version` | `str` | No | `'fabric.agent/v1alpha1'` | — | — |
| `metadata` | `MetadataConfig` | Yes | — | — | — |
| `harness` | `HarnessConfig` | Yes | — | — | — |
| `runtime` | `RuntimeConfig` | No | `RuntimeConfig()` | — | — |
| `environment` | `EnvironmentConfig \| None` | No | `None` | — | — |
| `models` | `dict[str, ModelConfig]` | No | `dict()` | — | — |
| `instructions` | `InstructionsConfig \| None` | No | `None` | — | — |
| `mcp` | `McpConfig \| None` | No | `None` | — | — |
| `skills` | `SkillConfig \| None` | No | `None` | — | — |
| `telemetry` | `TelemetryConfig \| None` | No | `None` | — | — |
| `relay` | `RelayConfig \| dict[str, Any] \| None` | No | `None` | — | — |
| `tools` | `ToolsConfig \| None` | No | `None` | — | — |

---

### <kbd>property</kbd> extra_fields

Return fields preserved by the extension point for this model.

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


### <kbd>method</kbd> `add_mcp_server`

```python
def add_mcp_server(
    name: str,
    *,
    transport: str,
    url: str,
    exposure: Literal['harness_native', 'fabric_managed'] = 'harness_native',
    extra_fields: Mapping[str, Any] | None = None,
) -> Self
```

Add or replace a named MCP server and return this config.

---


### <kbd>method</kbd> `add_skill_path`

```python
def add_skill_path(path: str | Path) -> Self
```

Add a skill path and return this config.

---


### <kbd>method</kbd> `block_tools`

```python
def block_tools(*tools: str) -> Self
```

Block adapter-native tool names and return this config.

---


### <kbd>method</kbd> `enable_relay`

```python
def enable_relay(
    *,
    project: str | None = None,
    output_dir: str | Path | None = None,
    observability: RelayObservabilityConfig | Mapping[str, Any] | None = None,
    components: Sequence[RelayComponentConfig | Mapping[str, Any]] | None = None,
    policy: RelayConfigPolicy | Mapping[str, Any] | None = None,
) -> Self
```

Enable NeMo Relay telemetry and return this config.

---


### <kbd>classmethod</kbd> `from_mapping`

```python
def from_mapping(value: Mapping[str, Any]) -> Self
```

Validate the public agent config mapping shape.

---


### <kbd>method</kbd> `remove_mcp_server`

```python
def remove_mcp_server(name: str) -> Self
```

Remove a named MCP server and return this config.

---


### <kbd>method</kbd> `remove_skill_path`

```python
def remove_skill_path(path: str | Path) -> Self
```

Remove a skill path and return this config.

---


### <kbd>method</kbd> `to_mapping`

```python
def to_mapping() -> dict[str, Any]
```

Return a detached mapping matching the Rust ``FabricConfig`` schema.


---


## <kbd>class</kbd> `RunRequest`

One validated NeMo Fabric invocation request.



### Fields

The model defines the following fields:

| Field | Type | Required | Default | Constraints | Description |
| --- | --- | --- | --- | --- | --- |
| `input` | `Any` | No | `''` | — | — |
| `request_id` | `str` | No | `<generated>` | `MinLen(min_length=1)` | — |
| `context` | `dict[str, Any]` | No | `dict()` | — | — |
| `overrides` | `dict[str, Any] \| None` | No | `None` | — | — |

---

### <kbd>property</kbd> extra_fields

Return fields preserved by the extension point for this model.

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


### <kbd>classmethod</kbd> `from_mapping`

```python
def from_mapping(value: Mapping[str, Any]) -> Self
```

Validate a mapping using this Pydantic model.

---


### <kbd>method</kbd> `to_mapping`

```python
def to_mapping() -> dict[str, Any]
```

Return a detached request mapping for the Rust runtime.




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
