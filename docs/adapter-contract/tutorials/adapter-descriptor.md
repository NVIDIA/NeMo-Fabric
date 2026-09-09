{/*
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
*/}

# Stage 1: Describe the Adapter

An Adapter Descriptor tells NVIDIA NeMo Fabric how to locate an adapter and
which contract surface it implements. NeMo Fabric reads and validates this
record during planning without importing or starting the adapter.

<Note>
Adapter Descriptor filenames end in `.fabric-adapter.json`.
</Note>

## Prerequisites

Before you start, complete the following:

1. Read the [Adapter Contract overview](../README.md) to understand where the
   descriptor sits in the six-stage authoring flow.
2. Choose the runtime binding your adapter uses: `python`, `process`, `http`, or
   `native_plugin`.
3. Keep the canonical
   [`adapter-descriptor.schema.json`](https://github.com/NVIDIA/NeMo-Fabric/blob/main/schemas/adapter-contract/adapter-descriptor.schema.json)
   open for exact fields, defaults, and constraints.

## Concepts Overview

The primary descriptor fields are:

| Field | Purpose |
| --- | --- |
| `contract_version` | Selects the complete negotiated adapter contract. Use `fabric.adapter/v1alpha2`. |
| `adapter_id` | Identifies the adapter implementation during selection and planning. |
| `adapter_kind` | Selects the runtime binding: `python`, `process`, `http`, or `native_plugin`. |
| `runner` | Supplies binding-specific startup metadata, such as a Python module. |
| `requirements` | Describes binaries, environment-variable names, files, services, or plugin hooks for diagnostics. |
| `config` | Declares the normalized fields the adapter applies and target-native files it generates. |
| `capabilities` | Declares optional runtime operations implemented through the adapter binding. |
| `telemetry` | Declares telemetry outputs and integration modes the adapter produces or forwards. |
| `target_types` | Declares registered target types a shared adapter can load. Omit it for a direct harness or dedicated-agent adapter. |

<Note title="Keep Claims Exact">
Declare only behavior implemented through the adapter boundary. A target's
native cancellation or streaming feature does not become a NeMo Fabric
capability until the adapter binding implements the corresponding contract.
Relay-backed ATOF streaming does not require `capabilities.streaming`; that
flag is reserved for the optional native OpenAI streaming operation.
</Note>

## To Implement the Adapter Descriptor

Work through each step in order, verifying your progress at each checkpoint.

### 1. Create a Minimum Descriptor

The following descriptor is enough to declare an in-process Python adapter
that accepts an empty `AgentConfig` and implements only the required lifecycle:

```json
{
  "contract_version": "fabric.adapter/v1alpha2",
  "adapter_id": "com.acme.fabric.example",
  "adapter_kind": "python",
  "runner": {
    "module": "acme_fabric_adapter.runtime"
  }
}
```

Use a globally stable `adapter_id`. Treat it as a machine identifier, not a
display name. Adapters receive `AgentConfig`; `FabricConfig` never crosses the
southbound boundary.

**Success Check**: Planning can read and validate the descriptor's metadata
without importing or starting the adapter.

### 2. Declare Accepted Configuration

Add a `config.accepts` value only after the implementation applies that field.
For example, the following adapter accepts named models, model endpoints,
system instructions, and a target-applied turn limit:

```json
"config": {
  "accepts": [
    "models",
    "models.base_url",
    "instructions.system",
    "runtime.max_turns"
  ],
  "system_instruction_modes": ["replace", "append"]
}
```

Accepting a parent does not automatically accept every optional child. For
example, `models` accepts the base model block, while
`models.temperature` and `models.base_url` are separate declarations. Use the
schema enum for the exact accepted values.

Planning rejects configured normalized behavior outside the declared surface.
NeMo Fabric does not silently remove unsupported fields.

When an adapter accepts `instructions.system`, declare the exact supported
modes in `config.system_instruction_modes`. `replace` discards the harness
default system instruction. `append` preserves the harness default and adds the
configured content after it. New descriptors must declare the supported modes
explicitly. For compatibility with descriptors created before mode discovery,
an omitted `system_instruction_modes` value means `replace` only.

Mode declarations must be nonempty, unique, and accompanied by
`instructions.system` in `config.accepts`. Planning rejects an unsupported
configured mode at `instructions.system.mode` before the adapter starts.

**Success Check**: A configured field outside `config.accepts` fails planning
instead of being silently ignored.

### 3. Add Adapter-Owned Schemas

Use an Adapter Descriptor schema for target-specific data that cannot be
validated by the normalized contract alone:

| Descriptor Field | Validates |
| --- | --- |
| `settings_schema` | `FabricConfig.harness.settings` for this adapter. |
| `model_schema` | Every configured model role, including provider compatibility and closed model settings. |
| `tool_definition_schema` | Every normalized named tool or tool-group definition. |
| `extension_schemas` | Adapter-owned data at named southbound extension points. |

The following closed settings schema permits one optional command timeout:

```json
"settings_schema": {
  "type": "object",
  "properties": {
    "command_timeout": {
      "type": "integer",
      "minimum": 1
    }
  },
  "additionalProperties": false
}
```

Use a closed object with no properties when the adapter accepts
`harness.settings` but has no settings. Omit the schema when the adapter does
not support that configuration surface.

Schemas must be valid, self-contained JSON Schema objects. NeMo Fabric does not
load arbitrary HTTP or file references during planning. Use
`additionalProperties: false` unless an intentionally open compatibility
surface is part of the adapter contract.

**Success Check**: Each declared schema is a self-contained JSON Schema object
with no external references.

### 4. Register Targets for a Shared Adapter

A shared framework adapter separates its static implementation descriptor from
the custom agents it can load. The Adapter Descriptor declares the supported
target type:

```json
{
  "contract_version": "fabric.adapter/v1alpha2",
  "adapter_id": "com.acme.fabric.framework",
  "adapter_kind": "python",
  "target_types": ["workflow"],
  "runner": {
    "module": "acme_framework_adapter.runtime"
  },
  "config": {
    "accepts": ["models", "tools.definitions"]
  }
}
```

Each separately installed workflow publishes one `*.fabric-target.json`
Adapter Target Descriptor. The target record selects its adapter, fixes the
adapter-scoped entry point, and validates that target's workflow settings:

```json
{
  "contract_version": "fabric.adapter/v1alpha2",
  "type": "workflow",
  "id": "com.acme.email-phishing",
  "adapter_id": "com.acme.fabric.framework",
  "spec": {
    "entrypoint": {
      "kind": "factory",
      "ref": "acme.agent.react"
    },
    "settings_schema": {
      "type": "object",
      "properties": {
        "llm_name": {
          "type": "string",
          "minLength": 1
        }
      },
      "required": ["llm_name"],
      "additionalProperties": false
    }
  }
}
```

The Adapter Target Descriptor uses the same `contract_version` as the Adapter
Descriptor. It does not introduce another contract or schema version.
`workflow.target_id` selects the target by `id`; consumer configuration does
not repeat its adapter-specific entry point.

Use the canonical
[`adapter-target-descriptor.schema.json`](https://github.com/NVIDIA/NeMo-Fabric/blob/main/schemas/adapter-contract/adapter-target-descriptor.schema.json)
for the complete target record.

**Success Check**: Planning resolves a `workflow.target_id` to exactly one
registered target that names this adapter.

## Summary

In this tutorial, you have:

- Created a minimum Adapter Descriptor that NeMo Fabric validates during
  planning without importing adapter code.
- Declared the normalized configuration surface your adapter applies through
  `config.accepts` and `system_instruction_modes`.
- Added adapter-owned JSON Schemas for target-specific data.
- Registered targets so a shared framework adapter can load separately
  installed custom agents.

## Next Steps

With the descriptor in place, continue through the adapter authoring stages:

<CardGroup cols={2}>

<Card title="Map the normalized configuration" href="normalized-configuration.md">

Continue to Stage 2 and implement only the fields listed in `config.accepts`.
</Card>

<Card title="Adapter Descriptor schema" href="https://github.com/NVIDIA/NeMo-Fabric/blob/main/schemas/adapter-contract/adapter-descriptor.schema.json">

Review the canonical schema for exact fields, defaults, and constraints.
</Card>

</CardGroup>
