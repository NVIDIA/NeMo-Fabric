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
| `skills` | Declares the structured skill features the adapter implements: `skills.directories` loading, bundled-skill selection, and duplicate policies. |
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

Accept `runtime.keep_state` only when the adapter keeps all of its native state in `RuntimeContext.state_dir` when that field is present. Every runtime of the agent then receives the same folder, so a later runtime finds the earlier runtime's sessions, memory, and other harness files. On Linux and macOS, the adapter host inherits the folder's lock. Detached children with inherited lock descriptors can keep the state locked after the host exits and prevent later runtimes from starting. A detached child without the lock can outlive the host and access the state folder after another runtime acquires the lock. Ensure that all children stop accessing retained state before the lock is released.

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

### 3. Declare Structured Skill Support

An adapter that lists `skills` in `config.accepts` receives `skills.paths`.
The structured fields `skills.directories`, `skills.bundled`, and
`skills.duplicates` need a top-level `skills` block. Each sub-block is
optional. Leave out a sub-block when the adapter does not implement that
feature. Planning rejects a configured field that needs a missing sub-block
and names the field.

The following block declares support for all three features:

```json
"skills": {
  "directory_loading": {
    "supported_entries": ["skill", "folder"],
    "skill_file": "SKILL.md",
    "discovery": {
      "traversal": "recursive",
      "nested_skills": "stop_at_skill",
      "depth_limit": {"kind": "fixed", "levels": 6}
    },
    "parsing": {
      "required_frontmatter_fields": ["description"],
      "name_fallback": "directory_name",
      "frontmatter_errors": "reject"
    }
  },
  "bundled": {
    "default": "enabled",
    "supports_enable": true,
    "supports_disable": true,
    "supports_include": true,
    "supports_exclude": true
  },
  "duplicates": {
    "supported_policies": ["reject", "first_wins", "last_wins"],
    "native": "The harness keeps the first copy it finds.",
    "name_source": "frontmatter"
  }
}
```

The sub-blocks describe what the adapter implements, not settings a caller
chooses:

| Field | Meaning |
| --- | --- |
| `directory_loading.supported_entries` | The entry kinds the adapter loads: `skill`, `folder`, or both. Non-empty and unique. |
| `directory_loading.skill_file` | The file that marks a skill folder. Must be `SKILL.md`. |
| `directory_loading.discovery.traversal` | `immediate_children` searches only the folder's direct children. `recursive` searches every level. |
| `directory_loading.discovery.nested_skills` | `include` also finds skills inside a skill's folder. `stop_at_skill` stops at a skill's folder. |
| `directory_loading.discovery.depth_limit` | Required. `{"kind": "fixed", "levels": N}` with `N` at least 1, where the configured folder is depth 0. `{"kind": "no_fixed_limit"}` has no depth cap, but exclusions still apply. `{"kind": "adapter_defined"}` means the adapter documents its own bounds. |
| `directory_loading.parsing.required_frontmatter_fields` | Front-matter fields a skill must have to load. Can be empty. |
| `directory_loading.parsing.name_fallback` | `directory_name` uses the folder name when the front matter has no `name`. `none` skips the skill. |
| `directory_loading.parsing.frontmatter_errors` | `recover_key_value` recovers simple `key: value` lines. `reject` rejects a skill whose front matter does not parse. `adapter_defined` means the adapter documents its own handling. |
| `bundled.default` | What loads when `skills.bundled` is left out: `enabled`, `disabled`, or `none` when the harness ships no bundled skills. |
| `bundled.supports_enable`, `.supports_disable`, `.supports_include`, `.supports_exclude` | Which `skills.bundled` requests the adapter implements, including the empty-list meaning of `include` and `exclude`. All four are required. |
| `duplicates.supported_policies` | The `skills.duplicates` values the adapter implements. Non-empty and unique. |
| `duplicates.native` | Optional. A short description of what the harness does when `skills.duplicates` is left out. |
| `duplicates.name_source` | Optional. Whether names are compared by `frontmatter` name or by `directory` name. |

Every field in `directory_loading` is required, so a descriptor cannot claim
unlimited discovery by accident. Declare a value only after the adapter
implements it. Refer to
[Configure Skills](../../integrations/harness/skills.mdx) for the behavior
callers see.

**Success Check**: A configured `skills.directories`, `skills.bundled`, or
`skills.duplicates` value that the `skills` block does not cover fails
planning with the field path.

### 4. Add Adapter-Owned Schemas

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

### 5. Register Targets for a Shared Adapter

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
- Declared the structured skill features your adapter implements in the
  `skills` block.
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
