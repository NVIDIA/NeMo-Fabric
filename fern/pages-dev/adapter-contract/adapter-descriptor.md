{/*
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
*/}

# Adapter Descriptors

NVIDIA NeMo Fabric resolves two independent records without importing adapter
code:

- An **Adapter Descriptor** describes one adapter implementation: how to start
  it, which normalized configuration it accepts, and which runtime operations
  it supports.
- An **Adapter Target Descriptor** describes one registered target implemented
  by an adapter. A workflow target supplies its entry point and the schema for
  its per-workflow settings.

Both records use the same `contract_version`. They do not have independent
package, schema, or target versions.

## Adapter Descriptor

Adapter Descriptor filenames end in `.fabric-adapter.json`.

| Field | Description |
| --- | --- |
| `contract_version` | Adapter Contract version. Use `fabric.adapter/v1alpha2`. |
| `adapter_id` | Globally stable adapter implementation ID. |
| `adapter_kind` | Runtime binding: `python`, `process`, `http`, or `native_plugin`. |
| `runner` | Binding-specific launch metadata, such as a Python module. |
| `target_types` | Target types this adapter can load, such as `workflow`. Omit for a harness-only adapter. |
| `requirements` | Binaries, environment-variable names, files, services, or plugin hooks checked by diagnostics. |
| `config` | Southbound input and normalized fields the adapter accepts. |
| `capabilities` | Optional adapter APIs implemented through the selected runtime binding. |
| `telemetry` | Telemetry outputs and integration modes implemented by the adapter. |

New adapters use `config.input: agent_config`. `config.accepts` declares only
normalized fields the implementation can apply. NeMo Fabric rejects configured
behavior outside that surface; it does not silently drop it.

`config.accepts` lists normalized fields the adapter can enforce. Planning
rejects configured fields that the adapter cannot apply. The current values
include models, model endpoint and temperature, system instructions, turn
limit, enabled/blocked tools, named tool definitions, MCP, MCP authentication
modes, MCP filters, and skills. Refer to the
[`AdapterConfigField` schema](https://github.com/NVIDIA/NeMo-Fabric/blob/main/schemas/adapter-contract/adapter-descriptor.schema.json)
for exact wire values.

## Descriptor Schemas

Adapter-owned schemas are static:

| Field | Validates |
| --- | --- |
| `settings_schema` | `FabricConfig.harness.settings` |
| `model_schema` | Every configured model role |
| `tool_definition_schema` | Every normalized named tool definition |
| `extension_schemas` | Named southbound extension points |

## Minimal Python Descriptor

The following is a complete minimal descriptor for an in-process Python adapter:

```json
{
  "contract_version": "fabric.adapter/v1alpha2",
  "adapter_id": "com.acme.fabric.example",
  "adapter_kind": "python",
  "runner": {"module": "acme_fabric_adapter.adapter"},
  "settings_schema": {
    "type": "object",
    "properties": {},
    "additionalProperties": false
  },
  "config": {
    "input": "agent_config",
    "accepts": ["models", "instructions.system"]
  }
}
```

## Adapter Target Descriptor

Adapter Target Descriptor filenames end in `.fabric-target.json`. Targets are
registered independently so a shared adapter can load workflows installed by
other packages.

| Field | Description |
| --- | --- |
| `contract_version` | The same Adapter Contract version used by the adapter. |
| `id` | Globally stable target ID selected by consumer configuration. |
| `adapter_id` | Adapter that implements the target. |
| `type` | Target semantics. The current contract supports `workflow`. |
| `spec` | Type-specific resolution and validation metadata. |

A workflow target owns the entry point. `FabricConfig` selects the target by
ID and supplies settings; it does not repeat adapter-specific entry-point
semantics.

```json
{
  "contract_version": "fabric.adapter/v1alpha2",
  "id": "com.acme.email-phishing",
  "adapter_id": "nvidia.fabric.nat",
  "type": "workflow",
  "spec": {
    "entrypoint": {
      "kind": "factory",
      "ref": "fabric.agent.react"
    },
    "settings_schema": {
      "type": "object",
      "properties": {
        "llm_name": {"type": "string", "minLength": 1}
      },
      "required": ["llm_name"],
      "additionalProperties": false
    }
  }
}
```

Planning resolves the target first, obtains its `adapter_id`, resolves the
Adapter Descriptor, validates both records, and projects the target entry point
plus consumer settings into `AgentConfig.workflow`.

## Schema Rules

Descriptor schemas must be valid, self-contained JSON Schema objects. NeMo
Fabric does not load HTTP or file references. Object-valued configuration
schemas must accept an object root. Prefer `additionalProperties: false` so
typos and stale settings fail during planning.

Do not advertise an optional capability merely because the underlying target
supports it. Advertise only behavior implemented through the adapter boundary.
Relay-backed ATOF streaming is NeMo Fabric-owned and does not require native
adapter streaming.

See [Registration and Discovery](registration-and-discovery.md) for package and
lookup rules and [Normalized Configuration](normalized-configuration.md) for
projection semantics.
