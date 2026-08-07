{/*
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
*/}

# NVIDIA NeMo Fabric Adapter Descriptor

`fabric-adapter.json` makes an adapter discoverable and validates its public
surface without importing or executing adapter code. The selected descriptor
is included in `RunPlan` and is authoritative for the adapter's runner,
configuration support, schemas, requirements, telemetry, and capabilities.

## Core Fields

| Field | Description |
| --- | --- |
| `contract_version` | Adapter contract implemented by the package. Use `fabric.adapter/v1alpha1`. |
| `adapter_id` | Globally stable implementation ID, normally a reverse-domain name. |
| `harness` | Stable machine-readable adapter-target ID. |
| `adapter_kind` | Execution binding: `python`, `process`, `http`, or `native_plugin`. Python and process hosts are implemented locally today. |
| `runner` | Binding-specific launch metadata, such as a Python `module`. |
| `requirements` | Binaries, environment-variable names, files, services, or plugin hooks checked during diagnostics. |
| `config` | Southbound input mode and normalized fields the adapter accepts. |
| `capabilities` | Runtime operations claimed by the adapter. Fabric intersects these claims with the selected runtime implementation. |
| `telemetry` | Provider-specific outputs and integration modes the adapter supports. |

Use `config.input: agent_config` for the typed southbound boundary. Omitting
`config.input` selects the legacy `fabric_config` input and should be limited
to adapters that have not migrated.

`config.accepts` lists normalized fields the adapter can enforce. Planning
rejects configured fields that the adapter cannot apply. The current values
include models, model endpoint and temperature, system instructions, turn
limit, enabled/blocked tools, named tool definitions, MCP and MCP filters, and
skills. Refer to the
[`AdapterConfigField` schema](https://github.com/NVIDIA/NeMo-Fabric/blob/main/schemas/adapter-contract/adapter-descriptor.schema.json)
for exact wire values.

## Descriptor Schemas

| Field | Validates |
| --- | --- |
| `settings_schema` | `FabricConfig.harness.settings` |
| `workflow_schema` | The complete `FabricConfig.workflow` block |
| `tool_definition_schema` | Every entry in `FabricConfig.tools.definitions` |
| `extension_schemas` | Named `extensions` maps at southbound extension points |

Schemas must be valid, self-contained JSON Schema objects. NeMo Fabric does
not load HTTP or file references from a descriptor. Object-valued configuration
schemas must accept an object root. Prefer closed schemas with
`additionalProperties: false`; use an open object only where arbitrary target
settings are an intentional compatibility surface.

If an adapter does not support settings, publish a closed empty
`settings_schema`. A configured workflow fails planning when the descriptor
does not publish `workflow_schema`. Named tool definitions similarly fail when
`tool_definition_schema` is absent.

## Minimal Python Descriptor

The following example declares the minimum Python adapter metadata:

```json
{
  "contract_version": "fabric.adapter/v1alpha1",
  "adapter_id": "com.acme.fabric.example",
  "harness": "example",
  "adapter_kind": "python",
  "runner": {
    "module": "acme_fabric_adapter.adapter"
  },
  "settings_schema": {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {},
    "additionalProperties": false
  },
  "requirements": {},
  "config": {
    "input": "agent_config",
    "accepts": ["models", "instructions.system"]
  },
  "capabilities": {
    "service": false,
    "streaming": false,
    "updates": false,
    "cancellation": false
  }
}
```

Do not advertise an optional capability merely because the underlying target
supports it. Advertise only behavior implemented through the current NeMo Fabric
runtime binding. Relay-backed ATOF streaming is NeMo Fabric-provided and does not
require `capabilities.streaming`.

See [Registration and Discovery](registration-and-discovery.md) for packaging
the descriptor and [Normalized Configuration](normalized-configuration.md)
for schema ownership.
