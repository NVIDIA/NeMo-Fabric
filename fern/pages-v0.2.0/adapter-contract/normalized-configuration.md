{/*
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
*/}

# Normalized Configuration

Consumers author `FabricConfig`. After validation, adapter resolution, path
resolution, and capability planning, NVIDIA NeMo Fabric projects the applicable fields
into `AgentConfig`. Adapters consume this resolved southbound projection; they
do not need to understand consumer-only planning fields.

```text
FabricConfig + Adapter Descriptor + optional Adapter Target Descriptor
                              |
                              v
                  resolve | validate | project
                              |
                              v
                         AgentConfig
                              |
                              v
                    adapter translation
```

`FabricConfig` remains the source of truth for consumer authoring.
`AgentConfig` is the source of truth for adapter-facing fields and validation.

## AgentConfig Blocks

`AgentConfig` contains these adapter-facing blocks:

| Block | Adapter-facing purpose |
| --- | --- |
| `harness` | Target-specific settings selected with this adapter. |
| `models` | Named model roles with provider, model, credential-variable name, endpoint, temperature, and provider settings. |
| `instructions` | Portable instructions the adapter can apply. |
| `runtime` | Target behavior such as the maximum number of turns. |
| `skills` | Skill paths resolved for the task environment. |
| `mcp` | Named MCP servers, HTTP authentication metadata and custom headers, and effective per-server tool policy. |
| `tools` | Named tool or tool-group definitions plus effective selection and blocking policy. |
| `workflow` | Custom-agent or workflow entry point and construction settings. |
| `extensions` | Adapter-owned fields validated at a declared extension point. |

Use the generated
[`AgentConfig` JSON Schema](https://github.com/NVIDIA/NeMo-Fabric/blob/0.2.0-rc.1/schemas/adapter-contract/agent-config.schema.json)
for exact fields and constraints. Python adapters can import matching
dataclasses from `nemo_fabric_adapter_contract.models`. MCP authentication is
decoded as `McpOAuth2Config` or `McpServiceAccountConfig` before the adapter
receives `AgentMcpServerConfig`.

## Projection Rules

The descriptor controls the projection:

- Scalar normalized fields are included only when `config.accepts` declares
  that the adapter can apply them.
- Every configured model role is validated against `model_schema` when the
  selected descriptor publishes one.
- Resolved native skills and MCP servers come from the capability plan.
- An authenticated MCP server requires the adapter to declare
  `mcp.auth.oauth2` or `mcp.auth.service_account`, matching the configured
  authentication type.
- HTTP MCP authentication metadata and custom headers remain attached to each
  projected server. Authentication contains credential environment-variable
  names, not resolved secret values.
- `harness.settings` is validated against the Adapter Descriptor.
- `workflow.settings` is validated against the selected Adapter Target
  Descriptor. The target's entry point is projected into `AgentConfig` with
  those settings.
- Named tool definitions are validated individually against
  `tool_definition_schema` before they are projected.
- NeMo Fabric-owned metadata, installation policy, environment preparation,
  invocation timeout, artifacts, Relay configuration, and planning details do
  not become `AgentConfig` fields. Runtime-owned values arrive through
  `RuntimeContext` instead.

Unsupported configured behavior fails planning; it is not silently dropped.
An absent optional field preserves the adapter target's default where the
field's contract says so. For example, `tools.enabled: null` preserves the
target default, while an empty list explicitly selects no named tools.

`model_schema` uses the same self-contained JSON Schema vocabulary as the other
descriptor schemas. Use it for statically knowable provider compatibility and
closed `ModelConfig.settings` validation. Do not use it for credential checks,
provider reachability, or model availability; those remain startup concerns.

## Extensions

Every extensible southbound block has a named `extensions` map. An adapter
publishes its schema under the matching key in
`AdapterDescriptor.extension_schemas`, such as `model`, `mcp_server`,
`workflow`, or `run_result`.

Use extensions only when normalized fields cannot express the behavior:

1. Define a closed typed model for the adapter-owned data.
2. Publish its JSON Schema at the exact extension point.
3. Set the extension through the block's `set_extensions(...)` helper. Adapters
   using the optional Pydantic integration can supply a typed extension model.
4. Reject extension data when the descriptor does not declare that extension
   point or the value does not satisfy its schema.

Do not use `extensions` to bypass an unsupported normalized field. An extension
belongs to one adapter target and is not portable across adapters unless those
adapters intentionally implement the same namespaced contract.

## Validation Ownership

Planning validates static shape and compatibility without executing adapter
code. Adapter startup validates facts available only inside the task
environment, such as imports, installed factories, credentials, and live
service reachability. Startup errors must identify the failing field without
including secret values.

`AgentConfig` contains credential environment-variable names, not secret
values. Environment values can be present in `RuntimeContext.environment.env`;
do not persist or log the unredacted context.
