---
name: nemo-fabric-build-adapter
description: Build, migrate, review, and maintain third-party NVIDIA NeMo Fabric adapters against the public adapter contract. Use when creating an adapter package or fabric-adapter.json descriptor, mapping AgentConfig into an agent harness or custom-agent runtime, implementing start/invoke/stop, declaring schemas and capabilities, packaging adapter discovery metadata, or assessing adapter conformance. Do not use for consumer applications that only call the NVIDIA NeMo Fabric SDK.
---

# Build an NVIDIA NeMo Fabric Adapter

Build against the published southbound contract. Keep the adapter thin: let
NeMo Fabric own planning and consumer-facing behavior, and let the adapter own
only target translation and lifecycle state.

## Read the Contract

Read the current
[adapter contract](https://github.com/NVIDIA/NeMo-Fabric/tree/main/docs/adapter-contract)
before changing code. Always read the overview, descriptor, normalized
configuration, execution, results, registration, and conformance pages. Read
the custom-agents page when the target loads application-defined agents or
workflows.

Use the committed
[adapter-contract JSON Schemas](https://github.com/NVIDIA/NeMo-Fabric/tree/main/schemas/adapter-contract)
or the
schemas installed with the matching NeMo Fabric release for exact wire shapes.
Do not reconstruct a schema from examples or copy field lists into adapter
code.

## Establish the Boundary

Establish the adapter boundary before defining its descriptor:

1. Identify the adapter target and its stable harness ID.
2. Reuse one adapter across agents built for the same target. Do not create an
   adapter per custom agent.
3. List the normalized fields the target can actually enforce.
4. Separate target-wide `harness.settings`, per-agent `workflow.settings`, and
   typed `extensions`.
5. Keep installation, environment preparation, Relay orchestration, caller
   scheduling, and consumer result enrichment outside the adapter.

If the requested behavior cannot be expressed by the current contract, surface
the gap. Do not silently consume an unsupported northbound field or hide it in
an unrelated extension.

## Define the Descriptor First

Create one self-contained `fabric-adapter.json` before implementing target
translation:

- Set the current `contract_version`, a globally stable `adapter_id`, the
  target `harness`, `adapter_kind`, and runner binding.
- Set `config.input` to `agent_config` for a new adapter.
- Declare only normalized `config.accepts` fields the implementation enforces.
- Publish closed `settings_schema`, `model_schema`, `workflow_schema`,
  `tool_definition_schema`, and `extension_schemas` where applicable. Use
  `model_schema` only for static model/provider compatibility and model settings;
  keep credential validity and provider availability in startup validation.
- Declare runtime requirements and telemetry outputs without secret values.
- Leave optional capability flags false unless the installed NeMo Fabric runtime
  exposes and tests that adapter operation. Relay-backed ATOF streaming does
  not require adapter-native streaming.

Validate descriptor schemas without importing adapter code. Keep all schema
references local to the descriptor document; do not rely on HTTP or file
references.

## Package Discovery Metadata

Install the descriptor in the standard shared-data location. For setuptools:

```toml
[tool.setuptools.data-files]
"share/nemo-fabric/adapters/acme" = ["fabric-adapter.json"]
```

Depend on `nemo-fabric-adapter-contract` for typed standard-library dataclasses.
Install its optional `pydantic` extra only for Pydantic interoperability. Add
`nemo-fabric-adapters-common` only if the adapter chooses its lifecycle or
Relay helpers. A bare adapter package should not depend on the NeMo Fabric
runtime.

For a TypeScript adapter, depend on
`nemo-fabric-adapter-contract`. Import the negotiated descriptor,
configuration, and runtime-context types from the package root. Import request
and result types from `nemo-fabric-adapter-contract/preview` only when
working on the future typed invocation boundary. TypeScript types do not
validate data received from a process or network boundary; validate untrusted
values against the JSON Schemas included with the package.

## Map AgentConfig

Accept a validated `AgentConfig` and translate each declared field once at the
adapter boundary:

- Resolve named model roles into target-native model clients or settings.
- Apply normalized instructions and runtime limits only when declared.
- Convert MCP servers, tool definitions, tool policy, and skills into native
  target constructs.
- Resolve workflow entry points and construction settings during `start` in
  the task environment.
- Read identity, environment, artifacts, and telemetry from `RuntimeContext`,
  not from workflow settings.

Reject unsupported values with stable, safe error codes. Do not log complete
configs, environment values, headers, credentials, or arbitrary user input.

Use typed extension models and publish their schemas at the exact descriptor
extension point. Never treat `extensions` as an unchecked dictionary escape
hatch.

## Implement the Lifecycle

Implement exactly one `start`, zero or more ordered `invoke` operations, and
one `stop` for each NeMo Fabric runtime.

- Construct and retain target state in `start`.
- Translate one request and one terminal outcome in `invoke`.
- Make `stop` safe after partial startup and failed invocation.
- Isolate mutable state between independent runtimes.
- Do not add an adapter streaming method for Relay-backed
  `Runtime.invoke_stream()`; execute ordinary `invoke` and use the provided
  telemetry context.

For a Python adapter that opts into the common host:

```python
from nemo_fabric_adapter_contract.models import AgentConfig
from nemo_fabric_adapters.common import lifecycle


class TargetRuntime:
    async def start(self, payload):
        config: AgentConfig = payload["config"]
        ...

    async def invoke(self, payload):
        ...

    async def stop(self):
        ...


def main() -> None:
    lifecycle.serve(TargetRuntime, config_loader=AgentConfig.from_mapping)
```

Keep current host request/result conversion in dedicated functions. The
published `AgentRunRequest` and `AgentRunResult` types are preview-only and are
not part of the negotiated contract. Do not return `AgentRunResult` from the
current local host: it is treated as ordinary JSON, including `status: failed`.

## Handle Custom Agents

Use `workflow.entrypoint.kind` for resolution semantics and `ref` for the
factory identity. Bound supported combinations with `workflow_schema`.

- Map NeMo Fabric-defined `factory` intents to target-native factories.
- Resolve `python_entrypoint` only from the fixed `nemo_fabric.agents` group.
- Resolve `python_module` only as an importable module with `create_agent`.
- Never load a filesystem path from `ref`.
- Supply factories an adapter-defined build context containing already
  resolved native values; do not require custom agents to parse `FabricConfig`
  or `AgentConfig`.

## Validate Before Handoff

Complete these checks before handing off an adapter:

1. Install the built wheel in an isolated adapter environment.
2. Confirm discovery from `share/nemo-fabric/adapters` and inspect the resolved
   descriptor in `Fabric().plan(...)`.
3. Exercise one accepted normalized config and rejection for unsupported
   fields and each declared schema.
4. Run `doctor(...)` with both missing and satisfied requirements.
5. Test start, success, target failure, malformed output, repeated invocation,
   stop, partial-start cleanup, EOF cleanup, and two-runtime isolation.
6. Test Relay correlation if telemetry support is claimed.
7. Report the adapter package version, contract version, required-profile
   result, and every optional capability as supported or unsupported.

Do not claim automated NeMo Fabric conformance until the published conformance
suite exists and the exact adapter release passes it.
