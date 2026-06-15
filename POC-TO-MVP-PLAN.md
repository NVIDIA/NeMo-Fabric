# POC To MVP Plan

This plan turns the current NeMo Fabric POC into an MVP that proves the contract
with Harbor first while keeping the API shape suitable for future Platform and
evaluation consumers.

## MVP Definition

The Fabric MVP lets consumers configure an agent once, vary it through profiles,
map unified capability config into harness-native forms, and run Hermes through
a stable Python SDK and CLI with artifacts and Relay telemetry.

The MVP is not a full production release for every harness. It is the smallest
complete slice that proves the Fabric contract is stable enough for Harbor and
future consumer work.

## Current POC Baseline

Already implemented:

- Rust core crate with typed config, profile resolution, adapter descriptors,
  run planning, lifecycle handles, normalized results, artifacts, and errors.
- CLI for `validate`, `inspect`, `plan`, `doctor`, and `run`.
- Python SDK package with native Rust bindings and explicit CLI fallback.
- Agent directory support with canonical `agent.yaml`, `profiles/`,
  `skills/`, and optional package-local `adapters/`.
- Ordered multi-profile application.
- Repository-maintained adapter descriptors plus package-local adapter
  discovery from `adapters/<adapter-name>/fabric-adapter.json`.
- Hermes SDK adapter path with Fabric-to-Hermes config mapping.
- Hermes CLI adapter path with Fabric-to-Hermes config mapping and one-shot
  invocation.
- Harbor consumer wrapper at `nemo_fabric.integrations.harbor:FabricAgent`.
- Capability routing into harness-native vs Fabric-managed buckets.
- Workspace patch/status artifact capture, including untracked files.
- Relay config pass-through and Hermes-native Relay smoke path.
- Test-only Hermes shim fixtures for dependency-free CLI/SDK and patch-artifact
  smokes.

## MVP Scope

### In Scope

- Stable Python SDK as the primary consumer API.
- CLI for local debugging, CI, and smoke tests.
- Versioned base config and profile config.
- Multiple profiles applied in caller-provided order.
- EffectiveConfig and RunPlan as the core resolved contract.
- Capability mapping for:
  - skills
  - tools
  - MCP
  - telemetry
- Harness-native capability generation where the adapter supports it.
- Clear rejection or explicit `fabric_managed` planning when Fabric cannot map
  a capability at runtime.
- Real Hermes adapter path.
- Minimal session lifecycle and cancellation semantics.
- Relay telemetry configuration and artifact discovery.
- Evaluation smoke integration with Harbor through the Python SDK.

### Out Of Scope For MVP

- Generic Fabric-managed MCP/tool proxy runtime for all harnesses.
- Full environment provisioning. Consumers provide the prepared environment.
- HTTP service surface.
- Production-grade support for every priority harness.
- Multi-modal input/output contracts.
- Installed third-party adapter package registry beyond package-local
  descriptors.

## Workstreams

The MVP has one serial bridge: get a minimal Hermes adapter working end-to-end
with Harbor. Once that is working, richer adapter work and richer consumer API
work can proceed in parallel.

## Repository Layout Boundary

Use repo layout to keep harnesses, adapters, and consumers distinct:

- `agents/` or agent package dirs: Fabric-managed agent harness configs.
- `adapters/`: harness adapters such as Hermes.
- `integrations/harbor/`: consumer integration glue for Harbor.
- `tests/integration/harbor/`: proof that the Harbor consumer integration works.

The Harbor vertical slice should live under `integrations/harbor/` and
`tests/integration/harbor/`, not under `examples/`, so it is not confused with a
Fabric-managed agent package.

### 1. Core Contract And Profile Resolution

Goal: make the Rust core contract stable enough for adapters and consumers to
depend on.

Tasks:

- Finalize public config structs for `agent.yaml`, profile files, and
  EffectiveConfig.
- Generate and commit JSON Schema for agent config, profile config, adapter
  descriptors, RunRequest, RunResult, and ArtifactManifest.
- Keep `harness.adapter_id` as the stable selector.
- Keep adapter descriptors as the source of harness family, adapter kind,
  supported runtime modes, supported transports, requirements, artifacts, and
  telemetry support.
- Add conformance tests for descriptor validation and profile stacking.
- Keep ordered multi-profile support.
- Add examples for varying:
  - harness: Hermes adapter variants
  - model: default model vs alternate model
  - MCP: native vs unsupported/managed planning
  - telemetry: disabled vs Relay enabled
  - runtime: one-shot vs session
- Tighten error messages for unknown adapter, invalid profile stack, unsupported
  transport, unsupported runtime mode, missing requirement, and invalid
  capability mapping.

MVP acceptance:

- A consumer can validate an agent package before running it.
- A consumer can resolve the same base config with different profile stacks.
- Adapter authors can tell from schema and tests what Fabric expects.

### 2. Minimal Hermes And Harbor Vertical Slice

Goal: prove the Fabric contract with the smallest working path: Hermes through
Fabric, invoked by Harbor, with a verifier checking the produced patch.

Tasks:

- Remove hard-coded local paths from the Hermes profile.
- Keep Hermes installation explicit:
  - preinstalled environment
  - image-provided environment
  - documented local dev venv
- Run Hermes through `FabricClient.run(...)` or the equivalent CLI path.
- Use Harbor `FabricAgent(BaseAgent)` wrapper.
- Run one Harbor SWE-Bench Verified task through Fabric.
- Run the Harbor verifier against the Fabric-produced patch.

MVP acceptance:

- Harbor can invoke Fabric without Hermes-specific launch code.
- A Fabric-produced patch is visible to Harbor's verifier.
- The run produces a result, artifact manifest, logs, and workspace patch/status
  artifacts.

### 3. Adapter Track

Goal: make the Hermes adapter real enough that it is not just a Harbor-specific
shim.

Tasks:

- Use current Hermes hooks and native Relay support.
- Convert Fabric model, skills, MCP, telemetry, workspace, and artifact config into
  Hermes-native configuration.
- Define the MVP capability set: `skills`, `tools`, `mcp`, and `telemetry`.
- Declare supported capability areas in the adapter descriptor.
- Preserve the capability plan shape: `native`, `managed`, and `routes`.
- Make unsupported runtime capability paths fail clearly unless explicitly
  planned as `fabric_managed`.
- Keep dependency-free shims in test fixtures and keep maintained adapters tied
  to real harness integration surfaces.
- Add a real Hermes run that produces inspectable output and Relay artifacts.
- Add minimal session support in the Hermes Python/native adapter. The
  process/CLI path can remain one-shot unless Hermes exposes a stable resumable
  CLI contract.
- Add best-effort cancellation:
  - adapter-native cancel if available
  - process termination for owned process runtimes
  - clear final status and artifacts

MVP acceptance:

- The same logical `skills`, `mcp`, and `telemetry` config can be resolved
  against Hermes adapter variants.
- The adapter output contains harness-native config artifacts or launch
  settings that reviewers can inspect.
- Unsupported mappings fail before invocation with actionable errors.
- Hermes can emit ATOF/ATIF through Relay when telemetry is enabled.
- The adapter does not require user-specific local paths.
- Consumers can start and stop a minimal Hermes session.
- Cancellation behavior is deterministic enough to test.

### 4. Consumer API Track

Goal: define a stable SDK/CLI surface that consumers can use while adapter
implementation continues to evolve underneath it.

Tasks:

- Keep the Python SDK as the primary consumer API.
- Keep the CLI as local debugging and executable documentation.
- Define SDK calls for:
  - load/validate an agent package
  - resolve profiles
  - plan a run
  - run one-shot
  - start/invoke/stop a minimal session
- Add SDK and CLI smoke tests that mirror README/design-doc snippets.
- Keep Harbor integration on the SDK path.
- Leave Platform integration as a follow-up open item.

MVP acceptance:

- Consumers can use the SDK for one-shot runs without depending on
  Hermes-specific code.
- Consumers can inspect EffectiveConfig, RunPlan, RunResult, and
  ArtifactManifest.
- CLI and SDK behavior match for profile resolution and one-shot execution.

### 5. Telemetry, Artifacts, And Release Readiness

Goal: make Relay configuration and artifact discovery first-class in the MVP.

Tasks:

- Keep Relay config fully inside Fabric config.
- Pass Relay config and metadata to adapters.
- Discover ATOF/ATIF artifacts from adapter output or known Relay output dirs.
- Preserve native harness telemetry behavior; do not reimplement Relay inside
  Fabric.
- Add tests for telemetry disabled/enabled profile variation.
- Add README examples for telemetry config.
- Keep README current as supported paths change.
- Add schema snapshot tests.
- Add final smoke checklist for CLI, SDK, real Hermes, Relay artifacts, and
  Harbor e2e.

MVP acceptance:

- Enabling telemetry in a profile results in Relay config being passed to the
  harness/adapter.
- Produced ATOF/ATIF files appear in the ArtifactManifest or telemetry
  reference.
- ArtifactManifest includes output, logs, patches, and telemetry references
  where available.
- README documents the supported happy path and known limitations.

## Execution Order

1. Harden the core contract and schema snapshots.
2. Build the minimal Hermes plus Harbor vertical slice.
3. Run adapter track and consumer API track in parallel.
4. Harden Relay telemetry and artifact discovery for Hermes.
5. Add final schema, smoke, README, and TODO cleanup.

After steps 1 and 2 are complete, steps 3, 4, and 5 can proceed in parallel
with light coordination around SDK return types, ArtifactManifest, and Relay
artifact discovery.

## MVP Review Checklist

Before calling the MVP complete, verify:

- `cargo test` passes.
- `cargo check -p fabric-python` passes.
- CLI smoke passes.
- SDK smoke passes.
- Real Hermes smoke passes in a documented clean environment.
- Relay smoke produces ATOF/ATIF artifacts.
- Profile stacking is tested through SDK and CLI.
- Capability mapping is tested for Hermes adapter variants.
- ArtifactManifest includes output, logs, patches, and telemetry references where
  available.
- README documents the supported happy path and the known limitations.

## Open Decisions

- What exact Platform smoke path is the first follow-up integration target.

## Recommended Next Steps

1. Finish the core contract work: schema generation, schema snapshot tests,
   adapter descriptor validation, and profile stacking tests.
   Status: mostly complete. Schema generation, schema snapshots, descriptor
   validation, and profile stacking tests are in place. Remaining work should
   be limited to review-driven tightening.
2. Make the minimal Hermes path reproducible: remove local-path assumptions and
   document explicit install modes.
   Status: in progress. Hermes SDK and CLI adapter paths exist, the main
   example path no longer depends on a local Gym Hermes venv, and install/run
   expectations are documented. Next pickup item is a real installed-Hermes CLI
   smoke.
3. Map Fabric capabilities into Hermes-native config for model, workspace,
   skills, MCP, tools, telemetry, and artifacts.
   Status: partially complete. Current mapping materializes Hermes config for
   model, workspace, skills, MCP, toolsets, telemetry hooks, and artifacts.
   Next pickup item is validating the mapping against real Hermes SDK and CLI
   behavior and closing any gaps.
4. Add the Fabric-owned Harbor consumer shim under `integrations/harbor/`.
5. Add `tests/integration/harbor/` with a Harbor SWE-Bench Verified smoke and
   verifier check.
6. After the Harbor vertical slice works, run the adapter track, consumer API
   track, and telemetry/artifact readiness track in parallel.
