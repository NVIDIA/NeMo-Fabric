<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Fabric MVP Plan

This plan turns the current NeMo Fabric codebase into a focused MVP. The MVP
proves the stable Fabric API surface first: configure an agent once, vary it
through profiles, map Fabric capabilities into Hermes, run through Fabric, and
return normalized results through SDK and CLI surfaces that can be consumed by
NeMo Platform and other orchestrators.

## MVP Goal

The MVP should prove that Fabric can be the harness-management layer between
consumer systems and agent harness runtimes.

For the MVP:

- Hermes is the first MVP harness target. Codex and additional priority
  harnesses follow in later milestones.
- Python SDK is the primary integration surface for Platform and other
  orchestrators.
- CLI is the executable surface for local debugging, CI, and integration
  testing.
- Harbor is the first proof consumer, not the center of the MVP contract.
- `agent.yaml` and profile files are the portable file format.
- typed Python/Pydantic-style config is the preferred path for real consumers
  that already own a top-level job or deployment config.

## MVP Slice

The MVP slice is:

```text
Consumer job/deployment -> Fabric SDK or CLI -> Hermes adapter -> Hermes runtime
                                           |
                                           v
                             RunResult + ArtifactManifest + Relay refs
```

The consumer owns job scheduling, environment preparation, task semantics, and
domain-specific verification. Fabric owns agent config/profile resolution,
harness invocation, capability mapping, normalized results, artifact discovery,
and telemetry pass-through.

Harbor validates this shape as a proof consumer: Harbor owns task
materialization, environment lifecycle, verifier execution, reward calculation,
and benchmark job layout, while Fabric owns the selected harness invocation.

## Current Baseline

The repo already contains the core shape of the MVP:

- Rust core crate with typed config, profile resolution, adapter descriptors,
  run planning, runtime handles, normalized results, artifacts, and errors.
- JSON Schema generation and committed schema snapshots.
- CLI commands for validation, inspection, planning, doctor checks, schema
  generation, and running.
- Python package with native Rust bindings plus CLI fallback.
- SDK support for both agent-package paths and typed/in-memory config.
- Session-mode SDK lifecycle support with a stable `session_id` resume key for
  both agent-package paths and typed/in-memory config.
- Agent package examples with `agent.yaml`, `profiles/`, `skills/`, and
  workspace fixtures.
- Ordered multi-profile resolution.
- Repository-maintained Hermes SDK and Hermes CLI adapters.
- Package-local adapter descriptor discovery for custom agent packages.
- Hermes capability mapping for model, workspace, skills, MCP, tools,
  telemetry hooks, and artifacts.
- Harbor proof wrapper at `nemo_fabric.integrations.harbor:FabricAgent`.
- Workspace patch/status artifact capture.
- Relay config pass-through and a Hermes Relay smoke path.

## In Scope

- Stable SDK and CLI behavior for Hermes-backed one-shot runs.
- Minimal session lifecycle shape where Hermes support is available.
- EffectiveConfig and RunPlan as the resolved core contract.
- Multiple profiles applied in caller-provided order.
- A reusable config-variation test matrix that proves the same agent can vary
  model, runtime, skills, tools, MCP, telemetry, workspace, artifacts, and
  harness adapter through profiles.
- Capability mapping for:
  - skills
  - tools
  - MCP
  - telemetry
- Harness-native config generation for supported Hermes surfaces.
- Clear validation failures for unsupported adapters, requirements, and
  capability mappings.
- Relay telemetry configuration pass-through.
- ArtifactManifest entries for output, logs, patches, and telemetry references
  where available.
- Consumer integration smoke with a Fabric-managed Hermes run.

## In Scope, Deferred Until Base MVP Is Stable

- Final adapter contract definition.
- Third-party adapter support. The base MVP supports built-in adapters only.
- Harbor SWE-Bench Verified smoke with verifier as the first evaluation proof
  once the environment is available.
- Harbor integration beyond the proof already included in the POC.

## Out Of Scope

- Completing all priority harnesses in the MVP cut. Codex, Claude Code, Cursor,
  OpenClaw, and Deep Agents are planned follow-on harnesses after the Hermes
  slice is stable.
- Generic Fabric-managed MCP/tool proxy runtime.
- Full environment provisioning. Consumers provide prepared environments.
- Production Platform integration.
- External third-party adapter package registry.
- Multi-modal input/output contracts.

## Repository Layout

The repo layout separates Fabric-owned concepts:

- `crates/fabric-core/`: config, schema, planning, runtime contract, and core
  types.
- `crates/fabric-cli/`: `fabric` command-line surface.
- `crates/fabric-python/`: native Python bindings for the Rust core.
- `python/src/nemo_fabric/`: Python SDK and consumer integrations.
- `adapters/hermes-sdk/`: Hermes SDK adapter implementation; this is the
  primary inline Python path for SDK consumers.
- `adapters/hermes-cli/`: Hermes CLI adapter implementation.
- `integrations/harbor/`: Harbor consumer integration notes.
- `examples/`: portable agent packages and config examples.
- `tests/`: CLI, adapter, Relay, local e2e, and SWE-Bench-style smokes.
- `python/tests/`: SDK and Harbor integration smokes.
- `schemas/`: committed schema snapshots.

## Workstreams

### 1. Core Contract

Status:

- Base MVP contract is mostly complete.
- Schema snapshots, profile resolution, ordered profile stacking, typed SDK
  config, YAML package config, adapter descriptor validation, planning, doctor
  checks, and CLI/SDK smoke coverage are already present.
- Consumers can validate and plan without running.
- The same base config can be resolved with different ordered profile stacks.
- Adapters receive EffectiveConfig/RunPlan, not raw profile files.
- The full adapter contract definition is deferred until the base MVP is
  stable; the base MVP keeps only the minimal descriptor fields Fabric already
  uses.
- Normalized trajectory structures and policy hooks for auditability are
  deferred until Fabric owns those contracts directly.

How to maintain:

- Keep schema snapshots current as the contract evolves.
- Tighten error messages where review or smoke tests show ambiguity.
- Keep SDK typed-config behavior and YAML package behavior aligned when new
  config fields are added.
- Add or update tests whenever the config contract changes.

### 2. Hermes Adapter Readiness

Status:

- Base Hermes SDK and Hermes CLI adapter work is in place.
- `hermes-sdk` is the inline Python adapter path for SDK consumers. The Python
  SDK imports the adapter callable directly and preserves the async SDK shape.
- `hermes-cli` is the process-backed path for CLI/debug and environment-backed
  consumers. Fabric launches the wrapper process and captures stdout, stderr,
  exit status, logs, and artifacts.
- The `hermes-cli` process path now gets a per-invocation `FABRIC_HOME` and
  `FABRIC_INVOCATION` file. The launcher reads that invocation file, maps Fabric
  config into Hermes-native config, and then invokes the real `hermes` CLI.
- Both paths return the normalized Fabric `RunResult` shape.
- Fabric model, workspace, skills, MCP, tools, telemetry, and artifact config
  remains visible in generated Hermes-native config or launch settings.
- Unsupported Hermes MCP mappings with no target fail before invocation.
- Session-mode adapters receive Fabric's stable session key from
  `runtime_context.session_id` when supplied, or `runtime_context.runtime_id`
  as the default. Hermes CLI maps that Fabric key onto Hermes session id/title
  for resume.
- Relay-backed Hermes CLI tests now cover ATOF/ATIF artifact references and
  generated Relay config in the process-backed path.
- SDK and CLI smoke coverage asserts normalized `RunResult` parity for shared
  fields across both inline Python and process-backed adapter paths.

Next steps:

- Review the Hermes adapter implementations for maintainability and alignment
  with the minimal descriptor fields Fabric currently uses.
- Test Hermes SDK and CLI paths with more representative inputs.
- Add testing for harness-native events, artifacts, and logs.

### 3. Config Variation Matrix

Status:

- Ordered profile stacking is implemented.
- `examples/code-review-agent` includes Hermes SDK, Hermes CLI, local env, MCP,
  and Relay-oriented profile examples.
- CLI and SDK smoke tests cover profile resolution and multi-profile planning.
- Hermes capability mapping exists for model, workspace, skills, MCP, tools,
  telemetry, and artifacts.
- Generated Hermes config checks confirm enabled skills, tools, MCP, telemetry,
  workspace, and artifact settings.
- Negative tests cover unsupported mappings failing before invocation.
- Relay-enabled Hermes CLI runs now assert emitted ATOF/ATIF artifact files and
  manifest visibility where the harness and adapter support it.

Next steps:

- Turn the example profiles into an explicit variation matrix for Hermes.
- Add missing profile variations where useful, including alternate model,
  toolset, workspace, artifact, and telemetry combinations.
- Test both Hermes SDK and Hermes CLI against the applicable matrix.
- Add checks for harness-native events, artifacts, and logs.

Config mapping and actual runtime behavior are related but not identical.
Fabric should prove that capability config is mapped into the harness-native
surface, and trajectory tests should prove whether the harness actually exposed
or used that capability during a run.

After the Hermes matrix is stable, each new harness should reuse the same
example shape while keeping the base `agent.yaml` stable.

### 4. SDK And CLI API

Status:

- Base Python SDK and CLI surfaces are in place.
- SDK supports agent-package paths and typed/in-memory config.
- CLI supports validate, inspect, plan, doctor, schema generation, and run.
- SDK session APIs cover `start`, `start_config`, `invoke`, `stream`, `cancel`,
  and `stop` for `runtime.mode: session`, including caller-provided
  `session_id` propagation.
- CLI includes `fabric chat` for local interactive session-mode debugging with
  explicit `--session-id`, `/info`, `/verbose`, and oneshot-profile rejection.
- SDK and CLI can plan and run Hermes without callers importing
  Hermes-specific code.
- CLI and SDK smoke tests cover core planning and run paths.
- README examples for plan, doctor, typed config, SDK sessions, and CLI chat are
  mirrored by executable smoke tests to prevent documented API drift.
- Typed config is a first-class SDK path and is covered without requiring an
  agent directory.
- The core SDK is covered as consumer-neutral and dependency-free; Harbor,
  Hermes, Relay, and adapter packages stay out of a plain `import nemo_fabric`.

Next steps:

- Define an SDK API doc that flushes out the APIs and request/response schema
  for each API.
- Keep Python SDK as the primary API for consumers.
- Keep CLI behavior aligned with SDK behavior for the same config/profile stack.

### 5. Telemetry And Artifacts

Status:

- Base artifact capture is in place for output, logs, generated harness config,
  workspace patch/status, and telemetry references where available.
- Relay config pass-through exists for Hermes profiles.
- Native harness outputs are preserved separately from Relay outputs.
- SDK, CLI, and Harbor-facing paths expose ArtifactManifest data.
- Relay artifact discovery is hardened for ATOF/ATIF outputs when telemetry is
  enabled.
- Relay-enabled profiles have tests for inspectable telemetry outputs or clear
  telemetry references.
- ArtifactManifest remains populated with output, logs, patch/status, native
  harness artifacts, and telemetry references where available.
- Relay-disabled smoke coverage verifies native output and native observability
  stay available without Relay.
- SDK, CLI, and Harbor-facing smoke paths cover ArtifactManifest visibility.

### 6. Consumer Proof: Harbor

Status: optional/stretch goal.

Goal: validate the SDK/CLI contract through one real evaluation consumer after
the SDK/CLI and Hermes paths are stable.

Current status:

- Keep `nemo_fabric.integrations.harbor:FabricAgent` as the Harbor entrypoint.
- Keep Harbor-specific usage in `integrations/harbor/README.md`.
- Lightweight Harbor integration smoke coverage validates command construction,
  Fabric metadata propagation, and ArtifactManifest handoff with a fake Harbor
  environment.

Next steps:

- Run the lightweight Harbor smoke in a clean environment.
- Run one Harbor SWE-Bench Verified task through Fabric.
- Run the Harbor verifier against the Fabric-produced patch.

Success criteria:

- Harbor can invoke Fabric without Hermes-specific launch code.
- Fabric result metadata is copied into Harbor context metadata.
- The Fabric-produced patch is visible to Harbor's verifier.
- Harbor remains responsible for datasets, environments, verifier, and rewards.
- No Harbor-specific assumption leaks into Fabric core, SDK, or Hermes adapters.

## Execution Order

1. Keep core contract/schema tests green while making small contract fixes.
2. Done: SDK and CLI behavior for typed config and agent-package config.
3. In progress: finish Hermes SDK and CLI reproducibility in clean environments.
4. Run the Hermes config-variation matrix across model, runtime, skills, tools,
   MCP, telemetry, workspace, artifacts, and harness adapter profiles.
5. Done: harden Relay telemetry and ArtifactManifest discovery for Hermes runs.
6. After the SDK/CLI and Hermes path are stable, split follow-up work into
   adapter, consumer API, and telemetry/artifact readiness tracks.
7. Stretch: run the Harbor lightweight smoke from a clean install.
8. Stretch: run a Harbor SWE-Bench Verified smoke and verifier path as the
   first evaluation proof.

## Review Checklist

Before calling the MVP complete:

- `cargo test --workspace` passes.
- `cargo fmt --check` passes.
- Python SDK smoke passes.
- CLI smoke passes.
- CLI chat smoke passes for session-mode profiles.
- real Hermes SDK smoke passes in a documented clean environment.
- real Hermes CLI smoke passes in a documented clean environment.
- Hermes config-variation matrix passes for supported profile combinations.
- typed config SDK smoke passes without requiring an agent directory.
- Harbor lightweight smoke passes.
- Harbor SWE-Bench Verified smoke runs through Fabric.
- Relay-enabled run produces ATOF/ATIF outputs or telemetry references.
- ArtifactManifest includes output, logs, patches, and telemetry references
  where available.
- README and integration docs describe only supported paths.

## Open Decisions

- What exact Platform smoke path should validate SDK consumption.
- Which SDK calls must be async in the first MVP cut versus immediately after.
