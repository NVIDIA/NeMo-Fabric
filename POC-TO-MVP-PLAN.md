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
- Dependency-free local environment, CLI, SDK, config-mapping, and Harbor smoke
  tests.

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
- Clear validation failures for unsupported runtime modes, transports,
  adapters, requirements, and capability mappings.
- Relay telemetry configuration pass-through.
- ArtifactManifest entries for output, logs, patches, and telemetry references
  where available.
- Consumer integration smoke with a Fabric-managed Hermes run.
- Harbor SWE-Bench Verified smoke with verifier as the first evaluation proof
  once the environment is available.

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
- `adapters/hermes-sdk/`: Hermes SDK adapter implementation.
- `adapters/hermes-cli/`: Hermes CLI adapter implementation.
- `integrations/harbor/`: Harbor consumer integration notes.
- `examples/`: portable agent packages and config examples.
- `tests/`: CLI, adapter, Relay, local e2e, and SWE-Bench-style smokes.
- `python/tests/`: SDK and Harbor integration smokes.
- `schemas/`: committed schema snapshots.

## Workstreams

### 1. Core Contract

Goal: make the Rust core contract stable enough for adapter and consumer work.

Remaining work:

- Keep schema snapshots current.
- Tighten error messages for invalid profile stacks, unsupported capability
  mappings, missing requirements, and unknown adapters.
- Keep SDK typed-config behavior and YAML package behavior aligned.
- Add or update tests whenever the config contract changes.

Acceptance:

- Consumers can validate and plan without running.
- The same base config can be resolved with different ordered profile stacks.
- Adapters receive EffectiveConfig/RunPlan, not raw profile files.

### 2. Hermes Adapter Readiness

Goal: make Hermes adapter behavior reproducible and inspectable.

Remaining work:

- Validate Hermes SDK adapter behavior against a clean installed Hermes
  environment.
- Validate Hermes CLI adapter behavior against a clean installed Hermes CLI
  environment.
- Remove any remaining local path assumptions.
- Keep install modes explicit:
  - preinstalled
  - image-provided
  - local development venv
- Persist adapter-generated Hermes config artifacts where useful for review.
- Keep dependency-free test shims in fixtures, not as product adapters.

Acceptance:

- Hermes SDK and CLI paths can run from documented clean environments.
- Fabric model, workspace, skills, MCP, tools, telemetry, and artifact config
  are visible in generated Hermes-native config or launch settings.
- Unsupported mappings fail before invocation with actionable errors.

### 3. Config Variation Matrix

Goal: prove Fabric profiles can vary the same logical agent across capabilities
and harnesses without rewriting the base agent package.

Hermes is the first target for the full variation matrix. The example
`examples/code-review-agent` should test:

- harness adapter variation: Hermes SDK and Hermes CLI profiles both resolve
  and run where supported;
- model variation: default model and alternate model profiles map into
  Hermes-native config;
- runtime variation: one-shot/CLI and session/library paths are planned where
  the selected adapter supports them;
- workspace and artifact variation: profile-specific workspace and artifact
  locations are respected;
- skills variation: base skill directories and profile-added skill directories
  map into Hermes config;
- tools variation: command allowlists or toolsets map into Hermes-native tool
  config, or fail clearly if unsupported;
- MCP variation: MCP server definitions map into Hermes-native MCP config when
  supported;
- telemetry variation: Relay-disabled and Relay-enabled profiles produce the
  expected adapter config and ArtifactManifest references;
- output variation: stdout/stderr logs, generated harness config, patch/status,
  and telemetry references are captured where available.

After the Hermes matrix is stable, each new harness should reuse the same
example shape:

- add a harness profile for the new adapter, such as Codex or Claude Code;
- add only harness-specific profile fields where the adapter requires them;
- run `plan` and `doctor` across the shared variation profiles;
- run the supported execution subset for that harness;
- require unsupported variations to fail during planning or doctor checks with
  actionable errors.

Acceptance:

- The base example agent remains stable while profiles vary harness and
  capabilities.
- Hermes SDK and Hermes CLI pass the full applicable variation matrix.
- Follow-on harnesses can be added by contributing profiles and adapter tests
  without changing the base example contract.

### 4. SDK And CLI API

Goal: give Platform and other consumers a stable surface while adapter
implementation evolves.

Remaining work:

- Keep Python SDK as the primary API.
- Support SDK calls from typed config, agent directory, or single config file.
- Keep CLI behavior aligned with SDK behavior.
- Keep plan/doctor/run examples in the README accurate.
- Decide which async SDK methods are required for MVP versus follow-up.
- Keep the API independent of Harbor-specific concepts.

Acceptance:

- A consumer can plan and run Hermes without importing Hermes-specific code.
- A consumer can inspect EffectiveConfig, RunPlan, RunResult, and
  ArtifactManifest.
- CLI and SDK produce equivalent plans for the same config/profile stack.
- Platform can construct the Fabric agent slice from its own job/deployment
  config without materializing an agent directory.

### 5. Telemetry And Artifacts

Goal: make telemetry and artifacts reviewable in the MVP path.

Remaining work:

- Pass Relay config and metadata through to Hermes adapters.
- Discover Relay ATOF/ATIF outputs when telemetry is enabled.
- Preserve native harness outputs without forcing Relay to replace them.
- Capture stdout/stderr logs as artifacts for process-backed runs.
- Keep ArtifactManifest populated with output, logs, patch/status, and
  telemetry references where available.

Acceptance:

- Enabling Relay in a profile produces inspectable telemetry outputs or clear
  telemetry references.
- Disabling Relay still produces useful native output and logs.
- Artifacts are visible to SDK, CLI, and Harbor consumers.

### 6. Consumer Proof: Harbor

Goal: validate the SDK/CLI contract through one real evaluation consumer.

Remaining work:

- Keep `nemo_fabric.integrations.harbor:FabricAgent` as the Harbor entrypoint.
- Keep Harbor-specific usage in `integrations/harbor/README.md`.
- Run the lightweight Harbor smoke in a clean environment.
- Run one Harbor SWE-Bench Verified task through Fabric.
- Run the Harbor verifier against the Fabric-produced patch.

Acceptance:

- Harbor can invoke Fabric without Hermes-specific launch code.
- Fabric result metadata is copied into Harbor context metadata.
- The Fabric-produced patch is visible to Harbor's verifier.
- Harbor remains responsible for datasets, environments, verifier, and rewards.
- No Harbor-specific assumption leaks into Fabric core, SDK, or Hermes adapters.

## Execution Order

1. Keep core contract/schema tests green while making small contract fixes.
2. Finish SDK and CLI behavior for typed config and agent-package config.
3. Finish Hermes SDK and CLI reproducibility in clean environments.
4. Run the Hermes config-variation matrix across model, runtime, skills, tools,
   MCP, telemetry, workspace, artifacts, and harness adapter profiles.
5. Harden Relay telemetry and ArtifactManifest discovery for Hermes runs.
6. Run the Harbor lightweight smoke from a clean install.
7. Run a Harbor SWE-Bench Verified smoke and verifier path as the first
   evaluation proof.
8. After the SDK/CLI and Hermes path are stable, split follow-up work into
   adapter, consumer API, and telemetry/artifact readiness tracks.

## Review Checklist

Before calling the MVP complete:

- `cargo test --workspace` passes.
- `cargo fmt --check` passes.
- Python SDK smoke passes.
- CLI smoke passes.
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
