<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Adapter Capability And Policy Contract

Use this reference while designing the descriptor, adapter boundary, and tests.
It describes the current repository behavior; verify the named Rust and Python
types again when those contracts change.

## Configuration Precedence

Apply configuration in this order:

1. Use the complete normalized `FabricConfig` in the adapter invocation as the
   source of truth.
2. Use `capability_plan` and `telemetry_plan` for routes and provider decisions
   that Fabric already resolved.
3. Use `runtime_context` for resolved workspace, artifact, telemetry, and
   correlation context.
4. Use `harness.settings` only for harness-specific behavior with no normalized
   equivalent.
5. Use adapter defaults only when neither Fabric nor the harness-specific
   contract supplies a value.

At the runner layer, `harness.settings` overrides matching descriptor `runner`
keys. This override is for runner selection such as `python`, `python_env`,
`args`, `cwd`, and `env`; it does not authorize a harness setting to replace a
normalized model, tool, MCP, skill, telemetry, environment, or runtime field.

Validate the entire supported `harness.settings` boundary. Reject known
duplicates with a message such as:

```text
harness.settings.mcp_servers is not supported; use FabricConfig.mcp
```

Reject unknown passthrough keys unless a narrow, documented harness SDK
passthrough is intentionally supported and validated against the SDK call
signature. Never let passthrough override Fabric-owned arguments.

## Capability Mapping

| Surface | Authoritative input | Descriptor declaration | Required adapter behavior and evidence |
| --- | --- | --- | --- |
| Models | `config.models` and the selected alias | Add `models` to `config.accepts` only when consumed. | Map provider, model, temperature, `api_key_env`, and provider settings deliberately. Reject unsupported providers and missing dynamic credentials before a model call. Test default and nondefault values plus rejection. |
| Tool policy | `config.tools.blocked`, `capability_plan.tools`, and `capability_plan.native.tools_configured` | `tools.blocked` is the exact claim used by core routing. Add broader `tools` only for other consumed tool fields. | Enforce every blocked name across direct tools, plugins, delegated agents, and subprocesses. If complete enforcement is impossible, omit the claim so planning routes it as unsupported. Test allowed and denied calls, aliases, and delegated paths. |
| MCP | `capability_plan.native.mcp_servers` | Add `mcp` to `config.accepts`. | Map only `harness_native` servers routed by Fabric. Validate every supported transport, URL/command, and extra field. Reject malformed or unsupported transports. `fabric_managed` exposure is currently unsupported by core and must not be silently converted. |
| Skills | `capability_plan.native.skill_paths` | Add `skills` to `config.accepts`. | Resolve from the provided paths, validate each skill directory and manifest, prevent name collisions, and register or stage skills in an invocation- or runtime-scoped location. Test missing paths, malformed manifests, duplicates, and cleanup. |
| Telemetry | `telemetry_plan` plus normalized `config.telemetry` and `config.relay` | Add `telemetry` to `config.accepts`; enumerate each implemented provider under `telemetry.providers`, with exact `outputs` and `integration_modes`. | Reject an enabled provider absent from the descriptor. For Relay, use the generated config path and shared helpers; for native telemetry, consume only the planned provider config. Test disabled, enabled, unsupported, startup failure, artifact, and cleanup paths. |
| Environment | `runtime_context.environment` | Add `environment` to `config.accepts` when the adapter actively consumes it; this declaration does not add runtime provider support. | Use the resolved workspace and artifact paths. Current `python` and `process` dispatch rejects non-local providers, so do not claim Docker, sandbox, or service execution without core support. Test path resolution, isolation, and unsupported providers at the appropriate layer. |
| Runtime and lifecycle | `runtime_context.runtime_id`, request IDs, and normalized runtime config | Add `runtime` when consumed. Set lifecycle booleans only for operations implemented end to end. | Treat IDs as opaque. Persist harness state by Fabric runtime ID, reject identity drift, serialize turns as required by the harness, and release invocation-owned resources. Generic multi-turn reuse does not imply `service`, `streaming`, `updates`, or `cancellation`. Test one-shot, two turns, independent runtimes, invalid state, failure, and cleanup. |
| Artifacts | `runtime_context.artifacts`, environment artifacts, and `config.runtime.artifacts` | List only harness-native files the adapter creates in `config.generates`; add `runtime` or `environment` to `accepts` when those fields drive generation. | Write only below the resolved root, use stable logical names, media types, and safe paths. Core collects its own stdout, stderr, and workspace patch. It currently promotes only existing Relay `atof`/`atif` entries from adapter output. Test file existence, containment, uniqueness, and missing-file behavior. |

`config.accepts` is both an inventory and, for `tools.blocked`, `mcp`, and
`skills`, an input to core capability routing. Keep exact strings stable and
assert the complete list in the descriptor test so accidental expansion is
review-visible.

## Unsupported Configuration Policy

Classify unsupported input before invoking the harness:

- **Core-routed unsupported capability:** omit the corresponding descriptor
  claim. Confirm `plan` or `doctor` reports the unsupported route. Blocked-tool
  policy fails before runtime start; MCP and skills currently produce warning
  routes, so the adapter must still consume only the native routed section.
- **Adapter-specific unsupported value:** return a normalized configuration
  error naming the field, received value, and supported alternatives. Do not
  remove, rewrite, or downgrade it silently.
- **Modeled but unimplemented lifecycle:** keep the descriptor boolean false and
  reject attempts to use the operation. Implement core dispatch before claiming
  it.
- **Version-dependent harness feature:** preflight the installed version and
  fail with the supported range. Do not use reflection to silently discard
  required arguments; reflection is acceptable only for truly optional
  compatibility inputs documented and tested across supported versions.

## Preflight And Authentication

Split preflight into fixed and dynamic checks:

- Put fixed binaries, environment variables, files, services, and plugin hooks
  in descriptor `requirements` when every run needs them. `doctor` can check
  local binaries, environment variables, and files; services and hooks are
  currently warnings rather than active probes.
- Resolve the Python interpreter through
  `harness.settings.python`, `harness.settings.python_env`, `ADAPTER_PYTHON`, the
  active virtual environment, the Fabric host interpreter, then `python3` as
  implemented by core. The chosen interpreter must contain both the adapter
  package and harness dependencies.
- Derive model credentials from the selected `ModelConfig.api_key_env`, with a
  documented provider default only when the config omits it. Test custom names,
  missing values, and provider-specific remapping.
- Check optional telemetry packages, CLI versions, hook availability, and
  service health only when the selected configuration activates them.

Never put credential values in a descriptor, generated config fixture, result,
error, event, log, or snapshot.

## Adapter Output Boundary

The core wraps adapter stdout in a normalized `RunResult`. The subprocess exit
code controls `RunResult.status`; parsed stdout becomes `RunResult.output`.
Therefore:

- Emit one JSON object and no other stdout.
- Return `completed: true` and `failed: false` only after a valid terminal
  harness result.
- On expected adapter or harness failure, emit `completed: false`,
  `failed: true`, a structured error, and exit nonzero.
- Catch unexpected exceptions at the adapter boundary and emit a stable internal
  error without raw diagnostics. Record detailed diagnostics only in redacted
  logs.
- Do not mistake adapter-output events for Fabric lifecycle events. Fabric adds
  `runtime_start`, `invocation_start`, `invocation_end`, artifact, and stop
  events separately.

Test the module entry point as a subprocess at least once. A direct `run()` unit
test does not catch stdout contamination, exit-code drift, interpreter
selection, package data omissions, or descriptor-resolution failures.
