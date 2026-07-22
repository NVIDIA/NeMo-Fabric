---
name: contribute-adapter
description: Add or substantially change a NeMo Fabric harness adapter, including its architecture, descriptor claims, package wiring, capability and policy mapping, runtime behavior, and validation.
license: Apache-2.0
---

# Contribute An Adapter To NVIDIA NeMo Fabric

## Companion Guidance

Use `karpathy-guidelines` to keep the change scoped. Use `python-tests` for test
design, `maintain-packaging` for package or dependency changes,
`contribute-docs` and `review-doc-style` for public text, `validate-change` for
the validation matrix, and `prepare-pr` for review handoff.

Do not use this skill for a consumer application that only selects an existing
adapter. Use the consumer `nemo-fabric-integrate` skill for that work.

## Define The Contract

Read the authoritative surfaces and the closest existing adapter before
editing:

- `schemas/adapter-descriptor.schema.json` and the descriptor types in
  `crates/fabric-core/src/config.rs`.
- Invocation construction, subprocess dispatch, result wrapping, and artifact
  collection in `crates/fabric-core/src/runtime.rs`.
- Planning and preflight behavior in `crates/fabric-core/src/config.rs` and
  `crates/fabric-core/src/doctor.rs`.
- Public models in `python/src/nemo_fabric/models.py` and result types in
  `python/src/nemo_fabric/types.py`.
- Shared utilities in `adapters/common/` and the closest adapter by harness API
  and lifecycle.

Decide the following before implementation:

1. Choose `python` for a Python-owned module or CLI and `process` for a
   language-neutral executable. Core runtime dispatch does not yet implement
   `http` or `native_plugin`.
2. Decide whether invocations are independent or resume harness state for the
   same `runtime_context.runtime_id`.
3. List the normalized fields the adapter supports, partially supports, or
   rejects.
4. Identify fixed requirements, dynamic credentials, telemetry, workspace,
   state, and artifact needs.

## Implement The Narrowest Adapter

- Follow the closest adapter package layout. Reuse `adapters/common/` only when
  its contract fits; do not add an abstraction for one adapter.
- Create only the package surfaces that adapter needs: license link, README,
  descriptor, package metadata, module entry point, and lockfile.
- Start with the narrowest truthful `fabric-adapter.json`. Keep
  `config.accepts`, `config.generates`, requirements, telemetry declarations,
  and lifecycle capabilities synchronized with implementation and tests.
- Use the complete Fabric invocation for adapters that consume normalized
  config or runtime context. Treat `config`, `capability_plan`,
  `telemetry_plan`, and `runtime_context` as authoritative. Reserve
  `harness.settings` for harness-specific behavior.
- Reject configured values or policies the adapter cannot honor. Never ignore
  them, silently weaken policy, or substitute another provider.
- Validate fixed dependencies through descriptor requirements and validate
  selected versions, hooks, and credentials before invoking the harness. Never
  expose credential values in output, errors, events, logs, or fixtures.
- Emit one JSON object on stdout and diagnostics on stderr. Return a nonzero
  exit code for failures so core result normalization remains accurate.
- Scope workspace, generated config, state, sessions, and artifacts to the
  resolved runtime context. Stateful adapters must isolate Fabric runtime IDs.
- Keep adapter output stable across harness versions. Normalize the primary
  response, structured errors, measured usage, harness events, session IDs, and
  artifact paths without duplicating Fabric lifecycle events.

Wire a new package into the same root extras, sources, `justfile` recipes,
descriptor packaging, catalogs, and CI enumerations as comparable adapters.
Use `maintain-packaging` to select the applicable surfaces and regenerate
lockfiles or package artifacts.

## Map Capabilities And Policy

Use this table as the minimum capability review. Omit claims that the adapter
cannot implement and test end to end.

| Surface | Fabric input | Adapter responsibility |
| --- | --- | --- |
| Models | `config.models` and selected alias | Map supported provider settings and credential-variable names; reject unsupported providers. |
| Tool policy | `config.tools.blocked` and `capability_plan.tools` | Claim `tools.blocked` only when every harness tool path enforces it. |
| MCP | `capability_plan.native.mcp_servers` | Claim `mcp` only for supported native transports; reject unsupported routes or fields. |
| Skills | `capability_plan.native.skill_paths` | Validate and stage skill paths without cross-runtime collisions. |
| Telemetry | `telemetry_plan` and normalized telemetry config | Declare only providers, outputs, and integration modes the adapter implements. |
| Environment and lifecycle | `runtime_context` | Use resolved workspace, runtime ID, telemetry, and artifact context; claim only lifecycle operations implemented end to end. |
| Artifacts | Resolved artifact roots and normalized runtime config | Write within the resolved root and declare only files the adapter generates. |

Descriptor claims participate in planning and routing. Assert their exact
values in focused tests so capability expansion is review-visible.

## Add Focused Evidence

Add only tests that prove adapter behavior or descriptor claims. Cover the
applicable cases:

- Descriptor shape and exact advertised capabilities.
- Positive mapping for each claimed normalized surface.
- Rejection of unsupported capability values and unenforceable policy.
- Successful and failed harness result normalization without secret leakage.
- One-shot execution, plus resume and runtime isolation when stateful.
- A subprocess test of the packaged entry point to catch stdout, exit-code,
  interpreter, descriptor, and package-data failures.

Provide a credential-free fixture that exercises `plan`, `doctor`, and `run`.
Keep live-harness tests opt-in when they require credentials, but retain a
deterministic end-to-end path for CI. Inspect a built wheel when package data or
metadata changes.

Document installation, supported configuration, harness-only settings,
credentials, lifecycle, telemetry, artifacts, limitations, and focused test
commands in the adapter README. Update docs and examples only when they expose
the changed behavior.

Finish with `validate-change`, then confirm that package, docs, fixtures, and
generated artifacts agree.
