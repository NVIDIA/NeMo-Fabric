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

Read the authoritative surfaces before editing. Use the closest existing
adapter only for harness-specific patterns, not to infer the core contract:

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

1. Choose the harness boundary: Python SDK or module, CLI/process, or service.
   Use `python` for Python-owned execution and `process` for an executable. Core
   runtime dispatch does not yet implement `http` service or `native_plugin`
   execution.
2. Place a repository adapter under `adapters/<name>`. Define its install extra
   and packaged descriptor. Choose the matching `harness.resolution` strategy,
   and document repository or `base_dir` descriptor discovery; wheel package
   data alone does not add a descriptor to core discovery.
3. Decide whether invocations are one-shot and independent or stateful and
   resume harness state for the same `runtime_context.runtime_id`.
4. List the normalized fields the adapter supports, partially supports, or
   rejects.
5. Identify fixed requirements, dynamic credentials, telemetry, workspace,
   state, and artifact needs.

## Implement The Narrowest Adapter

- Use the existing Fabric `python` or `process` runner and normalized
  request/result contracts. Reuse `adapters/common/` only when its contract
  fits; do not add a runner or abstraction for one adapter.
- Create only the package surfaces that adapter needs: license link, README,
  descriptor, package metadata, `src/nemo_fabric_adapters/<name>` module entry
  point, lockfile, and focused tests under `tests/adapters`. Keep the package
  independent, small, and self-contained.
- Start with the narrowest truthful `fabric-adapter.json`. Keep
  `config.accepts`, `config.generates`, requirements, telemetry declarations,
  and lifecycle capabilities synchronized with implementation and tests.
- Use the complete Fabric invocation for adapters that consume normalized
  config or runtime context. Treat `config`, `capability_plan`,
  `telemetry_plan`, and `runtime_context` as authoritative. Reserve
  `harness.settings` for harness-specific behavior.
- Apply precedence in this order: normalized `config`; Fabric-resolved plans and
  runtime context; harness-specific settings; descriptor and adapter defaults.
  Reject duplicates or unsupported behavior with an actionable error that
  names the field and supported alternatives. Never silently drop configuration.
- Run dependency and authentication preflight: declare fixed dependencies in
  descriptor requirements, then validate selected versions, hooks, and
  credentials before invoking the harness. Never expose credential values in
  output, errors, events, logs, or fixtures.
- Forward only required system variables, selected credential variables,
  telemetry variables, and documented harness-specific environment. Never
  forward or log unrelated environment values.
- Emit one JSON object on stdout and diagnostics on stderr. Return a nonzero
  exit code for failures so core result normalization remains accurate.
- Scope workspace, generated config, state, sessions, and artifacts to the
  resolved runtime context. Stateful adapters must isolate Fabric runtime IDs.
- Implement only the harness lifecycle hooks needed for start, invoke, resume,
  and cleanup; do not claim unsupported service, streaming, update, or
  cancellation behavior.
- Keep adapter output stable across harness versions. Normalize the primary
  response, structured errors, measured usage, harness events, session IDs, and
  artifact paths without duplicating Fabric lifecycle events.

Wire a public Python package into root optional extras, the adapter dependency
group, `[tool.uv.sources]`, `python_projects` in `justfile`, and applicable
catalogs and CI enumerations. Ship its descriptor under
`share/nemo-fabric/adapters/<name>`. Use `maintain-packaging` to regenerate
lockfiles and package artifacts.

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
commands in the adapter README. Provide a canonical typed SDK example and, when
the harness requires YAML, a canonical harness-native YAML fixture. Update docs
and examples together when they expose the changed behavior.

## Validate And Review

Run `validate-change` and the applicable adapter commands:

```bash
uv run --no-sync pytest tests/adapters/test_<name>_adapter.py
just test-python
just lock-python && just wheels  # Package or dependency changes.
cargo run -p nemo-fabric-core --example generate-schemas -- schemas  # Schema changes.
cargo fmt --all -- --check && just test-rust  # Rust changes.
just docs  # Docs-site or generated-reference changes.
uv run pre-commit run --all-files --show-diff-on-failure
git diff --check
```

Before handoff, confirm:

- [ ] Descriptor claims match implementation and positive, negative, and lifecycle evidence.
- [ ] Precedence, actionable rejection, forwarding, isolation, results, telemetry, and artifacts are tested.
- [ ] Package, installation, resolution, docs, SDK/YAML examples, fixtures, and generated artifacts agree.
- [ ] Generated files came from repository commands, applicable CI catalogs enumerate the adapter, and the diff has no contract drift or unrelated changes.
