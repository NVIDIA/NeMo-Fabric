---
name: contribute-adapter
description: Add or substantially change a NeMo Fabric harness adapter, including adapter architecture, descriptor claims, package wiring, normalized config and capability mapping, runtime/session behavior, policy enforcement, telemetry, artifacts, documentation, and adapter validation.
license: Apache-2.0
metadata:
  author: NVIDIA Corporation and Affiliates
---

# Contribute An Adapter To NVIDIA NeMo Fabric

Build a production-ready adapter without reconstructing the contract from each
existing implementation. Keep the adapter package small and independent, make
normalized Fabric configuration authoritative, and prove every advertised
capability.

## Companion Guidance

Use `karpathy-guidelines` while implementing and reviewing the change. Use
`python-tests` for Python tests, `maintain-packaging` for dependency or package
metadata changes, `contribute-docs` and `review-doc-style` for public text, and
`validate-change` before `prepare-pr`.

Do not use this skill for a consumer application that only selects an existing
adapter. Use the consumer `nemo-fabric-integrate` skill for that work.

## Define The Contract First

Write down the following decisions before editing:

1. Choose the harness boundary: CLI process, Python SDK, external service, or
   harness-native plugin.
2. Choose one-shot behavior and whether ordered invocations on one Fabric
   runtime resume harness state.
3. List each normalized config area the adapter implements, partially
   implements, or rejects.
4. List fixed dependencies, dynamic credentials, child-process environment,
   workspace, state, telemetry, and artifact needs.
5. Define success with a credential-free planning path, focused adapter tests,
   and an end-to-end invocation or deterministic fixture.

Use the current execution boundary, not an aspirational one:

| Adapter kind | Use for | Current execution status |
| --- | --- | --- |
| `python` | A Python SDK or Python-owned CLI wrapper | Implemented as `python -m <module>` in a subprocess for each invocation. |
| `process` | A language-neutral CLI or executable | Implemented as a supervised subprocess. Use `stdin_payload: fabric_request` when the adapter needs the normalized invocation. |
| `http` | An already-running service | Modeled, but runtime dispatch is not implemented. Do not select it without implementing core dispatch and tests. |
| `native_plugin` | A harness-native plugin lifecycle | Modeled, but runtime dispatch is not implemented. Do not select it without implementing core dispatch and tests. |

Fabric's logical runtime can receive multiple ordered invocations even though a
`python` or `process` adapter launches once per invocation. Stateful adapters
must persist and validate harness session identity using
`runtime_context.runtime_id`. That resume behavior does not by itself justify
setting descriptor `capabilities.service`, `streaming`, `updates`, or
`cancellation`.

The current core `stop_runtime` path does not call a Python- or
process-adapter-specific stop hook. Release invocation-owned resources before
the adapter process exits, and document any durable session state that remains
after Fabric stops the logical runtime.

## Inspect The Authoritative Surfaces

Read these before implementing:

- `schemas/adapter-descriptor.schema.json` and the descriptor types in
  `crates/fabric-core/src/config.rs`.
- `AdapterInvocation`, the selected runner settings, result wrapping, and
  artifact collection in `crates/fabric-core/src/runtime.rs`.
- Planning and preflight behavior in `crates/fabric-core/src/config.rs` and
  `crates/fabric-core/src/doctor.rs`.
- The public authoring models in `python/src/nemo_fabric/models.py` and result
  types in `python/src/nemo_fabric/types.py`.
- `adapters/common/` and the closest existing adapter by lifecycle and harness
  API. Reuse shared utilities when their contract fits; do not copy them into a
  new package.
- [capability-and-policy.md](references/capability-and-policy.md) for the
  normalized mapping and rejection rules.

## Scaffold The Repository Surfaces

Create only the surfaces the adapter needs. A normal Python adapter has this
shape:

```text
adapters/<name>/
├── LICENSE -> ../../LICENSE
├── README.md
├── fabric-adapter.json
├── pyproject.toml
├── src/nemo_fabric_adapters/<name>/
│   ├── __init__.py
│   └── adapter.py
└── uv.lock
```

Add `testing.md` only when live credentials or external harness setup cannot be
explained concisely in the README. Do not add a package-local abstraction for a
single adapter.

Wire a public Python adapter into all applicable package surfaces:

- Add `nemo-fabric-adapters-<name>` to the root optional extras, adapter
  dependency group, and `[tool.uv.sources]` in `pyproject.toml`. Preserve the
  established `adapters-<name>` extra and short harness alias when both are part
  of the public install surface.
- Add the project to `python_projects` in `justfile` so lock, build, wheel, and
  version recipes include it.
- Package `fabric-adapter.json` under
  `share/nemo-fabric/adapters/<name>` and keep runtime dependencies narrowly
  pinned.
- Add the adapter to install docs, root adapter entry points, relevant examples,
  CLI preset assets, descriptor-drift tests, and CI matrices that enumerate
  adapters when those catalogs expose the new adapter.
- Update the consumer `nemo-fabric-integrate` skill when its supported adapter
  extras, examples, or public behavior inventory changes.
- Regenerate every affected lockfile. Do not edit generated lockfiles or API
  references by hand.

The current registry scans repository descriptors and
`<base_dir>/adapters/**/fabric-adapter.json`. The Python package must still ship
its descriptor, but do not assume package data alone changes core discovery.
Add a local descriptor to a consumer or fixture base directory when that path
must resolve outside the repository, or change discovery as an explicit core
contract change with Rust and Python coverage.

## Implement The Descriptor And Entry Point

Start with the narrowest truthful `fabric-adapter.json`:

- Use contract version `fabric.adapter/v1alpha1`, a stable globally unique
  `adapter_id`, and a stable harness name.
- Put runner defaults in `runner`. Caller `harness.settings` overrides runner
  keys; descriptor-relative paths resolve from the descriptor directory, while
  caller overrides resolve from `base_dir`.
- Declare fixed `requirements.binaries`, `env`, `files`, `services`, and
  `plugin_hooks` only when they are required for every applicable run. Dynamic
  model credentials belong to the selected model config and adapter preflight.
- Keep `config.accepts`, `config.generates`, telemetry providers, outputs,
  integration modes, and lifecycle capabilities synchronized with code and
  tests. Omission is safer than an unproved claim.

For a Python adapter, expose `run(payload) -> dict`, a `main()` that loads one
invocation, and a `python -m` entry point. Print exactly one JSON object to
stdout. Capture or redirect harness stdout and send diagnostics to stderr so
logs cannot corrupt the result. Exit nonzero for a normalized failed result.

For a process adapter, prefer the complete `fabric_request` stdin payload. Use
the raw `input` payload only for a deliberately minimal adapter that needs none
of Fabric's config, runtime, policy, telemetry, or artifact context.

## Preserve Configuration Authority And Isolation

Treat these payload fields as authoritative: `config`, `capability_plan`,
`telemetry_plan`, `runtime_context`, `request`, and `base_dir`.

- Use normalized config for models, tools, MCP, skills, telemetry, environment,
  runtime, and artifacts. Reserve `harness.settings` for settings unique to the
  harness.
- Reject any harness setting that duplicates a normalized field. Name the
  correct `FabricConfig` path in the error.
- Reject a configured capability, transport, provider, or value that the
  adapter cannot honor. Never ignore it, weaken policy, or fall back to a
  different provider silently.
- Resolve the workspace from `runtime_context.environment.workspace`, with
  `base_dir` only as the documented fallback. Keep state and generated config
  under a runtime-scoped directory. Never share session files between Fabric
  runtime IDs.
- Validate dependencies, supported versions, hooks, and dynamic credentials
  before invoking the harness. Return actionable, stable error codes without
  exposing secrets or raw provider payloads.
- Build a deliberate child environment. Forward only required system variables,
  the configured `ModelConfig.api_key_env`, declared telemetry variables, and
  validated harness-specific environment. Do not log environment values or an
  unredacted Fabric payload.
- Clean up subprocesses, temporary config, telemetry gateways, SDK clients, and
  checkpointers on success, failure, timeout, and cancellation. Preserve the
  original failure when cleanup also fails.

## Normalize Results And Lifecycle

Keep the adapter's JSON output stable across harness versions:

- Include the harness, adapter mode, primary response, `completed`, `failed`, a
  structured error, harness events, usage when the harness reports it, and
  session identifiers needed for diagnostics.
- Use errors shaped as `code`, `message`, `retryable`, and optional safe
  `metadata`. Do not leak credentials, headers, prompts, or raw exceptions.
- Report only measured usage and cost. On resumed sessions, aggregate the
  current invocation rather than replayed history.
- Keep harness events in adapter output and let Fabric add its own lifecycle
  events to `RunResult.events`.
- Write artifacts inside the resolved artifact root. Use the shared Relay
  helpers for Relay config and artifact discovery. The core currently promotes
  existing `atof` and `atif` entries from `relay_artifacts`; do not claim generic
  artifact promotion that the runtime does not implement.
- Persist a harness session ID only after a valid turn and reject corrupt state
  or an unexpected identity change. Prove that independent Fabric runtimes do
  not share state.

## Add Focused Evidence

Add tests only where they prove adapter behavior or a descriptor claim. Avoid
duplicating generic runtime tests.

Cover the following applicable cases in `tests/adapters/test_<name>_adapter.py`:

- Descriptor shape, exact `config.accepts`, requirements, telemetry, and
  lifecycle claims.
- Positive mapping for every claimed normalized capability.
- Rejection of unsupported transports, providers, duplicated settings, invalid
  paths, missing credentials, malformed input, and unenforceable policy.
- Success and harness failure normalization, usage/event extraction, and
  cleanup failure behavior.
- Credential and environment allowlisting without mutating the parent process.
- One-shot lifecycle plus session resume, isolation, and cleanup when stateful.

Add or update a canonical typed SDK variant and a credential-free fixture that
exercises `plan`, `doctor`, and `run`. Add a two-turn runtime smoke for a
stateful adapter. Keep a live-harness smoke opt-in when it requires credentials;
CI must still have deterministic coverage for the adapter boundary. Update a
harness-native YAML fixture only when the adapter actually generates YAML. Do
not introduce a public Fabric YAML authoring format; the current public
authoring contract is an in-memory `FabricConfig`.

## Document The Supported Contract

In the adapter README, document:

- Installation, supported Python and harness versions, adapter ID, resolution,
  and interpreter selection.
- Supported normalized fields, harness-only settings, and explicit precedence.
- Model providers, required credential variable names, environment forwarding,
  workspace behavior, and security boundaries.
- One-shot and stateful behavior, concurrency assumptions, result/error shape,
  telemetry modes, artifacts, and limitations.
- A canonical typed SDK example and exact focused and live-test commands.

Update `README.md`, `docs/getting-started/install.mdx`, `docs/index.yml`,
adapter/integration docs, and examples only when they are entry points for the
new or changed public behavior. Use `contribute-docs` and `review-doc-style` for
the final pass.

## Generate And Validate

Run focused checks first, then the full matrix appropriate to the changed
surfaces:

```bash
uv run --no-sync pytest tests/adapters/test_<name>_adapter.py
cargo test -p nemo-fabric-core adapter --locked

# After Python package metadata or dependency changes.
just lock-python
just wheels

# After an intentional Rust schema contract change.
cargo run -p nemo-fabric-core --example generate-schemas -- schemas

# Required full validation for a new adapter.
cargo fmt --all -- --check
just --fmt --check
just test-rust
just test-python
uv run pre-commit run --all-files --show-diff-on-failure
git diff --check

# When docs-site content or generated API references changed.
just docs
```

Inspect the built adapter wheel to confirm that it contains the package,
descriptor, README metadata, and license. Run the deterministic end-to-end
fixture through the built environment, not only through source imports. Run the
opt-in real-harness smoke when credentials and dependencies are available, and
state explicitly when it was not run.

These commands mirror the repository's pre-commit, Python, Rust, packaging, and
Fern documentation CI. Follow `validate-change` when a change expands into core,
schema, bindings, dependencies, or another integration.

## Final Review

- [ ] Every descriptor claim has implementation and positive/negative evidence.
- [ ] Normalized config wins; duplicate or unsupported behavior fails loudly.
- [ ] Fixed and dynamic preflight checks produce actionable errors.
- [ ] Credentials and environment values are forwarded narrowly and never logged.
- [ ] Workspaces, state, sessions, generated config, and artifacts are isolated.
- [ ] One-shot, resume, failure, timeout/cancellation, and cleanup paths are covered as applicable.
- [ ] Results, errors, events, usage, telemetry, and artifacts are stable and normalized.
- [ ] Package extras, sources, recipes, lockfiles, wheel contents, docs, examples, fixtures, and CI enumerations agree.
- [ ] Generated files came from repository commands and show no unexplained drift.
- [ ] The final diff contains no unrelated cleanup or speculative abstraction.
