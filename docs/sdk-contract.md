<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# NeMo Fabric — SDK Interface & Configuration Contract

> **Status:** Draft for Platform-team review (FABRIC-2).
> Most of this contract is **implemented today**; anything that is not yet
> available is called out under [Pending / planned](#7-pending--planned). This
> document is the human-facing contract; the machine-readable contract lives in
> [`schemas/`](../schemas) and the per-method signatures in the generated API
> reference.

## 1. Purpose & audience

This document defines the surface that consumers build against: the **SDK
interface** (what you call) and the **configuration contract** (what you pass
and what you get back). It is written for the teams that consume Fabric —
**Platform**, **Harbor/Gym**, and **local** users — to review before they
depend on it.

It is deliberately *not*:

- the **generated API reference** (which documents exact call signatures from
  the SDK docstrings), or
- the raw **JSON schemas** in [`schemas/`](../schemas) (which are the
  machine-readable contract, generated from the Rust types).

Those two are the authoritative low-level artifacts; this document ties them
together into the model, the guarantees, and the consumption story.

| Artifact | What it is | Source of truth for |
| --- | --- | --- |
| This document | Contract narrative + design intent | The model and its guarantees |
| `schemas/*.json` | Generated JSON Schema snapshots | Field-level config/result shapes |
| API reference | Generated from SDK docstrings | Exact method signatures |

## 2. Design principles

These are the guarantees the contract is designed to keep.

1. **No mandatory load-and-run model.** Unlike NAT, Fabric does not require a
   single nested YAML that is loaded and run as one unit. A config may be an
   agent directory, a single file, or an **in-memory typed object**, and
   validating, planning, diagnosing, and running are **separate verbs** rather
   than one entry point.
2. **Simple for multiple consumers.** One small client surface serves Platform,
   Harbor/Gym, and local users. The core surface carries **no consumer-specific
   types** — harness- and platform-specific concerns live behind adapters and
   config, not in the SDK signatures.
3. **Extensible.** New harnesses, profiles, and capability mappings are added
   through **descriptors and config**, without changing the SDK surface.
4. **Config is not immutable after startup.** Fabric does not impose a blanket
   rule that configuration is frozen once a run starts. Where a harness
   supports safe runtime changes, the SDK shape should leave room for them.
   This is **not day-1 behavior**, but the contract is designed not to preclude
   it.

## 3. Configuration contract

### 3.1 The config model

Configuration resolves in stages:

```
agent config  ──(apply profiles)──▶  effective config  ──(plan)──▶  run plan
```

- **Agent config** (`agent.yaml`) is the portable base.
- **Profiles** are overlays applied on top (e.g. a local vs. hosted profile).
- **Effective config** is the merged result after profile resolution.
- **Run plan** is the executable plan derived from the effective config and the
  selected adapter.

### 3.2 Typed-config first (no agent directory required)

A consumer does **not** need an agent directory on disk. The `*_config` methods
accept an in-memory typed/dict config directly, with an optional `base_dir` for
resolving relative references. This is the primary path for Platform, which
constructs config programmatically rather than shipping a package.

### 3.3 Contract surfaces

The committed schemas in [`schemas/`](../schemas) are generated from the Rust
types (`fabric schema`) and checked against drift in CI. They group as:

| Group | Schemas |
| --- | --- |
| Config inputs | `agent`, `profile`, `adapter-descriptor` |
| Resolved | `effective-config`, `run-plan` |
| Adapter boundary | `adapter-invocation`, `runtime-context` |
| Per-invocation input | `run-request` |
| Results | `run-result`, `artifact-manifest`, `error-info`, `fabric-event` |
| Lifecycle handles | `environment-handle`, `runtime-handle`, `invocation-handle` |

### 3.4 Versioning & evolution

Schemas are **generated**, not hand-maintained: the Rust contract types are the
source, `fabric schema --output-dir schemas` regenerates the snapshots, and
`cargo test` fails if the committed snapshots drift from the types. Evolution is
expected to be additive, with `agent.yaml` carrying a version field for the
portable config.

## 4. SDK interface (today)

`FabricClient` is the single entry point. It uses the native (PyO3) bindings
when installed and falls back to invoking the `fabric` CLI in source trees;
construction takes an optional `command` and `cwd` to force/parameterize the
CLI path.

| Method | Sync/Async | Input | Notes |
| --- | --- | --- | --- |
| `validate` | sync | path | Validate an agent dir or config file |
| `inspect` | sync | path | Return the effective config |
| `plan` | sync | path | Resolve into a run plan |
| `plan_config` | sync | typed config | In-memory config → plan (**native-only**) |
| `doctor` | async | path | Diagnose a plan without running |
| `doctor_config` | async | typed config | Diagnose in-memory config (**native-only**) |
| `run` | async | path | Run through the selected adapter |
| `run_config` | async | typed config | Run in-memory config (**native-only**) |

Two axes describe the surface: **path-based vs. typed-config** (`*_config`), and
**sync vs. async** (planning/inspection are sync; anything that touches a
harness is async). The `*_config` variants currently require the native
extension and raise `FabricNativeUnavailableError` on the CLI fallback.

### 4.1 Result contract

`run`/`run_config` return a normalized `RunResult` regardless of which adapter
ran:

- `status` — `Succeeded` | `Failed` | `Cancelled`
- `output` — normalized harness output
- `artifacts` — an `ArtifactManifest` (`root` + typed `ArtifactRef`s)
- `events` — `FabricEvent`s (lifecycle/progress)
- `error` — `ErrorInfo` (`stage`, `code`, `message`, `retryable`) when failed
- identifiers — `runtime_id`, `invocation_id`, `request_id`
- `telemetry`, `metadata`

`ErrorInfo.stage` is one of `Config`, `Plan`, `Prepare`, `Start`, `Invoke`,
`Stop`, `Release`, `Artifact`, so a consumer can tell *where* a run failed
without parsing messages.

### 4.2 Consumer-neutral by construction

The signatures above name no Platform-, Harbor-, or Hermes-specific type. The
harness is selected by config (`harness.adapter_id`) and everything
harness-specific is mediated by the adapter, so the same calls serve every
consumer.

## 5. Consumption per consumer

- **Platform** — build a typed config in memory and call
  `plan_config` / `run_config`; no agent directory on disk.
- **Harbor/Gym** — point `plan` / `run` at an agent package directory and select
  a profile.
- **Local** — use the CLI verbs directly, or `FabricClient` with an explicit
  `command`.

## 6. Extensibility

- **Adapters.** A harness is integrated by adding a `fabric-adapter.json`
  descriptor — no SDK change. Key fields: `adapter_id`, `adapter_kind`
  (`process` | `http` | `python` | `native_plugin`), `runner` (how to launch),
  `requirements` (`env`, `binaries`), `config` (`accepts` / `generates`), and
  `telemetry.supports`. Maintained today: **Hermes SDK** (inline `python`) and
  **Hermes CLI** (`process`).
- **Profiles.** Kebab-case YAML overlays selected by name.
- **Capability mappings.** A descriptor declares which capability areas it
  `accepts` (e.g. tools, mcp, skills, telemetry) and the harness-native files it
  `generates`, so capabilities map onto each harness without consumer code.

## 7. Pending / planned

These are **not available today** and are surfaced here so reviewers see the
full intended shape. None are day-1.

- **Async session boundary** — `start`, `invoke`, `stream`, `cancel`, `stop`
  for resumable, multi-turn runtimes. Today only single-shot `run` is exposed;
  an internal `RuntimeAdapter` trait (`start` / `invoke` / `stop`) already
  exists in the core but is not surfaced to the SDK or CLI. Tracked under
  FABRIC-10 / runtime-modes work.
- **Runtime handle lifecycle** — exposing `RuntimeHandle` / `InvocationHandle`
  to consumers so a runtime can be held open across invocations.
- **Runtime config mutation** — concrete hooks for the safe-runtime-change room
  described in principle 4, gated on harness support.

## 8. Relationship to other artifacts

- **Generated API reference** — exact, always-current method signatures
  (generated from the SDK docstrings; see the Fern docs).
- **`schemas/`** — the machine-readable field-level contract.
- **CLI ↔ SDK parity** — the CLI verbs mirror the SDK methods, and normalized
  result fields are shared across the inline (native) and process-backed (CLI)
  paths.
