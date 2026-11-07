---
name: contribute-api
description: Contribute a new NVIDIA NeMo Fabric public API surface safely, with Rust, CLI, Python, TypeScript, schema, adapter, and documentation parity in mind
author: NVIDIA Corporation and Affiliates
license: Apache-2.0
---


# Contribute A New API Surface

## Companion Guidance

Use `karpathy-guidelines` alongside this skill for implementation or review
work. Keep changes scoped, surface assumptions, and define focused validation
before editing.

Use this skill when contributing a public API addition or behavior change to the
runtime or bindings.

## Default Guidance

- Start from the shared Rust core behavior first
- Decide whether the CLI, PyO3 binding, Python SDK, type stubs, schemas, or the
  Python and TypeScript adapter-contract bindings must expose the new surface
- Keep every affected public surface in parity
- `RunRequest.relay_session_root` is forwarded as the typed `AgentRunRequest.relay_session_root` field. Keep Rust, Python, TypeScript, and schema representations in parity. Context keys do not control Relay propagation; preserve the adapter fallback for unusable UUIDs.
- OpenShell task commands serialize public SDK config/request/handle/result mappings. Keep forwarding buffered and preserve task-generated IDs, normalized failed results, telemetry references, and declared artifacts. Do not normalize an SDK result as an adapter result. Task paths remain POSIX strings; host artifact destinations stay separate. Unknown outcomes bind the immutable sandbox identity through reattachment and must not be replayed.
- Preserve normalized `runtime.timeout_seconds` across host and task SDKs: positive finite values enforce the invocation deadline; omission/null disables the total invocation deadline. Keep lifecycle, health, and cleanup budgets separate, and do not replace disabled invocation deadlines with transport defaults.
- Keep `runtime.response_limits` SDK-owned across host/task execution. Preserve inclusive byte/file budgets, explicit limit diagnostics, and transport allowances derived from configured budgets. Recheck convenience-run responses after cleanup adds stop events or errors, and preserve structured collection-time limit failures. Do not add adapter capability requirements or independent compressed-file ceilings. Use `tests/e2e/response-limits-checklist.md` when changing these paths.
- Update docs and examples in the same branch

## Minimum Acceptance

- Public behavior is clearly described
- Every affected public surface is covered
- The validation matrix matches the changed surfaces
- PR notes explain the user-facing change

## References

- `validate-change`
- `review-doc-style`
- `docs/python-sdk-contract.md`
- `schemas/SCHEMA.md`
- `justfile`

- Software provenance uses `FabricConfig.provenance` and optional `RunResult.execution`. Keep Rust, shared Python evidence dataclasses in `nemo-fabric-evidence`, SDK result mappings, generated TypeScript types and the canonical `schemas/evidence/execution-evidence.schema.json` aligned. The shared schema is generated from Rust and consumed by both NB and SB bindings; keep the TypeScript adapter-contract schema copy byte-identical. Preserve strict adapter result shapes: lifecycle evidence is carried beside the outcome. Optional report failure must not replace invocation outcomes.
