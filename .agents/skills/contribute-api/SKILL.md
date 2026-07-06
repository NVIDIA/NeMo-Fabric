---
name: contribute-api
description: Contribute a new NeMo Fabric public API surface safely, with Rust, CLI, Python, schema, adapter, and documentation parity in mind
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
- Decide whether the CLI, PyO3 binding, Python SDK, type stubs, schemas, or
  adapter contract must expose the new surface
- Keep every affected public surface in parity
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
- `justfile`
