---
name: validate-change
description: Choose and run the right NeMo Fabric validation matrix for a change instead of using one fixed test list
author: NVIDIA Corporation and Affiliates
license: Apache-2.0
---


# Validate a Change

## Companion Guidance

Use `karpathy-guidelines` alongside this skill for implementation or review
work. Keep changes scoped, surface assumptions, and define focused validation
before editing.

Use this skill to choose the smallest validation set that still covers the
surfaces touched by a change.

## Mandatory Rules

- Format changed files with the language-native formatter before the final
  test pass.
- If Rust code changed, run `cargo fmt --all -- --check` and `just test-rust`.
- If Python code or a Python-facing adapter changed, run `just test-python`.
- If `crates/fabric-core` changed in a way exposed through Python, run both the
  Rust and Python suites.
- If the PyO3 bridge or package metadata changed, run `just build-python` and
  `cargo check -p fabric-python --locked`.
- If public configuration types changed, confirm the schema snapshot tests in
  `just test-rust` pass and review generated schema diffs.
- If an adapter or integration changed, run its focused tests or smoke path in
  addition to the language suite.
- If documentation or examples changed, run `just docs` when practical and
  verify documented commands against the current repository.
- If code changes alter APIs, commands, paths, packaging behavior, telemetry
  semantics, or documented best practices, update dependent maintainer skills
  in the same branch.

## Start With the Change Shape

- **Rust core, CLI, or shared runtime semantics changed**
  Run Rust formatting and tests. Add Python tests when the behavior is exposed
  through the SDK, and run relevant CLI smoke tests for CLI behavior.
- **Python SDK or PyO3 binding changed**
  Use `python-tests`, run focused pytest tests first, then run
  `just test-python`. Rebuild with `just build-python` when native code or
  packaging changed.
- **Adapter behavior changed**
  Run the focused adapter tests under `tests/`, then `just test-python`. Add the
  dependency-free smoke path when launch, environment, or lifecycle behavior
  changed.
- **Harbor integration changed**
  Run `tests/test_harbor_runner.py`, relevant Python integration smokes, and
  `just test-python`.
- **Schema or public contract changed**
  Run both language suites and review changes under `schemas/` and generated API
  references.
- **Documentation-only change**
  Use `contribute-docs` and `review-doc-style`. Run `just docs` for docs-site or
  generated-reference changes.
- **CI or packaging changed**
  Use `maintain-ci` or `maintain-packaging`, then run the recipes and checks
  whose behavior changed.

## Core Validation Matrix

```bash
just test-rust
just test-python
```

## Common Targeted Commands

```bash
# Rust
just build-rust
just test-rust
cargo fmt --all -- --check
cargo check -p fabric-python --locked

# Python and native extension
just build-python
just test-python
uv run --no-sync pytest -k "<pattern>"

# Documentation
just docs

# Justfile and patch hygiene
just --fmt --check
git diff --check
```

Use `.github/workflows/ci_python.yml` as the source of truth for the
dependency-free smoke list. Gated integration smokes require their documented
credentials, sibling checkouts, services, or Docker environment; do not report
them as run unless those prerequisites were present.

## Hygiene Checks

Before review or handoff:

- Verify README and docs entry points still match current package names and
  paths.
- Verify examples use current `just` recipes and public commands.
- Call out checks that were not run and why.
- Run `git diff --check`.

## References

- Python test guidance: `python-tests`
- Build and test recipes: `justfile`
- Python CI: `.github/workflows/ci_python.yml`
- Rust CI: `.github/workflows/ci_rust.yml`
- Documentation CI: `.github/workflows/fern-docs.yml`
- Public Python contract: `docs/python-sdk-contract.md`
