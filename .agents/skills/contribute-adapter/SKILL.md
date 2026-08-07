---
name: contribute-adapter
description: Add or substantially change a first-party NVIDIA NeMo Fabric adapter, including repository packaging, discovery metadata, catalogs, CI wiring, and validation.
license: Apache-2.0
---

# Contribute a First-Party NVIDIA NeMo Fabric Adapter

Use the public
[`nemo-fabric-build-adapter`](../../../skills/nemo-fabric-build-adapter/SKILL.md)
skill for adapter-contract semantics, descriptor design, configuration mapping,
lifecycle behavior, and conformance evidence. This maintainer skill adds only
the repository integration required for adapters shipped by NVIDIA NeMo
Fabric.

Do not use this skill for a consumer that selects an existing adapter. Use the
consumer `nemo-fabric-integrate` skill instead.

## Companion Guidance

Use `karpathy-guidelines` to keep the change scoped, `python-tests` for test
design, `maintain-packaging` for package or dependency changes,
`contribute-docs` and `review-doc-style` for public text, `validate-change` for
the validation matrix, and `prepare-pr` for review handoff.

## Repository Integration

Follow these repository-specific requirements after applying the public skill:

1. Place the adapter under `adapters/<name>/` with `LICENSE -> ../../LICENSE`,
   `README.md`, `fabric-adapter.json`, language-native package and lock files,
   a source entry point, and focused tests.
2. Give each Python leaf adapter a small base installation, a `harness` extra
   for supported target packages, and a `full` extra for package-installable
   integrations. Add a `relay` extra only when the adapter imports NVIDIA NeMo
   Relay Python APIs.
3. Add one canonical root extra that delegates to the matching leaf adapter
   and its `harness` extra. Keep `nemo-fabric-runtime` an exact-version,
   unconditional root dependency.
4. Add the package to the root adapter-test dependency group,
   `[tool.uv.sources]`, `python_projects` in `justfile`, applicable catalogs,
   and CI enumerations. Ship its descriptor under
   `share/nemo-fabric/adapters/<name>`.
5. Regenerate lockfiles and inspect the root and leaf wheel metadata. Verify
   root-to-leaf delegation and every published leaf extra.

Keep descriptor claims, implementation, focused tests, public documentation,
catalog entries, and packaged metadata synchronized. Start with the narrowest
truthful capability set.

## Repository Evidence

In addition to the evidence required by the public skill, include:

- A subprocess test of the packaged entry point.
- Exact descriptor assertions for every claimed capability.
- A credential-free fixture that exercises `plan`, `doctor`, and `run`.
- Wheel inspection when package data, dependencies, or extras change.
- Deterministic CI coverage; keep credentialed live-target tests opt-in.

## Validation

Use `validate-change` to select the complete matrix. The common adapter checks
are:

```bash
uv sync --group adapter-tests
uv run --no-sync pytest tests/adapters/test_<name>*.py
just test-python
just lock-python && just wheels
cargo run -p nemo-fabric-core --example generate-schemas -- schemas
cargo fmt --all -- --check && just test-rust
just docs
uv run pre-commit run --all-files --show-diff-on-failure
git diff --check
```

## References

- Public contract: `docs/adapter-contract/` and `schemas/adapter-contract/`.
- Descriptor schema: `schemas/adapter-contract/adapter-descriptor.schema.json`.
- Repository packaging: root `pyproject.toml`, `justfile`, adapter catalogs,
  and CI workflows.
- Shared first-party host patterns: `adapters/common/` and the closest adapter
  with the same target boundary.
