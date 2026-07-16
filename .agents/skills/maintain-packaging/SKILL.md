---
name: maintain-packaging
description: Maintain NeMo Fabric Rust and Python dependencies, package metadata, module paths, native artifacts, lockfiles, license evidence, and release-facing build surfaces
author: NVIDIA Corporation and Affiliates
license: Apache-2.0
---


# Maintain Release And Packaging Surfaces

## Companion Guidance

Use `karpathy-guidelines` alongside this skill for implementation or review
work. Keep changes scoped, surface assumptions, and define focused validation
before editing.

Use this skill when a change affects how NeMo Fabric is built, packaged, named, or
consumed outside the source tree.

## Audit Areas

- Rust `Cargo.toml` package names and workspace metadata
- Python and maturin packaging in `pyproject.toml`
- Python package metadata in `python/pyproject.toml`
- Native extension naming and placement under `python/src/nemo_fabric`
- Dependency resolution in `Cargo.lock` and `uv.lock`
- Documentation tooling metadata in `docs/package.json` and
  `docs/package-lock.json`
- CI workflows, install commands, and example commands
- `justfile` build, test, clean, and documentation recipes

## Dependency Selection

Treat every direct dependency as a long-lived API, supply-chain, and licensing
commitment.

- First prefer the standard library, an existing dependency, or a small local
  implementation when it keeps the behavior clear and maintainable.
- When multiple dependencies satisfy the technical requirement, prefer the
  maintained OSS option with clear SPDX metadata, a smaller transitive graph,
  and permissive terms such as Apache-2.0, MIT, BSD, or ISC.
- Inspect the resolved transitive graph, not only the direct package license.
- Treat `UNKNOWN`, non-SPDX/custom, proprietary or source-available terms, and
  copyleft or network-copyleft terms as explicit review points. Do not silently
  accept or reject them; route them to the dependency approvers with the
  distribution and linkage context.
- Record the functional need, viable alternatives considered, why the selected
  dependency is the narrowest fit, and any unresolved licensing question.
- Run
  `uv run --no-project python scripts/licensing/license_diff.py --base-ref origin/main`
  after updating manifests and lockfiles, then review added packages and license
  changes.
- Regenerate the attribution files with the named pre-commit hooks instead of
  editing generated output:

  ```bash
  uv run pre-commit run --all-files attributions-rust
  uv run pre-commit run --all-files attributions-python
  ```

The license diff is evidence for reviewers. Dependency approvers make
compatibility decisions using the distribution and linkage context.

## Checklist

- [ ] Package names, import paths, and module names are internally consistent
- [ ] Generated artifacts still land where downstream consumers expect
- [ ] Docs and examples use the current install/import/build commands
- [ ] CI references the same package names as local workflows
- [ ] Public packaging changes are reflected in release-facing docs
- [ ] Workspace, Python, and lockfile versions remain aligned where required
- [ ] The editable maturin build still produces `nemo_fabric._native`
- [ ] New dependencies are necessary, maintained, and narrower than the viable
      alternatives
- [ ] Direct and transitive license changes were reviewed from the resolved
      lockfiles
- [ ] Licensing uncertainties are called out for dependency approver review
- [ ] Changed attribution files are regenerated and included

## References

- `pyproject.toml`
- `python/pyproject.toml`
- `Cargo.toml`
- `Cargo.lock`
- `uv.lock`
- `docs/package.json`
- `docs/package-lock.json`
- `.github/workflows/ci_python.yml`
- `.github/workflows/ci_rust.yml`
- `.pre-commit-config.yaml`
- `scripts/licensing/license_diff.py`
- `justfile`
