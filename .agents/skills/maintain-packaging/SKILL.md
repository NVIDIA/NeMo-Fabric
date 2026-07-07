---
name: maintain-packaging
description: Maintain NeMo Fabric Rust and Python package metadata, module paths, native artifacts, lockfiles, and release-facing build surfaces
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

## Checklist

- [ ] Package names, import paths, and module names are internally consistent
- [ ] Generated artifacts still land where downstream consumers expect
- [ ] Docs and examples use the current install/import/build commands
- [ ] CI references the same package names as local workflows
- [ ] Public packaging changes are reflected in release-facing docs
- [ ] Workspace, Python, and lockfile versions remain aligned where required
- [ ] The editable maturin build still produces `nemo_fabric._native`

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
- `justfile`
