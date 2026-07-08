---
name: update-project-version
description: Update the NeMo Fabric release version across Cargo, setuptools package metadata, internal Python dependency pins, integration metadata, and lockfiles. Use when bumping, synchronizing, or auditing Fabric package versions for a release.
author: NVIDIA Corporation and Affiliates
license: Apache-2.0
---


# Update Project Version

## Companion Guidance

Use `karpathy-guidelines` alongside this skill for implementation or review
work. Keep changes scoped, surface assumptions, and define focused validation
before editing.

Use this skill when changing the NeMo Fabric version, including
pre-release or build-metadata variants used during packaging.

## Source Of Truth

- `Cargo.toml` `[workspace.package].version` is the source of truth for the Rust
  workspace and Python build versioning.
- Keep `Cargo.toml` `[workspace.dependencies]` self-references aligned when the
  workspace version changes.
- `python/pyproject.toml` is the exception among the Python projects: do not add
  a literal `project.version`. Keep `project.dynamic = ["version"]` because
  Maturin derives `nemo-fabric-runtime`'s version from
  `crates/fabric-python/Cargo.toml`, which inherits the workspace version.
- The setuptools projects do not derive their versions from Cargo. Update the
  literal `project.version` in every one of these files:
  - `pyproject.toml`
  - `adapters/**/pyproject.toml`
- Keep internal Python package requirement pins aligned with the Python release
  version:
  - All `nemo-fabric-* == <version>` requirements in the root
    `pyproject.toml` optional dependencies.
  - Each adapter's `nemo-fabric-adapters-common == <version>` dependency.

For a normal release, use the same `X.Y.Z` string everywhere. For a prerelease
or build-metadata version, use valid Cargo SemVer in `Cargo.toml` and the
equivalent PEP 440 version in explicit Python metadata. Confirm that the
Maturin-built runtime and setuptools-built packages resolve to equivalent
versions rather than blindly copying incompatible syntax.

## Workflow

1. Read the current version from `Cargo.toml` and decide the exact Cargo and
   Python target version strings.
2. Run `just set-version <cargo-version>`. The recipe converts supported Cargo
   SemVer prereleases to PEP 440 and updates:
   - `Cargo.toml` `[workspace.package].version`
   - `Cargo.toml` `workspace.dependencies.fabric-core.version`
   - All five setuptools `project.version` fields
   - Every internal `nemo-fabric-*` exact-version requirement
   - `FabricAgent.version()`
   - `Cargo.lock` through Cargo metadata resolution
   - The root, runtime, and adapter `uv.lock` files through `just lock-python`
3. Confirm that `python/pyproject.toml` remains dynamic and unchanged.
4. Audit references to the old version with targeted searches. Distinguish
   package-version surfaces from examples and unrelated dependency versions.

If editing the helper code, keep these contracts aligned:

- `set_project_version` must call the Cargo, Python project, and Harbor
  integration version helpers.
- `set_cargo_workspace_version` must update the workspace version and the
  `fabric-core` workspace dependency, then verify every `fabric-*` workspace
  package through Cargo metadata.
- `set_python_project_versions` must update all five explicit setuptools
  versions and all internal exact-version pins while rejecting a static version
  in `python/pyproject.toml`.
- `set_harbor_integration_version` must update `FabricAgent.version()`.
- The `set-version` recipe must run `just lock-python` after source metadata is
  updated.

## Validation

- Inspect Cargo version fields:
  `rg -n '^version =|fabric-core = \{ path = .*version =' Cargo.toml`
- Inspect explicit Python versions and internal pins:
  `rg -n '^version =|nemo-fabric-[a-z-]+ == ' pyproject.toml adapters/*/pyproject.toml`
- Confirm the runtime remains dynamic:
  `rg -n 'dynamic = \["version"\]' python/pyproject.toml`
- Run `cargo check --workspace --locked`.
- Run `just build-python` to verify all Python package metadata resolves.
- Run `just test-python` when the integration version or Python packaging
  behavior changes materially.
- Run `just wheels` for release-facing validation of every Python wheel.
- Run `git diff --check`.

## Avoid

- Updating only `Cargo.toml` and leaving the setuptools packages stale.
- Adding a literal version to `python/pyproject.toml`; Maturin owns that version.
- Updating Python package versions without their exact internal dependency pins.
- Forgetting `Cargo.lock`, the root `uv.lock`, or per-project `uv.lock` files.
- Blind repository-wide replacement of version-like strings.

## References

- `Cargo.toml`
- `Cargo.lock`
- `pyproject.toml`
- `uv.lock`
- `python/pyproject.toml`
- `python/uv.lock`
- `adapters/**/pyproject.toml`
- `adapters/**/uv.lock`
- `justfile`
