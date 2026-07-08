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
- Keep `FabricAgent.version()` in
  `python/src/nemo_fabric/integrations/harbor/__init__.py` aligned with the
  released Python package version.

For a normal release, use the same `X.Y.Z` string everywhere. For a prerelease
or build-metadata version, use valid Cargo SemVer in `Cargo.toml` and the
equivalent PEP 440 version in explicit Python metadata. Confirm that the
Maturin-built runtime and setuptools-built packages resolve to equivalent
versions rather than blindly copying incompatible syntax.

## Workflow

1. Read the current version from `Cargo.toml` and decide the exact target
   version string.
2. Run `just set-version <version>` to update release-version source files:
   - `[workspace.package].version`
3. Update all setuptools based `pyproject.toml` files listed above:
   - Set each `project.version` to the Python target version.
   - Update every internal `nemo-fabric-*` exact-version requirement.
4. Leave `python/pyproject.toml` dynamic and unchanged unless its Maturin
   configuration itself needs correction.
5. Update `FabricAgent.version()` to the Python target version.
6. Refresh generated dependency state:
   - Run `cargo check --workspace` to update `Cargo.lock` workspace package
     entries.
   - Run `just lock-python` to update the root, runtime, and adapter
     `uv.lock` files.
7. Audit references to the old version with targeted searches. Distinguish
   package-version surfaces from examples and unrelated dependency versions.

## Validation

- Inspect Cargo version fields:
  `rg -n '^version =|fabric-core = \{ path = .*version =' Cargo.toml`
- Inspect explicit Python versions and internal pins:
  `rg -n '^version =|nemo-fabric-[a-z-]+ == ' pyproject.toml adapters/*/pyproject.toml`
- Confirm the runtime remains dynamic:
  `rg -n 'dynamic = \["version"\]' python/pyproject.toml`
- Confirm the Harbor integration version:
  `rg -n 'return "[0-9]' python/src/nemo_fabric/integrations/harbor/__init__.py`
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
