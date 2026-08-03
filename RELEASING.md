<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Releasing NeMo Fabric

This document is the maintainer playbook for cutting NeMo Fabric releases. It
describes the release contract, the version files that must be updated, the tag
format that CI accepts, the package surfaces that are published, and the checks
to run before and after a tag push.

## Source Of Truth

This section defines where release history and release-facing details are maintained.

- There is no `CHANGELOG.md` in this repository.
- The documentation site has a release-notes landing page for current
  documentation-visible release status.
- The source of truth for complete release history and tag-specific release
  notes is always GitHub Releases for this repository.

Do not copy full GitHub Release notes into `CHANGELOG.md` or the docs site.
The docs release-notes page can summarize support status and point users to
GitHub Releases.

## Published Surfaces

The release pipeline publishes these package surfaces from a tag push:

| Ecosystem | Published Surface |
|---|---|
| crates.io | `nemo-fabric-core`, `nemo-fabric-cli` |
| GitHub Actions | `nemo-fabric`, `nemo-fabric-runtime`, `nemo-fabric-adapters-common`, `nemo-fabric-adapters-claude`, `nemo-fabric-adapters-codex`, `nemo-fabric-adapters-deepagents`, and `nemo-fabric-adapters-hermes` wheel artifacts |
| Fern | The documentation site |

## Version Model

NeMo Fabric versions are anchored on the workspace SemVer in the repository root
`Cargo.toml`.

- The root `Cargo.toml` `workspace.package.version` is the canonical release
  version for the Rust workspace.
- The root `Cargo.toml` `workspace.dependencies` entry for
  `nemo-fabric-core` must stay aligned with that same version.
- The root `pyproject.toml` and every `adapters/**/pyproject.toml` carry the
  Python package versions and internal dependency pins and must stay aligned
  with the same release version.
- The `nemo-fabric-runtime` Python package version is derived at packaging time.
  `python/pyproject.toml` stays `dynamic = ["version"]` in the repository, and
  Maturin derives the version from `crates/fabric-python/Cargo.toml`, which
  inherits the workspace version.

## Release Tags

Release tags use SemVer with a leading `v`.

- Use `v0.1.0` for stable releases.
- Use `v0.1.0-rc.1` for prereleases.
- Do not use tags such as `0.1.0` or `0.1.0-rc.1`.

CI rejects tags that do not match the required format.

The tag text must match the version that the packaging jobs publish.

Release tags for a frozen release line should be created from the matching
`release/*` branch, not from `main`.

## Code Freeze

When code freeze begins for a target release, create a release branch from the
latest `main` commit. Name the branch from the target release major and minor
version. The `prepare-code-freeze` skill automates this process.

These examples assume `upstream` is the NVIDIA repository remote
(`NVIDIA/NeMo-Fabric`). The `origin` remote is usually a maintainer's personal
fork.

```bash
git fetch upstream main
git checkout -b release/0.2 upstream/main
git push upstream release/0.2
```

After creating the release branch, open a PR against `main` that does the
following:

1. Bump all package versions on `main` to the next release line:

   ```bash
   just set-version <next-version>
   ```

New PRs that must go into the upcoming release must target the new `release/*`
branch. Changes intended for later releases should continue to target `main`.

## Before You Cut A Release

Before you create a release tag, confirm the following:

1. The intended release commit is already on the release branch you intend to
   tag. For frozen release lines, tag the matching `release/*` branch.
2. The release commit contains the final version bump, docs updates, and any
   public API changes that belong in the release.
3. The working tree you use for local validation is clean or disposable.

## Prepare The Release Commit

Update the versioned source files in the release PR or release-prep commit.
Prefer the repository helper:

```bash
just set-version <release-version>
```

The helper updates:

1. The root [`Cargo.toml`](Cargo.toml) workspace version.
2. The root [`Cargo.toml`](Cargo.toml) `workspace.dependencies` versions for
   `nemo-fabric-core`.
3. [`pyproject.toml`](pyproject.toml), every `adapters/**/pyproject.toml`, and
   their internal dependency pins to the same release version.
4. [`Cargo.lock`](Cargo.lock), [`uv.lock`](uv.lock), and every Python project
   lockfile.
Review docs and snippets that mention explicit versions, including:

- [`README.md`](README.md)
- [`CONTRIBUTING.md`](CONTRIBUTING.md)
- [`docs/getting-started/install.mdx`](docs/getting-started/install.mdx)
- Any binding README or example that pins a release number

Do not commit a static Python package version into `python/pyproject.toml` just
to cut the release. Maturin derives that version from Cargo during the build.

## Local Validation

Run the checks that match the surfaces affected by the release. For a normal
repository release, the safest baseline is:

```bash
uv run pre-commit run --all-files
just test-rust
just test-python
just docs
```

If you want to validate the Python packaging recipe before pushing a tag, run:

```bash
just set-version 0.1.0
just wheels
```

Be aware that the `set-version` helper intentionally rewrites version fields in
place. In a disposable CI workspace that is fine.
In a local checkout, restore those temporary manifest edits before continuing if
you are not committing them.

## Prepare Release Notes

Before cutting the final release tag, prepare the release notes (OK to skip for
RC and alpha tags). You can perform these steps manually or use the
[`draft-release-notes`](.agents/skills/draft-release-notes/SKILL.md) skill.

Confirm the target release version from the release branch and package
metadata, then gather read-only evidence with explicit release refs:

```bash
python3 .agents/skills/draft-release-notes/scripts/collect_release_evidence.py \
  --previous release/<previous-major>.<previous-minor> \
  --current HEAD \
  --version <major>.<minor>
```

Treat the report as an evidence index, not publication-ready copy. Verify every
candidate claim in the changed public docs, API types, command help, or source.
Prioritize breaking changes, migrations, user-visible features, and ongoing
support limitations.

Update [`docs/about-nemo-fabric/release-notes.mdx`](docs/about-nemo-fabric/release-notes.mdx). Preserve the MDX front matter and JSX SPDX comment. State the full history
is available in GitHub Releases..

Review product names, commands, package names, support claims, and links, then
validate the draft:

```bash
git diff --check
just docs
```

## Cut The Tag

After the release commit is merged and validated, create and push the raw
SemVer tag:

```bash
git fetch upstream release/0.1
git checkout release/0.1
git pull --ff-only upstream release/0.1
git tag -as -m "$(date +"%B %Y") Release" v0.1.0

# Verify the tag is correct
git tag -l v0.1.0
git show v0.1.0

git push upstream v0.1.0
```

Use the prerelease form when needed:

```bash
git tag -as -m "v0.1.0-rc.1 Release" v0.1.0-rc.1

# Check the tag
git tag -l v0.1.0-rc.1
git show v0.1.0-rc.1

# Push the tag
git push upstream v0.1.0-rc.1
```

## What CI Does On A Tag Push

Pushing a valid tag triggers
[`.github/workflows/ci_python.yml`](.github/workflows/ci_python.yml),
[`.github/workflows/publish_rust.yml`](.github/workflows/publish_rust.yml), and
[`.github/workflows/fern-docs.yml`](.github/workflows/fern-docs.yml).

The release pipeline then:

1. Validates the tag format with `just set-version` or
   `normalize_release_tag.py`.
2. Builds platform `nemo-fabric-runtime` wheels and pure-Python
   `nemo-fabric` and adapter wheels with the exact tag version, then uploads
   them as GitHub Actions artifacts.
3. Publishes `nemo-fabric-core` and `nemo-fabric-cli` to crates.io through
   trusted publishing for stable, beta, and RC tags. Alpha tags are not
   published to crates.io.
4. Publishes Fern documentation versions for stable, beta, and RC tags. Alpha
   tags do not publish a separate documentation version.

The workflow boundary is split intentionally:

- [`.github/workflows/ci_python.yml`](.github/workflows/ci_python.yml) produces
  Python wheel artifacts.
- [`.github/workflows/fern-docs.yml`](.github/workflows/fern-docs.yml) validates
  and publishes Fern documentation independently from package CI.
- [`.github/workflows/publish_rust.yml`](.github/workflows/publish_rust.yml)
  owns crates.io publication decisions and credentials.


## Publish The GitHub Release Entry

- [ ] Click the "Releases" button on the right at the repo homepage
- [ ] Click "Tags" and select the tag you just pushed
- [ ] Click "Create release from tag"
- [ ] Click the "Generate release notes" button which will pre-populate the body with information about new contributors and a link to the full diff.

## Post-Release Checks

After the release is live, verify:

1. The `nemo-fabric-core` and `nemo-fabric-cli` crates are visible on crates.io.
2. The Python wheels are available on https://pypi.org/.
3. The Python wheels are available on https://pypi.nvidia.com/
4. The Fern documentation site shows the expected version and release notes.
5. The GitHub Release page is complete and accurate.
