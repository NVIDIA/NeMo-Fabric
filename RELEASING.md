<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Releasing NVIDIA NeMo Fabric

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
| npm | `@nvidia/nemo-fabric-adapter-contract` |
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
- `typescript/adapter-contract/package.json` and its lockfile carry the npm
  adapter-contract package version and must stay aligned with the same release
  version. The package version is independent of the
  `fabric.adapter/v1alpha2` wire contract version.
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

## Patch Releases

Cut a patch release from the existing release branch for that major and minor
line. Do not create another release branch or run the code-freeze workflow.

Set the exact patch version, previous stable tag, and existing release branch:

```bash
export RELEASE_VERSION=0.1.1
export PREVIOUS_RELEASE_TAG=v0.1.0
export RELEASE_BRANCH=release/0.1

git fetch upstream "${RELEASE_BRANCH}" --tags
git log --oneline "${PREVIOUS_RELEASE_TAG}..upstream/${RELEASE_BRANCH}"
```

Open the fix or release-preparation PR against `${RELEASE_BRANCH}`. The PR must
contain the intended patch changes and run `just set-version <release-version>`
so the release branch contains the final package version before tagging. Verify
the commit range from `${PREVIOUS_RELEASE_TAG}` contains only changes intended
for the patch release. Changes required on `main` should be handled separately;
do not mix a forward merge into the patch release.

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
4. [`typescript/adapter-contract/package.json`](typescript/adapter-contract/package.json)
   and its npm lockfile.
5. [`Cargo.lock`](Cargo.lock), [`uv.lock`](uv.lock), and every Python project
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
just test-typescript
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

## Bootstrap npm Trusted Publishing

The npm package must exist before npm can bind it to a GitHub trusted publisher.
This is a one-time bootstrap for
`@nvidia/nemo-fabric-adapter-contract`; normal releases use OpenID Connect (OIDC)
and do not use an npm write token in GitHub Actions.

Before the first TypeScript package release:

1. Create and protect the GitHub `npmjs` environment. Require the release
   approvers who should authorize registry publication, and restrict deployment
   tags to `v*`.
2. Commit the intended prerelease version on the release branch. From that
   exact clean commit, run the version helper as an idempotency check and run
   the same package checks used by CI. Use the real first release candidate
   rather than a disposable version because npm versions are immutable. Use
   Node.js 24 and npm 11.5.1 or later, matching the trusted-publishing workflow,
   so the later integrity reconciliation uses the same packaging toolchain:

   ```bash
   just set-version 0.2.0-rc.1
   git diff --exit-code
   just test-typescript
   cd typescript/adapter-contract
   npm login
   npm publish --access public --tag next
   npm logout
   ```

   The publisher needs write access to the `@nvidia` scope and account-level
   two-factor authentication. Do not push the matching release tag yet.
3. In the npm package settings, configure the single trusted publisher with
   these exact, case-sensitive values:

   - Organization or user: `NVIDIA`
   - Repository: `NeMo-Fabric`
   - Workflow filename: `publish_typescript.yml`
   - Environment: `npmjs`
   - Allowed action: `npm publish`

4. Push the signed tag for that already-published release candidate and approve
   the `npmjs` environment when prompted. The workflow exits without
   republishing only when its repack has byte-identical integrity and the
   `next` dist-tag is exact. A mismatch fails closed because npm versions are
   immutable; do not overwrite or weaken the check. Use the next genuine
   release version for the first OIDC publication instead.
5. After a later release publishes through OIDC, confirm its provenance on npm.
   In the npm package settings, require two-factor authentication and disallow
   token publication. Then remove or revoke any local or automation credentials
   used for bootstrap.

The workflow publishes stable versions with the `latest` dist-tag and beta or
RC versions with `next`. Alpha versions are not published. A retry skips only
when the immutable package version, packed artifact integrity, and expected
dist-tag all match. If any of them differs, the workflow fails so a maintainer
can inspect and repair the registry state explicitly. Publication also fails
rather than moving `latest` or `next` backward when cutting a patch from an
older release line.

## Cut An RC Tag

After the release commit is merged and validated, create and push a signed,
annotated tag. Set the complete release-candidate version:

```bash
export RELEASE_VERSION=0.1.0-rc.1
export RELEASE_BRANCH=release/0.1
export RELEASE_TAG="v${RELEASE_VERSION}"
echo "Cutting release tag ${RELEASE_TAG} for release branch ${RELEASE_BRANCH}"

git fetch upstream "${RELEASE_BRANCH}" --tags
git switch "${RELEASE_BRANCH}"
git pull --ff-only upstream "${RELEASE_BRANCH}"

test -z "$(git status --porcelain)"
RELEASE_SHA="$(git rev-parse HEAD)"
REMOTE_RELEASE_SHA="$(git rev-parse "upstream/${RELEASE_BRANCH}^{commit}")"
test "${RELEASE_SHA}" = "${REMOTE_RELEASE_SHA}"
test "$(just normalize-release-tag "${RELEASE_TAG}")" = "${RELEASE_VERSION}"

if git ls-remote --exit-code --tags upstream "refs/tags/${RELEASE_TAG}" >/dev/null; then
  echo "Error: remote tag ${RELEASE_TAG} already exists" >&2
  exit 1
fi

git tag -s -a \
  -m "NVIDIA NeMo Fabric ${RELEASE_VERSION}" \
  "${RELEASE_TAG}" \
  "${RELEASE_SHA}"

git tag -v "${RELEASE_TAG}"
git show "${RELEASE_TAG}"
test "$(git rev-parse "${RELEASE_TAG}^{commit}")" = "${RELEASE_SHA}"

git push upstream "refs/tags/${RELEASE_TAG}"
```


## Prepare Release Notes

Before cutting the final release tag, prepare the release notes (OK to skip for
RC and alpha tags). You can perform these steps manually or use the
[`draft-release-notes`](.agents/skills/draft-release-notes/SKILL.md) skill.

Confirm the exact target release version from the release branch and package
metadata, then gather read-only evidence with explicit release refs. For a
patch release, use the previous stable tag as `--previous`. For a new release
line, use the previous release branch or tag:

```bash
python3 .agents/skills/draft-release-notes/scripts/collect_release_evidence.py \
  --previous <previous-release-tag-or-branch> \
  --current HEAD \
  --version <release-version>
```

Treat the report as an evidence index, not publication-ready copy. Verify every
candidate claim in the changed public docs, API types, command help, or source.
Prioritize breaking changes, migrations, user-visible features, and ongoing
support limitations.

Draft the authoritative GitHub Release body for every stable release. Summarize
the user-visible changes, compatibility or migration requirements, known
limitations, and verified fixes. For a patch release, state the affected
behavior and whether public APIs, configuration, or dependency contracts
changed.

Update [`docs/about-nemo-fabric/release-notes.mdx`](docs/about-nemo-fabric/release-notes.mdx)
only when the release changes the documentation-visible summary, compatibility
guidance, support status, or limitations. A patch release that does not change
those surfaces can leave the page unchanged. Preserve the MDX front matter and
JSX SPDX comment when editing it, and state that the full history is available
in GitHub Releases.

Review product names, commands, package names, support claims, and links. If the
documentation page changed, validate it with:

```bash
git diff --check
just docs
```

## Cut The Tag

After the release commit is merged and validated, create and push the signed,
annotated release tag. Set the exact stable release version and matching release
branch:

```bash
export RELEASE_VERSION=0.1.0
export RELEASE_BRANCH=release/0.1
export RELEASE_TAG="v${RELEASE_VERSION}"
echo "Cutting release tag ${RELEASE_TAG} for release branch ${RELEASE_BRANCH}"

git fetch upstream "${RELEASE_BRANCH}" --tags
git switch "${RELEASE_BRANCH}"
git pull --ff-only upstream "${RELEASE_BRANCH}"

test -z "$(git status --porcelain)"
RELEASE_SHA="$(git rev-parse HEAD)"
REMOTE_RELEASE_SHA="$(git rev-parse "upstream/${RELEASE_BRANCH}^{commit}")"
test "${RELEASE_SHA}" = "${REMOTE_RELEASE_SHA}"
test "$(just normalize-release-tag "${RELEASE_TAG}")" = "${RELEASE_VERSION}"
CURRENT_VERSION="$(sed -n 's/^version = "\(.*\)"$/\1/p' Cargo.toml | head -n 1)"
test "${CURRENT_VERSION}" = "${RELEASE_VERSION}"

if git ls-remote --exit-code --tags upstream "refs/tags/${RELEASE_TAG}" >/dev/null; then
  echo "Error: remote tag ${RELEASE_TAG} already exists" >&2
  exit 1
fi

git tag -s -a \
  -m "NVIDIA NeMo Fabric ${RELEASE_VERSION}" \
  "${RELEASE_TAG}" \
  "${RELEASE_SHA}"

git tag -v "${RELEASE_TAG}"
git show "${RELEASE_TAG}"
test "$(git rev-parse "${RELEASE_TAG}^{commit}")" = "${RELEASE_SHA}"

git push upstream "refs/tags/${RELEASE_TAG}"
```


## What CI Does On A Tag Push

Pushing a valid tag triggers :

| Workflow | Trigger |
|---|---|
| [`.github/workflows/ci_python.yml`](.github/workflows/ci_python.yml) | For all tags including alpha |
| [`.github/workflows/publish_rust.yml`](.github/workflows/publish_rust.yml) | For RC, beta and release tags |
| [`.github/workflows/publish_typescript.yml`](.github/workflows/publish_typescript.yml) | For RC, beta and release tags |
| [`.github/workflows/fern-docs.yml`](.github/workflows/fern-docs.yml) | For RC, beta and release tags |

The release pipeline then:

1. Validates the tag format with `just set-version` or
   `normalize_release_tag.py`.
2. Builds platform `nemo-fabric-runtime` wheels and pure-Python
   `nemo-fabric` and adapter wheels with the exact tag version, then uploads
   them as GitHub Actions artifacts.
3. Publishes `nemo-fabric-core` and `nemo-fabric-cli` to crates.io through
   trusted publishing for stable, beta, and RC tags. Alpha tags are not
   published to crates.io.
4. Publishes `@nvidia/nemo-fabric-adapter-contract` to npm through trusted
   publishing for stable, beta, and RC tags. Stable releases use the `latest`
   dist-tag; beta and RC releases use `next`. Alpha tags are not published to
   npm.
5. Publishes Fern documentation versions for stable, beta, and RC tags. Alpha
   tags do not publish a separate documentation version.

The workflow boundary is split intentionally:

- [`.github/workflows/ci_python.yml`](.github/workflows/ci_python.yml) produces
  Python wheel artifacts.
- [`.github/workflows/fern-docs.yml`](.github/workflows/fern-docs.yml) validates
  and publishes Fern documentation independently from package CI.
- [`.github/workflows/publish_rust.yml`](.github/workflows/publish_rust.yml)
  owns crates.io publication decisions and credentials.
- [`.github/workflows/publish_typescript.yml`](.github/workflows/publish_typescript.yml)
  owns npm publication decisions and requests a short-lived npm credential
  through GitHub OIDC. It does not receive an npm write token.


## Publish The GitHub Release Entry

1. Open **Releases** from the repository page.
2. Select **Tags**, select the tag that you pushed, and select **Create release
   from tag**.
3. Set the release title to `NVIDIA NeMo Fabric <release-version>`.
4. Paste the verified GitHub Release body prepared before tagging. You can use
   **Generate release notes** as an evidence source for contributors, included
   pull requests, and the full comparison link, but review and curate the
   generated text before publishing.
5. Confirm the selected tag and target commit match the signed tag that you
   verified locally. Publish a stable release as the latest release and mark a
   prerelease appropriately.

## Post-Release Checks

After the release is live, verify:

1. The `nemo-fabric-core` and `nemo-fabric-cli` crates are visible on crates.io.
2. The Python wheels are available on PyPI:
   - [`nemo-fabric`](https://pypi.org/project/nemo-fabric/)
   - [`nemo-fabric-runtime`](https://pypi.org/project/nemo-fabric-runtime/) Ensure that a wheel exists for each supported platform.
   - [`nemo-fabric-adapters-common`](https://pypi.org/project/nemo-fabric-adapters-common/)
   - [`nemo-fabric-adapters-claude`](https://pypi.org/project/nemo-fabric-adapters-claude/)
   - [`nemo-fabric-adapters-codex`](https://pypi.org/project/nemo-fabric-adapters-codex/)
   - [`nemo-fabric-adapters-deepagents`](https://pypi.org/project/nemo-fabric-adapters-deepagents/)
   - [`nemo-fabric-adapters-hermes`](https://pypi.org/project/nemo-fabric-adapters-hermes/)
3. The Python wheels are available on NVIDIA PyPI:
   - [`nemo-fabric`](https://pypi.nvidia.com/nemo-fabric/)
   - [`nemo-fabric-runtime`](https://pypi.nvidia.com/nemo-fabric-runtime/) Ensure that a wheel exists for each supported platform.
   - [`nemo-fabric-adapters-common`](https://pypi.nvidia.com/nemo-fabric-adapters-common/)
   - [`nemo-fabric-adapters-claude`](https://pypi.nvidia.com/nemo-fabric-adapters-claude/)
   - [`nemo-fabric-adapters-codex`](https://pypi.nvidia.com/nemo-fabric-adapters-codex/)
   - [`nemo-fabric-adapters-deepagents`](https://pypi.nvidia.com/nemo-fabric-adapters-deepagents/)
   - [`nemo-fabric-adapters-hermes`](https://pypi.nvidia.com/nemo-fabric-adapters-hermes/)
4. The TypeScript contract package is visible on npm with the expected version,
   dist-tag, and provenance:

   ```bash
   npm view "@nvidia/nemo-fabric-adapter-contract@<release-version>" version
   npm view "@nvidia/nemo-fabric-adapter-contract" dist-tags
   ```

5. The Fern documentation site shows the expected version and release notes.
6. The GitHub Release page is complete and accurate.
