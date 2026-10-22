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

The release process publishes these package surfaces from canonical and
package-specific tag pushes:

| Ecosystem | Published Surface |
|---|---|
| crates.io | `nemo-fabric-core`, `nemo-fabric-cli` |
| npm | `nemo-fabric-adapter-contract`, `nemo-fabric-adapters-common`, `nemo-fabric-adapters-droid`, `nemo-fabric-adapters-pi`, `nemo-fabric-adapters-opencode`, `nemo-fabric-adapters-qwen`, `nemo-fabric-adapters-kiro` |
| GitHub Actions | `nemo-fabric`, `nemo-fabric-runtime`, `nemo-fabric-collector`, `nemo-fabric-adapter-catalog`, `nemo-fabric-adapters-common`, `nemo-fabric-adapters-claude`, `nemo-fabric-adapters-codex`, `nemo-fabric-adapters-deepagents`, `nemo-fabric-adapters-hermes`, and `nemo-fabric-adapters-nooa` wheel artifacts |
| Fern | The documentation site |

## Version Model

NeMo Fabric versions are anchored on the workspace SemVer in the repository root
`Cargo.toml`.

- The root `Cargo.toml` `workspace.package.version` is the canonical release
  version for the Rust workspace.
- The root `Cargo.toml` `workspace.dependencies` entry for
  `nemo-fabric-core` must stay aligned with that same version.
- `sdk/python/nemo-fabric/pyproject.toml`,
  `sdk/python/nemo-fabric-collector/pyproject.toml`,
  `sdk/python/nemo-fabric-adapter-catalog/pyproject.toml`,
  `adapter-contract/python/pyproject.toml`, and every
  `adapters/python/*/pyproject.toml` carry the Python package versions and internal
  dependency pins and must stay aligned with the same release version. The
  root `pyproject.toml` is a private development coordinator and remains at
  `0.0.0`.
- The TypeScript contract, Common adapter support package, bundled TypeScript
  adapters, private adapter workspace, exact internal dependency pins, and their
  lockfiles all use the canonical NeMo Fabric release version. The package
  version is independent of the `fabric.adapter/v1alpha2` wire contract version.
- The `nemo-fabric-runtime` Python package version is derived at packaging time.
  `sdk/python/nemo-fabric-runtime/pyproject.toml` stays
  `dynamic = ["version"]` in the repository, and Maturin derives the version
  from `crates/fabric-python/Cargo.toml`, which inherits the workspace version.

## Release Tags

Release tags use SemVer with a leading `v`.

- Use `v0.1.0` for stable releases.
- Use `v0.1.0-rc.1` for prereleases.
- Do not use tags such as `0.1.0` or `0.1.0-rc.1`.

CI rejects tags that do not match the required format.

The tag text must match the version that the packaging jobs publish.

Release tags for a frozen release line should be created from the matching
`release/*` branch, not from `main`.


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


## Cut An RC Tag

RC Tags should be created when soon after code freeze to allow for QA testing of the upcoming release. The `create-rc-tag` skill automates this process.

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
BASE_RELEASE_VERSION="${RELEASE_VERSION%%-*}"
CURRENT_VERSION="$(sed -n 's/^version = "\(.*\)"$/\1/p' Cargo.toml | head -n 1)"
test "${CURRENT_VERSION}" = "${BASE_RELEASE_VERSION}"

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


## Before You Cut A Release

Before you create a release tag, confirm the following:

1. The intended release commit is already on the release branch you intend to
   tag. For frozen release lines, tag the matching `release/*` branch.
2. The release commit contains the final stable base version, docs updates, and
   any public API changes that belong in the release. Prerelease versions are
   derived from beta or RC tags in disposable workflow checkouts.
3. The working tree you use for local validation is clean or disposable.

## Prepare The Release Commit

Update the versioned source files in the release PR or release-prep commit to
the stable base version for the release line. Prefer the repository helper:

```bash
just set-version <stable-base-version>
# For example: just set-version 0.1.0
```

For beta and RC tags, keep the committed source metadata at the corresponding
stable base version. The tag workflows normalize the prerelease tag and stamp
ecosystem-specific package metadata in their disposable checkouts. Do not
commit RC-specific versions or internal dependency pins to the release branch.

The adapter catalog is a separate, metadata-only wheel. `just set-version`
regenerates its resource bundle after stamping both ecosystems. `just wheels`
checks freshness and builds it with the other distributions. Run
`just adapter-catalog` after descriptor edits; do not edit the generated bundle
by hand. Catalog resources do not register execution runners.

The catalog follows the existing nightly and release publication cadence:
GitHub Actions builds it in `python-wheels-*`, GitLab collects its
platform-independent wheel, and KitMaker publishes it to PyPI using the
`nemo-fabric-adapter-catalog` project. That project must exist in KitMaker before
the first publication. No separate package tag or manual upload is required.
PR CI tests sdist-to-wheel builds, isolated installation without harness SDKs,
and release dispatch with external writes mocked.

The helper updates:

1. The root [`Cargo.toml`](Cargo.toml) workspace version.
2. The root [`Cargo.toml`](Cargo.toml) `workspace.dependencies` versions for
   `nemo-fabric-core`.
3. [`sdk/python/nemo-fabric/pyproject.toml`](sdk/python/nemo-fabric/pyproject.toml),
   [`sdk/python/nemo-fabric-adapter-catalog/pyproject.toml`](sdk/python/nemo-fabric-adapter-catalog/pyproject.toml),
   [`adapter-contract/python/pyproject.toml`](adapter-contract/python/pyproject.toml),
   every `adapters/python/*/pyproject.toml`, and their internal dependency pins to
   the same release version.
4. The TypeScript contract and adapter package manifests, their exact internal
   dependency pins, and both npm lockfiles.
5. [`Cargo.lock`](Cargo.lock), [`uv.lock`](uv.lock), and every Python project
   lockfile.

Review docs and snippets that mention explicit versions, including:

- [`README.md`](README.md)
- [`CONTRIBUTING.md`](CONTRIBUTING.md)
- [`docs/getting-started/install.mdx`](docs/getting-started/install.mdx)
- Any binding README or example that pins a release number

Do not commit a static Python package version into
`sdk/python/nemo-fabric-runtime/pyproject.toml` just to cut the release.
Maturin derives that version from Cargo during the build.

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

An npm package must exist before npm can bind it to a GitHub trusted publisher.
For every new public TypeScript package, complete bootstrap during its contribution
PR, before cutting a release tag. TypeScript CI checks public package metadata;
the release workflow checks every package before uploading any package. HTTP 404
means the package is absent. Other HTTP or network failures mean the check could
not establish existence and must be investigated or retried.

List public packages from the contract and adapter workspace manifests
(no static package-name list):

```bash
python3 scripts/ci/check_npm_bootstrap.py --list-directories
python3 scripts/ci/check_npm_bootstrap.py
```

The bootstrap check discovers future public workspaces automatically and skips
private workspaces. Publication retains its explicit package list, aligned with
release version stamping and build/test coverage. When adding a package, update
all three surfaces before enabling its publication. The discovered directory list
is informational; it does not select uploads. A successful check proves public
existence only, not ownership, trusted-publisher settings, or the presence of a
proposed release version.
Contributors without npm permissions should ask an NVIDIA npm maintainer to
complete the following steps, then rerun the PR check.

1. Read the exact `name` from the new package's `package.json`. On an authorized
   maintainer's machine, authenticate with npm browser login and account-level
   two-factor authentication. Never add an npm write token to CI.
2. In a temporary directory outside the source package, prepare an inert bootstrap:
   a `package.json` with that exact name, version `0.0.0`, license `Apache-2.0`,
   and `files: ["README.md", "LICENSE"]`; a README explaining that this version
   reserves the package name and contains no usable implementation; and the
   repository's Apache-2.0 LICENSE. Include no dependencies, entry points,
   executables, lifecycle scripts, or source code. Inspect `npm pack --dry-run`
   and publish from that directory with:

   ```bash
   npm publish --registry https://registry.npmjs.org --access public --tag next --ignore-scripts
   ```

   Bootstrap only an absent package. npm versions are immutable: do not publish
   `0.0.0` again or manually publish a supported version. npm may initialize
   `latest` to the first `0.0.0` and reject removing its only `latest` tag; leave
   it in place. RC/beta releases move `next`; the first stable replaces `latest`.
3. Add another NVIDIA maintainer with `npm owner add <account> <package-name>`.
   Owner/trust changes require authenticated browser 2FA. The invited owner must
   accept the email invitation before access is granted. Verify accepted owners
   with `npm owner ls <package-name>`.
4. Protect GitHub's `npmjs` environment with release approvers and canonical
   `v*` deployment tags. Configure each package's trusted publisher with the
   case-sensitive organization `NVIDIA`, repository `NeMo-Fabric`, workflow
   filename `publish_typescript.yml`, environment `npmjs`, and allowed action
   `npm publish`. With npm CLI 11.16.0, configure and read back:

   ```bash
   npm trust github <package-name> --file publish_typescript.yml --repo NVIDIA/NeMo-Fabric --env npmjs --allow-publish --yes
   npm trust list <package-name>
   npm view <package-name> name --registry https://registry.npmjs.org
   ```

   See the [npm trust CLI reference](https://docs.npmjs.com/cli/v11/commands/npm-trust/)
   for administration commands. Authenticated trust readback must confirm all
   settings; public `npm view` and CI cannot verify this private configuration. Require package 2FA and disallow
   token publication in npm settings. Remove or revoke local bootstrap credentials
   after administration is complete.
   Recheck trust immediately before release: npm expires unvalidated
   configurations after 48 hours without a successful publish. If bootstrap
   happened earlier, an npm maintainer may need to recreate expired trust near
   release; do not manually publish a supported version to validate it. See
   [npm's trust expiry announcement](https://github.blog/changelog/2026-10-02-unvalidated-npm-trusted-publishing-configurations-now-expire/).
5. Cut the canonical release tag and approve the `npmjs` environment. Supported
   versions are published through GitHub OIDC. Confirm release provenance on npm.
   If a release job already failed because bootstrap or trust settings were missing,
   repair those settings and rerun the failed job on the existing tag. Do not recut
   tags or manually publish supported versions. Retry skips a version only when
   repacking proves byte-identical integrity and the expected dist-tag matches;
   inspect mismatches rather than weakening the immutable-version checks.

The workflows publish stable versions with the `latest` dist-tag and beta and
RC versions with `next`. Alpha versions are validated but are not published to
the public npm registry. Stable releases do not move the prerelease dist-tag. A
retry skips only when the immutable package version, packed artifact integrity,
and expected dist-tag all match. If any of them differs, the workflow fails so
a maintainer can inspect and repair the registry state explicitly. Publication
also fails rather than moving `latest` or `next` backward when cutting a patch
from an older release line.



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

## Publish the TypeScript Adapter Packages

Pushing the canonical beta, RC, or stable tag publishes the TypeScript contract
and bundled TypeScript adapter packages in dependency order. The workflow
submits every package before it starts deployment verification, then verifies
them in the same order. Do not create package-specific npm tags. The nightly
workflow runs the TypeScript tests against alpha tags without publishing to npm.


## What CI Does on a Canonical Tag Push

Pushing a valid canonical tag triggers:

| Workflow | Trigger |
|---|---|
| [`.github/workflows/ci_python.yml`](.github/workflows/ci_python.yml) | For all tags including alpha |
| [`.github/workflows/publish_rust.yml`](.github/workflows/publish_rust.yml) | For RC, beta and release tags |
| [`.github/workflows/ci_typescript.yml`](.github/workflows/ci_typescript.yml) | For nightly alpha tags and normal pull request/main validation |
| [`.github/workflows/publish_typescript.yml`](.github/workflows/publish_typescript.yml) | Publishes and verifies the TypeScript contract and bundled adapters for beta, RC, and stable tags |
| [`.github/workflows/fern-docs.yml`](.github/workflows/fern-docs.yml) | For RC, beta and release tags |

The release pipeline then:

1. Normalizes and validates the tag format, then stamps ecosystem-specific
   package metadata in each disposable workflow checkout.
2. Builds platform `nemo-fabric-runtime` wheels and pure-Python
   `nemo-fabric`, `nemo-fabric-collector`, and adapter wheels with the exact tag
   version, then uploads them as GitHub Actions artifacts.
3. Publishes `nemo-fabric-core` and `nemo-fabric-cli` to crates.io through
   trusted publishing for stable, beta, and RC tags. Alpha tags are not
   published to crates.io.
4. Publishes the contract and bundled TypeScript adapter packages to npm from
   the canonical tag in dependency order, then verifies the deployment in that order. Stable
   releases use the `latest` dist-tag and beta and RC releases use `next`.
   Alpha tags validate the packages without publishing them.
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
  owns TypeScript contract and adapter npm publication decisions. It requests a
  short-lived credential through GitHub OIDC and does not receive an npm write
  token.


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
   - [`nemo-fabric-collector`](https://pypi.org/project/nemo-fabric-collector/)
   - [`nemo-fabric-adapters-common`](https://pypi.org/project/nemo-fabric-adapters-common/)
   - [`nemo-fabric-adapters-claude`](https://pypi.org/project/nemo-fabric-adapters-claude/)
   - [`nemo-fabric-adapters-codex`](https://pypi.org/project/nemo-fabric-adapters-codex/)
   - [`nemo-fabric-adapters-deepagents`](https://pypi.org/project/nemo-fabric-adapters-deepagents/)
   - [`nemo-fabric-adapters-hermes`](https://pypi.org/project/nemo-fabric-adapters-hermes/)
   - [`nemo-fabric-adapters-nooa`](https://pypi.org/project/nemo-fabric-adapters-nooa/)
3. The Python wheels are available on NVIDIA PyPI:
   - [`nemo-fabric`](https://pypi.nvidia.com/nemo-fabric/)
   - [`nemo-fabric-runtime`](https://pypi.nvidia.com/nemo-fabric-runtime/) Ensure that a wheel exists for each supported platform.
   - [`nemo-fabric-collector`](https://pypi.nvidia.com/nemo-fabric-collector/)
   - [`nemo-fabric-adapters-common`](https://pypi.nvidia.com/nemo-fabric-adapters-common/)
   - [`nemo-fabric-adapters-claude`](https://pypi.nvidia.com/nemo-fabric-adapters-claude/)
   - [`nemo-fabric-adapters-codex`](https://pypi.nvidia.com/nemo-fabric-adapters-codex/)
   - [`nemo-fabric-adapters-deepagents`](https://pypi.nvidia.com/nemo-fabric-adapters-deepagents/)
   - [`nemo-fabric-adapters-hermes`](https://pypi.nvidia.com/nemo-fabric-adapters-hermes/)
   - [`nemo-fabric-adapters-nooa`](https://pypi.nvidia.com/nemo-fabric-adapters-nooa/)
4. The TypeScript contract and bundled adapter packages are visible on npm with the
   expected version, dist-tag, and provenance:

   ```bash
   (
     set -euo pipefail
     npmjs_registry="https://registry.npmjs.org/"
     for package in \
       nemo-fabric-adapter-contract \
       nemo-fabric-adapters-common \
       nemo-fabric-adapters-droid \
       nemo-fabric-adapters-pi \
       nemo-fabric-adapters-opencode \
       nemo-fabric-adapters-qwen \
       nemo-fabric-adapters-kiro; do
       npm view "${package}@<release-version>" version \
         --registry="$npmjs_registry"
       npm view "$package" dist-tags --registry="$npmjs_registry"
     done
     verification_dir="$(mktemp -d)"
     trap 'rm -rf "$verification_dir"' EXIT
     cd "$verification_dir"
     npm init --yes --registry="$npmjs_registry"
     npm install --ignore-scripts --save-exact \
       --registry="$npmjs_registry" \
       "nemo-fabric-adapter-contract@<release-version>" \
       "nemo-fabric-adapters-common@<release-version>" \
       "nemo-fabric-adapters-droid@<release-version>" \
       "nemo-fabric-adapters-pi@<release-version>" \
       "nemo-fabric-adapters-opencode@<release-version>" \
       "nemo-fabric-adapters-qwen@<release-version>" \
       "nemo-fabric-adapters-kiro@<release-version>"
     npm audit signatures --registry="$npmjs_registry"
   )
   ```

5. The Fern documentation site shows the expected version and release notes.
6. The GitHub Release page is complete and accurate.
