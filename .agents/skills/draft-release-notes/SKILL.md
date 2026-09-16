---
name: draft-release-notes
description: Compare NVIDIA NeMo Fabric release refs and draft the authoritative GitHub Release body plus any warranted documentation-site release-note update. Use when preparing a stable release, creating patch-release notes, updating docs/about-nemo-fabric/release-notes.mdx, or gathering verified release evidence.
author: NVIDIA Corporation and Affiliates
license: Apache-2.0
---

# Draft Release Notes

Draft release notes from verified repository evidence. Keep complete,
tag-specific release history in GitHub Releases. Update the documentation-site
release-notes page only when the documentation-visible release summary changes.

## Gather Evidence

Run the read-only helper with explicit release refs and the exact target release
version:

```bash
python3 .agents/skills/draft-release-notes/scripts/collect_release_evidence.py \
  --previous <previous-release-tag-or-branch> \
  --current HEAD \
  --version <release-version>
```

For a patch release, use the previous stable tag as `--previous`. For a new
release line, use the previous release branch or tag. The report verifies both
refs, inspects the release-notes page at each ref, identifies version text, and
groups commits into review candidates. Treat the groups as an evidence index,
not publication-ready copy.

## Workflow

1. Establish the release branch and target version. Preserve unrelated
   working-tree changes; do not stash or discard them.
   - If the user supplied a version, derive `release/<major>.<minor>` and
     check out that branch.
   - If the user did not supply a version, require the current branch to match
     `release/<major>.<minor>`; do not infer a release line from package
     metadata or Git history.
   - Read the target version from `Cargo.toml`. When the user supplied a
     version, require it to match the package version.

   ```bash
   RELEASE_VERSION_INPUT="${RELEASE_VERSION_INPUT:-}"
   if [[ -n "${RELEASE_VERSION_INPUT}" ]]; then
     if [[ ! "${RELEASE_VERSION_INPUT}" =~ ^[0-9]+\.[0-9]+(\.[0-9]+)?$ ]]; then
       echo "Error: release version must be <major>.<minor> or <major>.<minor>.<patch>" >&2
       exit 1
     fi
     RELEASE_LINE="$(printf '%s' "${RELEASE_VERSION_INPUT}" | cut -d. -f1,2)"
     if [[ "${RELEASE_VERSION_INPUT}" =~ ^[0-9]+\.[0-9]+$ ]]; then
       EXPECTED_VERSION="${RELEASE_VERSION_INPUT}.0"
     else
       EXPECTED_VERSION="${RELEASE_VERSION_INPUT}"
     fi
     RELEASE_BRANCH="release/${RELEASE_LINE}"
     git switch "${RELEASE_BRANCH}"
   else
     RELEASE_BRANCH="$(git branch --show-current)"
     if [[ ! "${RELEASE_BRANCH}" =~ ^release/([0-9]+\.[0-9]+)$ ]]; then
       echo "Error: supply a release version or check out release/<major>.<minor>" >&2
       exit 1
     fi
     RELEASE_LINE="${BASH_REMATCH[1]}"
   fi

   TARGET_VERSION="$(python -c "import tomllib; print(tomllib.load(open('Cargo.toml', 'rb'))['workspace']['package']['version'])")"
   if [[ -n "${RELEASE_VERSION_INPUT}" ]] && [[ "${TARGET_VERSION}" != "${EXPECTED_VERSION}" ]]; then
     echo "Error: ${RELEASE_BRANCH} declares ${TARGET_VERSION}, not ${EXPECTED_VERSION}" >&2
     exit 1
   fi
   ```
2. Create a new local branch from the release branch for the release-notes
   work. Let the command fail rather than replacing an existing branch:

   ```bash
   git switch -c "docs/create-release-notes-${TARGET_VERSION}" "${RELEASE_BRANCH}"
   ```
3. Run the helper. It reports an absent prior release-notes page without
   failing, which is expected for early release branches.
4. Verify each candidate claim in the changed public docs, API types, command
   help, or source before including it. Prioritize breaking changes, migrations,
   user-visible features, and ongoing support limitations.
5. For a patch release, identify the affected behavior and state whether public
   APIs, configuration, or dependency contracts changed.
6. Update only this page unless the release changes its route or entry point:
   - `docs/about-nemo-fabric/release-notes.mdx`
   Leave it unchanged when a patch release does not alter the
   documentation-visible summary, compatibility guidance, support status, or
   limitations.
7. Keep the existing page role:
   - `release-notes.mdx` gives the current-release summary, compatibility notes,
     scope, and curated feature links.
   - `release-notes.mdx` groups notable changes by user-facing theme.
   - `release-notes.mdx` records current limitations.
8. Preserve MDX front matter and the JSX SPDX comment. State the full history
   is available in GitHub Releases. Do not create a changelog.
9. Update the `Roadmap` section in [README.md](../../../README.md), removing items implemented or fixed by changes in the current release.
10. Run the validation checks detailed in [Validate](#validate).
11. If `docs/about-nemo-fabric/release-notes.mdx` or `README.md` changed, commit the
    release-notes page and README with a signed-off commit after validation
    succeeds:

    ```bash
    git add docs/about-nemo-fabric/release-notes.mdx README.md
    git commit -sm "Drafting release notes for v${TARGET_VERSION}"
    git push -u upstream HEAD
    ```
12. If `docs/about-nemo-fabric/release-notes.mdx` or `README.md` changed, use the `prepare-pr`
    skill to open a pull request from the release-notes branch to
    `${RELEASE_BRANCH}`. Use this title:

    ```text
    docs: Release notes for v${TARGET_VERSION}
    ```

    If `docs/about-nemo-fabric/release-notes.mdx` and `README.md` are unchanged, skip steps 11
    and 12. Do not create a release-notes commit or pull request.

## Validate

Run the helper for the target release and review every public claim. If the
documentation page changed, run:

```bash
git diff --check
just docs
```

Check product names, commands, package names, support claims, and links against
the current repository before handing off the draft.
