---
name: create-release-tag
description: Create and push a signed, annotated NeMo Fabric stable release tag, then prepare unpublished GitHub Release and team announcement drafts for review. Use when cutting a stable release tag; not for beta or release-candidate tags.
author: NVIDIA Corporation and Affiliates
license: Apache-2.0
---

# Create a Release Tag

Require the user to provide the stable release version in exact
`<major>.<minor>.<patch>` form. Do not infer it from package metadata or a
branch name, and do not edit package metadata while creating the tag.

```bash
RELEASE_VERSION="<user-provided-major.minor.patch>"
if [[ ! "${RELEASE_VERSION}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "Error: release version must be <major>.<minor>.<patch>" >&2
  exit 1
fi
RELEASE_BRANCH="release/$(printf '%s' "${RELEASE_VERSION}" | cut -d. -f1,2)"
RELEASE_TAG="v${RELEASE_VERSION}"

if ! git ls-remote --exit-code --heads upstream "refs/heads/${RELEASE_BRANCH}" >/dev/null; then
  echo "Error: remote release branch ${RELEASE_BRANCH} does not exist on upstream" >&2
  exit 1
fi

git fetch upstream "${RELEASE_BRANCH}" --tags
```

## Create and Verify the Tag

Assume `upstream` is the NVIDIA repository remote. Preserve user changes: do
not stash, discard, or switch away from a dirty worktree. The remote release
branch check must succeed before switching to the release branch.

Run the remaining commands in one noninteractive Bash session. Record the
user's original checkout and install this exit trap before switching branches.
The trap returns to the original named branch (or detached commit) after the
workflow succeeds or encounters an error, but only if this workflow changed the
checkout.

```bash
set -euo pipefail
ORIGINAL_BRANCH="$(git branch --show-current)"
ORIGINAL_HEAD="$(git rev-parse --verify HEAD)"
CHECKOUT_CHANGED=false

restore_checkout() {
  local outcome=$?
  trap - EXIT
  if [[ "${CHECKOUT_CHANGED}" == true ]]; then
    if [[ -n "${ORIGINAL_BRANCH}" ]]; then
      git switch "${ORIGINAL_BRANCH}" || outcome=1
    else
      git switch --detach "${ORIGINAL_HEAD}" || outcome=1
    fi
  fi
  exit "${outcome}"
}
trap restore_checkout EXIT
```

Require the selected release commit and package version to match upstream. Do
not create the tag when either the local or upstream repository already has a
tag with the exact release name.

```bash
if [[ "${ORIGINAL_BRANCH}" != "${RELEASE_BRANCH}" ]]; then
  git switch "${RELEASE_BRANCH}"
  CHECKOUT_CHANGED=true
fi
git pull --ff-only upstream "${RELEASE_BRANCH}"

test -z "$(git status --porcelain)"
RELEASE_SHA="$(git rev-parse HEAD)"
REMOTE_RELEASE_SHA="$(git rev-parse "upstream/${RELEASE_BRANCH}^{commit}")"
test "${RELEASE_SHA}" = "${REMOTE_RELEASE_SHA}"
test "$(just normalize-release-tag "${RELEASE_TAG}")" = "${RELEASE_VERSION}"
CURRENT_VERSION="$(python -c "import tomllib; print(tomllib.load(open('Cargo.toml', 'rb'))['workspace']['package']['version'])")"
test "${CURRENT_VERSION}" = "${RELEASE_VERSION}"

if git rev-parse --verify --quiet "refs/tags/${RELEASE_TAG}" >/dev/null; then
  echo "Error: local tag ${RELEASE_TAG} already exists" >&2
  exit 1
fi
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

Stop on any failed check. Do not force-update, replace, or delete an existing
local or remote tag.

## Draft the GitHub Release

After the tag push succeeds, create a draft GitHub Release for that exact tag.
Do not publish the release. Determine the previous release tag from GitHub's
latest published, non-draft, non-prerelease release. Do not infer it from local
tag ordering.

```bash
GITHUB_REPOSITORY="NVIDIA/NeMo-Fabric"
PREVIOUS_RELEASE_TAG="$(
  gh api "repos/${GITHUB_REPOSITORY}/releases/latest" --jq '.tag_name'
)"
test -n "${PREVIOUS_RELEASE_TAG}"
test "${PREVIOUS_RELEASE_TAG}" != "${RELEASE_TAG}"

if ! git rev-parse --verify --quiet "refs/tags/${PREVIOUS_RELEASE_TAG}" >/dev/null; then
  git fetch upstream "refs/tags/${PREVIOUS_RELEASE_TAG}:refs/tags/${PREVIOUS_RELEASE_TAG}"
fi

RELEASE_NOTES_URL="https://docs.nvidia.com/nemo/fabric/about-nemo-fabric/release-notes"
COMPARISON_URL="https://github.com/${GITHUB_REPOSITORY}/compare/${PREVIOUS_RELEASE_TAG}...${RELEASE_TAG}"
```

Read the tagged release-notes page so the draft describes the content that was
actually released, not later working-tree edits:

```bash
git show \
  "${RELEASE_TAG}:docs/about-nemo-fabric/release-notes.mdx"
```

Also gather the comparison evidence with the existing read-only helper:

```bash
python3 .agents/skills/draft-release-notes/scripts/collect_release_evidence.py \
  --previous "${PREVIOUS_RELEASE_TAG}" \
  --current "${RELEASE_TAG}" \
  --version "${RELEASE_VERSION}"
```

Use GitHub's generated notes only to identify included pull requests and
potential new contributors. Verify that evidence before including it; do not
substitute the generated text for the curated release-notes page.

```bash
gh api --method POST \
  "repos/${GITHUB_REPOSITORY}/releases/generate-notes" \
  -f "tag_name=${RELEASE_TAG}" \
  -f "previous_tag_name=${PREVIOUS_RELEASE_TAG}"
```

Draft a condensed release body rather than reproducing the full release-notes
page. Include only the highest-impact user-facing changes, keep each bullet
brief, and use the detailed release-notes link for supporting context. Do not
copy full paragraphs or exhaustive lists from the source page.

Use a temporary Markdown file with this exact structure:

```markdown
# NVIDIA NeMo Fabric <version>

<One or two sentences summarizing the release.>

## At a Glance

### New

- <Notable new capability>

### Changed

- <Notable behavior or experience change>

### Breaking Changes

- **<Affected area>:** <What changed and what users must do.>
  [Migration details](<absolute-documentation-url>)

## Learn More

- [Detailed release notes](<absolute-documentation-url>)
- [Full changelog](<comparison-url>)

## New Contributors

- <Contributor and contribution>

```

Replace `<version>` with `${RELEASE_VERSION}`, without the leading `v`. Derive
the summary, `New`, `Changed`, and `Breaking Changes` content from the tagged
`docs/about-nemo-fabric/release-notes.mdx`. Link migration items to the most
specific absolute documentation URL available. Use `${RELEASE_NOTES_URL}` for
the detailed release-notes link and `${COMPARISON_URL}` for both changelog
links. List only verified first-time contributors from the comparison. Use
`- None.` under any category with no applicable entries so every heading in the
template remains present.

Review the completed body for unsupported claims, unresolved placeholders, and
relative URLs. Then create and verify the draft. `--verify-tag` is required so
GitHub cannot create a different tag implicitly. The title must be the exact tag
name.

```bash
RELEASE_BODY_PATH="<temporary-markdown-file>"

if gh release view "${RELEASE_TAG}" \
  --repo "${GITHUB_REPOSITORY}" >/dev/null 2>&1; then
  echo "Error: a GitHub Release for ${RELEASE_TAG} already exists" >&2
  exit 1
fi

gh release create "${RELEASE_TAG}" \
  --repo "${GITHUB_REPOSITORY}" \
  --draft \
  --title "${RELEASE_TAG}" \
  --notes-file "${RELEASE_BODY_PATH}" \
  --verify-tag

test "$(gh release view "${RELEASE_TAG}" \
  --repo "${GITHUB_REPOSITORY}" \
  --json isDraft \
  --jq '.isDraft')" = "true"
test "$(gh release view "${RELEASE_TAG}" \
  --repo "${GITHUB_REPOSITORY}" \
  --json name \
  --jq '.name')" = "${RELEASE_TAG}"

DRAFT_EDIT_URL="https://github.com/${GITHUB_REPOSITORY}/releases/edit/${RELEASE_TAG}"
printf 'Review the draft release before publishing: %s\n' "${DRAFT_EDIT_URL}"
```

Present `${DRAFT_EDIT_URL}` to the user as a clickable link. Tell the user to
review the title and body before clicking **Publish release**. Do not run
`gh release edit --draft=false`, call the publish API, or otherwise publish the
release as part of this skill.

## Draft the Team Release Announcement

Draft an announcement suitable for a team communication platform. Make it shorter and more condensed than the GitHub Release body: use a one-line description and include only the most important features. Do not copy paragraphs, exhaustive lists, contributor details, or changelog details from the release notes.

Create the `.tmp` directory if needed and save the announcement at the exact
path `.tmp/draft-${RELEASE_VERSION}release-announce.md`:

```bash
mkdir -p .tmp
ANNOUNCEMENT_PATH=".tmp/draft-${RELEASE_VERSION}release-announce.md"
TAGGED_RELEASE_NOTES_URL="https://github.com/${GITHUB_REPOSITORY}/blob/${RELEASE_TAG}/docs/about-nemo-fabric/release-notes.mdx"
```

Use this structure:

```markdown
:mega: NVIDIA NeMo Fabric <version> is here! :mega:

<short one-line description>

What’s new in <version>?
<!-- List the most important features here there should be 3-4 bullet points -->
* <emoji> <key feature>

<if there are breaking changes>
:warning: Breaking change: <change>
:warning: Breaking change: <change>
</if there are breaking changes>

Learn more and get started:

:book: Release Notes: <link to tagged release-notes.mdx file in GitHub>
:computer: GitHub: https://github.com/NVIDIA/NeMo-Fabric
```

Replace `<version>` with `${RELEASE_VERSION}`, without the leading `v`, and use
`${TAGGED_RELEASE_NOTES_URL}` for the release-notes link. Select only the
highest-impact features from the tagged release-notes page and give each a
relevant emoji. Include one `:warning:` line per verified breaking change;
omit the entire breaking-change block when there are none. Remove every
placeholder before saving the file.

Present `${ANNOUNCEMENT_PATH}` to the user together with `${DRAFT_EDIT_URL}`.
Do not post or send the announcement as part of this skill.
