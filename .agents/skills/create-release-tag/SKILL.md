---
name: create-release-tag
description: Create and push a signed, annotated NeMo Fabric stable release tag from its validated release branch. Use when cutting a stable release tag; not for beta or release-candidate tags.
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
The trap returns to the original named branch (or detached commit) after a
successful tag or any error, but only if this workflow changed the checkout.

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
