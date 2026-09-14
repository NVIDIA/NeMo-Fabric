---
name: create-beta-tag
description: Create and push a signed, annotated NeMo Fabric beta tag from its release branch or from validated main. Use when cutting a beta tag; not for RC or stable release tags.
author: NVIDIA Corporation and Affiliates
license: Apache-2.0
---

# Create a Beta Tag

Require the user to provide the stable release version in either
`<major>.<minor>` or `<major>.<minor>.<patch>` form. Treat two components as
`<major>.<minor>.0`. Do not infer the version from package metadata or a branch
name, and do not edit package metadata while creating the tag.

```bash
RELEASE_VERSION_INPUT="<user-provided-major.minor[.patch]>"
if [[ "${RELEASE_VERSION_INPUT}" =~ ^[0-9]+\.[0-9]+$ ]]; then
  RELEASE_BASE_VERSION="${RELEASE_VERSION_INPUT}.0"
elif [[ "${RELEASE_VERSION_INPUT}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  RELEASE_BASE_VERSION="${RELEASE_VERSION_INPUT}"
else
  echo "Error: release version must be <major>.<minor> or <major>.<minor>.<patch>" >&2
  exit 1
fi
RELEASE_BRANCH="release/$(printf '%s' "${RELEASE_BASE_VERSION}" | cut -d. -f1,2)"
```

## Select the tag source

Prefer the corresponding remote release branch. If it does not exist, use
`main` only when `upstream/main:Cargo.toml` declares the requested base version.
Stop if neither condition holds.

```bash
if git ls-remote --exit-code --heads upstream "refs/heads/${RELEASE_BRANCH}" >/dev/null; then
  TAG_SOURCE_BRANCH="${RELEASE_BRANCH}"
else
  TAG_SOURCE_BRANCH='main'
fi

git fetch upstream "${TAG_SOURCE_BRANCH}" --tags

if [[ "${TAG_SOURCE_BRANCH}" == 'main' ]]; then
  UPSTREAM_MAIN_VERSION="$(
    git show 'upstream/main:Cargo.toml' |
      python -c "import sys, tomllib; print(tomllib.loads(sys.stdin.read())['workspace']['package']['version'])"
  )"
  test "${UPSTREAM_MAIN_VERSION}" = "${RELEASE_BASE_VERSION}"
fi
```

If the user provides a beta number, use it only after checking that it is a
positive integer and its remote tag does not already exist. Otherwise, select
one greater than the largest numeric beta suffix for this exact base version,
starting at `1`:

```bash
LAST_BETA_NUM="$(
  git tag --list "v${RELEASE_BASE_VERSION}-beta.*" |
    awk -v prefix="v${RELEASE_BASE_VERSION}-beta." '
      index($0, prefix) == 1 {
        suffix = substr($0, length(prefix) + 1)
        if (suffix ~ /^[0-9]+$/ && suffix + 0 > max) max = suffix + 0
      }
      END { if (max) print max }
    '
)"
BETA_NUM="${LAST_BETA_NUM:+$((LAST_BETA_NUM + 1))}"
BETA_NUM="${BETA_NUM:-1}"
RELEASE_VERSION="${RELEASE_BASE_VERSION}-beta.${BETA_NUM}"
RELEASE_TAG="v${RELEASE_VERSION}"
```

## Create and verify the tag

Assume `upstream` is the NVIDIA repository remote. Preserve user changes: do
not stash, discard, or switch away from a dirty worktree. Require the checked
out commit to equal the selected upstream branch. This also verifies the local
package version when tagging from a release branch, and re-verifies the
`upstream/main` version condition after switching to `main`.

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

```bash
if [[ "${ORIGINAL_BRANCH}" != "${TAG_SOURCE_BRANCH}" ]]; then
  git switch "${TAG_SOURCE_BRANCH}"
  CHECKOUT_CHANGED=true
fi
git pull --ff-only upstream "${TAG_SOURCE_BRANCH}"

test -z "$(git status --porcelain)"
RELEASE_SHA="$(git rev-parse HEAD)"
REMOTE_RELEASE_SHA="$(git rev-parse "upstream/${TAG_SOURCE_BRANCH}^{commit}")"
test "${RELEASE_SHA}" = "${REMOTE_RELEASE_SHA}"
test "$(just normalize-release-tag "${RELEASE_TAG}")" = "${RELEASE_VERSION}"
CURRENT_VERSION="$(python -c "import tomllib; print(tomllib.load(open('Cargo.toml', 'rb'))['workspace']['package']['version'])")"
test "${CURRENT_VERSION}" = "${RELEASE_BASE_VERSION}"

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
