---
name: create-rc-tag
description: Create and push a signed, annotated NeMo Fabric release-candidate tag from its validated release branch. Use when cutting an RC tag; not for code freezes or stable release tags.
author: NVIDIA Corporation and Affiliates
license: Apache-2.0
---

# Create an RC Tag

Create the tag only after the release branch has been created.

## Determine the version

Require the user to provide the stable release version in either
`<major>.<minor>` or `<major>.<minor>.<patch>` form. Treat a two-component
version as `<major>.<minor>.0`. Do not infer the version from `Cargo.toml` or
a branch name, and do not edit `Cargo.toml` while cutting the tag. Normalize
the supplied version into `RELEASE_BASE_VERSION`:

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

if ! git ls-remote --exit-code --heads upstream "refs/heads/${RELEASE_BRANCH}" >/dev/null; then
  echo "Error: remote release branch ${RELEASE_BRANCH} does not exist on upstream" >&2
  exit 1
fi

git fetch upstream "${RELEASE_BRANCH}" --tags
```

If the user provides an RC number, use it. Otherwise, after fetching tags,
select one greater than the largest numeric RC suffix for this exact base
version. Start at `1` when none exists:

```bash
LAST_RC_NUM="$(
  git tag --list "v${RELEASE_BASE_VERSION}-rc.*" |
    awk -v prefix="v${RELEASE_BASE_VERSION}-rc." '
      index($0, prefix) == 1 {
        suffix = substr($0, length(prefix) + 1)
        if (suffix ~ /^[0-9]+$/ && suffix + 0 > max) max = suffix + 0
      }
      END { if (max) print max }
    '
)"
RC_NUM="${LAST_RC_NUM:+$((LAST_RC_NUM + 1))}"
RC_NUM="${RC_NUM:-1}"
RELEASE_VERSION="${RELEASE_BASE_VERSION}-rc.${RC_NUM}"
RELEASE_TAG="v${RELEASE_VERSION}"
```

Use an explicitly supplied RC number only after checking it is a positive
integer and that its remote tag does not already exist.

## Create and verify the tag

Assume `upstream` is the NVIDIA repository remote. Preserve user changes: do
not stash, discard, or switch away from a dirty worktree. The preceding remote
branch check must succeed before switching to the release branch. Require its
checked-out commit to equal the upstream branch before tagging:

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
