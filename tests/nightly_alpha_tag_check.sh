#!/usr/bin/env bash
# Test the git ls-remote tag-existence parsing used by the nightly alpha tag workflow.

set -euo pipefail

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT
cd "$tmpdir"

# Set up an "origin" repository with a commit, a lightweight tag, and an annotated tag.
git init origin >/dev/null
cd origin
git config user.email "test@example.com"
git config user.name "Test"
echo "init" > file
git add file
git commit -m "init" >/dev/null
commit_sha="$(git rev-parse HEAD)"
git tag lightweight
git tag -a annotated -m "annotated" >/dev/null
cd ..

# Set up a local clone to run git ls-remote against origin.
git clone origin local >/dev/null 2>&1
cd local

# Replicate the lookup logic from the workflow.
lookup_tag() {
  local tag="$1"
  local existing_sha
  existing_sha="$(
    git ls-remote --tags origin "refs/tags/${tag}^{}" "refs/tags/${tag}" 2>/dev/null |
      awk 'index($2, "^{}") { print $1; found=1; exit } END { if (!found && NF) print $1 }' || true
  )"
  printf '%s\n' "$existing_sha"
}

# Lightweight tag resolves to the commit SHA.
[[ "$(lookup_tag lightweight)" == "$commit_sha" ]]

# Annotated tag resolves to the peeled commit SHA.
[[ "$(lookup_tag annotated)" == "$commit_sha" ]]
