---
name: prepare-code-freeze
description: Prepare a NeMo Fabric code freeze by creating a release branch, deciding whether frozen-line nightly alpha tags are required, bumping main to the next version, updating current-version documentation, and opening the required PR. Use when starting, preparing, or automating a NeMo Fabric code freeze.
author: NVIDIA Corporation and Affiliates
license: Apache-2.0
---

# Prepare Code Freeze

## Companion Guidance

Use `update-project-version` for version semantics, `maintain-ci` if the freeze
changes nightly or release-branch automation, and `prepare-pr` before opening
the PR.

## Workflow

Assume `upstream` is the NVIDIA repository remote (`NVIDIA/NeMo-Fabric`) and
`origin` is a maintainer's fork.

1. Confirm the target release version from `upstream/main:Cargo.toml`. Derive
   the release branch as `release/<major>.<minor>`.
2. Prompt for `<next-version>` if the user did not provide it. This is the
   version that `main` moves to after the release branch is cut.
3. Fetch the latest `main` and create the release branch from `upstream/main`:

   ```bash
   git fetch upstream main
   git branch release/<major>.<minor> upstream/main
   git push upstream release/<major>.<minor>
   ```

   If the remote release branch already exists, verify it points where expected
   before continuing.
4. Create a PR branch from latest `upstream/main`, for example
   `docs/code-freeze-<major>.<minor>`.
5. Run `just set-version <next-version>` to bump all release-versioned package
   surfaces on `main`.
6. Search documentation source for references to the old version and update
   current-version install commands, package examples, and configuration
   examples to `<next-version>` where appropriate:

   ```bash
   rg -n '<old-version>' README.md docs examples adapters \
     --glob '!docs/reference/api/**' || true
   ```

   Review matches before changing them. Leave intentional historical references
   alone, such as release notes, changelogs, generated build output, and
   third-party dependency attribution entries.
7. Validate with targeted checks:

   ```bash
   just set-version <next-version>
   cargo check --workspace --locked
   just build-all
   just wheels
   just --fmt --check
   rg -n '<old-version>' README.md docs examples adapters \
     --glob '!docs/reference/api/**' || true
   git diff --check
   ```

   Any remaining documentation matches for `<old-version>` should be intentional
   and called out in the PR description.
9. Open a PR targeting `main` using `.github/pull_request_template.md`. The PR
   must mention:
   - the new release branch
   - the nightly alpha branch config update
   - the `just set-version <next-version>` bump
   - documentation old-version reference updates or intentional leftovers
   - that release-bound PRs now target the new `release/*` branch

## Guardrails

- Do not create release tags. Code freeze only creates the branch and the main
  PR.
- Do not target the code-freeze PR at the release branch. It targets `main`.
- Do not leave uncommitted user changes mixed into the code-freeze PR branch.
