<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NeMo Fabric Docs Website Branch

This orphan branch is the Fern publishing branch for NeMo Fabric. It contains
CI-managed generated documentation and a small set of branch-local maintenance
files.

## Edit Policy

Author documentation on `main` or on a pull-request branch based on `main`.
Do not manually edit generated content on this branch.

The source of truth is:

- `docs/` on `main` for Markdown and MDX content and `docs/index.yml` for
  navigation.
- `fern/` on `main` for shared Fern configuration and the temporary cover.
- `scripts/docs/sync_fern_docs_branch.py` on `main` for generating this branch.

Only these branch-local files should be edited directly on `docs-website`:

- `.gitignore`
- `README.md`
- `.github/workflows/publish-fern-docs.yml`

## Branch Contents

- `fern/pages-main/` and `fern/versions/main.yml` contain the generated
  bleeding-edge documentation.
- `fern/pages-vX.Y.Z/` and `fern/versions/vX.Y.Z.yml` contain release
  snapshots.
- `fern/pages-soon/` and `fern/versions/soon.yml` contain the temporary public
  cover.
- `fern/docs.yml` preserves the accumulated version list.

The temporary cover remains the default until the first stable docs release.
A stable release makes that release the default and removes the cover from the
version selector. Beta and release-candidate tags do not replace the cover or
the current stable default.

## Publishing

The source workflow on `main` normally syncs this branch and publishes the
site. The branch-local `Publish Fern Docs` workflow is a manual and direct-push
fallback. Both workflows use the `FERN_TOKEN` secret from the `fern` GitHub
environment.

## Recovery

Regenerate this branch from a validated source checkout instead of editing
generated files by hand:

```bash
python scripts/docs/sync_fern_docs_branch.py sync-main \
  --source-root . \
  --target-root /path/to/docs-website-checkout
```
