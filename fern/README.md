<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NeMo Fabric Fern Documentation

NeMo Fabric uses Fern to publish documentation at
`nvidia-nemo-fabric.docs.buildwithfern.com/nemo/fabric` and
`docs.nvidia.com/nemo/fabric`.

## Branch Model

Documentation is authored on `main`:

- `docs/` contains Markdown and MDX page source.
- `docs/index.yml` contains source navigation.
- `fern/` contains local Fern configuration and the temporary cover page.

The orphan `docs-website` branch is mostly CI-managed. It stores the generated
Fern publishing layout and retains release snapshots after their source branch
or tag is no longer active:

- `fern/pages-main/` contains generated bleeding-edge pages.
- `fern/versions/main.yml` contains rewritten bleeding-edge navigation.
- `fern/pages-vX.Y.Z/` and `fern/versions/vX.Y.Z.yml` contain release snapshots.
- `fern/docs.yml` preserves the accumulated version list.
- `.github/workflows/publish-fern-docs.yml` provides a branch-local manual and
  direct-push publishing fallback.

The source workflow creates `docs-website` as an orphan branch if it does not
exist. Files under `scripts/docs/docs_website_branch/` seed its branch-local
README, ignore rules, and fallback workflow. After bootstrap, update those
three files through a pull request that targets `docs-website`.

## Site Configuration

Keep these settings in `docs.yml`:

- `instances[].multi-source: true` limits publishing to the Fabric subpath on
  the shared NVIDIA documentation domain.
- `global-theme: nvidia` applies the shared NVIDIA documentation theme.
- `logo.right-text: NeMo Fabric` replaces the generic theme label with the
  product name.

Do not copy product-specific redirects or custom components from another NeMo
site unless Fabric needs them.

## Temporary Cover

The public root displays a temporary cover while the complete documentation is
available at `/main/`. The cover uses a separate version so it is never copied
into a release snapshot:

- `versions/soon.yml` maps the documentation root to the cover page.
- `pages-soon/index.mdx` uses Fern's custom layout for the centered design.
- `docs.yml` makes the cover the default and lists `Main · preview` separately.

The first stable docs release automatically makes that release the default and
removes the cover from the version selector. Beta and release-candidate docs
retain the current default. To remove the cover before a stable release, update
the preserved product configuration on `docs-website` and publish that branch.

## Publishing and Versioning

`.github/workflows/fern-docs.yml` handles previews, `main` publishing, and
release snapshots:

- Pushes to `pull-request/**` validate the source, assemble the same generated
  layout used for publishing, and create a stable Fern preview.
- Pushes to `main` regenerate API references, sync `docs-website`, commit any
  generated changes, and publish the bleeding-edge docs.
- Raw SemVer tags such as `0.1.0`, `0.1.0-beta.1`, and `0.1.0-rc.1` create or
  replace a public snapshot displayed with a leading `v`.

Stable tags use `availability: stable` and update `Latest`. Beta and
release-candidate tags use `availability: beta`, replace the snapshot for the
same base version, and do not update `Latest`. Alpha tags and tags with a
leading `v` are not published.

Version snapshots are generated from the selected tag checkout, not from the
current contents of `docs-website`. The helper also rewrites internal `/main/`
links and GitHub `main` links to the released version.

## Local Validation

Run the normal docs check from the repository root:

```bash
just docs
```

To inspect the generated publishing layout locally, use an empty temporary
directory or a checkout of `docs-website`:

```bash
python scripts/docs/sync_fern_docs_branch.py sync-main \
  --source-root . \
  --target-root /path/to/docs-website-checkout
```

To test a release snapshot without publishing it, run:

```bash
python scripts/docs/sync_fern_docs_branch.py release-version \
  --source-root . \
  --target-root /path/to/docs-website-checkout \
  --tag 0.1.0
```
