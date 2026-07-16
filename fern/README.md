<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NeMo Fabric Fern Documentation

This directory contains the Fern configuration and versioned documentation
source for NeMo Fabric.

## Site Configuration

The site publishes to the `/nemo/fabric` subpath of the shared
`docs.nvidia.com` domain. Keep the following settings in `docs.yml`:

- Set `instances[].multi-source: true` so a publish updates only the Fabric
  subpath. Keep the `/nemo/fabric` basepath identical in `url` and
  `custom-domain`.
- Set `global-theme: nvidia` to inherit the shared NVIDIA documentation theme.
- Set `logo.right-text: NeMo Fabric` to replace the theme's generic
  documentation label with the product name.

Do not copy product-specific redirect, generated-library, or custom-component
configuration from another NeMo site unless Fabric needs that feature.

## Temporary Cover Version

The public root currently displays a temporary cover page while the full
documentation remains available for preview. This pattern has three parts:

- `docs.yml` lists `Coming soon` first because Fern uses the first version as
  the default version for unversioned URLs. It lists `Main · preview` second,
  which keeps the bleeding-edge documentation at `/main/...`.
- `versions/soon.yml` contains one navigation entry with an empty slug. That
  entry resolves to the documentation root.
- `versions/soon/pages/index.mdx` uses Fern's `custom` page layout. The custom
  layout removes the generated article heading and content constraints so the
  MDX can render one centered heading with page-scoped CSS.

Keep the cover in a separate version instead of adding it to `main`. Release
snapshots copy `main`, so separating the cover prevents temporary launch
content from entering a release snapshot.

### Reuse the Cover Pattern

To reuse this pattern in another versioned Fern site:

1. Create a version configuration that contains one page with `slug: ""`.
2. Add the temporary version as the first entry under `versions` in
   `docs.yml`.
3. Add the real documentation version after it and set an explicit slug such
   as `main`.
4. Set `layout: custom` in the cover page frontmatter and render the page with
   one `<h1>` element. Scope any inline CSS to a cover-specific class.
5. Run `just docs` and verify both the root page and the real documentation
   version in the Fern preview.

Use a Fern custom React component only when multiple pages or sites must share
the same cover implementation. For a single temporary page, page-local MDX and
CSS keep the implementation self-contained.

### Remove the Cover Version

When the documentation is ready to replace the cover page, remove the
`Coming soon` entry from `docs.yml` and delete `versions/soon.yml` and
`versions/soon/`. Keep the `Main · preview` version.

## Layout

- `versions/main.yml` defines navigation for bleeding-edge documentation.
- `versions/main/pages/` contains documentation for the `main` branch.
- `versions/soon.yml` temporarily defines the public cover page.
- `versions/<release>.yml` defines navigation for an immutable SemVer release.
- `versions/<release>/pages/` contains the corresponding release snapshot.

After the temporary cover is removed and before the first stable release,
`main` is the only configured version. After the first stable release, the
stable snapshot becomes the first version in `docs.yml`, so unversioned URLs
resolve to stable documentation while `/main/...` remains the bleeding-edge
site.

## Local Validation

From the repository root, run:

```bash
just docs
```

This command regenerates the Python and Rust API references under
`versions/main/pages/reference/` and runs `fern check`.

## Create a Release Snapshot

Use the raw SemVer release tag without a leading `v`, consistent with
`CONTRIBUTING.md`. For a release such as `0.1.0`:

1. Run `just docs` and commit the regenerated `main` reference pages.
2. Copy `versions/main/pages/` to `versions/0.1.0/pages/`.
3. Copy `versions/main.yml` to `versions/0.1.0.yml` and replace
   `./main/pages/` with `./0.1.0/pages/` in the snapshot navigation.
4. In the copied pages, replace version-scoped links beginning with `/main/`
   with `/0.1.0/`. Do not rewrite external GitHub links that contain `main`.
5. Create `versions/latest.yml` as a symlink to `0.1.0.yml`.
6. Add `Latest · 0.1.0` first in the `versions:` list in `docs.yml`, keep
   `Main · preview` second, and add the immutable `0.1.0` entry after it.
7. Run `just docs` again to validate all configured versions.

For later releases, create a new snapshot and retarget `latest.yml`. Do not
regenerate or edit an older snapshot except for a deliberate documentation
backport.
