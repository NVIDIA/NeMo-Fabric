---
name: contribute-docs
description: Contribute documentation or example changes that stay aligned with NeMo Fabric public behavior
author: NVIDIA Corporation and Affiliates
license: Apache-2.0
---


# Contribute Docs Or Examples

## Companion Guidance

Use `karpathy-guidelines` alongside this skill for implementation or review
work. Keep changes scoped, surface assumptions, and define focused validation
before editing.

Use this skill for docs-only or example-heavy changes.

## Rules

- Prefer the documented public API, not internal shortcuts
- Keep package names, repo references, and build commands current
- Update entry-point docs when examples or reading paths change
- In MDX files, top-of-file comments must use JSX comment delimiters:
  `{/*` to open and `*/}` to close. Do not use HTML comments for MDX SPDX
  headers.

## Checklist

- [ ] `README.md` or `docs/index.yml` updated when entry points changed
- [ ] Relevant getting-started or reference docs updated
- [ ] Example commands still match current package names and paths
- [ ] Relevant adapter or example `README.md` files updated when examples
      or adapters have changed.
- [ ] New or regenerated MDX files use `{/* ... */}` for top-of-file SPDX comments
- [ ] Run `just docs` when the docs site changed

## References

- `README.md`
- `docs/index.yml`
- `review-doc-style`
