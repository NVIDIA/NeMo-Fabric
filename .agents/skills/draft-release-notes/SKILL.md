---
name: draft-release-notes
description: Compare NVIDIA NeMo Fabric release refs and draft the authoritative GitHub Release body plus any warranted documentation-site release-note update. Use when preparing a stable release, creating patch-release notes, updating docs/about-nemo-fabric/release-notes.mdx, or gathering verified release evidence.
author: NVIDIA Corporation and Affiliates
license: Apache-2.0
---

# Draft Release Notes

Draft release notes from verified repository evidence. Keep complete,
tag-specific release history in GitHub Releases. Update the documentation-site
release-notes page only when the documentation-visible release summary changes.

## Gather Evidence

Run the read-only helper with explicit release refs and the exact target release
version:

```bash
python3 .agents/skills/draft-release-notes/scripts/collect_release_evidence.py \
  --previous <previous-release-tag-or-branch> \
  --current HEAD \
  --version <release-version>
```

For a patch release, use the previous stable tag as `--previous`. For a new
release line, use the previous release branch or tag. The report verifies both
refs, inspects the release-notes page at each ref, identifies version text, and
groups commits into review candidates. Treat the groups as an evidence index,
not publication-ready copy.

## Workflow

1. Confirm the target release version from the release branch and package
   metadata. Preserve unrelated working-tree changes.
2. Run the helper. It reports an absent prior release-notes page without
   failing, which is expected for early release branches.
3. Verify each candidate claim in the changed public docs, API types, command
   help, or source before including it. Prioritize breaking changes, migrations,
   user-visible features, and ongoing support limitations.
4. Draft the GitHub Release body for every stable release. Include:
   - a concise user-facing overview
   - breaking changes, migrations, and compatibility requirements
   - verified features and fixes grouped by user-facing theme
   - current limitations that materially affect the release
   - links to included pull requests and the full comparison
5. For a patch release, identify the affected behavior and state whether public
   APIs, configuration, or dependency contracts changed.
6. Update only this page unless the release changes its route or entry point:
   - `docs/about-nemo-fabric/release-notes.mdx`
   Leave it unchanged when a patch release does not alter the
   documentation-visible summary, compatibility guidance, support status, or
   limitations.
7. Keep the existing page role:
   - `release-notes.mdx` gives the current-release summary, compatibility notes,
     scope, and curated feature links.
   - `release-notes.mdx` groups notable changes by user-facing theme.
   - `release-notes.mdx` records current limitations.
8. Preserve MDX front matter and the JSX SPDX comment. State the full history
   is available in GitHub Releases. Do not create a changelog.

## Validate

Run the helper for the target release and review every public claim. If the
documentation page changed, run:

```bash
git diff --check
just docs
```

Check product names, commands, package names, support claims, and links against
the current repository before handing off the draft.
