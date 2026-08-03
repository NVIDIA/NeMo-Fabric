---
name: draft-release-notes
description: Compare NeMo Fabric release branches and draft or update the documentation-site release notes. Use when preparing a release-notes update under docs/about-nemo-fabric/release-notes.mdx, reviewing release-to-release changes, or gathering evidence for a current release summary.
---

# Draft Release Notes

Draft the documentation-site release-notes page from verified repository
evidence. Keep complete, PR-by-PR release history in GitHub Releases.

## Gather Evidence

Run the read-only helper with explicit release refs and the target minor version:

```bash
python3 .agents/skills/draft-release-notes/scripts/collect_release_evidence.py \
  --previous release/<previous-major>.<previous-minor> \
  --current HEAD \
  --version <major>.<minor>
```

The report verifies both refs, compares their release-notes pages, identifies
version text currently present in that page, and groups commits into review
candidates. Treat the groups as an evidence index, not publication-ready copy.

## Workflow

1. Confirm the target release version from the release branch and package
   metadata. Preserve unrelated working-tree changes.
2. Run the helper. It reports an absent prior release-notes page without
   failing, which is expected for early release branches.
3. Verify each candidate claim in the changed public docs, API types, command
   help, or source before including it. Prioritize breaking changes, migrations,
   user-visible features, and ongoing support limitations.
4. Update only this page unless the release changes its route or entry point:
   - `docs/about-nemo-fabric/release-notes.mdx`
5. Keep the existing page role:
   - `release-notes.mdx` gives the current-release summary, compatibility notes,
     scope, and curated feature links.
   - `release-notes.mdx` groups notable changes by user-facing theme.
   - `release-notes.mdx` records current limitations.
6. Preserve MDX front matter and the JSX SPDX comment. State the full history
   is available in GitHub Releases; do not create a changelog or GitHub Release
   body from this skill.

## Validate

Run the helper for the target release and for an early branch without release
notes when available. Review public claims, then run:

```bash
git diff --check
just docs
```

Check product names, commands, package names, support claims, and links against
the current repository before handing off the draft.
