<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NeMo Fabric Maintainer Skills

This directory is the maintainer skill set for developing NeMo Fabric itself.
Use these skills for repository work such as:

- Contributing or changing public API surfaces across Rust, CLI, Python, schema,
  and adapters.
- Writing and validating Python tests and choosing the right validation matrix.
- Maintaining CI workflows, packaging, versions, and release surfaces.
- Reviewing documentation for NVIDIA style and preparing pull requests.

These skills may reference repository internals, build and test commands, and
contribution workflows.

Consumer-facing integration skills live in the top-level
[`skills/`](../../skills/README.md) directory so they can be exported separately
for application developers and integrators. Keep external usage guidance out of
this set.

## Skills

This table lists the maintainer skills in this set and what each covers.

| Skill | Purpose |
|---|---|
| [`contribute-api`](contribute-api/SKILL.md) | Add a public API surface with Rust, CLI, Python, schema, adapter, and documentation parity. |
| [`small-fix`](small-fix/SKILL.md) | Make a small, reviewable bug fix without widening scope. |
| [`contribute-docs`](contribute-docs/SKILL.md) | Change documentation or examples in step with public behavior. |
| [`review-doc-style`](review-doc-style/SKILL.md) | Review documentation and public text for NVIDIA technical-writing style. |
| [`validate-change`](validate-change/SKILL.md) | Choose and run the right validation matrix for a change. |
| [`python-tests`](python-tests/SKILL.md) | Write Python tests for Fabric. |
| [`maintain-ci`](maintain-ci/SKILL.md) | Maintain GitHub Actions workflows with pinned actions and local validation. |
| [`maintain-packaging`](maintain-packaging/SKILL.md) | Maintain package metadata, native artifacts, lockfiles, and release surfaces. |
| [`update-project-version`](update-project-version/SKILL.md) | Bump and synchronize release versions across packaging. |
| [`prepare-pr`](prepare-pr/SKILL.md) | Prepare, open, or edit a pull request with the right scope and review handoff. |
| [`karpathy-guidelines`](karpathy-guidelines/SKILL.md) | Behavioral coding guidelines; use as a companion to the others. |

## Discovery And Conventions

- Coding agents auto-discover this set from `.agents/skills/`. For Claude Code,
  `.claude/skills` is a symlink to this directory, exposing the same maintainer
  set without mixing in consumer skills.
- **Naming:** maintainer skills use descriptive, task-based names (for example
  `contribute-api`, `validate-change`).
- **Frontmatter:** each `SKILL.md` begins with YAML frontmatter containing at
  least `name` and `description`.
