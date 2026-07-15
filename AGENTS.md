<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AGENTS.md

## Documentation And Contribution Workflow

These workflow notes keep public documentation, examples, and PR preparation aligned
with repository expectations.

- Update user-facing entry points when public behavior, package names (`nemo-fabric` / `nemo_fabric`, `fabric` CLI), examples, or supported bindings change: `README.md`, the Fern docs under `docs/` (navigation in `docs/index.yml`, site config in `fern/docs.yml`), and the adapter/integration READMEs (`adapters/*/README.md`, `integrations/*/README.md`, `examples/README.md`).
- Keep the Python/Rust binding contract current when the public API changes: `docs/python-sdk-contract.md`, the JSON Schema notes in `schemas/SCHEMA.md`, and the generated references under `docs/reference/api/`. Regenerate docs with `just docs` after changing the docs site.
- Keep release- and packaging-process details in maintainer surfaces (currently the `maintain-packaging` skill at `.agents/skills/maintain-packaging/SKILL.md`). Do not move release-history policy into user-facing docs. There is no `RELEASING.md` or `CHANGELOG.md` yet; add release-history policy there if those files are introduced rather than into user docs.
- Keep the stable public wrapper `scripts/generate_api_docs.sh` at the `scripts/` root in docs and examples. Reference namespaced helper paths under `scripts/docs/` only when documenting internal maintenance work.
- Use branch prefixes for your work: `feat/`, `fix/`, `docs/`, `test/`, or `refactor/`.
- Name branches after the work, never the Linear ticket. Do not embed ticket IDs or slugs in the branch name (e.g. use `feat/notebooks-onboarding`, not `feat/fabric-70-notebooks-onboarding`). This rule has historically been overlooked, so double-check the branch name before pushing or opening a PR.
- Use Conventional Commit PR titles (`<type>: <summary>`) as required by `.coderabbit.yaml` and the `prepare-pr` skill; reserve `fix` for actual product bugs, not CI, docs, or chores.
- Use signed-off commits for PR work: `git commit -s`.
- When creating a pull request from the current branch, target the upstream repository rather than a fork.
- Before creating, opening, publishing, or editing a pull request, read `.github/pull_request_template.md` and use it as the PR body skeleton (or `gh pr create --template .github/pull_request_template.md`). Preserve its visible headings (`Overview`, `Where should the reviewer start?`, `Related Issues`) and its contribution checkboxes; fill the sections instead of replacing them with a generic summary.
- If repo-local PR guidance such as the `prepare-pr` skill conflicts with generic GitHub connector or plugin guidance, follow the repo-local PR guidance for PR body format and review handoff details.
- PR descriptions should include what changed, why, how it was tested, and any breaking changes within the repository template format.
