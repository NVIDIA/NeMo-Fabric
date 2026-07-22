<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NeMo Fabric Consumer Skills

These are user-facing skills for integrating NeMo Fabric into your own
application, service, evaluation harness, or platform through the public Python
SDK. They are intended for external application developers and integrators — not
for developing NeMo Fabric itself.

If you are contributing to NeMo Fabric — changing core, bindings, adapters,
documentation, CI, or packaging — use the
[maintainer skills](../.agents/skills/README.md) in `.agents/skills/` instead.

## Portability

Consumer skills are self-contained and exportable. Each skill depends only on
supported public interfaces (the `nemo_fabric` Python package) and public
documentation URLs, never on repository-internal paths.

- Cross-links point to the published documentation and public example URLs on
  GitHub, not to files inside this checkout. Skill-specific material is bundled
  under each skill's own `references/`.
- Skills do not depend on repository internals — their links are absolute or
  bundled, so they resolve when copied out.

## Using A Consumer Skill In Your Project

Copy the skill directory — for example `nemo-fabric-integrate/`, including its
`references/` — into the place your coding agent discovers skills **in your own
project**. Do not rely on this repository's maintainer wiring (its `.claude/skills`
symlink or `.agents/skills/` set); those serve NeMo Fabric's own contributors.

- **Claude Code:** place it at `.claude/skills/nemo-fabric-integrate/` in your
  project, or `~/.claude/skills/nemo-fabric-integrate/` to use it across
  projects. Claude Code discovers `SKILL.md` files under those directories.
- **OpenAI Codex:** place it at
  `<your-project>/.agents/skills/nemo-fabric-integrate/` in your project, or
  `$CODEX_HOME/skills/nemo-fabric-integrate/` to use it across projects.
- **Other agents:** each skill is a portable `SKILL.md` bundle — put it wherever
  your agent loads skills, or reference its `SKILL.md` directly from your agent
  instructions. Confirm discovery with a prompt that should trigger the skill.

## Start Here

| Skill | Use it when |
|---|---|
| [`nemo-fabric-integrate`](nemo-fabric-integrate/SKILL.md) | You are adding NeMo Fabric to a consumer application, service, evaluation harness, or platform through the typed Python SDK — building an in-memory `FabricConfig`, choosing the single-invocation convenience API or an explicitly started runtime, validating with `plan`/`doctor`, and consuming normalized results. |

## Conventions

- **Naming:** consumer skills are prefixed with the product name,
  `nemo-fabric-<topic>`.
- **Frontmatter:** each `SKILL.md` begins with YAML frontmatter containing at
  least `name` and `description`. `SKILL.md` files do not carry an SPDX header;
  every other file, including this README and bundled `references/`, does.
- **Self-containment:** keep a skill usable outside this repository. Link to
  public documentation and example URLs, and bundle any skill-specific reference
  material under the skill's own `references/`.
