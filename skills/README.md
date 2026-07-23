<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Fabric Integration Skills

These skills help external developers integrate with NeMo Fabric through its
public contracts. Consumer integration skills connect an application, service,
evaluation harness, or platform to NeMo Fabric through the public Python SDK.
Harness integration skills will help harness authors build adapters that are
compatible with NeMo Fabric.

Harness integrations are separate adapters that connect NeMo Fabric to agent
harnesses such as Claude Code, Codex, Hermes Agent, and LangChain Deep Agents.
Refer to the [harness integration guides](../adapters/README.md) when you need to
configure or compare those adapters.

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

## Using an Integration Skill in Your Project

For example, copy the individual skill directory
`integrations/consumer/nemo-fabric-integrate/`, including its `references/`,
into the place your coding agent discovers skills **in your own project**. Copy
the skill bundle itself, not its `consumer/` or `harness/` category directory.
Do not rely on this repository's maintainer wiring (its `.claude/skills` symlink
or `.agents/skills/` set); those serve NeMo Fabric's own contributors.

- **Claude Code:** place it at `.claude/skills/nemo-fabric-integrate/` in your
  project, or `~/.claude/skills/nemo-fabric-integrate/` to use it across
  projects. Claude Code discovers `SKILL.md` files under those directories.
- **OpenAI Codex:** place it at
  `<your-project>/.agents/skills/nemo-fabric-integrate/` in your project, or
  `$CODEX_HOME/skills/nemo-fabric-integrate/` to use it across projects.
- **Other agents:** each skill is a portable `SKILL.md` bundle — put it wherever
  your agent loads skills, or reference its `SKILL.md` directly from your agent
  instructions. Confirm discovery with a prompt that should trigger the skill.

## Consumer Integrations

Consumer integration skills live under `integrations/consumer/`. They help
software on the consumer side call NeMo Fabric through its public SDK.

| Skill | Use it when |
|---|---|
| [`nemo-fabric-integrate`](integrations/consumer/nemo-fabric-integrate/SKILL.md) | You are adding NeMo Fabric to a consumer application, service, evaluation harness, or platform through the typed Python SDK — building an in-memory `FabricConfig`, choosing the single-invocation convenience API or an explicitly started runtime, validating with `plan`/`doctor`, and consuming normalized results. |

## Harness Integrations

Harness integration skills belong under `integrations/harness/`. A forthcoming
adapter-authoring skill will guide third-party harness authors through the
published adapter contract so they can make their harnesses Fabric-ready.

## Conventions

- **Naming:** integration skills are prefixed with the product name,
  `nemo-fabric-<topic>`.
- **Frontmatter:** each `SKILL.md` begins with YAML frontmatter containing at
  least `name` and `description`. `SKILL.md` files do not carry an SPDX header;
  every other file, including this README and bundled `references/`, does.
- **Self-containment:** keep a skill usable outside this repository. Link to
  public documentation and example URLs, and bundle any skill-specific reference
  material under the skill's own `references/`.
