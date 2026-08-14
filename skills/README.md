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

Integration skills are self-contained and exportable. Consumer skills depend
on the public `nemo_fabric` package. Harness skills depend only on published
adapter-contract packages, schemas, and documentation. Neither depends on
repository-internal paths.

- Cross-links point to the published documentation and public example URLs on
  GitHub, not to files inside this checkout. Skill-specific material is bundled
  under each skill's own `references/`.
- Skills do not depend on repository internals — their links are absolute or
  bundled, so they resolve when copied out.

## Using an Integration Skill in Your Project

Copy an individual skill directory, such as `nemo-fabric-integrate/` or
`nemo-fabric-build-adapter/`, into the place your coding agent discovers skills
**in your own project**. Include any bundled resources. Do not rely on this
repository's maintainer wiring (its `.claude/skills` symlink or
`.agents/skills/` set); those serve NeMo Fabric's own contributors.

- **Claude Code:** place the skill at `.claude/skills/<skill-name>/` in your
  project, or `~/.claude/skills/<skill-name>/` to use it across projects.
  Claude Code discovers `SKILL.md` files under those directories.
- **OpenAI Codex:** place it at
  `<your-project>/.agents/skills/<skill-name>/` in your project, or
  `~/.agents/skills/<skill-name>/` to use it across projects.
- **Other agents:** each skill is a portable `SKILL.md` bundle — put it wherever
  your agent loads skills, or reference its `SKILL.md` directly from your agent
  instructions. Confirm discovery with a prompt that should trigger the skill.

## Consumer Integrations

Consumer integration skills live directly under `skills/` so each bundle can be
validated and published independently. The following skill helps software on
the consumer side call NeMo Fabric through its public SDK:

| Skill | Use It When |
|---|---|
| [`nemo-fabric-integrate`](nemo-fabric-integrate/SKILL.md) | You are adding NeMo Fabric to a consumer application, service, evaluation harness, or platform through the typed Python SDK — building an in-memory `FabricConfig`, choosing the single-invocation convenience API or an explicitly started runtime, validating with `plan`/`doctor`, and consuming normalized results. |

## Harness Integrations

Harness integration skills also live directly under `skills/`:

| Skill | Use It When |
|---|---|
| [`nemo-fabric-build-adapter`](nemo-fabric-build-adapter/SKILL.md) | You are creating, migrating, reviewing, or maintaining a third-party adapter, descriptor, normalized configuration mapping, lifecycle implementation, custom-agent loader, or conformance report. |

## Conventions

- **Naming:** integration skills are prefixed with the product name,
  `nemo-fabric-<topic>`.
- **Release collection:** before publishing release-bound skill updates to an
  external skills registry, collect the merged public integration-skill changes
  from `skills/` and preserve their source PR references.
- **Frontmatter:** each `SKILL.md` begins with YAML frontmatter containing at
  least `name` and `description`. `SKILL.md` files do not carry an SPDX header;
  every other file, including this README and bundled `references/`, does.
- **Self-containment:** keep a skill usable outside this repository. Link to
  public documentation and example URLs, and bundle any skill-specific reference
  material under the skill's own `references/`.
