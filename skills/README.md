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

Consumer skills are self-contained and exportable: copy a skill directory into
your own project and it keeps working.

- They depend only on supported public interfaces (the `nemo_fabric` Python
  package) and the published documentation at
  <https://docs.nvidia.com/nemo/fabric>, never on repository-internal paths.
- Cross-links point to published docs and public example URLs, not to files
  inside this checkout. Skill-specific material is bundled under each skill's own
  `references/`.
- They do not reference maintainer skills, contribution commands, or repository
  internals.

## Start Here

| Skill | Use it when |
|---|---|
| [`nemo-fabric-integrate`](nemo-fabric-integrate/SKILL.md) | You are adding NeMo Fabric to a consumer application, service, evaluation harness, or platform through the typed Python SDK — building an in-memory `FabricConfig`, choosing one-shot versus stateful-runtime execution, validating with `plan`/`doctor`, and consuming normalized results. |

## Conventions

- **Naming:** consumer skills are prefixed with the product name,
  `nemo-fabric-<topic>`.
- **Frontmatter:** each `SKILL.md` begins with YAML frontmatter containing at
  least `name` and `description`. `SKILL.md` files do not carry an SPDX header;
  every other file, including this README and bundled `references/`, does.
- **Self-containment:** keep a skill usable outside this repository. Link to the
  published docs and public example URLs, and bundle any skill-specific reference
  material under the skill's own `references/`.
