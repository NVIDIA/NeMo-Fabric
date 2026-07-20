<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Temporary Workarounds

Track repository workarounds that must be removed after an upstream dependency
ships. Each entry must identify an upstream reference, removal condition, and
cleanup validation.

## Installed Adapter Descriptor Metadata

- **Status:** Waiting for adapter wheel metadata support
- **Added:** July 19, 2026
- **Affected implementation:** `crates/fabric-core/src/config.rs`,
  `crates/fabric-cli/src/presets.rs`, `crates/fabric-cli/assets/adapters/`, and
  the Python binding
- **Reason:** The published CLI crate currently carries package-local copies of
  adapter descriptors because Cargo cannot package descriptors from the
  repository-level adapter directories. A drift test keeps those copies aligned
  with the canonical descriptors, but the duplication is a packaging
  workaround. CLI presets stage the embedded descriptors under a temporary
  `adapters/` directory, and core also probes a compile-time repository path.
- **Upstream resolution:** NeMo Fabric adapter distributions advertise their
  descriptors through installed wheel metadata (internal packaging milestone;
  no external dependency).
- **Removal condition:** Python can discover installed adapter descriptors from
  wheel metadata and pass them to core through an explicit, typed adapter
  registry.
- **Cleanup:** Add adapter registrations to `ResolveContext`; remove implicit
  repository and `base_dir/adapters` discovery from core; have the Python
  binding register wheel-owned descriptors; revisit and remove the duplicated
  CLI descriptor assets once wheel metadata is consumable; retain embedded or
  staged files only for standalone CLI assets such as the scripted runner; and
  validate both installed-wheel and standalone Rust CLI behavior before
  removing this entry.

## NeMo Relay 0.6.x Request Decoding Release

- **Status:** Waiting for an upstream release
- **Added:** July 16, 2026
- **Affected documentation:** `adapters/codex/README.md` and
  `docs/integrations/codex.mdx`
- **Reason:** Released NeMo Relay versions do not yet decode the
  `zstd`-compressed request bodies emitted by the Codex SDK, so semantic Relay
  artifacts require a source installation.
- **Upstream resolution:**
  [NVIDIA/NeMo-Relay#452](https://github.com/NVIDIA/NeMo-Relay/pull/452), merged
  as `fe144d0d23e483c8216537118304e306abc20837`
- **Removal condition:** A published `nemo-relay-cli` version in Fabric's
  supported `>=0.6.0,<0.7.0` range contains the merged request-decoding fix.
- **Cleanup:** Replace the pinned source-install instructions with the released
  CLI installation, run the Codex Relay end-to-end test, update both affected
  documentation files, and remove this entry.

## NeMo Fabric PyPI Availability

- **Status:** Waiting for a published release
- **Added:** July 16, 2026
- **Affected documentation:** `skills/nemo-fabric-integrate/SKILL.md` (install
  section)
- **Reason:** `nemo-fabric` is not yet published on PyPI, so the consumer
  integration skill installs from a source checkout (`just build-all`) or locally
  built wheels (`just wheels` plus `uv pip install --find-links`).
- **Upstream resolution:** NeMo Fabric's own first PyPI release (internal
  milestone; no external dependency).
- **Removal condition:** `nemo-fabric` and its adapter extras are published on
  PyPI in a version the skill can target.
- **Cleanup:** Replace the source and wheel install steps in the skill with the
  published `pip install "nemo-fabric[...]"` instructions and remove this entry.

# Planned Work

## Example Authoring Skill

- **Status:** Not started
- **Purpose:** Add a guided workflow for users who need help designing a new
  example, selecting the closest preset or maintained example, customizing the
  generated code, and adding documentation and tests.
- **Boundary:** Keep `nemo-fabric example init` as the deterministic operation
  for users who already know which example and variant they want to customize.
  The skill provides authoring judgment and validation around that operation.
- **Implementation:** Have the skill invoke or reuse `example init` as its
  starting point, then modify and validate the generated application. Do not
  create a separate template collection or scaffolding implementation in the
  skill.
- **Completion condition:** The skill can guide a user from an example idea to
  a runnable, documented, and tested Python or Rust example while keeping the
  CLI scaffold as the single source for generated starter code.
