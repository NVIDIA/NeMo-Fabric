<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Temporary Workarounds

Track repository workarounds that must be removed after an upstream dependency
ships. Each entry must identify an upstream reference, removal condition, and
cleanup validation.

## NeMo Relay 0.6.x Request Decoding Release

- **Status:** Waiting for an upstream release
- **Added:** July 16, 2026
- **Affected documentation:** `adapters/codex/README.md` and
  `fern/versions/main/pages/integrations/codex.mdx`
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
