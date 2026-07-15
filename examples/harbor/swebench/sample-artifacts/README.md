<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Qualified Relay Sample

This bundle comes from a successful Harbor run of
`swe-bench/django__django-13741` on July 14, 2026. Fabric reported `succeeded`,
Harbor's verifier returned reward `1.0`, and Relay emitted both ATOF and ATIF.

The [`hermes-relay/`](hermes-relay/) directory contains:

- Relay's complete `events.atof.jsonl` and native ATIF trajectory;
- `trajectory.json`, the byte-identical ATIF copy promoted to Harbor's canonical
  path;
- concise Fabric, Harbor verifier, and telemetry summaries; and
- the resulting workspace patch.

The complete telemetry files use Git LFS. The telemetry quality gate scanned
the recorded artifacts for obvious credential patterns before they were added.
Runtime and request identifiers in the summaries are omitted because they are
not part of the portable artifact contract.
