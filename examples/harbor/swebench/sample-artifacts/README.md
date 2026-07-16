<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Relay Sample Artifacts

These bundles come from successful Harbor runs of
`swe-bench/django__django-13741` through Hermes and Claude. Fabric reported
`succeeded`, Harbor's verifier returned reward `1.0` for both runs, and Relay
emitted both ATOF and ATIF.

Each harness directory contains:

- Relay's complete `events.atof.jsonl` and native ATIF trajectory;
- `trajectory.json`, the byte-identical ATIF copy promoted to Harbor's canonical
  path;
- concise Fabric, Harbor verifier, and telemetry summaries; and
- the resulting workspace patch.

The complete telemetry files use Git LFS. The telemetry quality gate scanned
the recorded artifacts for obvious credential patterns before they were added.
Runtime and request identifiers in the summaries are omitted because they are
not part of the portable artifact contract.
