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

The complete telemetry files and workspace patches use Git LFS. Use these
samples to compare Harbor's canonical trajectory with Relay's direct ATOF and
ATIF output before running the example yourself.
