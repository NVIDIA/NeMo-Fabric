<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Qualified Harness Samples

These compact bundles were curated from successful Harbor runs of
`swe-bench/django__django-13741` on July 14, 2026. Both runs completed without
an orchestration exception, Fabric reported `succeeded`, and Harbor's verifier
returned reward `1.0`.

The files are review aids, not complete trial directories. Runtime IDs, request
IDs, host paths, prompts, verbose logs, full Claude events, and credentials are
excluded. `workspace.patch` is the relevant source-code portion of the collected
patch; a harness may have produced other incidental files during its run.

| Harness | Model Used for This Run | Bundle |
| --- | --- | --- |
| Hermes | self-hosted `nvidia/nemotron-3-nano` | [`hermes/`](hermes/) |
| Claude | `anthropic/claude-sonnet-4-5` | [`claude/`](claude/) |

Telemetry was intentionally disabled in these baseline runs, so both telemetry
summaries demonstrate the expected `not_emitted` state. Use the Hermes Relay
variant in the parent guide to produce and validate ATOF and ATIF.
