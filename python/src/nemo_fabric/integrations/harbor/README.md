<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Harbor Integration Internals

The public entrypoint is `nemo_fabric.integrations.harbor:FabricAgent`.
`fabric_agent.py` owns the Harbor `BaseAgent` boundary, `models.py` owns the
strict transport contract, and `runner.py` calls `Fabric.run()` directly inside
the task environment with one typed `FabricConfig`. The primary path constructs
that config from Harbor agent inputs; an explicit `module:callable` factory is
available for advanced consumers. `telemetry.py` validates ATOF/ATIF and
publishes Harbor's canonical `agent/trajectory.json`.

Usage, configuration, examples, and operational guidance live in the canonical
[`examples/harbor/README.md`](../../../../../examples/harbor/README.md).
