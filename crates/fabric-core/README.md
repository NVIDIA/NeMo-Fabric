<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Fabric Core

`nemo-fabric-core` provides the core configuration and runtime contracts for
NVIDIA NeMo Fabric, a runtime execution layer for agents.

Add the crate to a Rust project:

```bash
cargo add nemo-fabric-core
```

This crate provides typed agent configuration, validation, run planning,
runtime lifecycle operations, normalized results and artifact manifests,
telemetry references, diagnostics, and JSON Schema generation.

For architecture, configuration concepts, adapters, and examples, refer to the
[NVIDIA NeMo Fabric repository](https://github.com/NVIDIA/nemo-fabric).

## Planning remote descriptor snapshots

`resolve_run_plan_from_descriptors` accepts the canonical resolved descriptors
returned by discovery and a public `FabricConfig`. It uses only that supplied
inventory, never local or bundled fallbacks. The same planner validates native
settings, models, extension schemas and supported normalized fields. Hosts can
therefore plan an image snapshot without starting containers or reconstructing
adapter-specific compatibility rules. Snapshot validation does not prove the
remote runtime is ready.
