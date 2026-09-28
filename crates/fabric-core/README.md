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

## Planning Against a Descriptor Catalog

`discover_descriptors` returns the `DescriptorCatalog` that planning would use:
every adapter and target descriptor, with its sources.
`resolve_run_plan_from_descriptors` plans a public `FabricConfig` against a
supplied catalog and never falls back to local or bundled descriptors. It uses
the same validation for native settings, models, extension schemas, and
normalized fields. A host can therefore read a catalog inside an image and plan
for that image without starting it or reconstructing adapter-specific rules.
A valid plan does not establish that the image's runtime is ready.
