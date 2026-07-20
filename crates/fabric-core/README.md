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
