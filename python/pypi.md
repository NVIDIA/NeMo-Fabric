<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Fabric Runtime

[![License](https://img.shields.io/github/license/NVIDIA/NeMo-Fabric)](https://github.com/NVIDIA/NeMo-Fabric/blob/main/LICENSE)
[![GitHub](https://img.shields.io/badge/github-repo-blue?logo=github)](https://github.com/NVIDIA/NeMo-Fabric/)
[![Release](https://img.shields.io/github/v/release/NVIDIA/NeMo-Fabric?color=green)](https://github.com/NVIDIA/NeMo-Fabric/releases)

`nemo-fabric-runtime` provides the Python SDK and native Rust bindings for
NVIDIA NeMo Fabric, a runtime execution layer for agents.

Typically this is installed using the [`nemo-fabric`](https://pypi.org/project/nemo-fabric/) meta-package.

```bash
pip install nemo-fabric
```

Is equivalent to:

```bash
pip install nemo-fabric-runtime
```

The package exposes the `nemo_fabric` Python module for typed agent
configuration, validation, run planning, runtime lifecycle management,
normalized results, artifacts, diagnostics, and telemetry references.

Refer to the [NeMo Fabric documentation](https://nvidia-nemo-fabric.docs.buildwithfern.com/nemo/fabric)
for installation and usage guidance. Source code is available in the
[NVIDIA NeMo Fabric repository](https://github.com/NVIDIA/nemo-fabric/).
