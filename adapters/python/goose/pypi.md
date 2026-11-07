<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Fabric Goose Adapter

[![License](https://img.shields.io/github/license/NVIDIA/NeMo-Fabric)](https://github.com/NVIDIA/NeMo-Fabric/blob/main/LICENSE)
[![GitHub](https://img.shields.io/badge/github-repo-blue?logo=github)](https://github.com/NVIDIA/NeMo-Fabric/)

`nemo-fabric-adapters-goose` provides the Goose adapter for NVIDIA NeMo Fabric.
It runs the Goose CLI headless and keeps one Goose session for each NeMo Fabric
runtime.

Install the Goose CLI 1.57.0 from the
[Goose release](https://github.com/aaif-goose/goose/releases/tag/v1.57.0), then
install the adapter:

```bash
pip install "nemo-fabric[goose]"
```

The adapter does not install Goose. Relay telemetry is not supported.

Refer to the [NVIDIA NeMo Fabric repository](https://github.com/NVIDIA/NeMo-Fabric/tree/main/adapters/python/goose)
for configuration and usage instructions.
