<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# OpenHands Adapter for NVIDIA NeMo Fabric

This package provides the OpenHands SDK adapter for NVIDIA NeMo Fabric. It maps
normalized NeMo Fabric configuration into one persistent OpenHands conversation
per runtime.

Install the tested OpenHands packages and the adapter:

```bash
pip install "openhands-sdk>=1.49.5,<1.50.0" "openhands-tools>=1.49.5,<1.50.0"
pip install nemo-fabric-adapters-openhands
```

Refer to the [NVIDIA NeMo Fabric repository](https://github.com/NVIDIA/NeMo-Fabric/tree/main/adapters/python/openhands)
for configuration and usage instructions.
