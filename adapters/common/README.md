<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Fabric Adapter Utilities

`nemo-fabric-adapters-common` provides shared Python helpers for NVIDIA NeMo
Fabric adapter implementations. NeMo Fabric adapter packages normally install
this package as a dependency.

Install the package directly when developing an adapter:

```bash
pip install nemo-fabric-adapters-common
```

Alternately through the NeMo Fabric metapackage:

```bash
pip install "nemo-fabric[adapters-common]"
```

Refer to the [NeMo Fabric documentation](https://nvidia-nemo-fabric.docs.buildwithfern.com/nemo/fabric)
for adapter and configuration guidance. Source code is available in the
[NVIDIA NeMo Fabric repository](https://github.com/NVIDIA/nemo-fabric/).
