<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Fabric Collector

Standalone service for collecting and routing NVIDIA NeMo Relay records to
NVIDIA NeMo Fabric runtimes.

Embed the collector in an asynchronous Python application with:

```python
from nemo_fabric_collector import serve_collector

async with serve_collector(host="127.0.0.1", port=0) as base_url:
    print(base_url)
```
