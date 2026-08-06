<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Fabric LangGraph Adapter

`nemo-fabric-adapters-langgraph` loads a configured LangGraph graph factory,
compiles its graph once per NeMo Fabric runtime, and invokes the compiled graph
with the NeMo Fabric request input.

Install the adapter and a compatible LangGraph harness with:

```bash
pip install "nemo-fabric-adapters-langgraph[harness]"
```

For configuration and lifecycle details, see the
[LangGraph adapter guide](https://github.com/NVIDIA/NeMo-Fabric/tree/main/adapters/langgraph).
