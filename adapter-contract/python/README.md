<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Fabric Adapter Contract

`nemo-fabric-adapter-contract` provides the typed Python configuration and
execution contract implemented by NeMo Fabric adapters. It does not include a
lifecycle host, harness integration, or NeMo Relay integration.

Python adapters can use the package's standard-library dataclasses without an
additional runtime dependency. Adapters in other languages can consume the JSON
Schemas published by NeMo Fabric without a Python package dependency.

The dataclasses provide strict `from_mapping()` validation and JSON-compatible
`to_mapping()` serialization. Install the optional `pydantic` extra when an
adapter uses Pydantic models for typed extensions or wants Pydantic
interoperability. Both paths use the same contract dataclasses.

Import `AgentConfig` from the contract models module:

```python
from nemo_fabric_adapter_contract.models import AgentConfig
```

An adapter descriptor opts into the southbound configuration with
`config.input=agent_config`. Python adapters using the optional common
lifecycle host pass `AgentConfig.from_mapping` as the `config_loader`.

## Install

Install the package directly when developing a Python adapter:

```bash
pip install nemo-fabric-adapter-contract
```

Install the optional `pydantic` extra to enable Pydantic interoperability.

```bash
pip install "nemo-fabric-adapter-contract[pydantic]"
```

Refer to the [NeMo Fabric documentation](https://docs.nvidia.com/nemo/fabric)
for adapter and configuration guidance. Source code is available in the
[NVIDIA NeMo Fabric repository](https://github.com/NVIDIA/NeMo-Fabric).
