<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Fabric mini-SWE-agent Adapter

Use the `nvidia.fabric.mini-swe-agent` adapter to run mini-SWE-agent with
NVIDIA NeMo Fabric.

## Install

| Installation | Runtime | Adapter | Harness |
| --- | --- | --- | --- |
| `pip install "nemo-fabric[mini-swe-agent]"` | Yes | Yes | Yes |
| `pip install "nemo-fabric-adapters-mini-swe-agent[harness]"` | No | Yes | Yes |
| `pip install nemo-fabric-adapters-mini-swe-agent` | No | Yes | No |

The `harness` and `full` extras install the latest compatible mini-SWE-agent
2.x release.

## Configuration

The adapter supports `models`, `models.base_url`, `models.temperature`,
`instructions.system`, `runtime.max_turns`, and `environment.workspace`.
`runtime.timeout_seconds` sets the NVIDIA NeMo Fabric invocation deadline. Use
`harness.settings.timeout` to set the maximum duration of one command; the
default is `30` seconds.

Set `models.<role>.api_key_env` to the environment variable containing the
model-provider credential. This adapter does not support MCP, skills, tool
policy, streaming, or telemetry output.

```python
import asyncio

from nemo_fabric import EnvironmentConfig, Fabric, FabricConfig, HarnessConfig
from nemo_fabric import InstructionConfig, InstructionsConfig, MetadataConfig
from nemo_fabric import ModelConfig, RuntimeConfig

config = FabricConfig(
    metadata=MetadataConfig(name="mini-swe-agent"),
    harness=HarnessConfig(
        adapter_id="nvidia.fabric.mini-swe-agent",
        resolution="preinstalled",
        settings={"timeout": 30},
    ),
    models={
        "default": ModelConfig(
            provider="openai",
            model="gpt-5-mini",
            api_key_env="OPENAI_API_KEY",
        )
    },
    instructions=InstructionsConfig(
        system=InstructionConfig(content="Work carefully and verify the change.")
    ),
    runtime=RuntimeConfig(max_turns=50, timeout_seconds=1800),
    environment=EnvironmentConfig(provider="local", workspace="/workspace"),
)

async def main():
    return await Fabric().run(
        config, base_dir="/workspace", input="Fix the failing test."
    )


result = asyncio.run(main())
```

The result includes the submitted final text and API-call usage.
