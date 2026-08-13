<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Fabric mini-SWE-agent Adapter

The `nvidia.fabric.mini-swe-agent` adapter runs mini-SWE-agent's native
shell-only loop through NVIDIA NeMo Fabric. Select the harness by adapter ID;
the mini-SWE-agent Python package is an implementation detail.

## Install

| Installation | Runtime | Adapter | Harness |
| --- | --- | --- | --- |
| `pip install "nemo-fabric[mini-swe-agent]"` | Yes | Yes | Yes |
| `pip install "nemo-fabric-adapters-mini-swe-agent[harness]"` | No | Yes | Yes |
| `pip install nemo-fabric-adapters-mini-swe-agent` | No | Yes | No |

The `harness` and `full` extras install the latest compatible mini-SWE-agent
2.x release.

## Configuration

The adapter maps `models`, `models.base_url`, `models.temperature`,
`instructions.system`, `runtime.max_turns`, and `environment.workspace` to
mini-SWE-agent. `runtime.timeout_seconds` remains the NVIDIA NeMo Fabric
invocation deadline. Use the adapter-specific `harness.settings.timeout` to set
the maximum duration of one shell command; mini-SWE-agent defaults to `30`.

mini-SWE-agent uses LiteLLM. The adapter forwards a configured
`models.<role>.api_key_env` credential to LiteLLM without persisting it. Without
that setting, LiteLLM uses its native credential handling. The adapter does not
expose MCP, skills, tool policy, native streaming, or native telemetry.

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

Each invocation runs a fresh mini-SWE-agent loop in the runtime's workspace.
The output includes the submitted final text and API-call usage. A loop that
ends without submitting a final output produces a failed NVIDIA NeMo Fabric
result.
