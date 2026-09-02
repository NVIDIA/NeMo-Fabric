<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Fabric mini-SWE-agent Adapter

Use the `nvidia.fabric.mini-swe-agent` adapter to run mini-SWE-agent with
NVIDIA NeMo Fabric.

## Install

Choose an installation option based on the components required in the target
environment:

| Installation | Runtime | Adapter | Harness | Relay Python |
| --- | --- | --- | --- | --- |
| `pip install "nemo-fabric[mini-swe-agent]"` | Yes | Yes | Yes | No |
| `pip install "nemo-fabric[mini-swe-agent,relay]"` | Yes | Yes | Yes | Yes |
| `pip install "nemo-fabric-adapters-mini-swe-agent[harness]"` | No | Yes | Yes | No |
| `pip install "nemo-fabric-adapters-mini-swe-agent[relay]"` | No | Yes | No | Yes |
| `pip install "nemo-fabric-adapters-mini-swe-agent[full]"` | No | Yes | Yes | Yes |
| `pip install nemo-fabric-adapters-mini-swe-agent` | No | Yes | No | No |

The `harness` and `full` extras install the latest compatible mini-SWE-agent
2.x release.

## Configuration

The adapter supports `models`, `models.base_url`, `models.temperature`,
replacement `instructions.system`, `runtime.max_turns`, and
`environment.workspace`. It rejects `append` system instructions.
`runtime.timeout_seconds` sets the NVIDIA NeMo Fabric invocation deadline. Use
`harness.settings.timeout` to set the maximum duration of one command; the
default is `30` seconds.

Set `models.<role>.api_key_env` to the environment variable containing the
model-provider credential. This adapter does not support MCP, skills, tool
policy, or native OpenAI streaming.

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
            provider="nvidia",
            model="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
            api_key_env="NVIDIA_API_KEY",
            base_url="https://integrate.api.nvidia.com/v1",
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

## Continue a Task Across Invocations

Use one runtime for ordered invocations that should share mini-SWE-agent
conversation history:

```python
async def continue_task():
    async with await Fabric().start_runtime(
        config,
        base_dir="/workspace",
    ) as runtime:
        first = await runtime.invoke(
            input="Inspect calculator.py and identify the bug."
        )
        second = await runtime.invoke(input="Fix it and run the tests.")
    return first, second
```

For example, the first invocation can produce this conversation:

```text
system: You are a coding agent.
user: Inspect calculator.py and identify the bug.
assistant: I will inspect the file.
assistant action: bash {"command": "cat calculator.py"}
tool: def divide(a, b): return a // b
assistant: The divide function uses // instead of /.
assistant action: bash {"command": "printf 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\nThe divide function uses integer division.\\n'"}
tool: COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
      The divide function uses integer division.
exit: Submitted
```

Before the second model call, the adapter removes the terminal `exit` message
and appends the new input. The retained context resembles the following
conversation:

```text
system: You are a coding agent.
user: Inspect calculator.py and identify the bug.
assistant: I will inspect the file.
assistant action: bash {"command": "cat calculator.py"}
tool: def divide(a, b): return a // b
assistant: The divide function uses // instead of /.
assistant action: bash {"command": "printf 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\nThe divide function uses integer division.\\n'"}
tool: COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
      The divide function uses integer division.
user: Fix it and run the tests.
```

Assistant messages and output from commands executed through the `bash` tool
are part of the retained model context. Unrelated terminal output is not
retained. History lasts only for the same runtime and is discarded when the
runtime stops. The adapter does not currently truncate or summarize retained
history, so repeated invocations can increase model context usage.

## Relay Telemetry

When Relay is enabled, the adapter uses a Relay-specific mini-SWE-agent subclass
to emit an Agent scope for each invocation, Function scopes for agent steps,
and LLM and tool events for model queries and `bash` actions. Relay is imported
and the subclass is selected only for Relay-enabled runtimes; otherwise the
adapter uses the existing retaining agent without Relay instrumentation.

The adapter wraps each invocation with `nemo_relay.scope.scope(...)`. Because
mini-SWE-agent does not provide native observability hooks, the adapter-owned
subclass explicitly emits nested step, LLM, and `bash` tool events using Relay
handles. This produces a correlated Agent, Function, LLM, and tool scope
hierarchy without requiring upstream changes.

### Correlation IDs

The top-level `mini-swe-agent.request` Agent scope stores the NeMo Fabric
request and invocation IDs as Relay metadata:

| NeMo Fabric ID | Relay metadata key | Meaning |
| --- | --- | --- |
| `request_id` | `nemo_fabric_request_id` | Correlates the caller's logical request. A caller can provide the same value when it wants to correlate retries or related processing. |
| `invocation_id` | `nemo_fabric_invocation_id` | Identifies one concrete invocation attempt. NeMo Fabric assigns a new value to each invocation. |

Nested step, LLM, and `bash` tool events are correlated through the Relay scope
hierarchy; they do not repeat these metadata fields. The `runtime_id`
identifies the longer-lived retained runtime that can process multiple
invocations, but the mini-SWE-agent adapter does not currently add it to Relay
scope metadata.

Install both the harness and Relay:

```bash
pip install "nemo-fabric[mini-swe-agent,relay]"
```

Enable Relay on the configuration:

```python
config.enable_relay()
```

Relay-enabled runs support configured ATIF, OpenTelemetry, and OpenInference
outputs. `Runtime.invoke_stream()` also provides live ATOF records. Live
telemetry uses ordinary adapter `invoke()` and is independent of native OpenAI
streaming, so the adapter's `capabilities.streaming` value remains `false`.
