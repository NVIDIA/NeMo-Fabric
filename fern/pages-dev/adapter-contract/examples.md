{/*
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
*/}

# Examples and References

Use the example that matches the integration shape you chose. Review its
descriptor, lifecycle implementation, and consumer configuration, in that
order. The canonical schemas remain authoritative when an example omits a
field.

| Integration Shape | Start With | What It Demonstrates |
| --- | --- | --- |
| Harness adapter | [Hermes Agent descriptor](https://github.com/NVIDIA/NeMo-Fabric/blob/main/adapters/hermes/hermes.fabric-adapter.json) and [runtime](https://github.com/NVIDIA/NeMo-Fabric/blob/main/adapters/hermes/src/nemo_fabric_adapters/hermes/adapter.py) | A complete harness integration with normalized configuration, persistent state, and telemetry. |
| Minimum-surface harness adapter | [mini-SWE-agent descriptor](https://github.com/NVIDIA/NeMo-Fabric/blob/main/adapters/mini-swe-agent/mini-swe-agent.fabric-adapter.json) and [runtime](https://github.com/NVIDIA/NeMo-Fabric/blob/main/adapters/mini-swe-agent/src/nemo_fabric_adapters/mini_swe_agent/adapter.py) | The required typed config boundary and `start`/`invoke`/`stop` with little optional behavior. |
| Shared framework adapter | [NeMo Agent Toolkit adapter descriptor](https://github.com/NVIDIA/NeMo-Fabric/blob/main/external/nat/nat.fabric-adapter.json), [runtime](https://github.com/NVIDIA/NeMo-Fabric/blob/main/external/nat/src/nemo_fabric_adapters/nat/adapter.py), and [targets](https://github.com/NVIDIA/NeMo-Fabric/tree/main/external/nat/targets) | One adapter plus independently registered calculator and email-phishing workflows. |
| Dedicated custom-agent adapter | [LangGraph descriptor](https://github.com/NVIDIA/NeMo-Fabric/blob/main/examples/langgraph_custom_agent/adapter/email-phishing.fabric-adapter.json) and [runtime](https://github.com/NVIDIA/NeMo-Fabric/blob/main/examples/langgraph_custom_agent/adapter/runtime.py) | A custom graph with an adjacent adapter and no generic workflow loader. |

## Harness Adapter

Use the Hermes Agent adapter as the primary reference for integrating an
opinionated harness. Its descriptor and runtime demonstrate normalized models,
MCP, instructions, runtime settings, persistent session state, and Relay
telemetry through one harness adapter.

Use mini-SWE-agent as the secondary minimum-surface reference when you want to
understand the required boundary at a glance. Its package contains:

- A static Adapter Descriptor that accepts a small normalized config surface
- One runtime class with `start`, `invoke`, and `stop`
- The optional common Python lifecycle host
- One package README with a complete consumer configuration

The adapter retains the model, environment, agent, and conversation state in
one runtime instance. It does not implement MCP, skills, telemetry, or native
streaming, which keeps the required path visible.

## Shared Adapter With Registered Custom Agents

Start with the NeMo Agent Toolkit reference when one adapter must support many
custom agents built for a framework. Read these files in order:

1. [`nat.fabric-adapter.json`](https://github.com/NVIDIA/NeMo-Fabric/blob/main/external/nat/nat.fabric-adapter.json)
   declares `target_types: ["workflow"]`, normalized configuration, tool
   definition schema, and native OpenAI streaming.
2. [`email-phishing-analyzer.fabric-target.json`](https://github.com/NVIDIA/NeMo-Fabric/blob/main/external/nat/targets/email-phishing-analyzer.fabric-target.json)
   selects the shared adapter and supplies the workflow entry point and
   settings schema.
3. [`calculator.fabric-target.json`](https://github.com/NVIDIA/NeMo-Fabric/blob/main/external/nat/targets/calculator.fabric-target.json)
   shows a second target that reuses the same adapter contract.
4. [`adapter.py`](https://github.com/NVIDIA/NeMo-Fabric/blob/main/external/nat/src/nemo_fabric_adapters/nat/adapter.py)
   translates `AgentConfig` into NeMo Agent Toolkit configuration and owns the
   retained workflow lifecycle.

The consumer selects a target, not an entry point:

```python
FabricConfig(
    workflow=WorkflowConfig(
        target_id="nvidia.examples.nat.email-phishing-analyzer",
        settings={"llm_name": "default"},
    ),
    models={"default": ModelConfig(...)},
)
```

Planning resolves the target's adapter and projects the target-owned entry
point into `AgentConfig.workflow`. A second target changes the target ID and
settings schema without changing the shared Adapter Descriptor.

## Dedicated Custom-Agent Adapter

Start with the LangGraph email-phishing example when the custom application is
the clearest integration boundary. Its directories make ownership explicit:

- [`consumer/`](https://github.com/NVIDIA/NeMo-Fabric/tree/main/examples/langgraph_custom_agent/consumer)
  owns `FabricConfig` and invocation.
- [`adapter/`](https://github.com/NVIDIA/NeMo-Fabric/tree/main/examples/langgraph_custom_agent/adapter)
  receives `AgentConfig`, translates dependencies, and owns the lifecycle.
- [`agent/`](https://github.com/NVIDIA/NeMo-Fabric/tree/main/examples/langgraph_custom_agent/agent)
  contains application behavior without an NVIDIA NeMo Fabric dependency.

The descriptor directly selects this adapter and does not declare
`target_types` or accept `workflow`. Its minimum path maps one model and one
system instruction, compiles the graph during `start`, invokes the retained
graph, and releases its references during `stop`. Optional MCP and Relay
modules are adjacent additions rather than requirements of the minimum
lifecycle.

## Follow the Type Boundary

Across all examples, keep these ownership boundaries intact:

| Consumer Side | Adapter Side | Current v1alpha2 Behavior |
| --- | --- | --- |
| `FabricConfig` | `AgentConfig` | Planning resolves, validates, and projects config before `start`. |
| `RunRequest` | `AgentRunRequest` | NeMo Fabric projects caller input, context, and declared request extensions before invocation. Request identity is supplied through `RuntimeContext`. |
| `RunResult` | `AgentRunResult` | The adapter returns typed status, output, error, usage, artifacts, and declared extensions; NeMo Fabric validates and enriches them for the consumer. |

Use the
[`schemas/adapter-contract/` directory](https://github.com/NVIDIA/NeMo-Fabric/tree/main/schemas/adapter-contract)
for exact southbound wire shapes and [Stage 1](adapter-descriptor.md) to begin
your own descriptor.
