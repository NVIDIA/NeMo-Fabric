{/*
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
*/}

# Custom Agents

NVIDIA NeMo Fabric distinguishes reusable agent harnesses from custom agents.
An agent harness is an opinionated, reusable execution layer that supplies
agent behavior and integration points such as models, tools, MCP, skills,
state, or delegation. Customizing a harness means selecting and configuring
those supported components.

A custom agent is application-defined executable behavior built with a
lower-level framework or target runtime. Its execution model cannot be known
statically by NeMo Fabric. The shared adapter therefore resolves configuration,
loads the selected entry point, constructs a target-native agent, and retains
that agent for the NeMo Fabric runtime.

One adapter can support many custom agents built for the same adapter target.
Agent-specific `workflow.settings` still belongs to each `AgentConfig`, and the
descriptor must bound the workflow shapes that the shared adapter accepts.

## Workflow

`workflow` is the `AgentConfig` construct that selects and configures one
custom agent or workflow:

- `workflow.entrypoint.kind` selects well-known resolution semantics.
- `workflow.entrypoint.ref` identifies the factory within those semantics.
- `workflow.settings` contains construction settings for this agent.

Runtime/session identity comes from `RuntimeContext`, and per-invocation input
comes from `AgentRunRequest`. Neither belongs in workflow settings.

The descriptor's `workflow_schema` advertises the kinds, references, and
settings the adapter supports. A configured workflow fails planning if the
descriptor does not publish that schema.

## Resolution Kinds

The initial Python contract defines three resolution kinds. An adapter
implements only the kinds allowed by its descriptor.

| `kind` | Meaning of `ref` | Resolution |
| --- | --- | --- |
| `factory` | NeMo Fabric-defined agent intent, such as `fabric.agent.react` | The adapter maps the intent to a target-native factory. |
| `python_entrypoint` | Name in the `nemo_fabric.agents` Python entry-point group | The adapter resolves an installed entry point with `importlib.metadata`. |
| `python_module` | Importable dotted module name | The adapter imports the module and loads its `create_agent` factory. |

`ref` is never a filesystem path. Resolution and module import occur in the
task environment during `start`, not in the planning process.

### Fabric Factory Intent

Use a NeMo Fabric-defined factory intent for portable agent behavior:

```yaml
workflow:
  entrypoint:
    kind: factory
    ref: fabric.agent.react
```

NeMo Fabric owns the intent name and its portable semantics. Each adapter maps a
supported intent to its target-native implementation. The NAT reference
adapter currently demonstrates this mode for `fabric.agent.react`.

### Installed Python Entry Point

Register a custom agent factory in the fixed Python entry-point group:

```toml
[project.entry-points."nemo_fabric.agents"]
"acme.agent.phishing" = "acme_agents.phishing:create_agent"
```

Select the registered factory by name:

```yaml
workflow:
  entrypoint:
    kind: python_entrypoint
    ref: acme.agent.phishing
```

A missing or duplicate entry-point registration is a startup configuration
error.

### Installed Python Module

An importable Python module exposes a `create_agent` factory:

```python
# acme_agents/phishing.py
def create_agent(context):
    ...
```

Select that module with the matching workflow entry point:

```yaml
workflow:
  entrypoint:
    kind: python_module
    ref: acme_agents.phishing
```

## Shared Adapter Responsibilities

All resolution paths yield the adapter's internal factory abstraction. The
shared adapter performs the following steps:

1. Resolves normalized models, instructions, tools, MCP, skills, workflow
   settings, and runtime context into target-native values.
2. Resolves and calls the selected factory exactly once per NeMo Fabric runtime.
3. Supplies an adapter-defined build context rather than raw `FabricConfig`.
4. Retains the returned target-native agent for later invocations.
5. Translates requests and results and owns shutdown.

The custom agent factory assembles agent-specific behavior. It should not parse
`FabricConfig` or reimplement normalized configuration mapping. This keeps one
adapter reusable across custom agents and keeps consumer config independent of
the target framework.

`workflow.settings` can be an explicitly open compatibility object for an
existing target, but such a mode cannot promise harness variation. Prefer a
closed schema or bounded bindings that let the shared adapter map normalized
models, tools, and MCP independently of the agent-specific settings.
