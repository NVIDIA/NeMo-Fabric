<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Use NVIDIA-labs Object Oriented Agents (NOOA) InteractiveAgent with NVIDIA NeMo Fabric

The shared `nvidia.fabric.nooa` adapter runs registered NOOA `InteractiveAgent`
targets through the NeMo Fabric workflow contract. Target packages construct
their own agents, while the adapter provides a common lifecycle, queue
dispatcher, and result format.

Complete the [shared source and Relay setup](../README.md) before using this
adapter.

## How the Adapter Works

A target descriptor selects a Python factory. For each NeMo Fabric runtime, the
adapter performs the following operations:

1. Calls the factory during `start` and retains the returned agent.
2. Submits each invocation string to `user_messages`.
3. Runs the standard NOOA `race` / drain / `handle` dispatcher loop.
4. Converts `AgentMessage` events into the normalized response and message list.
5. Calls target-owned cleanup, `agent.close()`, or the queue manager's fallback
   shutdown during `stop`.

The adapter translates normalized models into NOOA `UnifiedLLM` clients. The
factory build context also receives the resolved system instruction, workspace,
artifact root, workflow settings, and exact skill paths.

The descriptor accepts normalized models, model endpoint and temperature fields,
`instructions.system`, `skills`, and whole MCP server configurations.

## Terminal Behavior

`WAIT` pauses the dispatcher until another NOOA channel wakes within the same
NeMo Fabric invocation. `DONE`, `NEED_INPUT`, and the legacy `GET_USER_INPUT`
end the invocation. The latter two return a successful adapter call with
`completed=false` and preserve the native terminal reason.

The normalized `messages` field contains ordered `{"content": "..."}` records.
The `response` field contains the final message or, when the agent emits no
`AgentMessage`, the terminal `RespondResult.explanation`.

## Register a Target

A separately installed target publishes a `*.fabric-target.json` descriptor:

```json
{
  "contract_version": "fabric.adapter/v1alpha2",
  "type": "workflow",
  "id": "com.example.nooa.my-agent",
  "adapter_id": "nvidia.fabric.nooa",
  "spec": {
    "entrypoint": {
      "kind": "interactive_agent_factory",
      "ref": "my_package.fabric_target:create_agent"
    },
    "settings_schema": {
      "type": "object",
      "properties": {},
      "additionalProperties": false
    }
  }
}
```

The entry-point reference uses `package.module:factory` syntax. The factory
receives an `InteractiveAgentBuildContext` and returns either an
`InteractiveAgent` or an `InteractiveAgentTarget` with explicit cleanup:

```python
from nemo_fabric_adapters.nooa import InteractiveAgentBuildContext
from nemo_fabric_adapters.nooa import InteractiveAgentTarget


async def create_agent(
    context: InteractiveAgentBuildContext,
) -> InteractiveAgentTarget:
    agent = MyInteractiveAgent(
        cwd=context.workspace,
        **context.config.workflow.settings,
    )
    return InteractiveAgentTarget(agent=agent, close=agent.close)
```

Most targets should return the bare agent and use the default prompt-turn
policy. A target whose external environment determines completion can provide
a `continue_after(agent, reason, explanation)` predicate. This predicate only
decides whether the adapter waits for another queue event after a terminal
result. The shared adapter retains control of dispatch, result validation,
message capture, and normalization.

The factory owns target-specific construction and dependency validation. The
adapter validates the returned object's public interactive-agent surface; it
does not import target-specific agent classes.

## Configure MCP Servers

The adapter creates NOOA MCP tools from each normalized server, registers them
as `mcp.<server-name>`, and activates the complete server. Stdio, server-sent
events (SSE), and streamable HTTP transports are supported. Network servers can
also use custom headers.

For example, add a streamable HTTP server to an existing configuration:

```python
config.add_mcp_server(
    "repository",
    transport="streamable-http",
    url="${REPOSITORY_MCP_URL}",
    custom_headers={"X-Tenant": "code-review"},
)
```

Use `config.remove_mcp_server("repository")` to run the same agent without that
server. This add-or-remove pattern provides whole-server capability variation.
The adapter does not declare normalized MCP authentication or per-method tool
filters, so NeMo Fabric rejects those configurations during planning.

The selected target must expose a NOOA-compatible skill registry through
`agent.skills.register()` and `agent.skills.activate()`. `CodingAgent` and the
registered ARC solver provide this surface. A custom `InteractiveAgent` can run
without a skill registry when its configuration does not include MCP servers.

## Relay Telemetry

Each Relay-enabled invocation installs NOOA's public `install_nemo_relay()`
middleware and opens one `nooa-interactive-agent-request` scope. The middleware
records nested agent-method, LLM, and `execute_python` scopes before the adapter
finalizes the current invocation's artifacts.

`Runtime.invoke_stream()` consumes the resulting ATOF records while the adapter
runs its ordinary `invoke` operation. This adapter does not implement native
model-response streaming.

## Run from Source

Complete the [shared source installation](../README.md#install-from-source).
Include this directory and the target descriptor's directory in
`FabricConfig.discovery.local_paths`.

### CodingAgent

[`coding-agent.fabric-target.json`](../targets/coding-agent.fabric-target.json)
registers `nvidia.nooa.coding-agent`. Its factory constructs
`nooa_cli.coding.CodingAgent` directly. Install `nooa-cli` in the adapter
environment before selecting this target. The factory uses the selected
`default` model, resolved workspace, portable system instruction, and
configured text skills. `CodingAgent.close()` owns its shell, skill registry,
queue jobs, and model shutdown. The adapter also finalizes its model clients
during runtime cleanup.

The maintained [code-review example](../../../examples/code_review_agent/README.md)
exposes this target as `--variant nooa`.

### ARC Solver

[`arc-solver.fabric-target.json`](../targets/arc-solver.fabric-target.json)
registers `nvidia.nooa.arc-solver`. Its factory constructs the markdown-backed
`MdArcSolverAgent`, which derives from `ArcSolverBase`. The memory-backed
variant is not registered.

The target uses `<artifact-root>/nooa-arc` as its run directory, or
`<workspace>/nooa-arc` when no artifact root is configured. An external ARC
harness must use the same directory for `states.jsonl` and `actions.jsonl`.
The factory fixes the agent-visible identity to `the game`; consumers cannot
configure a real game ID.

The target settings are `reflect_every`, `visual`, `png_scale`, and
`max_actions_per_turn`. Configure at most one skill directory, and ensure that
it contains `SKILL.md`.

The ARC launcher treats `DONE` as the end of one turn while the external
harness decides when the game finishes. The target continues after `DONE` until
the latest state is `WIN` or `GAME_OVER`, or the harness publishes a
`harness stopped:` note. This completion policy remains outside the shared
adapter. If the latest state is unavailable, the target ends the invocation
instead of waiting indefinitely for another queue event.

Expose the ARC example source before starting NeMo Fabric:

```bash
export PYTHONPATH="$PWD/external/nooa/src:$PWD/../labs-OO-Agents/examples/arc_agi_3${PYTHONPATH:+:$PYTHONPATH}"
```

This command assumes that the `labs-OO-Agents` checkout is a sibling of the
NVIDIA NeMo Fabric repository. Adjust the path when the checkout is elsewhere.

The deterministic NeMo Fabric test uses a finite test harness. A manual
full-game run requires the NOOA ARC harness, its `arc` optional dependencies,
and an external game service.

## Security Boundary

The adapter is an execution bridge, not a sandbox. A CodeAct target such as
`CodingAgent` can run generated Python and shell commands with the permissions
of its NeMo Fabric environment. Select an environment provider with the required
operating-system isolation.
