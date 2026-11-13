<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA-labs Object Oriented Agents (NOOA) Adapters for NVIDIA NeMo Fabric

This directory provides two ways to run NOOA agents through the NeMo Fabric
lifecycle contract. Choose the integration that matches the agent interface:

| Integration | Use It For |
| --- | --- |
| [InteractiveAgent adapter](docs/interactive-agent.md) | Registered `InteractiveAgent` implementations, including `CodingAgent` and `ArcSolverBase` targets |
| [BenchAgent adapter](docs/bench-agent.md) | `nooa_bench.BenchAgent` tasks and Harbor evaluations |

The shared `nvidia.fabric.nooa` adapter owns the common NOOA queue dispatcher.
It also maps normalized skill paths and whole MCP servers into compatible
registered targets. An MCP server that cannot be reached at runtime start is
skipped instead of failing startup and is reported under
`output.mcp.unavailable_servers`; see
[Configure MCP Servers](docs/interactive-agent.md#configure-mcp-servers).
The dedicated `nvidia.fabric.nooa.bench-agent` adapter maps the benchmark-native
`BenchAgent` task contract directly into a NeMo Fabric invocation. Its
[observed token usage](docs/bench-agent.md#observed-token-usage) preserves the
SDK's inclusive input convention and documents missing-request limitations.

## Install

NOOA requires Python 3.12 or 3.13. Install NeMo Fabric, the adapters, and the
tested NOOA packages together:

```bash
pip install "nemo-fabric[nooa]"
```

The `harness` and `full` extras constrain LiteLLM below 1.106 for NOOA and standalone BenchAgent runs. LiteLLM 1.106.4 and 1.106.5 import MCP types unavailable in MCP 1.x during ordinary tool dispatch. The tested 1.105.6 dispatch works without an MCP runtime.

For a split adapter environment, install
`nemo-fabric-adapters-nooa[harness]`. Use the adapter package without an extra
when the environment already manages compatible NOOA packages. The installed
wheel provides both adapter descriptors and the supported CodingAgent and ARC
target descriptors; no source-path or explicit discovery configuration is
required.

NOOA BenchAgent requires a Unix-like environment because its storage imports Python's `fcntl` module. Native Windows execution is unsupported.

## Configure Relay

Relay is optional. Install the adapter, tested NOOA packages, and compatible
Relay Python package together:

```bash
pip install "nemo-fabric-adapters-nooa[full]"
```

Use `nemo-fabric-adapters-nooa[relay]` when the environment already manages
the NOOA harness packages.

Session grouping requires the upstream propagation fix in
[NOOA PR #589](https://github.com/NVIDIA-NeMo/labs-OO-Agents/pull/589). The
current `harness` and `full` extras pin NOOA 0.0.10, which does not include that
fix. Fabric uses NOOA's agent-call middleware and retains only the tool-result
compatibility shim for Relay 0.11. Before merging or releasing this integration,
select compatible published NOOA and Relay versions, update their constraints
and the lockfile, and qualify the combined installation.

Both adapters declare Relay outputs for Agent Trajectory Interchange Format
(ATIF), OpenTelemetry, and OpenInference. NeMo Fabric supplies the generated
`FABRIC_RELAY_CONFIG_PATH`; the adapters reject ambient user or project plugin
configuration and activate only the generated document.

Both adapters accept `RunRequest.relay_session_root` to group ATIF trajectories
under one `session_id`. Each invocation keeps a distinct `trajectory_id` and
artifact. Without a usable session UUID, they use a usable UUID request ID or
preserve the existing Relay context; Relay creates a native root when no context
exists. `runtime_id` does not become an automatic conversation root. Per-span
OpenTelemetry attributes depend on the Relay exporter and are not guaranteed by
the adapter.

A Relay setup failure prevents agent execution and returns a failed result. If
artifact finalization fails after the agent completes, the adapter preserves
the functional result and marks telemetry as degraded. A leaked Relay scope
quarantines telemetry on later invocations.

Relay-backed `Runtime.invoke_stream()` returns raw Agent Trajectory
Observability Format (ATOF) records and a separate terminal result. Neither
adapter claims native model-response streaming.

### Strict Trace Capture

Strict receivers that require `session_id` need an explicit mapping from Relay's
native `gen_ai.conversation.id`. Add this mapping to the configured GenAI
endpoint, retaining its destination URL and authentication headers:

```json
{
  "type": "gen_ai",
  "attribute_mappings": [
    {"key": "gen_ai.conversation.id", "alias": "session_id"}
  ]
}
```

The Relay release must include both
[native-root conversation ID export](https://github.com/NVIDIA/NeMo-Relay/issues/1737)
and [GenAI attribute mapping support](https://github.com/NVIDIA/NeMo-Relay/pull/1613).
The declared Relay 0.11 range does not establish that these fixes are included;
a compatible release must be qualified before updating the package constraints.
The adapter does not inject session metadata or add this mapping to exporters.

## Choose an Environment

These adapters do not provide a sandbox. Agents run model-generated code and
shell commands with the permissions of their NeMo Fabric environment. Select an
environment provider with the isolation required by your workload.
