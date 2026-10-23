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
registered targets.
The dedicated `nvidia.fabric.nooa.bench-agent` adapter maps the benchmark-native
`BenchAgent` task contract directly into a NeMo Fabric invocation.

## Install

NOOA requires Python 3.12 or 3.13. Install NeMo Fabric, the adapters, and the
tested NOOA packages together:

```bash
pip install "nemo-fabric[nooa]"
```

For a split adapter environment, install
`nemo-fabric-adapters-nooa[harness]`. Use the adapter package without an extra
when the environment already manages compatible NOOA packages. The installed
wheel provides both adapter descriptors and the supported CodingAgent and ARC
target descriptors; no source-path or explicit discovery configuration is
required.

## Configure Relay

Relay is optional. Install the adapter, tested NOOA packages, and compatible
Relay Python package together:

```bash
pip install "nemo-fabric-adapters-nooa[full]"
```

Use `nemo-fabric-adapters-nooa[relay]` when the environment already manages
the NOOA harness packages.

Both adapters declare Relay outputs for Agent Trajectory Interchange Format
(ATIF), OpenTelemetry, and OpenInference. NeMo Fabric supplies the generated
`FABRIC_RELAY_CONFIG_PATH`; the adapters reject ambient user or project plugin
configuration and activate only the generated document.

A Relay setup failure prevents agent execution and returns a failed result. If
artifact finalization fails after the agent completes, the adapter preserves
the functional result and marks telemetry as degraded. A leaked Relay scope
quarantines telemetry on later invocations.

Relay-backed `Runtime.invoke_stream()` returns raw Agent Trajectory
Observability Format (ATOF) records and a separate terminal result. Neither
adapter claims native model-response streaming.

## Choose an Environment

These adapters do not provide a sandbox. Agents run model-generated code and
shell commands with the permissions of their NeMo Fabric environment. Select an
environment provider with the isolation required by your workload.


## Model usage

Both NOOA adapters collect calls through the model clients Fabric constructs.
For NOOA 0.0.10's managed JSON HTTP paths, the collector reads the original
response before LiteLLM can fill omitted fields with zero. It observes both
Completion and Responses clients, including sync and async calls, retries,
history summaries and local jobs that use those clients.

An omitted count stays unknown; a reported zero stays zero. Failed execution
keeps the usage of completed calls. A request without usable usage, an
unfinished call, or a pending owned summary or queue job leaves the affected
complete totals unknown. Available counts remain in the declared
`nemo_fabric.usage_accounting.v1` extension as observations with accounting gaps.
The request endpoint establishes the usage field names, so cache and total
remain independent when both primary counts are omitted. Repeated response IDs
within one managed call are counted once; conflicting reports retain an
observation with a scope gap.
HTTP status alone never establishes zero usage. Usage freezes when the
invocation returns. Late work keeps its original owner and never enters a later
invocation.

The collector does not wait for, cancel or disable background work. Streaming
HTTP and clients that bypass NOOA's managed transport have a scope gap. Their
normalized counts are not promoted because LiteLLM can estimate missing stream
usage. Models independently constructed by a target or tool are outside
this collector's managed-client boundary. They must not be treated as covered
by the configured clients' totals. Calls during runtime construction occur
outside an invocation.

Accounting model identity comes from each HTTP request, including call-level
model overrides. The native provider identity is recorded only for the native
OpenAI and Anthropic hosts; custom routes do not inherit native catalog prices.
Provider-reported cost is not inferred from tokens or LiteLLM estimates. Shared
NeMo Fabric normalization preserves tokens independently of cost.
